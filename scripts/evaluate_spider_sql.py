#!/usr/bin/env python3
"""
Apples-to-apples-ish Spider SQL evaluation aligned with Syncode/IterGen prompt format.

Dataset: Spider (local checkout required for databases)
Metrics:
  - execution accuracy (simple result-set equality; best-effort)
  - execution success rate
  - avg tokens / time

Prompt formatting (matches the papers' examples):
  db_id: <db_id>
  db_info: # <table> ( <col1> , <col2> , ... )
  # <table2> ( ... )
  # <fk.table.col> = <fk2.table.col>
  ...

  question: <question> Only output the SQL query.
  SQL:

Methods:
  - standard: unconstrained HF generate (greedy)
  - constrained: grammar-constrained decoding using VerifiedDecoderAgent.CSDHelpers.ConstrainedGeneration
  - csd: run a compiled CSD strategy (GeneratedCSD.default__.MyCSDStrategy) from a synthesis run dir

Example:
  python scripts/evaluate_spider_sql.py --spider-root /path/to/spider \
    --method csd --run-dir outputs/generated-csd/runs/20260105_215059_4ee3a0 \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --device cuda --limit 50 \
    --grammar grammars/sql_syncode.lark
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_csd_with_grammar import create_lark_dafny_parser  # noqa: E402
from scripts.evaluate_csd_performance import create_huggingface_lm, _select_sql_token_ids  # noqa: E402
from parsers.schema_to_grammar import create_spider_schema_grammar  # noqa: E402


def _check_cuda_available() -> bool:
    """
    Safely check if CUDA is available and working.
    Returns False if CUDA is not available or if there are version mismatches.
    """
    try:
        import torch
    except Exception as e:
        # If torch import itself fails due to CUDA issues, catch it here
        error_str = str(e).lower()
        if "cuda" in error_str or "cudart" in error_str or "undefined symbol" in error_str:
            print(f"⚠️  PyTorch import failed due to CUDA version mismatch: {e}", flush=True)
            return False
        # Re-raise if it's a different import error
        raise
    
    try:
        if not torch.cuda.is_available():
            return False
        # Try to create a small tensor on CUDA to verify it works
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "cuda" in error_str or "cudart" in error_str or "undefined symbol" in error_str:
            print(f"⚠️  CUDA check failed (version mismatch): {e}", flush=True)
        else:
            print(f"⚠️  CUDA check failed: {e}", flush=True)
        return False


def _resolve_device(device: str) -> str:
    """
    Resolve device string, checking availability.
    """
    if device == "cuda":
        if not _check_cuda_available():
            raise RuntimeError("CUDA requested but not available or has version mismatch.")
        return "cuda"
    return device


def _load_spider(spider_root: Path, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if split not in {"dev", "train"}:
        raise ValueError("split must be one of: dev, train")

    data_path = spider_root / f"{split}.json"
    tables_path = spider_root / "tables.json"
    if not data_path.exists() or not tables_path.exists():
        raise FileNotFoundError(
            f"Spider files not found. Expected {data_path} and {tables_path}. "
            "Download Spider from https://github.com/taoyds/spider"
        )

    data = json.loads(data_path.read_text())
    tables = json.loads(tables_path.read_text())

    tables_by_db = {t["db_id"]: t for t in tables}
    return data, tables_by_db


def _format_db_info(table_meta: Dict[str, Any]) -> str:
    """
    Build the paper-style db_info section.
    """
    # Spider tables.json keys: table_names_original, column_names_original, foreign_keys
    # column_names_original: list of [table_id, col_name]
    db_id = table_meta["db_id"]
    table_names = table_meta["table_names_original"]

    cols_by_table: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for tid, col in table_meta["column_names_original"]:
        if tid == -1:
            continue  # "*"
        cols_by_table[tid].append(col)

    lines = [f"db_id: {db_id}", "db_info:"]
    for tid, tname in enumerate(table_names):
        cols = cols_by_table.get(tid, [])
        cols_str = " , ".join(cols)
        lines.append(f"# {tname} ( {cols_str} )")

    # foreign_keys: list of [col_idx1, col_idx2]
    # map col index -> (table, col)
    col_names = table_meta["column_names_original"]
    for c1, c2 in table_meta.get("foreign_keys", []):
        t1, col1 = col_names[c1]
        t2, col2 = col_names[c2]
        if t1 == -1 or t2 == -1:
            continue
        lines.append(f"# {table_names[t1]}.{col1} = {table_names[t2]}.{col2}")

    return "\n".join(lines)


def _make_prompt(question: str, db_info: str) -> str:
    return f"{db_info}\n\nquestion: {question} Only output the SQL query.\nSQL:\n"


def _is_incomplete_sql(sql: str) -> bool:
    """
    Detect if SQL query appears incomplete.
    Checks for common patterns that indicate truncation.
    """
    sql_upper = sql.upper().strip()
    if not sql_upper:
        return True
    
    # Check for incomplete patterns (with word boundaries to avoid false positives)
    incomplete_patterns = [
        sql_upper.endswith("WHERE"),
        sql_upper.endswith("WHERE "),
        sql_upper.endswith(">"),
        sql_upper.endswith("<"),
        sql_upper.endswith("="),
        sql_upper.endswith("!="),
        sql_upper.endswith("<>"),
        sql_upper.endswith("AND"),
        sql_upper.endswith("OR"),
        sql_upper.endswith("SELECT"),
        sql_upper.endswith("FROM"),
        sql_upper.endswith("GROUP BY"),
        sql_upper.endswith("ORDER BY"),
        sql_upper.endswith("HAVING"),
        sql_upper.endswith("("),
        sql_upper.endswith("(SELECT"),
        sql_upper.endswith("IN ("),
        sql_upper.endswith("NOT IN ("),
        sql_upper.endswith("EXISTS ("),
        sql_upper.endswith("NOT EXISTS ("),
        sql_upper.endswith("CASE"),
        sql_upper.endswith("WHEN"),
        sql_upper.endswith("THEN"),
        sql_upper.endswith("ELSE"),
    ]
    
    # Check for unbalanced parentheses (indicates truncation)
    open_parens = sql.count("(")
    close_parens = sql.count(")")
    if open_parens > close_parens:
        return True
    
    # Check for very short queries that are likely incomplete
    if len(sql.split()) < 5:
        return True
    
    # Check if query ends with incomplete string literal (odd number of quotes)
    single_quotes = sql.count("'") - sql.count("\\'")
    double_quotes = sql.count('"') - sql.count('\\"')
    if (single_quotes % 2 != 0) or (double_quotes % 2 != 0):
        return True
    
    # Check for incomplete subquery patterns
    if sql_upper.count("SELECT") > sql_upper.count("FROM"):
        # More SELECTs than FROMs suggests incomplete subquery
        return True
    
    return any(incomplete_patterns)


def _extract_sql(text: str) -> str:
    """
    Best-effort extraction of a SQL query:
    - strip
    - take first line up to double-newline if present
    - remove markdown fences if present
    """
    s = text.strip()
    # remove fenced code blocks
    if "```" in s:
        parts = s.split("```")
        # take the largest middle chunk if exists
        if len(parts) >= 3:
            s = parts[1]
            # strip optional "sql" language tag line
            s = s.lstrip()
            if s.lower().startswith("sql"):
                s = s[3:].lstrip()
    # stop condition used in Syncode paper: \n\n
    if "\n\n" in s:
        s = s.split("\n\n", 1)[0]
    # keep first statement only
    if ";" in s:
        s = s.split(";", 1)[0]
    return s.strip()


def _execute_sql(db_path: Path, sql: str) -> Tuple[bool, Optional[List[Tuple[Any, ...]]], Optional[str]]:
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows, None
    except sqlite3.OperationalError as e:
        return False, None, f"OperationalError: {str(e)}"
    except sqlite3.ProgrammingError as e:
        return False, None, f"ProgrammingError: {str(e)}"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def _normalize_rows(rows: List[Tuple[Any, ...]]) -> Counter:
    """
    Normalize results for comparison:
    - convert floats with rounding
    - keep tuples
    - compare as multiset (Counter)
    """
    normed = []
    for r in rows:
        out = []
        for v in r:
            if isinstance(v, float):
                out.append(round(v, 6))
            else:
                out.append(v)
        normed.append(tuple(out))
    return Counter(normed)


def _exec_match(db_path: Path, pred_sql: str, gold_sql: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns: (execution_success, execution_match, error_message)
    """
    ok_gold, gold_rows, gold_err = _execute_sql(db_path, gold_sql)
    if not ok_gold:
        # If gold fails, skip scoring as match.
        return False, False, f"Gold SQL failed: {gold_err}"
    ok_pred, pred_rows, pred_err = _execute_sql(db_path, pred_sql)
    if not ok_pred:
        return False, False, pred_err
    return True, _normalize_rows(pred_rows) == _normalize_rows(gold_rows), None


def _make_chatml_instruction(prompt_text: str) -> str:
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"


def _run_standard(model_name: str, device: str, prompt_text: str, max_new_tokens: int) -> Tuple[str, int, float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Resolve device (check CUDA availability)
    device = _resolve_device(device)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    instruction = _make_chatml_instruction(prompt_text)
    inputs = tok(instruction, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    end = time.time()
    gen = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
    return gen, tokens, end - start


def _load_compiled_modules(run_dir: Path):
    # Try to read compiled_dir from success_report.json first
    module_dir = None
    success_report = run_dir / "success_report.json"
    if success_report.exists():
        try:
            report_data = json.loads(success_report.read_text())
            compiled_dir = report_data.get("compiled_dir")
            if compiled_dir:
                module_dir = Path(compiled_dir)
                if not module_dir.exists():
                    module_dir = None  # Fall back to default
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fall back to default name if not found in report
    if module_dir is None:
        module_dir = run_dir / "generated_csd"
    
    # Also try common alternative names if default doesn't exist
    if not module_dir.exists():
        for alt_name in ["sql_csd", "json_csd", "gsm_csd"]:
            alt_dir = run_dir / alt_name
            if alt_dir.exists():
                module_dir = alt_dir
                break
    
    if not module_dir.exists():
        raise FileNotFoundError(f"Compiled module directory not found: {module_dir}")
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import _dafny  # type: ignore
    import VerifiedDecoderAgent  # type: ignore
    import GeneratedCSD  # type: ignore
    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def _setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: Optional[int],
    grammar_file: Path,
    tables_by_db: Dict[str, Any],
    use_schema_grammar: bool = True,
    use_8bit: bool = True,
    use_4bit: bool = False,
):
    """
    Load model and setup Dafny environment once. Returns reusable objects.
    
    If use_schema_grammar is True, creates a parser cache that will generate
    schema-specific grammars per database. Otherwise uses a single generic grammar.
    
    If vocab_size is None, uses full tokenizer vocabulary (slower but more accurate).
    
    Args:
        use_8bit: Use 8-bit quantization to reduce memory (default: True)
        use_4bit: Use 4-bit quantization for even lower memory (overrides use_8bit)
    """
    # Resolve device (check CUDA availability)
    device = _resolve_device(device)
    
    _dafny, VerifiedDecoderAgent, GeneratedCSD = _load_compiled_modules(run_dir)

    # Curated token set for SQL (ensures keywords exist as single tokens)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Determine effective vocab size (full vocab if None)
    effective_vocab_size = vocab_size if vocab_size is not None else len(tok)
    
    # If using schema grammar and vocab_size is set, extend token set with schema names
    if vocab_size is not None and use_schema_grammar:
        token_ids = _select_sql_token_ids_with_schema(tok, vocab_size, tables_by_db)
    elif vocab_size is not None:
        token_ids = _select_sql_token_ids(tok, vocab_size)
    else:
        token_ids = None  # Use full vocabulary
    
    # Try to create LM, retry with 4-bit quantization if OOM
    try:
        lm = create_huggingface_lm(model_name, device, effective_vocab_size, VerifiedDecoderAgent, _dafny, token_ids=token_ids, use_8bit=use_8bit, use_4bit=use_4bit)
    except Exception as e:
        if device == "cuda" and ("out of memory" in str(e).lower()) and not use_4bit:
            print(f"⚠️  CUDA OOM during model loading: {e}", flush=True)
            print("   Retrying with 4-bit quantization...", flush=True)
            lm = create_huggingface_lm(model_name, device, effective_vocab_size, VerifiedDecoderAgent, _dafny, token_ids=token_ids, use_8bit=False, use_4bit=True)
        else:
            raise
    
    # Parser cache for schema-specific grammars
    parser_cache: Dict[str, Any] = {}
    
    def get_parser_for_db(db_id: str):
        """Get or create a schema-aware parser for the given database."""
        if db_id in parser_cache:
            return parser_cache[db_id]
        
        cache_size = len(parser_cache)
        if use_schema_grammar and db_id in tables_by_db:
            # Generate schema-specific grammar
            db_meta = tables_by_db[db_id]
            num_tables = len(db_meta.get("table_names_original", []))
            num_cols = len([c for _, c in db_meta.get("column_names_original", []) if c != "*"])
            print(f"  [Grammar #{cache_size + 1}] Building schema-aware grammar for db_id='{db_id}' ({num_tables} tables, {num_cols} cols)...", end="", flush=True)
            schema_grammar = create_spider_schema_grammar(db_meta, str(grammar_file))
            # Count how many names are in the grammar
            import re
            name_matches = re.findall(r'"([^"]+)"', schema_grammar.split('name:')[1].split('\n')[0] if 'name:' in schema_grammar else '')
            num_grammar_names = len([n for n in name_matches if n and not n.startswith('\\')])
            print(f" grammar generated ({num_grammar_names} names), creating parser...", end="", flush=True)
            LarkDafnyParser = create_lark_dafny_parser(schema_grammar, VerifiedDecoderAgent, _dafny, start="start")
            print(" parser created.", flush=True)
        else:
            # Use generic grammar
            if cache_size == 0:  # Only log once for generic grammar
                print(f"  Using generic grammar (not schema-aware)...", flush=True)
            LarkDafnyParser = create_lark_dafny_parser(str(grammar_file), VerifiedDecoderAgent, _dafny, start="start")
        
        parser = LarkDafnyParser(lm._Tokens)
        parser_cache[db_id] = parser
        return parser

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "get_parser_for_db": get_parser_for_db,
        "parser_cache": parser_cache,  # Expose cache for stats
    }


def _select_sql_token_ids_with_schema(tokenizer, max_tokens: Optional[int], tables_by_db: Dict[str, Any]) -> Optional[List[int]]:
    """
    Build a token set that includes SQL keywords AND all table/column names from the schema.
    If max_tokens is None, returns None to indicate "use all tokens".

    PRIORITY ORDER:
    1. Schema names (tables, columns) - CRITICAL for correctness
    2. SQL keywords and operators
    3. Common SQL tokens (numbers, punctuation, etc.)
    """
    if max_tokens is None:
        return None  # Use full vocabulary

    vocab = tokenizer.get_vocab()

    # 1. Collect ALL schema names with variations (HIGHEST PRIORITY)
    schema_token_ids = set()
    all_schema_names = set()

    for db_meta in tables_by_db.values():
        # Table names
        for tname in db_meta.get("table_names_original", []):
            all_schema_names.add(tname)
            all_schema_names.add(tname.lower())
            all_schema_names.add(tname.upper())
            # Add capitalized version too
            all_schema_names.add(tname.capitalize())

        # Column names
        for _, col_name in db_meta.get("column_names_original", []):
            if col_name and col_name != "*":
                all_schema_names.add(col_name)
                all_schema_names.add(col_name.lower())
                all_schema_names.add(col_name.upper())
                all_schema_names.add(col_name.capitalize())

    # Find tokens for all schema names (with prefix variations)
    for name in all_schema_names:
        # Try various tokenizer prefix styles
        for prefix in ["", " ", "Ġ", "▁"]:
            token_str = f"{prefix}{name}"
            if token_str in vocab:
                schema_token_ids.add(vocab[token_str])

        # Also try substrings (for when names are split into multiple tokens)
        # This helps with names like "country_id" that might tokenize as ["country", "_", "id"]
        for i in range(len(name)):
            for j in range(i+1, min(i+10, len(name)+1)):  # Check substrings up to 10 chars
                substr = name[i:j]
                if substr in vocab:
                    schema_token_ids.add(vocab[substr])
                if f" {substr}" in vocab:
                    schema_token_ids.add(vocab[f" {substr}"])

    # 2. Get base SQL tokens (keywords, operators, etc.)
    base_ids = _select_sql_token_ids(tokenizer, max_tokens)

    # 3. Combine with SCHEMA TOKENS FIRST (prioritize schema names)
    combined = list(schema_token_ids) + [tid for tid in base_ids if tid not in schema_token_ids]

    # Return up to max_tokens, with schema names prioritized
    result = combined[:max_tokens]

    # Log how many schema tokens we included
    num_schema = len([tid for tid in result if tid in schema_token_ids])
    print(f"  Token selection: {num_schema} schema tokens + {len(result) - num_schema} SQL tokens = {len(result)} total", flush=True)

    return result


def _run_dafny_method(
    env: dict,
    method: str,
    prompt_text: str,
    max_steps: int,
    db_id: str,
    timeout_seconds: Optional[float] = None,
) -> Tuple[str, int, float]:
    """
    Run generation using pre-loaded environment with schema-aware parser.
    
    Args:
        timeout_seconds: Maximum time to allow for generation (None = no timeout)
                        Note: This is a soft limit that logs warnings but doesn't interrupt
    """
    _dafny = env["_dafny"]
    VerifiedDecoderAgent = env["VerifiedDecoderAgent"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    
    # Get schema-specific parser for this database
    parser = env["get_parser_for_db"](db_id)

    lm.instruction_text = _make_chatml_instruction(prompt_text)
    dafny_prompt = _dafny.SeqWithoutIsStrInference([])

    start_t = time.time()
    
    # Run generation (note: we can't easily interrupt Dafny-compiled code, so timeout is advisory)
    if method == "csd":
        out_tokens = GeneratedCSD.default__.MyCSDStrategy(lm, parser, dafny_prompt, max_steps)
    elif method == "constrained":
        out_tokens = VerifiedDecoderAgent.CSDHelpers.ConstrainedGeneration(lm, parser, dafny_prompt, max_steps)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    end_t = time.time()
    elapsed = end_t - start_t
    
    # Warn if timeout exceeded (soft check)
    if timeout_seconds is not None and elapsed > timeout_seconds:
        print(f"  ⚠️  Warning: Generation took {elapsed:.1f}s (exceeded timeout of {timeout_seconds}s)", flush=True)

    output_list = [str(t) for t in out_tokens]
    output_text = "".join(output_list)
    return output_text, len(output_list), elapsed


@dataclass
class Metrics:
    n: int = 0
    exec_success: int = 0
    exec_match: int = 0
    total_tokens: int = 0
    total_time: float = 0.0


def main():
    ap = argparse.ArgumentParser(description="Evaluate on Spider with execution accuracy")
    ap.add_argument("--spider-root", type=Path, required=True, help="Path to Spider repo root (contains dev.json and database/)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--method", choices=["standard", "constrained", "csd"], required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="Required for methods: constrained/csd")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda", help="Device to use: 'cuda', 'cpu', or 'auto' (auto-detect)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="For standard")
    ap.add_argument("--max-steps", type=int, default=512, help="For constrained/csd (default: 512, increased from 256 for complex SQL queries)")
    ap.add_argument("--vocab-size", type=int, default=2000, help="Token vocab size limit (default: 2000 for better performance; None = use all tokens, but very slow). Lower values (1500-2000) are faster.")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "sql_no_subquery.lark")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--schema-aware", action="store_true", default=True,
                    help="Use schema-aware grammars that constrain table/column names (default: True)")
    ap.add_argument("--no-schema-aware", action="store_false", dest="schema_aware",
                    help="Disable schema-aware grammars (use generic grammar - may help diagnose accuracy issues)")
    ap.add_argument("--timeout", type=float, default=600.0, 
                    help="Maximum seconds per query (default: 600 = 10 minutes, set to 0 for no timeout)")
    ap.add_argument("--no-8bit", action="store_true", 
                    help="Disable 8-bit quantization (uses more memory but may be faster)")
    ap.add_argument("--use-4bit", action="store_true",
                    help="Use 4-bit quantization (lowest memory, requires bitsandbytes)")
    args = ap.parse_args()

    if args.method in {"constrained", "csd"} and args.run_dir is None:
        ap.error("--run-dir is required for method constrained/csd")

    data, tables_by_db = _load_spider(args.spider_root, args.split)
    n = min(args.limit, len(data))
    metrics = Metrics()

    # Auto-detect device if requested
    if args.device == "auto":
        if _check_cuda_available():
            args.device = "cuda"
            print("Auto-detected CUDA device", flush=True)
        else:
            raise RuntimeError("Auto-detection failed: CUDA not available. Please specify --device cpu explicitly if you want to use CPU.")
    else:
        # Resolve device early to show errors before model loading
        args.device = _resolve_device(args.device)

    # Load model once before the loop (for constrained/csd methods)
    dafny_env = None
    if args.method in {"constrained", "csd"}:
        print("Setting up Dafny environment and loading model (once)...")
        print(f"Configuration:")
        print(f"  Max steps: {args.max_steps} (increase with --max-steps if queries are truncated)")
        print(f"  Timeout: {args.timeout}s per query (0 = no timeout)")
        if args.schema_aware:
            unique_dbs = len(set(ex.get("db_id") for ex in data[:n] if ex.get("db_id") in tables_by_db))
            print(f"  Schema-aware: YES (will generate grammars for ~{unique_dbs} unique databases)")
            print(f"    → If generation is too slow, try --no-schema-aware for faster (but less accurate) generation")
        else:
            print(f"  Schema-aware: NO (using generic grammar - faster but less accurate)")
        if args.vocab_size is None:
            print("⚠️  Using full tokenizer vocabulary (no token limit) - this will be VERY SLOW!")
            print("   Consider using --vocab-size 2000 for better performance")
        else:
            print(f"  Vocab size: {args.vocab_size} tokens")
            if args.vocab_size > 2500:
                print(f"  ⚠️  Large vocab size ({args.vocab_size}) may be slow. Consider reducing to 1500-2000 for faster inference.")
        
        if args.device == "cuda":
            if args.use_4bit:
                print("  Quantization: 4-bit (lowest memory)")
            elif not args.no_8bit:
                print("  Quantization: 8-bit (reduces memory by ~50%)")
            else:
                print("  Quantization: None (uses more memory)")
                print("  💡 Tip: Use --use-4bit or remove --no-8bit for lower memory usage")
        dafny_env = _setup_dafny_environment(
            run_dir=args.run_dir,
            model_name=args.model,
            device=args.device,
            vocab_size=args.vocab_size,
            grammar_file=args.grammar,
            tables_by_db=tables_by_db,
            use_schema_grammar=args.schema_aware,
        )
        print("Model loaded. Starting evaluation...\n")

    # Track start time for ETA calculation
    eval_start_time = time.time()
    
    for i in range(n):
        ex = data[i]
        db_id = ex["db_id"]
        question = ex.get("question") or ex.get("utterance") or ex.get("query") or ""
        gold_sql = ex.get("query") or ex.get("sql") or ""
        if not question or not gold_sql:
            continue

        table_meta = tables_by_db.get(db_id)
        if table_meta is None:
            continue
        
        print(f"[{i+1}/{n}] Processing db_id='{db_id}'...", flush=True)
        db_info = _format_db_info(table_meta)
        prompt = _make_prompt(question, db_info)

        if args.method == "standard":
            out_text, tok_count, dt = _run_standard(args.model, args.device, prompt, args.max_new_tokens)
        else:
            timeout = args.timeout if args.timeout > 0 else None
            out_text, tok_count, dt = _run_dafny_method(
                env=dafny_env,
                method=args.method,
                prompt_text=prompt,
                max_steps=args.max_steps,
                db_id=db_id,
                timeout_seconds=timeout,
            )

        pred_sql = _extract_sql(out_text)
        
        # Check if SQL appears incomplete
        is_incomplete = _is_incomplete_sql(pred_sql)
        
        # Additional check: if very few tokens and query looks incomplete, mark as incomplete
        if tok_count < 15 and is_incomplete:
            # This is likely a parser/grammar issue causing early stopping
            # Check if query ends with a table name (common pattern when parser rejects continuations)
            sql_upper = pred_sql.upper().strip()
            if sql_upper.endswith("FROM") or (sql_upper.count("FROM") == 1 and not any(c in sql_upper for c in ["WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", ";"])):
                # Query has FROM but no clauses - parser likely returned empty ValidNextTokens
                is_incomplete = True

        db_path = args.spider_root / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            # Some Spider layouts use .db extension; try that too
            alt = args.spider_root / "database" / db_id / f"{db_id}.db"
            db_path = alt if alt.exists() else db_path

        ok_exec, ok_match, exec_error = _exec_match(db_path, pred_sql, gold_sql)
        
        # If query is incomplete, mark execution as failed
        if is_incomplete and ok_exec:
            ok_exec = False
            exec_error = "OperationalError: incomplete input (query appears truncated)"

        metrics.n += 1
        metrics.total_tokens += tok_count
        metrics.total_time += dt
        metrics.exec_success += 1 if ok_exec else 0
        metrics.exec_match += 1 if ok_match else 0

        # Calculate ETA
        avg_time_per_example = metrics.total_time / max(1, metrics.n)
        remaining_examples = n - metrics.n
        eta_seconds = avg_time_per_example * remaining_examples
        
        # Format ETA nicely
        def format_time(seconds):
            """Format seconds into human-readable time string."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        eta_str = format_time(eta_seconds) if remaining_examples > 0 else "done"
        elapsed = time.time() - eval_start_time
        elapsed_str = format_time(elapsed)

        # Print progress summary with ETA
        acc = 100.0 * metrics.exec_match / max(1, metrics.n)
        status_parts = [f"Generated {tok_count} tokens in {dt:.2f}s", f"Exec: {ok_exec}", f"Match: {ok_match}", f"Acc: {acc:.1f}%", f"Elapsed: {elapsed_str}", f"ETA: {eta_str}"]
        print(f"  -> {' | '.join(status_parts)}", flush=True)
        
        # Show failure details (always show for first few failures, or if verbose)
        # Show more details to diagnose accuracy issues
        if not ok_exec or not ok_match:
            # Show details for first 10 failures, or all if verbose
            show_details = args.verbose or metrics.n <= 10 or (metrics.n <= 30 and not ok_exec)
            if show_details:
                print(f"  ❌ FAILURE DETAILS:")
                print(f"     Question: {question[:100]}..." if len(question) > 100 else f"     Question: {question}")
                print(f"     Generated SQL: {pred_sql[:200]}..." if len(pred_sql) > 200 else f"     Generated SQL: {pred_sql}")
                print(f"     Expected SQL: {gold_sql[:200]}..." if len(gold_sql) > 200 else f"     Expected SQL: {gold_sql}")
                if exec_error:
                    print(f"     Execution error: {exec_error}")
                if is_incomplete:
                    print(f"     ⚠️  Query appears incomplete (truncated)")
                if tok_count < 10:
                    print(f"     ⚠️  Very few tokens generated ({tok_count}) - query may be incomplete")
                    print(f"     💡 Possible causes:")
                    print(f"        - Parser ValidNextTokens returned empty (no valid continuations)")
                    print(f"        - CSD strategy stopped early due to grammar constraints")
                    print(f"        - Tokenizer splitting table/column names incorrectly")
                    if args.method in {"constrained", "csd"}:
                        print(f"     💡 Try: --no-schema-aware (if using schema-aware) or increase --vocab-size")
                if dt > 30:
                    print(f"     ⚠️  Very slow generation ({dt:.1f}s) - possible grammar validation bottleneck")
                    print(f"     💡 ValidNextTokens checks all {args.vocab_size if args.vocab_size else 'vocab'} tokens per step")
                if tok_count >= args.max_steps * 0.9:
                    print(f"     ⚠️  Hit max_steps limit ({tok_count}/{args.max_steps}) - query may be truncated, consider increasing --max-steps")
                if dt > 600:
                    print(f"     ⚠️  Extremely slow generation ({dt/60:.1f} minutes) - consider using --no-schema-aware or reducing --vocab-size")
                # Check for specific patterns that suggest parser issues
                if "FROM" in pred_sql.upper() and not any(c in pred_sql.upper() for c in ["WHERE", "GROUP", "ORDER", "LIMIT", "HAVING"]):
                    # Query has FROM but no clauses - might be parser rejecting valid continuations
                    if tok_count < 20:
                        print(f"     💡 Query has FROM but stopped early - parser may be rejecting valid table/column names")
                print()

        if args.verbose and (not ok_match):
            print("=" * 80)
            print(f"[{i}] db_id={db_id}")
            print(f"Q: {question}")
            print(f"PRED: {pred_sql}")
            print(f"GOLD: {gold_sql}")
            print(f"exec_success={ok_exec} exec_match={ok_match}")

    print("\n" + "=" * 60)
    print("SPIDER SQL RESULTS (best-effort execution match)")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    if args.method != "standard":
        print(f"Run dir: {args.run_dir}")
        print(f"Grammar: {args.grammar}")
        print(f"Schema-aware: {args.schema_aware}")
        if dafny_env and "parser_cache" in dafny_env:
            parser_cache_size = len(dafny_env["parser_cache"])
            if parser_cache_size > 0:
                print(f"Unique grammars generated: {parser_cache_size}")
    print(f"Examples scored: {metrics.n}")
    print(f"Exec success (%): {100.0 * metrics.exec_success / max(1, metrics.n):.1f}")
    print(f"Exec accuracy (%): {100.0 * metrics.exec_match / max(1, metrics.n):.1f}")
    print(f"Avg tokens: {metrics.total_tokens / max(1, metrics.n):.1f}")
    print(f"Avg time (s): {metrics.total_time / max(1, metrics.n):.2f}")


if __name__ == "__main__":
    main()


