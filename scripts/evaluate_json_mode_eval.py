#!/usr/bin/env python3
"""
Apples-to-apples JSON evaluation aligned with Syncode paper:

- Dataset: JSON-Mode-Eval (NousResearch/json-mode-eval)
- Metric: JSON schema validation accuracy (and JSON parse validity)
- Decoding: greedy

Methods:
- standard: unconstrained HF generate
- constrained: grammar-constrained decoding using VerifiedDecoderAgent.CSDHelpers.ConstrainedGeneration
- csd: run a compiled CSD strategy (GeneratedCSD.default__.MyCSDStrategy) from a synthesis run dir

Example:
  python scripts/evaluate_json_mode_eval.py --method csd \
    --run-dir outputs/generated-csd/runs/20260105_215059_4ee3a0 \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --device cuda --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_csd_with_grammar import create_lark_dafny_parser  # noqa: E402
from scripts.evaluate_csd_performance import create_huggingface_lm  # noqa: E402
from parsers.schema_to_grammar import json_schema_to_lark_grammar  # noqa: E402


def _load_json_mode_eval(split: str):
    try:
        from datasets import load_dataset, get_dataset_split_names
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    available = []
    try:
        available = list(get_dataset_split_names("NousResearch/json-mode-eval"))
    except Exception:
        available = []

    # Many HF datasets expose only 'train'. If user requested 'test' but it's missing,
    # fall back to 'train' for convenience.
    if available and split not in available:
        if split == "test" and "train" in available:
            split = "train"
        else:
            raise ValueError(f'Unknown split "{split}". Should be one of {available}.')

    return load_dataset("NousResearch/json-mode-eval", split=split)


def _load_jsonschema_validator(schema_obj: Any):
    try:
        import jsonschema
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `jsonschema`. Install with: pip install jsonschema"
        ) from e
    return jsonschema.Draft7Validator(schema_obj)


def _extract_first_json(text: str) -> Optional[str]:
    """
    Best-effort extraction:
    - Finds first '{' or '['
    - Scans forward counting braces/brackets, respecting string literals.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue
        if c in "{[":
            stack.append("}" if c == "{" else "]")
            continue
        if c in "}]":
            if not stack or c != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : j + 1]
    return None


def _get_prompt_and_schema(example: Dict[str, Any]) -> Tuple[str, Any]:
    """
    JSON-Mode-Eval examples can vary slightly; we try common fields.
    """
    # Common dataset formats:
    # - {"prompt": "...", "schema": {...}}
    # - {"messages": [...], "schema": {...}}
    if "prompt" in example:
        prompt = example["prompt"]
    elif "messages" in example:
        # Join messages in a simple chat format.
        # Keep exact content to avoid changing semantics.
        msg_lines = []
        for m in example["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            msg_lines.append(f"{role}:\n{content}")
        prompt = "\n\n".join(msg_lines)
    else:
        raise KeyError(f"Unrecognized JSON-Mode-Eval example keys: {list(example.keys())}")

    schema = example.get("schema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            pass
    if schema is None:
        # Some variants store it as string
        schema_str = example.get("json_schema") or example.get("schema_str")
        if schema_str is not None:
            schema = json.loads(schema_str)
    if schema is None:
        raise KeyError(f"Schema missing in example keys: {list(example.keys())}")
    return prompt, schema


@dataclass
class Result:
    ok_json: bool
    ok_schema: bool
    output_text: str
    extracted_json: Optional[str]
    tokens: int
    time_s: float


def _make_chatml_instruction(prompt_text: str) -> str:
    # Keep it simple for Qwen-like chat models. Works reasonably for base models too.
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"


def _load_model_standard(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading standard model: {model_name} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    return model, tok


def _run_standard(model, tok, device: str, prompt_text: str, max_new_tokens: int) -> Tuple[str, int, float]:
    import torch
    
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
    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        raise FileNotFoundError(f"Compiled module directory not found: {module_dir}")
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import _dafny  # type: ignore
    import VerifiedDecoderAgent  # type: ignore
    import GeneratedCSD  # type: ignore
    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def _load_resources_csd(run_dir: Path, model_name: str, device: str, vocab_size: int, grammar_file: Path):
    print(f"Loading CSD resources from {run_dir}...")
    _dafny, VerifiedDecoderAgent, GeneratedCSD = _load_compiled_modules(run_dir)
    grammar_source = str(grammar_file)
    LarkDafnyParser = create_lark_dafny_parser(grammar_source, VerifiedDecoderAgent, _dafny, start="start")
    
    print(f"Loading model wrapper: {model_name} on {device}...")
    lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny)
    parser = LarkDafnyParser(lm._Tokens)
    
    return lm, parser, _dafny, VerifiedDecoderAgent, GeneratedCSD


def _load_resources_csd_base(run_dir: Path, model_name: str, device: str, vocab_size: int):
    """Load CSD resources without creating a parser (for schema-specific grammar mode)."""
    print(f"Loading CSD resources from {run_dir}...")
    _dafny, VerifiedDecoderAgent, GeneratedCSD = _load_compiled_modules(run_dir)
    
    print(f"Loading model wrapper: {model_name} on {device}...")
    lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny)
    
    return lm, _dafny, VerifiedDecoderAgent, GeneratedCSD


def _create_schema_parser(schema: Dict[str, Any], lm, VerifiedDecoderAgent, _dafny) -> Any:
    """Create a parser from a JSON schema by generating a schema-specific grammar."""
    try:
        grammar = json_schema_to_lark_grammar(schema)
        LarkDafnyParser = create_lark_dafny_parser(grammar, VerifiedDecoderAgent, _dafny, start="start")
        return LarkDafnyParser(lm._Tokens)
    except Exception as e:
        # Fallback to generic JSON grammar if schema conversion fails
        print(f"  Warning: Schema grammar generation failed ({e}), using generic JSON")
        return None


def _wrap_parser_with_min_tokens(parser, min_tokens: int, VerifiedDecoderAgent, _dafny):
    """Wrap a parser to prevent early closing braces and require minimum tokens."""
    if min_tokens <= 0:
        return parser
    
    class MinTokenParser(VerifiedDecoderAgent.Parser):
        """Parser wrapper that prevents early completion by filtering closing tokens."""
        
        def __init__(self, inner_parser, min_toks):
            super().__init__()
            self._inner = inner_parser
            self._min_tokens = min_toks
            # Tokens that close structures - we'll filter these early on
            self._closing_tokens = {'}', ']', '"}', '"]', ' }', ' ]', '}\n', ']\n'}
        
        def IsValidPrefix(self, prefix) -> bool:
            return self._inner.IsValidPrefix(prefix)
        
        def IsCompletePrefix(self, prefix) -> bool:
            # Only consider complete if we have enough tokens AND it's grammatically complete
            if len(prefix) < self._min_tokens:
                return False
            return self._inner.IsCompletePrefix(prefix)
        
        def ValidNextTokens(self, prefix):
            valid = self._inner.ValidNextTokens(prefix)
            
            # If we haven't reached min tokens, filter out closing tokens
            if len(prefix) < self._min_tokens - 5:  # Allow closing near the end
                filtered = []
                for tok in valid:
                    tok_str = str(tok).strip()
                    # Keep token if it's not purely a closing token
                    if tok_str not in self._closing_tokens and tok_str not in ['}', ']']:
                        filtered.append(tok)
                
                # Only use filtered if we have alternatives
                if filtered:
                    return _dafny.SeqWithoutIsStrInference(filtered)
            
            return valid
    
    return MinTokenParser(parser, min_tokens)


def _run_dafny_method(
    lm, parser, _dafny, VerifiedDecoderAgent, GeneratedCSD,
    method: str,
    prompt_text: str,
    max_steps: int,
    strip_whitespace: bool = False,
) -> Tuple[str, int, float]:

    lm.instruction_text = _make_chatml_instruction(prompt_text)
    dafny_prompt = _dafny.SeqWithoutIsStrInference([])

    start_t = time.time()
    if method == "csd":
        out_tokens = GeneratedCSD.default__.MyCSDStrategy(lm, parser, dafny_prompt, max_steps)
    elif method == "constrained":
        out_tokens = VerifiedDecoderAgent.CSDHelpers.ConstrainedGeneration(lm, parser, dafny_prompt, max_steps)
    else:
        raise ValueError(f"Unknown method: {method}")
    end_t = time.time()

    output_list = [str(t) for t in out_tokens]
    
    if strip_whitespace:
        # Strip leading/trailing whitespace from non-whitespace tokens
        # This fixes the issue where tokens like " s" become "s" in the output
        cleaned_list = []
        for tok in output_list:
            if tok.strip():  # Non-whitespace token
                cleaned_list.append(tok.strip())
            # Skip whitespace-only tokens (they're just grammar padding)
        output_text = "".join(cleaned_list)
    else:
        output_text = "".join(output_list)
    
    return output_text, len(output_list), end_t - start_t


def main():
    ap = argparse.ArgumentParser(description="Evaluate JSON-Mode-Eval with schema validation")
    ap.add_argument("--method", choices=["standard", "constrained", "csd"], required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="Required for methods: constrained/csd")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="train", help="HF split name (NousResearch/json-mode-eval is typically just 'train')")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="For standard")
    ap.add_argument("--max-steps", type=int, default=256, help="For constrained/csd")
    ap.add_argument("--vocab-size", type=int, default=1000, help="For constrained/csd")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "json.lark")
    ap.add_argument("--schema-grammar", action="store_true",
                    help="Generate schema-specific grammar for each example (experimental)")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Minimum tokens before allowing completion (prevents {} outputs)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.method in {"constrained", "csd"} and args.run_dir is None:
        ap.error("--run-dir is required for method constrained/csd")

    ds = _load_json_mode_eval(args.split)
    n = min(args.limit, len(ds))

    # Pre-load model resources ONCE to avoid OOM
    model_standard = None
    tok_standard = None
    csd_resources = None
    csd_base_resources = None  # For schema-grammar mode

    if args.method == "standard":
        model_standard, tok_standard = _load_model_standard(args.model, args.device)
    elif args.schema_grammar:
        # Load base resources without parser (will create per-schema parsers)
        csd_base_resources = _load_resources_csd_base(
            args.run_dir, args.model, args.device, args.vocab_size
        )
        # Also load generic parser as fallback
        lm, _dafny, VerifiedDecoderAgent, GeneratedCSD = csd_base_resources
        generic_grammar = str(args.grammar)
        LarkDafnyParser = create_lark_dafny_parser(generic_grammar, VerifiedDecoderAgent, _dafny, start="start")
        generic_parser = LarkDafnyParser(lm._Tokens)
        csd_resources = (lm, generic_parser, _dafny, VerifiedDecoderAgent, GeneratedCSD)
    else:
        csd_resources = _load_resources_csd(args.run_dir, args.model, args.device, args.vocab_size, args.grammar)

    ok_json = 0
    ok_schema = 0
    total = 0
    total_tokens = 0
    total_time = 0.0

    print(f"Starting evaluation of {n} examples...")

    for idx in range(n):
        ex = ds[idx]
        prompt, schema = _get_prompt_and_schema(ex)
        validator = _load_jsonschema_validator(schema)

        if args.method == "standard":
            out_text, tok_count, dt = _run_standard(model_standard, tok_standard, args.device, prompt, args.max_new_tokens)
        else:
            lm, generic_parser, _dafny, VerifiedDecoderAgent, GeneratedCSD = csd_resources
            
            # Use schema-specific parser if enabled
            if args.schema_grammar:
                schema_parser = _create_schema_parser(schema, lm, VerifiedDecoderAgent, _dafny)
                parser = schema_parser if schema_parser is not None else generic_parser
            else:
                parser = generic_parser
            
            # Apply minimum token requirement if specified
            if args.min_tokens > 0:
                parser = _wrap_parser_with_min_tokens(parser, args.min_tokens, VerifiedDecoderAgent, _dafny)
            
            # Use whitespace stripping for schema-specific grammars
            strip_ws = args.schema_grammar
            out_text, tok_count, dt = _run_dafny_method(
                lm, parser, _dafny, VerifiedDecoderAgent, GeneratedCSD,
                args.method, prompt, args.max_steps, strip_whitespace=strip_ws
            )

        extracted = _extract_first_json(out_text)
        parsed = None
        is_json = False
        is_schema = False
        if extracted is not None:
            try:
                parsed = json.loads(extracted)
                is_json = True
            except Exception:
                is_json = False

        if is_json:
            try:
                errs = list(validator.iter_errors(parsed))
                is_schema = len(errs) == 0
            except Exception:
                is_schema = False

        total += 1
        total_tokens += tok_count
        total_time += dt
        ok_json += 1 if is_json else 0
        ok_schema += 1 if is_schema else 0

        if args.verbose and (not is_schema):
            print("=" * 60)
            print(f"Example {idx}")
            print(f"JSON OK: {is_json} | Schema OK: {is_schema}")
            print(f"Output (first 400): {repr(out_text[:400])}")
            print(f"Extracted: {repr(extracted[:200]) if extracted else None}")
        elif (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{n} examples...")

    print("\n" + "=" * 60)
    print("JSON-MODE-EVAL RESULTS")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    if args.method != "standard":
        print(f"Run dir: {args.run_dir}")
        if args.schema_grammar:
            print(f"Grammar: schema-specific (per-example)")
        else:
            print(f"Grammar: {args.grammar}")
        if args.min_tokens > 0:
            print(f"Min tokens: {args.min_tokens}")
    print(f"Examples: {total}")
    print(f"Valid JSON (%): {100.0 * ok_json / max(1,total):.1f}")
    print(f"Schema valid (%): {100.0 * ok_schema / max(1,total):.1f}")
    print(f"Avg tokens: {total_tokens / max(1,total):.1f}")
    print(f"Avg time (s): {total_time / max(1,total):.2f}")


if __name__ == "__main__":
    main()
