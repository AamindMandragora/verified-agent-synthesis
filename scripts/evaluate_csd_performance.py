#!/usr/bin/env python3
"""
Performance evaluation script for Generated CSD.

Compares the generated CSD strategy against reported baselines (Syncode, IterGen)
using real HuggingFace models and mini-benchmarks.

Usage:
    python scripts/evaluate_csd_performance.py --run-dir outputs/generated-csd/runs/XXXXX --task json --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import sys
import time
import json
import torch
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the grammar runner utilities
from scripts.run_csd_with_grammar import (
    create_lark_dafny_parser,
    get_builtin_grammar,
    create_vocabulary
)

# --- Baseline Data (from papers) ---

@dataclass
class BaselineMetrics:
    accuracy: float  # Validity/Syntax accuracy
    execute_rate: Optional[float]  # Execution success (for SQL)
    tokens: float
    time: float

BASELINES = {
    "json": {
        "syncode": BaselineMetrics(accuracy=99.0, execute_rate=None, tokens=100.0, time=1.2), # Approx from paper
        "itergen": BaselineMetrics(accuracy=100.0, execute_rate=None, tokens=95.0, time=1.3),  # Approx
        "standard": BaselineMetrics(accuracy=41.0, execute_rate=None, tokens=150.0, time=0.8), # Standard/Baseline
    },
    "sql": {
        # Using Qwen2.5-1.5B numbers from IterGen paper Table 1
        "syncode": BaselineMetrics(accuracy=48.9, execute_rate=79.0, tokens=35.48, time=0.81),
        "itergen": BaselineMetrics(accuracy=49.7, execute_rate=81.5, tokens=42.41, time=1.14),
        "standard": BaselineMetrics(accuracy=48.2, execute_rate=78.1, tokens=35.79, time=0.64),
    }
}

# --- Mini-Benchmarks ---

JSON_PROMPTS = [
    "Generate a JSON object for a user profile with name, age, and email.",
    "Create a list of 3 items in JSON format, where each item has id and value.",
    "Output a JSON configuration object with debug(bool), timeout(int), and servers(list of strings).",
    "Generate a nested JSON object representing a file system structure.",
    "Create a JSON response with status, code, and a data object containing results."
]

SQL_PROMPTS = [
    "Given schema: table users(id, name, age). Write a SQL query to find all users older than 25.",
    "Given schema: table employees(id, dept_id, salary), table departments(id, name). Find average salary per department.",
    "Given schema: table products(id, name, price). Find the most expensive product.",
    "Given schema: table orders(id, customer_id, total). Count orders per customer.",
    "Given schema: table students(id, name, grade). Find students with grade > 90."
]


# --- Real LM Wrapper ---

def create_huggingface_lm(model_name: str, device: str, vocab_size: int, VerifiedDecoderAgent, _dafny, token_ids: Optional[List[int]] = None, use_8bit: bool = True, use_4bit: bool = False):
    """Create a Dafny-compatible LM backed by a real HuggingFace model.
    
    Args:
        use_8bit: Use 8-bit quantization (reduces memory by ~50%)
        use_4bit: Use 4-bit quantization (reduces memory by ~75%, overrides use_8bit)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name} on {device}...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Determine quantization settings
    load_kwargs = {
        "trust_remote_code": True,
    }
    
    if device == "cuda":
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                load_kwargs["device_map"] = "auto"
                print(" (4-bit quantized)", flush=True)
            except ImportError:
                print(" (bitsandbytes not available, falling back to 8-bit)", flush=True)
                use_4bit = False
                use_8bit = True
        elif use_8bit:
            try:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
                print(" (8-bit quantized)", flush=True)
            except Exception:
                print(" (8-bit not available, using float16)", flush=True)
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            print(" (float16)", flush=True)
    else:
        load_kwargs["torch_dtype"] = torch.float32
        print(" (float32, CPU)", flush=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device == "cuda" and "device_map" not in load_kwargs:
            model = model.to(device)
    except Exception as e:
        if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
            print(f"\n⚠️  CUDA OOM during model loading: {e}", flush=True)
            if device == "cuda" and not use_4bit:
                print("   Retrying with 4-bit quantization...", flush=True)
                try:
                    from transformers import BitsAndBytesConfig
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        ),
                        device_map="auto"
                    )
                    print("   ✓ Loaded with 4-bit quantization", flush=True)
                except Exception as e2:
                    print(f"   ⚠️  4-bit also failed: {e2}", flush=True)
                    print("   Falling back to CPU...", flush=True)
                    device = "cpu"
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    ).to(device)
            else:
                raise
        else:
            raise
    
    model.eval()

    def _build_token_set(token_ids_local: Optional[List[int]] = None):
        """
        Build a compact Dafny token set and mapping to model token ids.
        If token_ids is None, default to first vocab_size tokenizer ids.
        """
        if token_ids_local is None:
            token_ids_local = list(range(min(len(tokenizer), vocab_size)))
        # Dedupe while preserving order
        seen = set()
        token_ids_local = [i for i in token_ids_local if isinstance(i, int) and 0 <= i < len(tokenizer) and (i not in seen and not seen.add(i))]
        token_ids_local = token_ids_local[:vocab_size]

        dafny_tokens_local: List[str] = []
        model_id_to_dafny_idx_local: Dict[int, int] = {}
        for model_id in token_ids_local:
            try:
                text = tokenizer.decode([model_id])
            except Exception:
                text = f"<UNK_{model_id}>"
            dafny_tokens_local.append(text)
            model_id_to_dafny_idx_local[model_id] = len(dafny_tokens_local) - 1
        return dafny_tokens_local, model_id_to_dafny_idx_local

    class HuggingFaceLM(VerifiedDecoderAgent.LM):
        def __init__(self):
            super().__init__()
            self._Tokens = _dafny.SeqWithoutIsStrInference(dafny_tokens)
            self._Ids = _dafny.SeqWithoutIsStrInference(list(range(len(dafny_tokens))))
            self.Logits = _dafny.Array(None, len(dafny_tokens))
            for i in range(len(dafny_tokens)):
                self.Logits[i] = _dafny.BigRational(0)
            
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            # Determine actual device (for quantized models with device_map="auto")
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                # Model is split across devices, use first device
                first_device = list(model.hf_device_map.values())[0]
                if isinstance(first_device, (int, str)):
                    self.actual_device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device
                else:
                    self.actual_device = device
            elif hasattr(model, "device"):
                self.actual_device = str(model.device)
            else:
                self.actual_device = device
            self.model_id_to_dafny_idx = model_id_to_dafny_idx
            self.instruction_text = ""
            
        def GenerateLogits(self, input_prefix_tokens):
            """
            Run model forward pass on prefix.
            input_prefix_tokens: Dafny sequence of strings
            """
            # Convert Dafny tokens back to a string prompt
            prefix_text = ""
            # Iterate using range/length for Dafny sequence
            length = len(input_prefix_tokens)
            for i in range(length):
                prefix_text += str(input_prefix_tokens[i])
            
            full_text = self.instruction_text + prefix_text
            
            # Use actual device for inputs
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.actual_device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]
            
            # Reset logits to very low value
            min_val = _dafny.BigRational(-1e9)
            for k in range(self.Logits.length(0)):
                self.Logits[k] = min_val
            
            # Map model logits to Dafny logits
            for model_id, dafny_idx in self.model_id_to_dafny_idx.items():
                if model_id < len(next_token_logits):
                    val = float(next_token_logits[model_id].cpu().item())
                    self.Logits[dafny_idx] = _dafny.BigRational(val)

        def ChooseNextToken(self):
            """Greedy decoding (or sample) from Logits."""
            best_idx = 0
            best_logit = -1e10
            
            # Helper to read BigRational
            def br_to_float(br):
                s = str(br)
                if '/' in s:
                    n, d = s.split('/')
                    return float(n) / float(d)
                return float(s)

            for i in range(self.Logits.length(0)):
                val = br_to_float(self.Logits[i])
                if val > best_logit:
                    best_logit = val
                    best_idx = i
            
            return self._Tokens[best_idx]

    # Token set (possibly curated)
    dafny_tokens, model_id_to_dafny_idx = _build_token_set(token_ids)
    return HuggingFaceLM()


def _select_sql_token_ids(tokenizer, max_tokens: int) -> List[int]:
    """
    Build a curated token set that ensures SQL keywords like SELECT/FROM/WHERE
    exist as single tokens. This avoids getting stuck generating only whitespace
    when using a literal-keyword grammar with a subword tokenizer.
    """
    required_literals = ["SELECT", "FROM", "WHERE", "AND", "OR"]
    required_decoded = set(required_literals + [" " + w for w in required_literals] + ["*", ",", "(", ")", "=", "!=", "<", ">", "<=", ">="])

    # Find single-token matches for required decoded strings
    found_ids: List[int] = []
    for tok, tok_id in tokenizer.get_vocab().items():
        # Cheap filter before decode
        if not any(k in tok for k in ["SELECT", "FROM", "WHERE", "AND", "OR", "*", ",", "(", ")", "=", "!", "<", ">"]):
            continue
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if decoded in required_decoded:
            found_ids.append(tok_id)

    # Also include whitespace tokens explicitly (so grammar can ignore WS)
    for tok, tok_id in tokenizer.get_vocab().items():
        if tok_id in found_ids:
            continue
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if decoded in {" ", "\n", "\t"}:
            found_ids.append(tok_id)

    # Fill remaining budget with "safe" small tokens to build identifiers/numbers.
    # Important: tokenizers often include leading/trailing whitespace in the decoded string
    # (e.g. " 1", " age"), so we allow whitespace around otherwise-safe tokens.
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    safe_punct = set(" ,().=*<>!+-/\n\t")
    for tok_id in range(len(tokenizer)):
        if tok_id in found_ids:
            continue
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded:
            continue
        # Keep mostly short tokens to keep ValidNextTokens manageable
        if len(decoded) > 8:
            continue
        stripped = decoded.strip()
        if not stripped:
            continue
        # Accept tokens whose non-whitespace chars are all safe identifier/number chars
        # (with optional surrounding whitespace)
        if all((c in safe_chars) for c in stripped) and all((c in {" ", "\n", "\t"} or c in safe_chars) for c in decoded):
            found_ids.append(tok_id)
        # Accept punctuation-only tokens (including whitespace)
        elif all((c in safe_punct) for c in decoded):
            found_ids.append(tok_id)
        if len(found_ids) >= max_tokens:
            break

    # Dedupe preserving order, cap to max_tokens
    seen = set()
    out = [i for i in found_ids if (i not in seen and not seen.add(i))]
    return out[:max_tokens]


# --- Evaluation Loop ---

def run_evaluation(
    run_dir: Path,
    task: str,
    model_name: str,
    device: str,
    vocab_size: int,
    max_steps: int,
    debug: bool = False,
    num_cases: Optional[int] = None
):
    print(f"\nEvaluating CSD strategy from: {run_dir}")
    print(f"Task: {task.upper()}")
    print(f"Model: {model_name}")
    print("-" * 60)

    # Setup paths
    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        print(f"Error: Module directory not found: {module_dir}")
        return

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    try:
        import _dafny
        import VerifiedDecoderAgent
        import GeneratedCSD
        
        # Setup Grammar
        format_name = "json" if task == "json" else "sql"
        grammar_source = get_builtin_grammar(format_name)
        start_rule = "start"
        
        LarkDafnyParser = create_lark_dafny_parser(grammar_source, VerifiedDecoderAgent, _dafny, start_rule)
        
        # Setup Real LM
        token_ids = None
        if task == "sql":
            # Curated SQL token set ensures single-token keywords like SELECT/FROM exist.
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            token_ids = _select_sql_token_ids(tok, vocab_size)
        lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny, token_ids=token_ids)
        
        parser = LarkDafnyParser(lm._Tokens)
        
        prompts = JSON_PROMPTS if task == "json" else SQL_PROMPTS
        if num_cases is not None:
            prompts = prompts[: max(0, int(num_cases))]
        
        metrics = {
            "valid_count": 0,
            "valid_prefix_count": 0,
            "complete_count": 0,
            "total_tokens": 0,
            "total_time": 0,
            "count": 0
        }
        
        print(f"\nRunning {len(prompts)} test cases...")
        
        for i, prompt_text in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Generating for: {prompt_text[:40]}...")
            
            # Set instruction for the LM
            lm.instruction_text = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            
            # Start Generation with empty prefix (relative to instruction)
            dafny_prompt = _dafny.SeqWithoutIsStrInference([])
            
            start_time = time.time()
            try:
                output = GeneratedCSD.default__.MyCSDStrategy(lm, parser, dafny_prompt, max_steps)
            except Exception as e:
                print(f"  -> Error: {e}")
                metrics["count"] += 1 # Count as failure
                continue
                
            end_time = time.time()
            
            # Process output
            output_list = [str(t) for t in output]
            output_text = "".join(output_list)
            
            # Validate
            is_valid_prefix = parser._is_valid_prefix(output_text)
            is_complete = parser._is_complete(output_text)
            is_valid = is_valid_prefix and is_complete
            
            metrics["count"] += 1
            metrics["total_tokens"] += len(output_list)
            metrics["total_time"] += (end_time - start_time)
            if is_valid_prefix:
                metrics["valid_prefix_count"] += 1
            if is_complete:
                metrics["complete_count"] += 1
            if is_valid:
                metrics["valid_count"] += 1
            
            print(f"  -> Generated {len(output_list)} tokens in {end_time-start_time:.2f}s")
            print(f"  -> Valid: {is_valid}")
            if debug:
                non_ws_len = len(output_text.strip())
                print(f"  -> Prefix-valid: {is_valid_prefix} | Complete: {is_complete} | non-ws chars: {non_ws_len}")
                print(f"  -> Output (repr, first 200): {repr(output_text[:200])}")
            
        
        # --- Results & Comparison ---
        
        if metrics["count"] == 0:
            print("No results collected.")
            return

        avg_tokens = metrics["total_tokens"] / metrics["count"]
        avg_time = metrics["total_time"] / metrics["count"]
        accuracy = (metrics["valid_count"] / metrics["count"]) * 100.0
        valid_prefix_rate = (metrics["valid_prefix_count"] / metrics["count"]) * 100.0
        complete_rate = (metrics["complete_count"] / metrics["count"]) * 100.0
        
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({task.upper()})")
        print("=" * 60)
        print(f"{'Metric':<15} | {'My CSD':<15} | {'Syncode':<15} | {'IterGen':<15}")
        print("-" * 60)
        
        base = BASELINES[task]
        
        print(f"{'Accuracy (%)':<15} | {accuracy:<15.1f} | {base['syncode'].accuracy:<15.1f} | {base['itergen'].accuracy:<15.1f}")
        print(f"{'Prefix-valid %':<15} | {valid_prefix_rate:<15.1f} | {'-':<15} | {'-':<15}")
        print(f"{'Complete %':<15} | {complete_rate:<15.1f} | {'-':<15} | {'-':<15}")
        print(f"{'Tokens':<15} | {avg_tokens:<15.1f} | {base['syncode'].tokens:<15.1f} | {base['itergen'].tokens:<15.1f}")
        print(f"{'Time (s)':<15} | {avg_time:<15.2f} | {base['syncode'].time:<15.2f} | {base['itergen'].time:<15.2f}")
        print("-" * 60)
        
        if task == "sql":
            print("Note: 'Execute (%)' metric requires a SQL environment and is not measured here.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nEvaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSD Performance")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["json", "sql"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", help="Print per-sample details and output preview")
    parser.add_argument("--num-cases", type=int, default=None, help="Limit number of prompts evaluated (default: all)")
    
    args = parser.parse_args()
    
    run_evaluation(
        args.run_dir,
        args.task,
        args.model,
        args.device,
        args.vocab_size,
        args.max_steps,
        debug=args.debug,
        num_cases=args.num_cases
    )

if __name__ == "__main__":
    main()
