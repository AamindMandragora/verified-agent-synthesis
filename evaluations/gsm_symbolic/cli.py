#!/usr/bin/env python3
"""
GSM-Symbolic evaluation CLI.

Runs CSD or unconstrained baseline generation on GSM-Symbolic dataset.
Prompt formatting and answer extraction must be provided externally.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch

from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
from evaluations.gsm_symbolic.metrics import GSMMetrics
from evaluations.gsm_symbolic.generation import run_crane_csd, run_unconstrained
from evaluations.gsm_symbolic.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description="Evaluate GSM-Symbolic with CSD")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", choices=["main", "p1", "p2"], default="main",
                    help="GSM-Symbolic difficulty level")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max examples to evaluate")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="Max steps for generation")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark",
                    help="Grammar file for math validation")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-example details")
    ap.add_argument("--debug-delimiters", action="store_true",
                    help="Debug delimiter detection")
    ap.add_argument("--random-sample", action="store_true",
                    help="Randomly sample examples instead of taking first N")
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained baseline instead of CSD")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit quantization")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="Load model in 8-bit quantization")
    args = ap.parse_args()

    ds = load_gsm_symbolic(args.config, limit=args.limit, random_sample=args.random_sample)
    n = len(ds)
    metrics = GSMMetrics()

    print(f"Setting up Dafny environment...")
    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        device=args.device,
        grammar_file=args.grammar,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    eval_start_time = time.time()

    for i in range(n):
        example = ds[i]
        question = example.get("question", "")

        if not question:
            continue

        print(f"[{i+1}/{n}] Processing example...", flush=True)

        prompt = (
            "Solve the following math problem step by step. "
            "Assign a single-letter variable to each quantity and state its numeric value. "
            "Write each computation step as a SHORT expression inside << >> delimiters — "
            "one step per << >> window, closing >> before starting the next step.\n\n"
            "Example:\n"
            "Q: A store sells pens for $3 each and notebooks for $8 each. "
            "Bob buys 4 pens and 2 notebooks. How much does he spend?\n"
            "A: Let p = 3, n = 8, a = 4, b = 2.\n"
            "Pen total = <<a * p>>\n"
            "Notebook total = <<b * n>>\n"
            "Total spent = <<a * p + b * n>>\n"
            "The answer is a * p + b * n.\n\n"
            f"Q: {question}\nA:"
        )

        if args.unconstrained:
            out_text, tok_count, dt = run_unconstrained(
                dafny_env, prompt, args.max_steps,
                debug=args.verbose,
            )
            constrained_segments = []
        else:
            out_text, tok_count, dt, constrained_segments = run_crane_csd(
                dafny_env, prompt, args.max_steps, args.grammar,
                debug_delimiters=args.debug_delimiters,
            )

        # Evaluate format: output must contain at least one <<...>> segment
        is_valid_format = bool(re.search(r"<<[^<>]+>>", out_text))

        # Evaluate accuracy: extract expected answer, extract actual from last <<>> segment
        expected = None
        m = re.search(r"####\s*([-+]?\d*\.?\d+)", example.get("answer", ""))
        if m:
            expected = m.group(1)

        actual = None
        seg_matches = re.findall(r"<<\s*([^<>]+?)\s*>>", out_text)
        if seg_matches:
            last_expr = seg_matches[-1].strip()
            # Parse variable assignments from the output (e.g. "a = 5")
            var_values = {}
            for vm in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\b', out_text):
                try:
                    var_values[vm.group(1)] = float(vm.group(2))
                except ValueError:
                    pass
            # Substitute variables and evaluate
            if var_values:
                substituted = last_expr
                for var in sorted(var_values, key=len, reverse=True):
                    substituted = re.sub(r'\b' + re.escape(var) + r'\b', str(var_values[var]), substituted)
                if not re.search(r'[a-zA-Z_]', substituted):
                    try:
                        import ast, operator as op
                        _ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                                ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
                                ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos}
                        def _eval(node):
                            if isinstance(node, (ast.Constant, ast.Num)):
                                return float(getattr(node, 'value', getattr(node, 'n', 0)))
                            elif isinstance(node, ast.BinOp) and type(node.op) in _ops:
                                return _ops[type(node.op)](_eval(node.left), _eval(node.right))
                            elif isinstance(node, ast.UnaryOp) and type(node.op) in _ops:
                                return _ops[type(node.op)](_eval(node.operand))
                            raise ValueError
                        result = _eval(ast.parse(substituted.strip(), mode='eval').body)
                        val = int(result) if result == int(result) else result
                        actual = str(val)
                    except Exception:
                        pass

        is_correct = (actual is not None and expected is not None and
                      str(actual).strip() == str(expected).strip())

        metrics.update(
            is_correct=is_correct,
            is_valid_format=is_valid_format,
            token_count=tok_count,
            time_seconds=dt,
            constrained_segments=constrained_segments,
        )

        avg_time = metrics.total_time / metrics.n
        remaining = n - metrics.n
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | ETA: {eta_str}", flush=True)

        if args.verbose:
            print(f"  Question: {question}")
            print(f"  Expected: {expected} | Actual: {actual} | Correct: {is_correct}")
            print(f"  Output: {out_text}")
            print()

    print("\n" + "=" * 60)
    print("GSM-SYMBOLIC RESULTS")
    print("=" * 60)
    print(f"Method: {'Unconstrained Baseline' if args.unconstrained else 'CSD'}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print()
    print(metrics.summary())


if __name__ == "__main__":
    main()
