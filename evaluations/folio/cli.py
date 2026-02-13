#!/usr/bin/env python3
"""
FOLIO evaluation CLI.

Runs CSD or unconstrained baseline generation on FOLIO dataset.
Prompt formatting and answer extraction must be provided externally.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluations.folio.dataset import (
    load_folio,
    load_folio_from_json,
    create_synthetic_folio_examples,
)
from evaluations.common.parser_utils import create_lark_dafny_parser
from evaluations.folio.metrics import FOLIOMetrics
from evaluations.folio.generation import run_crane_csd, run_unconstrained
from evaluations.folio.environment import setup_dafny_environment, verify_critical_tokens

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    import os

    ap = argparse.ArgumentParser(description="Evaluate FOLIO with CSD")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="validation",
                    help="Dataset split to use")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max examples to evaluate")
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="Max steps for generation")
    ap.add_argument("--vocab-size", type=int, default=2000,
                    help="Token vocabulary size limit")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "folio.lark",
                    help="Grammar file for FOL validation")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-example details")
    ap.add_argument("--debug-delimiters", action="store_true",
                    help="Debug delimiter detection")
    ap.add_argument("--debug-csd", action="store_true",
                    help="Debug CSD constrained generation")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic examples for testing")
    ap.add_argument("--json-path", type=Path,
                    help="Load from local JSON file instead of HuggingFace")
    ap.add_argument("--output", type=Path,
                    help="Path to save metrics JSON")
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained baseline instead of CSD")
    args = ap.parse_args()

    if args.debug_csd:
        os.environ['CSD_MASK_DEBUG'] = '1'
        print("[DEBUG] CSD debug mode enabled")

    print("Loading dataset...")
    if args.synthetic:
        examples = create_synthetic_folio_examples()
        if args.limit and args.limit < len(examples):
            examples = examples[:args.limit]
    elif args.json_path:
        examples = load_folio_from_json(
            str(args.json_path),
            num_samples=args.limit,
        )
    else:
        examples = load_folio(
            split=args.split,
            num_samples=args.limit,
        )

    n = len(examples)
    print(f"Loaded {n} examples")

    metrics = FOLIOMetrics()

    print(f"Setting up Dafny environment...")
    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        device=args.device,
        vocab_size=args.vocab_size,
        grammar_file=args.grammar,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    eval_start_time = time.time()

    for i, example in enumerate(examples):
        problem = example.problem
        question = example.question
        gold_label = example.label

        print(f"[{i+1}/{n}] Processing example {example.id}...", flush=True)

        # The prompt is just the raw problem + question — no hardcoded formatting
        prompt = f"{problem}\n{question}"

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
                debug_csd=args.debug_csd,
            )

        metrics.update(
            predicted=None,
            gold=gold_label,
            is_correct=False,
            valid_structure=False,
            fol_segments=constrained_segments,
            time_seconds=dt,
            tokens=tok_count,
            example_id=example.id,
        )

        avg_time = metrics.total_time / metrics.total
        remaining = n - metrics.total
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | "
              f"Gold: {gold_label} | ETA: {eta_str}", flush=True)

        if args.verbose:
            print(f"  Problem: {problem}")
            print(f"  Question: {question}")
            print(f"  Output: {out_text[:500]}...")
            print()

    print("\n")
    metrics.print_summary()

    print(f"\nConfiguration:")
    print(f"  Method: {'Unconstrained' if args.unconstrained else 'CSD'}")
    print(f"  Model: {args.model}")
    print(f"  Split: {args.split}")
    print(f"  CSD Run: {args.run_dir}")
    print(f"  Grammar: {args.grammar}")

    if args.output:
        metrics.to_json(str(args.output))
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
