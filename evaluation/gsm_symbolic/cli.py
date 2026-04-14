#!/usr/bin/env python3
"""
GSM-Symbolic evaluation CLI.

Runs CSD or unconstrained baseline generation on GSM-Symbolic dataset.
Prompt formatting and answer extraction must be provided externally.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluation.gsm_symbolic.dataset import load_gsm_symbolic
from evaluation.gsm_symbolic.metrics import GSMMetrics
from evaluation.gsm_symbolic.generation import run_crane_csd, run_unconstrained
from evaluation.gsm_symbolic.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
)
from evaluation.common.cli_utils import add_common_eval_args

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description="Evaluate GSM-Symbolic with CSD")
    add_common_eval_args(
        ap,
        default_grammar=PROJECT_ROOT / "utils" / "grammars" / "gsm.lark",
        default_max_steps=1024,
    )
    ap.add_argument("--config", choices=["main", "p1", "p2"], default="main",
                    help="GSM-Symbolic difficulty level")
    ap.add_argument("--random-sample", action="store_true",
                    help="Randomly sample examples instead of taking first N")
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
        vocab_size=args.vocab_size,
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

        # The prompt is just the raw question — no hardcoded formatting
        prompt = question

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

        metrics.update(
            is_correct=False,
            is_valid_format=False,
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
