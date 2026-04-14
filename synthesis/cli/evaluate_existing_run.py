#!/usr/bin/env python3
"""
Evaluate an existing synthesized run with the current synthesis evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common.run_artifacts import find_compiled_module_dir
from evaluation.evaluator import Evaluator


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing synthesis run with the current evaluator.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a synthesis run directory.",
    )
    parser.add_argument(
        "--dataset",
        default="folio",
        choices=("folio", "gsm_symbolic", "pddl", "sygus_slia"),
        help="Dataset evaluator to use.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Model used for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=256,
        help="Maximum generation steps per example.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=3000,
        help="Evaluation vocabulary size.",
    )
    parser.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load the evaluation model in 4-bit mode.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the aggregate metrics line.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    compiled_path = find_compiled_module_dir(args.run_dir)
    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.model,
        device=args.device,
        vocab_size=args.vocab_size,
        sample_size=args.sample_size,
        max_steps=args.max_steps,
        load_in_4bit=args.load_in_4bit,
    )
    result = evaluator.evaluate_sample(
        compiled_module_path=compiled_path,
        sample_size=args.sample_size,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.summary_only:
        print(
            f"format_rate={result.format_rate:.2%}  "
            f"parse(syntax)_rate={result.syntax_rate:.2%}  "
            f"accuracy={result.accuracy:.2%}  "
            f"(n={result.num_examples})"
        )
        return 0

    print(f"Run dir: {args.run_dir}")
    print(f"Compiled module: {compiled_path}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Sample size: {args.sample_size}")
    print("Using the current synthesis evaluator\n")
    result.print_outputs_vs_expected()
    print()
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"Format rate: {result.format_rate:.1%}")
    print(f"Syntax rate: {result.syntax_rate:.1%}")
    if result.sample_outputs:
        print(result.get_detailed_samples(max_samples=min(args.sample_size, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
