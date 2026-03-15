#!/usr/bin/env python3
"""
Evaluate an existing CSD run using the same prompt and LM as synthesis evaluation.

Use this to test older working CSDs with the current (new) prompt and
first-token masking (plain text before << >>).

Example:
  python scripts/eval_csd_with_synthesis_prompt.py \\
    --run-dir outputs/generated-csd/runs/20260314_222155_2d1042 \\
    --model Qwen/Qwen2.5-Coder-3B-Instruct \\
    --sample-size 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_compiled_module_path(run_dir: Path) -> Path:
    """Return the directory containing GeneratedCSD.py (for use as compiled_module_path)."""
    run_dir = run_dir.resolve()
    if (run_dir / "GeneratedCSD.py").exists():
        return run_dir
    candidates = [
        run_dir / "folio_csd",
        run_dir / "gsm_crane_csd",
        run_dir / "fol_csd",
        run_dir / "generated_csd",
    ]
    for d in candidates:
        if d.exists() and (d / "GeneratedCSD.py").exists():
            return d
    found = list(run_dir.glob("*/GeneratedCSD.py"))
    if found:
        return found[0].parent
    raise FileNotFoundError(f"No compiled CSD module found in {run_dir}. Tried: {[str(c) for c in candidates]}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate an existing CSD run with the synthesis evaluator (same prompt + first-token masking)."
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a synthesis run directory (e.g. outputs/generated-csd/runs/20260314_222155_2d1042)",
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="HuggingFace model (default: Qwen/Qwen2.5-Coder-3B-Instruct)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of examples to evaluate (default: 5)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=256,
        help="Max generation steps per example (default: 256)",
    )
    ap.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load model in 4-bit for evaluation",
    )
    ap.add_argument(
        "--dataset",
        default="folio",
        choices=("folio", "gsm_symbolic"),
        help="Dataset (default: folio)",
    )
    args = ap.parse_args()

    compiled_path = _resolve_compiled_module_path(args.run_dir)
    print(f"Run dir: {args.run_dir}")
    print(f"Compiled module: {compiled_path}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Sample size: {args.sample_size}")
    print("Using synthesis evaluator (current prompt + first-token masking)\n")

    from synthesis.evaluator import Evaluator

    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.model,
        device=args.device,
        vocab_size=2000,
        sample_size=args.sample_size,
        max_steps=args.max_steps,
        load_in_4bit=args.load_in_4bit,
    )
    result = evaluator.evaluate_sample(
        compiled_module_path=compiled_path,
        sample_size=args.sample_size,
    )

    result.print_outputs_vs_expected()
    print()
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"Format rate: {result.format_rate:.1%}")
    print(f"Syntax rate: {result.syntax_rate:.1%}")
    if result.sample_outputs:
        print(result.get_detailed_samples(max_samples=min(args.sample_size, 10)))
    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)
