#!/usr/bin/env python3
"""
Generate a CSD from a dataset preset.

This complements the dataset/model shell wrappers under ``synthesis/shell/`` with a
single preset-driven entrypoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.presets import DATASET_PRESETS, MODEL_PRESETS, get_synthesis_preset, resolve_model_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a CSD using a built-in dataset preset.",
    )
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_PRESETS),
        help="Dataset preset to synthesize for.",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="qwen7b",
        help="Named model preset (ignored if --model is provided).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit model name to use instead of a model preset.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the preset output module name.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum refinement iterations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for synthesis.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device to pass through to run_synthesis.py.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Override the preset accuracy threshold.",
    )
    parser.add_argument(
        "--min-format-rate",
        type=float,
        default=None,
        help="Override the preset format threshold.",
    )
    parser.add_argument(
        "--min-syntax-rate",
        type=float,
        default=None,
        help="Override the preset syntax threshold.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Override the preset evaluation sample size.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="Override the preset evaluation max steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command instead of running it.",
    )
    return parser


def build_synthesis_command(
    *,
    dataset: str,
    model_name: str,
    output_name: str | None = None,
    max_iterations: int = 10,
    temperature: float = 0.7,
    device: str = "auto",
    min_accuracy: float | None = None,
    min_format_rate: float | None = None,
    min_syntax_rate: float | None = None,
    eval_sample_size: int | None = None,
    eval_max_steps: int | None = None,
) -> list[str]:
    """Build the `run_synthesis.py` command for a dataset preset."""
    preset = get_synthesis_preset(dataset)
    return [
        sys.executable,
        str(PROJECT_ROOT / "run_synthesis.py"),
        *preset.to_cli_args(
            model_name=model_name,
            max_iterations=max_iterations,
            temperature=temperature,
            device=device,
            output_name=output_name,
            min_accuracy=min_accuracy,
            min_format_rate=min_format_rate,
            min_syntax_rate=min_syntax_rate,
            eval_sample_size=eval_sample_size,
            eval_max_steps=eval_max_steps,
        ),
    ]


def main() -> int:
    args = _build_arg_parser().parse_args()

    model_name = resolve_model_name(model=args.model, model_preset=args.model_preset)
    preset = get_synthesis_preset(args.dataset)
    command = build_synthesis_command(
        dataset=args.dataset,
        model_name=model_name,
        output_name=args.output_name,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        device=args.device,
        min_accuracy=args.min_accuracy,
        min_format_rate=args.min_format_rate,
        min_syntax_rate=args.min_syntax_rate,
        eval_sample_size=args.eval_sample_size,
        eval_max_steps=args.eval_max_steps,
    )

    print(f"Dataset preset: {preset.dataset}")
    print(f"Model: {model_name}")
    print(f"Output name: {args.output_name or preset.output_name}")
    print(f"Command: {' '.join(command)}")

    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
