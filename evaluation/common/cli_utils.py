"""
Shared CLI argument groups for evaluation scripts.

Use add_common_eval_args() so GSM and FOLIO CLIs don't duplicate
run-dir, model, device, limit, max-steps, vocab-size, grammar, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def add_common_eval_args(
    parser: argparse.ArgumentParser,
    default_grammar: Path,
    default_max_steps: int = 1024,
) -> None:
    """
    Add common evaluation arguments to an ArgumentParser.

    Call this first, then add dataset-specific args (e.g. --config, --split, --debug-csd).

    Args:
        parser: Parser to add arguments to
        default_grammar: Default path to grammar file (e.g. utils/grammars/gsm.lark or utils/grammars/folio.lark)
        default_max_steps: Default max generation steps
    """
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to compiled CSD run directory",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max examples to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=default_max_steps,
        help="Max steps for generation",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=3000,
        help="Token vocabulary size limit",
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        default=default_grammar,
        help="Grammar file for validation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-example details",
    )
    parser.add_argument(
        "--debug-delimiters",
        action="store_true",
        help="Debug delimiter detection",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Run unconstrained baseline instead of CSD",
    )
