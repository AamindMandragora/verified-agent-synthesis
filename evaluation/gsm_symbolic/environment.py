"""
Environment setup for GSM-Symbolic evaluation.

Thin wrapper around common; uses start_rule="csd_start" and supports quantization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluation.common.environment import (
    load_compiled_modules,
    resolve_run_dir,
    setup_dafny_environment as _setup_dafny_environment,
    verify_critical_tokens,
)


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int,
    grammar_file: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Dict[str, Any]:
    """Setup Dafny environment with GSM grammar start rule and optional quantization."""
    return _setup_dafny_environment(
        run_dir=run_dir,
        model_name=model_name,
        device=device,
        vocab_size=vocab_size,
        grammar_file=grammar_file,
        start_rule="csd_start",
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        add_gsm_delimiter_tokens=True,
    )


__all__ = [
    "resolve_run_dir",
    "load_compiled_modules",
    "verify_critical_tokens",
    "setup_dafny_environment",
]
