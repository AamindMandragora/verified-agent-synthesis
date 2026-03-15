"""
Environment setup for FOLIO evaluation.

Thin wrapper around common; uses start_rule="start" for FOL grammar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from evaluations.common.environment import (
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
    **kwargs: Any,
) -> Dict[str, Any]:
    """Setup Dafny environment with FOL grammar start rule and FOL keywords as single tokens."""
    return _setup_dafny_environment(
        run_dir=run_dir,
        model_name=model_name,
        device=device,
        vocab_size=vocab_size,
        grammar_file=grammar_file,
        start_rule="start",
        add_fol_keyword_tokens=True,
        **kwargs,
    )


__all__ = [
    "resolve_run_dir",
    "load_compiled_modules",
    "verify_critical_tokens",
    "setup_dafny_environment",
]
