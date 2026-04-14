"""
Shared Dafny workspace setup for the Python-first synthesis pipeline.

The synthesis loop now treats Python sources as the source of truth.
This module prepares a temporary workspace containing:
- the generated Python strategy file (`GeneratedCSD.py`)
- Python contract dependencies such as `generation/csd/VerifiedAgentSynthesis.py`
- the transpiled Dafny workspace used by `dafny verify` / `dafny build`
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from verification.transpiler.transpiler import prepare_dafny_workspace_from_python


GENERATED_MODULE_FILENAME = "GeneratedCSD.py"
VERIFIED_AGENT_FILENAME = "VerifiedAgentSynthesis.py"


def get_verified_agent_synthesis_path() -> Path | None:
    """Return the Python contract source for VerifiedAgentSynthesis, or None if not found."""
    root = Path(__file__).resolve().parent.parent
    candidate = root / "generation" / "csd" / VERIFIED_AGENT_FILENAME
    return candidate if candidate.exists() else None


def check_dafny_available(dafny_path: str, timeout: int = 10) -> None:
    """Check that Dafny is installed and accessible."""
    try:
        result = subprocess.run(
            [dafny_path, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Dafny returned non-zero exit code: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            f"Dafny not found at '{dafny_path}'. "
            "Please install Dafny and ensure it's in your PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Dafny version check timed out")


def prepare_temp_dafny_dir(temp_path: Path, python_code: str) -> Tuple[Path, Path, Path]:
    """
    Materialize a temporary Dafny workspace from Python sources.

    Args:
        temp_path: Existing temp directory
        python_code: Complete contents of the generated `GeneratedCSD.py`

    Returns:
        `(source_file_path, cwd, generated_python_path)` where `source_file_path`
        is the transpiled main Dafny file, `cwd` is the temp workspace root,
        and `generated_python_path` is the written Python source file.
    """
    source_python = get_verified_agent_synthesis_path()
    if not source_python or not source_python.exists():
        raise FileNotFoundError(f"generation/csd/{VERIFIED_AGENT_FILENAME} not found")

    generated_python = temp_path / GENERATED_MODULE_FILENAME
    generated_python.write_text(python_code)
    shutil.copyfile(source_python, temp_path / VERIFIED_AGENT_FILENAME)

    workspace = temp_path / "dafny_workspace"
    workspace.mkdir(exist_ok=True)

    result = prepare_dafny_workspace_from_python(generated_python, workspace, axiomatize=False)
    if result.is_err():
        raise result.error

    return result.value, workspace, generated_python
