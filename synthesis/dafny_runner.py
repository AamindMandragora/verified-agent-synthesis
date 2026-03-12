"""
Shared Dafny path and temp-dir setup for verifier and compiler.

Provides get_verified_agent_synthesis_path(), check_dafny_available(),
and prepare_temp_dafny_dir() so verifier and compiler don't duplicate
logic for finding VerifiedAgentSynthesis.dfy and laying out temp directories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal, Tuple


def get_verified_agent_synthesis_path() -> Path | None:
    """
    Return path to VerifiedAgentSynthesis.dfy, or None if not found.

    Looks in proofs/ then dafny/ relative to the synthesis package root (repo root).
    """
    root = Path(__file__).resolve().parent.parent
    for candidate in (root / "proofs" / "VerifiedAgentSynthesis.dfy", root / "dafny" / "VerifiedAgentSynthesis.dfy"):
        if candidate.exists():
            return candidate
    return None


def check_dafny_available(dafny_path: str, timeout: int = 10) -> None:
    """
    Check that Dafny is installed and accessible.

    Raises:
        RuntimeError: If Dafny is not found or returns non-zero.
    """
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


def prepare_temp_dafny_dir(
    temp_path: Path,
    dafny_code: str,
    mode: Literal["verify", "compile"],
) -> Tuple[Path, Path]:
    """
    Write VerifiedAgentSynthesis.dfy and GeneratedCSD.dfy into temp_path for verify or compile.

    Args:
        temp_path: Existing temp directory to write into
        dafny_code: Contents of GeneratedCSD.dfy
        mode: "verify" or "compile" — layout differs (verify uses root + agents; compile uses agents + proofs)

    Returns:
        (source_file_path, cwd) where source_file_path is the GeneratedCSD.dfy to pass to Dafny,
        and cwd is the directory to run Dafny from.

    Raises:
        FileNotFoundError: If VerifiedAgentSynthesis.dfy is not found in proofs/ or dafny/
    """
    source_proof = get_verified_agent_synthesis_path()
    if not source_proof or not source_proof.exists():
        raise FileNotFoundError("VerifiedAgentSynthesis.dfy not found in proofs/ or dafny/")

    proof_text = source_proof.read_text()

    if mode == "verify":
        (temp_path / "VerifiedAgentSynthesis.dfy").write_text(proof_text)
        agents_dir = temp_path / "agents"
        agents_dir.mkdir(exist_ok=True)
        (agents_dir / "VerifiedAgentSynthesis.dfy").write_text(proof_text)
        source_file = temp_path / "GeneratedCSD.dfy"
        source_file.write_text(dafny_code)
        return source_file, temp_path

    # mode == "compile"
    agents_dir = temp_path / "agents"
    agents_dir.mkdir()
    proofs_dir = temp_path / "proofs"
    proofs_dir.mkdir()
    (proofs_dir / "VerifiedAgentSynthesis.dfy").write_text(proof_text)
    (agents_dir / "VerifiedAgentSynthesis.dfy").write_text(proof_text)
    source_file = agents_dir / "GeneratedCSD.dfy"
    source_file.write_text(dafny_code)
    return source_file, temp_path
