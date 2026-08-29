"""Hash the tracked source bytes that can change synthesis or evaluation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def execution_source_paths(repo: Path) -> tuple[str, ...]:
    """Return the tracked execution closure in stable order."""
    names = (
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    selected = [
        name
        for name in names
        if name
        and (
            name.startswith("synthesis/")
            or name.startswith("scripts/runtime/")
            or name.startswith("environment/benchmark_splits/")
            or name in {"run_all_tests.py", ".context/run_post14b_rebar_queue.py"}
        )
    ]
    return tuple(sorted(selected))


def execution_source_hashes(repo: Path) -> dict[str, str]:
    """Hash every source file in the execution closure."""
    return {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in execution_source_paths(repo)
    }


def execution_source_sha256(repo: Path) -> str:
    """Hash the complete path-to-content map for one source snapshot."""
    encoded = json.dumps(
        execution_source_hashes(repo), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
