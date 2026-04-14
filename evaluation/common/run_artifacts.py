"""
Helpers for locating synthesis run artifacts.

The repo writes synthesized runs directly under ``outputs/`` and many
tools need the same two operations:
- resolve the ``latest`` shortcut via ``latest_run.txt``
- find the compiled directory that contains ``GeneratedCSD.py``
"""

from __future__ import annotations

from pathlib import Path


COMPILED_MODULE_FALLBACK_SUBDIRS = (
    "generated_csd",
    "folio_csd",
    "gsm_crane_csd",
    "fol_csd",
    "pddl_csd",
    "sygus_slia_csd",
    "pipeline_smoke",
)


def _rewrite_legacy_run_path(path: Path) -> Path:
    """
    Rewrite the old ``outputs/generated-csd/runs/<run_id>`` layout to ``outputs/<run_id>``.
    """
    parts = path.parts
    for idx in range(len(parts) - 2):
        if parts[idx] == "generated-csd" and parts[idx + 1] == "runs":
            candidate = Path(*parts[:idx], *parts[idx + 2 :])
            return candidate
    return path


def resolve_run_dir(run_dir: Path) -> Path:
    """
    Resolve a run directory path, including the ``latest`` indirection.

    If ``run_dir`` points at ``.../latest`` and that path does not
    exist, this falls back to reading ``latest_run.txt`` from the parent.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        legacy_rewritten = _rewrite_legacy_run_path(run_dir)
        if legacy_rewritten.exists():
            return legacy_rewritten

    if run_dir.name != "latest" or run_dir.exists():
        return run_dir

    latest_txt = run_dir.parent / "latest_run.txt"
    if not latest_txt.exists():
        return run_dir

    actual_path = Path(latest_txt.read_text().strip())
    if not actual_path.exists():
        actual_path = _rewrite_legacy_run_path(actual_path)
    return actual_path if actual_path.exists() else run_dir


def find_compiled_module_dir(run_dir: Path) -> Path:
    """
    Return the directory containing ``GeneratedCSD.py`` for a synthesis run.
    """
    resolved_run_dir = resolve_run_dir(run_dir).resolve()
    if (resolved_run_dir / "GeneratedCSD.py").exists():
        return resolved_run_dir

    for subdir in COMPILED_MODULE_FALLBACK_SUBDIRS:
        candidate = resolved_run_dir / subdir
        if (candidate / "GeneratedCSD.py").exists():
            return candidate

    found = list(resolved_run_dir.glob("*/GeneratedCSD.py"))
    if found:
        return found[0].parent

    raise FileNotFoundError(f"Compiled module directory not found in {resolved_run_dir}")
