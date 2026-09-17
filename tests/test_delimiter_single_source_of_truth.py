"""One source of truth for "must this dataset emit a visible << >> span".

Two mechanisms used to answer this question:

  A. the CLI, routed through registry.resolve_require_delimiters(), which
     validates the dataset name and then honours the flag. Every benchmark's
     output carries visible << >> spans, so none of them can veto it.

  B. a central table    -- REQUIRE_DELIMITERS_BY_DATASET in run_constants.py,
     which hardcoded three answers and ignored the CLI flag.

A is the one we keep: it keeps --require-delimiters meaningful instead of
quietly making it dead.

B is deleted. The reason these tests exist rather than just deleting it: the
table is still live on the prompt-rendering refactor branches, where
run_synthesis.py reads REQUIRE_DELIMITERS_BY_DATASET[args.dataset] directly.
Merging that work would otherwise reintroduce a second answer that silently
overrides the benchmark and undoes the Spider fix. These tests turn that
silent revert into an obvious red test.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHESIS = REPO_ROOT / "synthesis"

RETIRED_TABLE = "REQUIRE_DELIMITERS_BY_DATASET"


def _source_files():
    """Every python and jinja source file under synthesis/, excluding vendored code."""
    for path in list(SYNTHESIS.rglob("*.py")) + list(SYNTHESIS.rglob("*.j2")):
        if "syncode" in path.parts:  # vendored third-party tree
            continue
        yield path


def test_retired_delimiter_table_is_gone():
    """The central table must not come back -- not even via a merge."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _source_files()
        if RETIRED_TABLE in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == [], (
        f"{RETIRED_TABLE} is back in: {offenders}. It is a second, competing "
        "answer to whether a dataset needs a visible << >> span, and it "
        "overrides the CLI flag. Route the call through "
        "registry.resolve_require_delimiters() instead. If this "
        "fired right after merging the prompt-rendering refactor, that branch's "
        "run_synthesis.py still reads the table -- fix it there, do not restore "
        "the table here."
    )


def test_run_synthesis_asks_the_benchmark():
    """The live wiring must go through the benchmark, not a lookup table."""
    source = (SYNTHESIS / "run_synthesis.py").read_text(encoding="utf-8")

    delimiter_lines = [
        line for line in source.splitlines() if "require_delimiters=" in line
    ]

    assert delimiter_lines, "run_synthesis.py no longer passes require_delimiters at all"
    for line in delimiter_lines:
        assert "resolve_require_delimiters" in line, (
            "run_synthesis.py sets require_delimiters without asking the "
            f"benchmark: {line.strip()!r}. Use "
            "resolve_require_delimiters(args.dataset, args.require_delimiters)."
        )


def test_evaluator_docstring_points_at_the_live_mechanism():
    """The docstring must not send a reader to the deleted table."""
    evaluator = SYNTHESIS / "evaluate" / "evaluator.py"
    source = evaluator.read_text(encoding="utf-8")

    assert RETIRED_TABLE not in source, (
        "synthesis/evaluate/evaluator.py still cites the deleted "
        f"{RETIRED_TABLE} table as the authority for require_delimiters."
    )
    assert "run_constants.py" not in source, (
        "synthesis/evaluate/evaluator.py still points at run_constants.py for "
        "the delimiter decision; that decision now comes from the CLI via "
        "registry.resolve_require_delimiters()."
    )
