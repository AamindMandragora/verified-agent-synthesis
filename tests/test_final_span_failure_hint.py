"""TDD tests for _final_span_failure_hint (Improvement 1).

Tests verify the three categories of delimiter-bearing syntax failures are
correctly classified and surfaced in author-facing feedback text:
  - final_span_unclosed: rfind('<<') > rfind('>>')
  - no_span_emitted: no '<<' anywhere in output
  - final_span_invalid: closed block but fails syntax check (e.g. contains '{' or '**')

RED: tests fail before _final_span_failure_hint is added to feedback_loop.py.
GREEN: tests pass once the function exists and is wired correctly.

Run on focal:
    cd /home/aadivyar/csd-generation
    python -m pytest tests/test_final_span_failure_hint.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

def _import_hint():
    """Import _final_span_failure_hint, skipping the test if not present."""
    try:
        from synthesis.evaluate.feedback_loop import _final_span_failure_hint
        return _final_span_failure_hint
    except ImportError as e:
        pytest.fail(
            f"Could not import _final_span_failure_hint from feedback_loop: {e}. "
            "This is the RED state — add _final_span_failure_hint to feedback_loop.py."
        )


# ---------------------------------------------------------------------------
# Helper: build a minimal sample_outputs list entry
# ---------------------------------------------------------------------------

def _sample(full_output: str, is_syntax_valid: bool) -> dict:
    return {
        "full_output": full_output,
        "is_syntax_valid": is_syntax_valid,
    }


# ---------------------------------------------------------------------------
# Test 1 — Returns empty string when require_delimiters=False
# ---------------------------------------------------------------------------

def test_returns_empty_when_delimiters_not_required():
    """Hint is suppressed when the eval doesn't require << >> spans."""
    fn = _import_hint()
    samples = [_sample("no span here", is_syntax_valid=False)]
    result = fn(require_delimiters=False, sample_outputs=samples)
    assert result == "", (
        "_final_span_failure_hint must return '' when require_delimiters=False"
    )


# ---------------------------------------------------------------------------
# Test 2 — Returns empty string when sample_outputs is empty / None
# ---------------------------------------------------------------------------

def test_returns_empty_when_no_samples():
    fn = _import_hint()
    assert fn(require_delimiters=True, sample_outputs=None) == ""
    assert fn(require_delimiters=True, sample_outputs=[]) == ""


# ---------------------------------------------------------------------------
# Test 3 — Returns empty string when ALL samples are syntax-valid
# ---------------------------------------------------------------------------

def test_returns_empty_when_all_syntax_valid():
    fn = _import_hint()
    samples = [
        _sample("step by step... <<n * 2>>", is_syntax_valid=True),
        _sample("answer: <<x + 1>>", is_syntax_valid=True),
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    assert result == "", (
        "_final_span_failure_hint must return '' when all samples pass syntax"
    )


# ---------------------------------------------------------------------------
# Test 4 — Detects final_span_unclosed (rfind('<<') > rfind('>>'))
# ---------------------------------------------------------------------------

def test_final_span_unclosed_detected():
    """Model emitted '<<' at the end but stopped without '>>'."""
    fn = _import_hint()
    samples = [
        _sample("The answer is <<", is_syntax_valid=False),           # trailing <<
        _sample("...result is <<n + 1>>. Final is <<", is_syntax_valid=False),  # second << after >>
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    assert result, "_final_span_failure_hint must return a non-empty hint"
    assert "final_span_unclosed" in result, (
        f"Expected 'final_span_unclosed' in hint but got:\n{result}"
    )
    assert "2" in result or "example" in result.lower(), (
        "Expected count or 'example(s)' language in hint"
    )


# ---------------------------------------------------------------------------
# Test 5 — Detects no_span_emitted (no '<< ' anywhere)
# ---------------------------------------------------------------------------

def test_no_span_emitted_detected():
    """Model never produced '<<' — constrained span trigger never fired."""
    fn = _import_hint()
    samples = [
        _sample("The total is 42.", is_syntax_valid=False),
        _sample("I think the answer is 7.", is_syntax_valid=False),
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    assert result, "_final_span_failure_hint must return a non-empty hint"
    assert "no_span_emitted" in result, (
        f"Expected 'no_span_emitted' in hint but got:\n{result}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Detects final_span_invalid (closed block present but syntax fails)
# ---------------------------------------------------------------------------

def test_final_span_invalid_detected():
    """Closed << >> block exists but fails syntax — e.g. contains braces or **."""
    fn = _import_hint()
    samples = [
        _sample("The answer is <<{daily} + n>>", is_syntax_valid=False),
        _sample("Result: <<n0 * (1 + r) ** d>>", is_syntax_valid=False),
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    assert result, "_final_span_failure_hint must return a non-empty hint"
    assert "final_span_invalid" in result, (
        f"Expected 'final_span_invalid' in hint but got:\n{result}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Mixed categories are all reported
# ---------------------------------------------------------------------------

def test_mixed_categories_all_reported():
    """When all three failure types appear, all three category labels appear."""
    fn = _import_hint()
    samples = [
        _sample("The total is 42.", is_syntax_valid=False),           # no_span
        _sample("Final answer is <<", is_syntax_valid=False),         # unclosed
        _sample("Result: <<{x} + 1>>", is_syntax_valid=False),        # invalid
        _sample("Step: <<n * 2>>", is_syntax_valid=True),             # valid (not counted)
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    assert "final_span_unclosed" in result
    assert "no_span_emitted" in result
    assert "final_span_invalid" in result


# ---------------------------------------------------------------------------
# Test 8 — Example output tails appear in hint text (up to 2 per category)
# ---------------------------------------------------------------------------

def test_output_tails_appear_in_hint():
    """The hint should include a snippet of example output per category."""
    fn = _import_hint()
    samples = [
        _sample("The total is 99.", is_syntax_valid=False),  # no_span
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    # Should contain some portion of the example text or a tail marker
    assert "99" in result or "output tail" in result.lower(), (
        "Expected example output tail to appear in hint text"
    )


# ---------------------------------------------------------------------------
# Test 9 — Hint is capped at 2 output tails per category
# ---------------------------------------------------------------------------

def test_at_most_two_output_tails_per_category():
    """Even with 5 unclosed examples, at most 2 output tails are shown."""
    fn = _import_hint()
    samples = [
        _sample(f"Answer {i} is <<", is_syntax_valid=False)
        for i in range(5)
    ]
    result = fn(require_delimiters=True, sample_outputs=samples)
    # Count occurrences of "output tail:" — should be ≤ 2
    n_tails = result.count("output tail:")
    assert n_tails <= 2, (
        f"Expected at most 2 'output tail:' entries per category, got {n_tails}"
    )



# ---------------------------------------------------------------------------
# Test 11 — Source-level: _final_span_failure_hint is wired into threshold_feedback
# ---------------------------------------------------------------------------

def test_final_span_failure_hint_wired_into_threshold_feedback():
    """AST check: threshold_feedback construction in synthesize() must call
    _final_span_failure_hint so the new signal reaches the author model."""
    import ast

    fb_path = _REPO / "synthesis" / "evaluate" / "feedback_loop.py"
    tree = ast.parse(fb_path.read_text())

    # Find all Call nodes that reference _final_span_failure_hint
    call_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_final_span_failure_hint":
                call_sites.append(node)
            elif isinstance(f, ast.Attribute) and f.attr == "_final_span_failure_hint":
                call_sites.append(node)

    assert len(call_sites) >= 1, (
        "feedback_loop.py must contain at least one call to _final_span_failure_hint. "
        "Wire it into threshold_feedback (and optionally evaluation_feedback) "
        "so the author model sees the new breakdown."
    )
