"""When delimiters are NOT required (--no-require-delimiters), the synthesis
author feedback must omit the visible `<<`/`>>` span diagnostics.

Those span statistics are meaningless when the model isn't asked to emit
spans, and were observed misleading the author into chasing a non-issue
(7B att18: the strategy emitted literal `<<`/`>>` markers, corrupting the SQL
and collapsing syntax to 0%). With --no-require-delimiters the feedback should
not mention spans at all.

get_feedback_summary(require_delimiters=False) must drop:
  - the "Contains << >>" header line
  - the whole "Structural Generation Metrics" block (all "visible `<<`" stats)
  - the "Answer extraction source" and "Span usefulness" diagnostic lines
...while still keeping the generally-useful Accuracy, Syntax Rate, and
"Correctness by syntax bucket" lines.

With require_delimiters=True (the default), all of the above must still appear,
so existing delimiter runs are unaffected.
"""
from __future__ import annotations

from synthesis.evaluate.evaluator import EvaluationResult


def _sample(output: str) -> dict:
    return {
        "question": "q",
        "question_full": "q",
        "expected": "SELECT 1",
        "actual": "SELECT 2",
        "full_output": output,
        "scored_output": output,
        "is_correct": False,
        "accuracy_applicable": True,
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": False,
        "is_syntax_valid": True,
        "syntax_rate": 1.0,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 1,
        "num_valid_visible_spans": 1,
        "has_extracted_answer": True,
        "token_count": 10,
        "time_seconds": 1.0,
        "hit_max_steps": False,
    }


def _result() -> EvaluationResult:
    samples = [_sample("SELECT <<x>> FROM t")]
    return EvaluationResult(
        success=False,
        accuracy=0.0,
        contains_delimiters=True,
        syntax_rate=1.0,
        num_examples=1,
        num_correct=0,
        total_time_seconds=1.0,
        sample_outputs=samples,
    )


# Lines that are delimiter/visible-span specific — noise under --no-require-delimiters.
# Callers: test_delimiter_diagnostics_*; API: get_feedback_summary text.
# User: "remove the contradiction through the contain delimiters thing"
DELIM_MARKERS = [
    "Text contains << >>",
    "Entered constrained mode",
    "Structural Generation Metrics",
    "Examples with visible `<<`",
    "Answer extraction source",
    "Span usefulness",
]
# Generally-useful lines that must survive regardless of delimiter mode.
KEEP_MARKERS = ["Accuracy:", "Syntax Rate:", "Correctness by syntax bucket"]


def test_delimiter_diagnostics_present_when_required():
    summary = _result().get_feedback_summary(require_delimiters=True)
    for marker in DELIM_MARKERS:
        assert marker in summary, f"expected {marker!r} when require_delimiters=True"
    for marker in KEEP_MARKERS:
        assert marker in summary, f"expected {marker!r} when require_delimiters=True"


def test_default_keeps_delimiter_diagnostics():
    # Default call (no arg) must stay backward-compatible: delimiters required.
    summary = _result().get_feedback_summary()
    assert "Structural Generation Metrics" in summary
    assert "Text contains << >>" in summary
    assert "Entered constrained mode" in summary


def test_delimiter_diagnostics_omitted_when_not_required():
    summary = _result().get_feedback_summary(require_delimiters=False)
    for marker in DELIM_MARKERS:
        assert marker not in summary, (
            f"{marker!r} must be omitted from author feedback when require_delimiters=False"
        )
    for marker in KEEP_MARKERS:
        assert marker in summary, (
            f"expected {marker!r} to remain when require_delimiters=False"
        )
