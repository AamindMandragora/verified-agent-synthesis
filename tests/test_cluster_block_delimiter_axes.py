"""The failure-cluster block must not report visible-span axes on a surface
that has no visible spans.

Second half of the same bug as tests/test_delimiter_feedback_gating.py. That
test covers the hand-written diagnostic blocks, which are correctly gated on
`require_delimiters`. But the feedback summary also carries a *cluster block*
built by synthesis/failure_taxonomy.py, which groups the wrong answers and
prints the axes they share. `render_cluster_block` takes no delimiter argument
at all, so it prints its span axes unconditionally:

    visible_delimiters: no (100%)
    zero_visible_spans: yes (100%)
    zero_valid_spans: yes (100%)

On Spider's token-0 surface those three are 100% by construction -- nothing can
ever emit a visible span there -- so they are constants, not symptoms. Worse,
the cluster block presents them as *what the failures have in common*, which is
the strongest possible phrasing for a fact that is equally true of the correct
answers. That is what sent the author chasing `<<`-detection branches.

Contract: when delimiters are not required, the cluster block keeps every axis
that still carries information (syntax, answer source, step budget, runtime)
and drops only the visible-span ones. When they are required, nothing changes.
"""

from __future__ import annotations

from synthesis.evaluate.evaluator import EvaluationResult


def _sample(index: int, *, visible: bool) -> dict:
    """One wrong answer. `visible` picks the surface it was produced on."""
    return {
        "question": f"q{index}",
        "question_full": f"q{index}",
        "expected": "SELECT 1",
        "actual": "SELECT 2",
        "full_output": "SELECT <<x>> FROM t" if visible else "SELECT 2",
        "scored_output": "SELECT <<x>> FROM t" if visible else "SELECT 2",
        "is_correct": False,
        "accuracy_applicable": True,
        "contains_delimiters": visible,
        "visible_delimiters": visible,
        "used_constrained_chunk": not visible,
        "is_syntax_valid": False,
        "syntax_rate": 0.0,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 1 if visible else 0,
        "num_valid_visible_spans": 1 if visible else 0,
        "has_extracted_answer": True,
        "token_count": 10,
        "time_seconds": 1.0,
        "hit_max_steps": False,
    }


def _result(*, visible: bool) -> EvaluationResult:
    samples = [_sample(i, visible=visible) for i in range(4)]
    return EvaluationResult(
        success=False,
        accuracy=0.0,
        contains_delimiters=visible,
        syntax_rate=0.0,
        num_examples=len(samples),
        num_correct=0,
        total_time_seconds=float(len(samples)),
        sample_outputs=samples,
    )


# Axes that only mean something on a surface that can emit visible spans.
SPAN_AXES = ["visible_delimiters", "zero_visible_spans", "zero_valid_spans"]

# Axes that stay informative either way -- dropping these would blind the author.
KEPT_AXES = ["syntax_valid", "answer_source", "hit_max_steps"]


def test_span_axes_are_dropped_when_delimiters_are_not_required():
    summary = _result(visible=False).get_feedback_summary(require_delimiters=False)
    for axis in SPAN_AXES:
        assert axis not in summary, (
            f"{axis!r} is 100% true by construction on a no-delimiter surface; "
            "reporting it as something the failures share tells the author to "
            "chase a constant"
        )


def test_the_cluster_block_still_reports_the_axes_that_matter():
    summary = _result(visible=False).get_feedback_summary(require_delimiters=False)
    assert "Failure clusters" in summary or "cluster" in summary.lower(), (
        "the cluster block itself must survive; only its span axes are dropped"
    )
    assert any(axis in summary for axis in KEPT_AXES), (
        f"expected at least one of {KEPT_AXES} to remain in the cluster block"
    )


def test_span_axes_survive_when_delimiters_are_required():
    """GSM really does emit visible spans, so there the axes are real evidence."""
    summary = _result(visible=True).get_feedback_summary(require_delimiters=True)
    assert any(axis in summary for axis in SPAN_AXES), (
        "on a delimiter surface the span axes are informative and must be kept"
    )


def test_render_cluster_block_takes_the_delimiter_surface_as_an_argument():
    """The leak existed because this function had no way to know the surface."""
    import inspect

    from synthesis.failure_taxonomy import render_cluster_block

    params = inspect.signature(render_cluster_block).parameters
    assert "require_delimiters" in params, (
        "render_cluster_block must be told whether visible spans are possible; "
        "without it the caller cannot suppress the span axes"
    )
