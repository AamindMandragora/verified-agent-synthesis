"""Red-first tests for the two approved CONTENT fixes to the eval-feedback family.

These are BUG-REPRODUCING tests: each must FAIL on the current code and PASS after
the fix. Unlike the golden characterization tests (test_feedback_summary.py), these
intentionally change behavior.

Fix 1a — `_CONSTRAINED_HELPERS` is missing 11 traced constrained-step helper names,
so a sample whose only constrained activity was e.g. BoostedConstrainedStep is
wrongly counted as "no_constrained_activity" / "examples_without_activity" in the
feedback the author LLM reads.

Fix C/D — `_summarize_failure_modes` classifies `<<`/`>>` failures with raw
substring-PRESENCE checks (`"<<" in out and ">>" not in out`). These are blind to
multiple spans, order, and counts: a multi-span output whose LAST span is
unterminated has both `<<` and `>>` present, so it is never flagged as
unterminated; and a multi-span output with one valid + one invalid closed span is
never flagged as malformed unless syntax_rate is exactly 0.0. The fix uses the
parser's real span accounting (delimiter counts + num_visible_spans /
num_valid_visible_spans, both already on the sample dict).
"""
import pytest

from synthesis.evaluate.evaluator import EvaluationResult


# ---------------------------------------------------------------------------
# Fix 1a: helpers missing from _CONSTRAINED_HELPERS
# ---------------------------------------------------------------------------

# Traced by _attach_helper_trace but absent from _CONSTRAINED_HELPERS today.
PREVIOUSLY_MISSING_CONSTRAINED_HELPERS = [
    "BoostedConstrainedStep",
    "PenalizedConstrainedStep",
    "RepetitionPenaltyStep",
    "TemperatureConstrainedStep",
    "AdaptiveConstrainedStepWithPenalties",
    "SpeculativeConstrainedRollout",
    "RolloutConstrainedWithPenalties",
    "ConstrainedGeneration",
    "CraneGeneration",
    "RollbackAndContinue",
    "RegenerateUnitOnCheckFailure",
]


@pytest.mark.parametrize("helper_name", PREVIOUSLY_MISSING_CONSTRAINED_HELPERS)
def test_constrained_step_helper_counts_as_constrained_activity(helper_name):
    """A sample whose only helper call is one of these real constrained-step
    helpers must count as constrained activity (RED today: all return False)."""
    sample = {"helper_trace": [{"helper": helper_name}]}
    assert EvaluationResult._sample_has_constrained_activity(sample) is True


def test_unconstrained_helper_still_not_constrained_activity():
    """Guard: purely free-generation helpers must STILL not count (no over-broadening)."""
    sample = {"helper_trace": [{"helper": "UnconstrainedChunk"}, {"helper": "UnconstrainedStep"}]}
    assert EvaluationResult._sample_has_constrained_activity(sample) is False


# ---------------------------------------------------------------------------
# Fix C/D: failure-mode bracket accounting
# ---------------------------------------------------------------------------

def _make_result(sample):
    """Minimal EvaluationResult carrying a single crafted sample."""
    return EvaluationResult(
        success=True,
        accuracy=0.0,
        contains_delimiters=True,
        syntax_rate=float(sample.get("syntax_rate", 0.0)),
        num_examples=1,
        num_correct=0,
        total_time_seconds=1.0,
        sample_outputs=[sample],
    )


def _mode_keys(result):
    return {mode for mode, _count, _detail in result._summarize_failure_modes()}


def test_unterminated_last_span_in_multi_span_output_is_flagged():
    """Two spans, first closed, last left open: `<<` and `>>` both appear, so the
    presence check misses it. Real accounting (count `<<` > count `>>`) catches it.
    RED today (misclassified as malformed instead of unterminated)."""
    sample = {
        "error": None,
        "full_output": "The answer is <<a + b>> and then <<c + d",
        "actual": "c + d",
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": True,
        "syntax_rate": 0.0,
        "is_correct": False,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 1,
        "num_valid_visible_spans": 1,
    }
    assert "unterminated_constrained_segment" in _mode_keys(_make_result(sample))


def test_partial_invalid_multi_span_output_is_flagged_malformed():
    """Two closed spans, one valid one invalid (num_valid < num_visible), with a
    nonzero syntax_rate. The `syntax_rate == 0.0` gate misses it today.
    RED today (not flagged malformed)."""
    sample = {
        "error": None,
        "full_output": "The result is <<good>> and also <<bad>> at the end",
        "actual": "bad",
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": True,
        "syntax_rate": 0.5,
        "is_correct": False,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 2,
        "num_valid_visible_spans": 1,
    }
    assert "malformed_constrained_content" in _mode_keys(_make_result(sample))


def test_unmatched_close_in_multi_span_output_is_flagged():
    """More `>>` than `<<` (a stray close) must be flagged as unmatched closure,
    even when a `<<` also appears elsewhere. RED today (presence check misses it)."""
    sample = {
        "error": None,
        "full_output": "stray close >> then a real one <<x>> and another close >>",
        "actual": "x",
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": True,
        "syntax_rate": 0.0,
        "is_correct": False,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 1,
        "num_valid_visible_spans": 1,
    }
    assert "premature_or_unmatched_closure" in _mode_keys(_make_result(sample))


# ---------------------------------------------------------------------------
# Sibling-site consistency: the representative-example picker
# (_pick_representative_samples_by_mode) classifies each sample with the SAME
# cascade as _summarize_failure_modes. If they drift, the counts the author
# sees and the example shown for a mode disagree. These pin that they agree.
# ---------------------------------------------------------------------------

_SIBLING_SAMPLES = [
    # unterminated last span in a multi-span output
    {
        "error": None,
        "full_output": "The answer is <<a + b>> and then <<c + d",
        "actual": "c + d",
        "contains_delimiters": True, "visible_delimiters": True,
"used_constrained_chunk": True,
        "syntax_rate": 0.0, "is_correct": False, "runtime_budget_exceeded": False,
        "num_visible_spans": 1, "num_valid_visible_spans": 1,
    },
    # partial-invalid closed spans, nonzero syntax_rate
    {
        "error": None,
        "full_output": "The result is <<good>> and also <<bad>> at the end",
        "actual": "bad",
        "contains_delimiters": True, "visible_delimiters": True,
"used_constrained_chunk": True,
        "syntax_rate": 0.5, "is_correct": False, "runtime_budget_exceeded": False,
        "num_visible_spans": 2, "num_valid_visible_spans": 1,
    },
    # stray close alongside a real span
    {
        "error": None,
        "full_output": "stray close >> then a real one <<x>> and another close >>",
        "actual": "x",
        "contains_delimiters": True, "visible_delimiters": True,
"used_constrained_chunk": True,
        "syntax_rate": 0.0, "is_correct": False, "runtime_budget_exceeded": False,
        "num_visible_spans": 1, "num_valid_visible_spans": 1,
    },
]


@pytest.mark.parametrize("sample", _SIBLING_SAMPLES)
def test_picker_and_summary_agree_on_modes(sample):
    """The representative-sample picker and the failure-mode summary must assign
    the SAME failure-mode keys to a sample (RED today: the picker still uses the
    old presence checks and disagrees on these multi-span cases)."""
    result = _make_result(sample)
    summary_modes = {mode for mode, _c, _d in result._summarize_failure_modes()}
    picker_modes = {mode for mode, _s in result._pick_representative_samples_by_mode()}
    assert picker_modes == summary_modes
