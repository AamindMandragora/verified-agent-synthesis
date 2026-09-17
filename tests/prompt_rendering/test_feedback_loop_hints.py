"""Golden (characterization) tests for the near-pure feedback_loop.py builders.

Covers the 5 delimiter/token hint functions + SynthesisExhaustionError.get_failure_summary,
per-branch (each hint's non-empty message branch + its empty guards).

REFACTOR GUARDS: GREEN on current code, must STAY byte-identical after each builder
is converted to the pydantic-model + Jinja pattern. These carry NO descriptive change.

Regenerate goldens from current code:  REGEN_GOLDENS=1 pytest <thisfile>
(only against known-good current code, never after a conversion).
"""
import os
import pathlib

import pytest

from synthesis.evaluate.feedback_loop import (
    FailureStage,
    SynthesisAttempt,
    SynthesisExhaustionError,
    _constraint_bypassed_hint,
    _delimiter_miss_hint,
    _final_span_failure_hint,
    _span_not_closed_hint,
    _token_cap_exhaustion_hint,
)
from synthesis.verify.verifier import VerificationResult
from synthesis.verify.compiler import CompilationResult
from synthesis.evaluate.evaluator import EvaluationResult

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "feedback_loop"


def _check(name: str, produced: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.golden.txt"
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(produced)
        pytest.skip(f"regenerated {path.name}")
    assert path.read_text() == produced


def _s(**kw):
    """One sample_outputs dict with the keys the hints inspect."""
    return kw


# ---------------------------------------------------------------------------
# _delimiter_miss_hint
# ---------------------------------------------------------------------------

def test_delimiter_miss_not_required_is_empty():
    _check("delim_miss__not_required", _delimiter_miss_hint(False, False, None))


def test_delimiter_miss_present_is_empty():
    _check("delim_miss__present", _delimiter_miss_hint(True, True, None))


def test_delimiter_miss_default_message():
    # require, no delimiters, no samples -> default force-delimiter message
    _check("delim_miss__default", _delimiter_miss_hint(True, False, None))


def test_delimiter_miss_open_not_closed_message():
    # >=20% of samples opened "<<" but never closed ">>"  -> root-cause-2 message
    samples = [
        _s(full_output="foo <<", contains_delimiters=False),
        _s(full_output="bar <<", contains_delimiters=False),
        _s(full_output="baz", contains_delimiters=False),
        _s(full_output="qux", contains_delimiters=False),
        _s(full_output="quux", contains_delimiters=False),
    ]
    _check("delim_miss__open_not_closed", _delimiter_miss_hint(True, False, samples))


# ---------------------------------------------------------------------------
# _token_cap_exhaustion_hint
# ---------------------------------------------------------------------------

def test_token_cap_no_samples_is_empty():
    _check("token_cap__no_samples", _token_cap_exhaustion_hint(None, 200))


def test_token_cap_below_threshold_is_empty():
    samples = [_s(hit_max_steps=True), _s(hit_max_steps=False), _s(hit_max_steps=False), _s(hit_max_steps=False)]
    _check("token_cap__below_threshold", _token_cap_exhaustion_hint(samples, 200))


def test_token_cap_message():
    samples = [_s(hit_max_steps=True), _s(hit_max_steps=True), _s(hit_max_steps=True), _s(hit_max_steps=False)]
    _check("token_cap__message", _token_cap_exhaustion_hint(samples, 200))


# ---------------------------------------------------------------------------
# _span_not_closed_hint
# ---------------------------------------------------------------------------

def test_span_not_closed_not_required_is_empty():
    _check("span_not_closed__not_required", _span_not_closed_hint(False, [_s(full_output="<<")]))


def test_span_not_closed_message():
    # 2/10 opened "<<" without ">>"  -> 20% >= 10% threshold
    samples = [_s(full_output="a <<", contains_delimiters=False),
               _s(full_output="b <<", contains_delimiters=False)]
    samples += [_s(full_output="ok <<x>>", contains_delimiters=True) for _ in range(8)]
    _check("span_not_closed__message", _span_not_closed_hint(True, samples))


# ---------------------------------------------------------------------------
# _constraint_bypassed_hint
# ---------------------------------------------------------------------------

def test_constraint_bypassed_delimiters_absent_is_empty():
    _check("constraint_bypassed__absent", _constraint_bypassed_hint(True, False, [_s(contains_delimiters=True)]))


def test_constraint_bypassed_message():
    # 5 relevant samples, only 1 engaged the constrained branch (4 bypassed)
    samples = [
        _s(contains_delimiters=True, used_constrained_chunk=True),
        _s(contains_delimiters=True, used_constrained_chunk=False),
        _s(contains_delimiters=True, used_constrained_chunk=False),
        _s(contains_delimiters=True, used_constrained_chunk=False),
        _s(contains_delimiters=True, used_constrained_chunk=False),
    ]
    _check("constraint_bypassed__message", _constraint_bypassed_hint(True, True, samples))


# ---------------------------------------------------------------------------
# _final_span_failure_hint
# ---------------------------------------------------------------------------

def test_final_span_failure_not_required_is_empty():
    _check("final_span__not_required", _final_span_failure_hint(False, [_s(full_output="<<")]))


def test_final_span_failure_all_categories():
    samples = [
        _s(full_output="reasoning then <<", is_syntax_valid=False),      # unclosed
        _s(full_output="no span here at all", is_syntax_valid=False),    # no_span
        _s(full_output="answer <<3 ** 2>>", is_syntax_valid=False),      # invalid (closed)
    ]
    _check("final_span__all_categories", _final_span_failure_hint(True, samples))


# ---------------------------------------------------------------------------
# SynthesisExhaustionError.get_failure_summary
# ---------------------------------------------------------------------------

def _attempt(n, failed_at=None, error_summary="", succeeded_stack=False):
    a = SynthesisAttempt(
        attempt_number=n, strategy_code="// s", full_dafny_code="// d", timestamp="2026-01-01T00:00:00",
        failed_at=failed_at, error_summary=error_summary,
    )
    if succeeded_stack:
        a.verification_result = VerificationResult(success=True)
        a.compilation_result = CompilationResult(success=True)
        a.eval_result = EvaluationResult(
            success=True, accuracy=1.0, contains_delimiters=True, syntax_rate=1.0,
            num_examples=1, num_correct=1, total_time_seconds=1.0,
        )
    return a


def test_get_failure_summary_no_attempts():
    _check("failure_summary__none", SynthesisExhaustionError("m", []).get_failure_summary())


def test_get_failure_summary_mixed():
    attempts = [
        _attempt(1, FailureStage.VERIFICATION, "verify blew up"),
        _attempt(2, FailureStage.EVALUATION, "e" * 250),  # >200 -> truncation branch
        _attempt(3, None, succeeded_stack=True),           # succeeded() True -> ✓ SUCCESS
    ]
    _check("failure_summary__mixed", SynthesisExhaustionError("m", attempts).get_failure_summary())


def test_get_failure_summary_with_report_path():
    attempts = [_attempt(1, FailureStage.COMPILATION, "compile fail")]
    err = SynthesisExhaustionError("m", attempts, report_path=pathlib.Path("/runs/r1/report.json"))
    _check("failure_summary__with_report", err.get_failure_summary())
