"""Greedy hill-climb search for the synthesis loop
(synthesis/evaluate/feedback_loop.py + synthesis/evaluate/evaluator.py).

Spec under test (planning/monotonic-search-plan.md):

1. The loop keeps an INCUMBENT — the best evaluated strategy so far. Every
   evaluated-and-rejected candidate is discarded: the next refinement's base
   is the incumbent's code, never the rejected candidate's.
2. Candidates are scored on slice A (the usual first-N train examples). A
   candidate whose slice-A shortfall beats the incumbent's slice-A shortfall
   is confirmed on slice B (the next N train indices, via the evaluator's new
   `sample_offset` argument). It replaces the incumbent only if the pooled
   A+B shortfall beats the incumbent's pooled A+B shortfall.
   shortfall = max(0, min_acc - acc) + max(0, min_syn - syn).
3. The first successfully evaluated candidate seeds the incumbent (A + B both
   scored), with no comparison to beat.
4. SUCCESS requires the full meets_threshold check to pass on slice A AND on
   slice B — a single-slice bar crossing no longer ends the run.
5. Slice-B numbers never appear in author-facing feedback; the incumbent is
   presented with its slice-A scores only. Rejections are announced to the
   author with a REJECTED marker.
6. The old Pareto-anchor/restart machinery is deleted outright.

All tests use fakes (no model, GPU, Dafny, or dataset); the file runs in
well under a second.
"""

from pathlib import Path

import pytest

from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.evaluate.feedback_loop import (
    FailureStage,
    SynthesisAttempt,
    SynthesisExhaustionError,
    SynthesisPipeline,
)
from synthesis.failure_taxonomy import fingerprint
from synthesis.run_synthesis import _load_initial_failure_ledger
from synthesis.verify.compiler import CompilationResult
from synthesis.verify.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Fakes (patterned on tests/test_attempt_cap_and_incremental_save.py)
# ---------------------------------------------------------------------------


class FakeGenerator:
    """Hands out a scripted sequence of strategy codes and records every
    refine_after_evaluation_failure call's kwargs for assertions."""

    def __init__(self, strategies):
        self._model = None
        self._vllm = None
        self.strategies = list(strategies)
        self.next_index = 1  # strategies[0] is the initial one
        self.generate_initial_calls = []
        self.refine_calls = []

    def set_synthesis_context(self, *args, **kwargs):
        pass

    def set_task_description(self, *args, **kwargs):
        pass

    def summarize_rationale_claim(self, rationale):
        return "summary"

    def generate_initial(
        self, task_description, allowed_helpers=None, start_inside_constrained=False
    ):
        self.generate_initial_calls.append(
            {
                "task_description": task_description,
                "allowed_helpers": allowed_helpers,
                "start_inside_constrained": start_inside_constrained,
            }
        )
        return self.strategies[0]

    def inject_strategy(self, strategy_code):
        return f"// dafny\n{strategy_code}"

    def _next(self):
        code = self.strategies[min(self.next_index, len(self.strategies) - 1)]
        self.next_index += 1
        return code

    def refine_after_verification_error(self, *args, **kwargs):
        return self._next()

    def refine_after_compilation_error(self, *args, **kwargs):
        return self._next()

    def refine_after_runtime_error(self, *args, **kwargs):
        return self._next()

    def refine_after_evaluation_failure(self, **kwargs):
        self.refine_calls.append(kwargs)
        return self._next()


class FakeVerifier:
    def verify(self, full_code):
        return VerificationResult(success=True)


class FakeCompiler:
    def __init__(self, dafny_path=None, output_dir=None, timeout=None, extra_args=None):
        self.dafny_path = dafny_path or "dafny"
        self.timeout = timeout or 120
        self.extra_args = list(extra_args) if extra_args else []

    def compile(self, full_code, output_name):
        return CompilationResult(
            success=True,
            output_dir=Path("/tmp/fake_compiled"),
            main_module_path=Path("/tmp/fake_compiled/mod.py"),
        )


@pytest.fixture(autouse=True)
def _patch_dafny_compiler(monkeypatch):
    import synthesis.evaluate.feedback_loop as feedback_loop_module

    monkeypatch.setattr(feedback_loop_module, "DafnyCompiler", FakeCompiler)


def _result(accuracy, syntax_rate, n=10, success=True, error=None):
    num_correct = round(accuracy * n)
    syntax_valid = round(syntax_rate * n)
    samples = [
        {
            "is_correct": i < num_correct,
            "is_syntax_valid": i < syntax_valid,
            "failure_location": "correct" if i < num_correct else "no_valid_visible_span",
            "time_seconds": 0.01,
        }
        for i in range(n)
    ]
    return EvaluationResult(
        success=success,
        accuracy=accuracy,
        contains_delimiters=True,
        syntax_rate=syntax_rate,
        num_examples=n,
        num_correct=num_correct,
        total_time_seconds=0.1,
        max_sample_time_seconds=0.01,
        sample_outputs=samples,
        error=error,
    )


class ScriptedEvaluator:
    """Returns a scripted EvaluationResult per call and records the
    (sample_size, sample_offset) of every call."""

    def __init__(self, results):
        self.sample_seed = 0
        self.results = list(results)
        self.calls = []  # list of dicts with sample_size / sample_offset
        self.model_name = "fake-model"
        self.dataset_name = "fake_dataset"
        self.max_steps = 1
        self.step_token_budget = 1

    def evaluate_sample(self, compiled_module_path, sample_size=None, sample_offset=0, **kwargs):
        self.calls.append({"sample_size": sample_size, "sample_offset": sample_offset})
        return self.results.pop(0)

    def split_provenance(self, bar_split_name=None):
        return {"split_file": None, "split_side": "train", "bar_split": bar_split_name}


def make_pipeline(tmp_path, evaluator, generator, max_iterations, min_accuracy=0.6, min_syntax_rate=0.9):
    return SynthesisPipeline(
        evaluator=evaluator,
        generator=generator,
        verifier=FakeVerifier(),
        compiler=FakeCompiler(),
        max_iterations=max_iterations,
        output_dir=tmp_path,
        save_reports=True,
        min_accuracy=min_accuracy,
        min_syntax_rate=min_syntax_rate,
        require_delimiters=False,
        eval_sample_size=10,
        eval_max_seconds_per_example=None,
        min_examples_before_threshold_stop=None,
        max_attempt_seconds=None,
        refinement_beam_size=1,
    )


# ---------------------------------------------------------------------------
# Pure acceptance math
# ---------------------------------------------------------------------------


def test_shortfall_is_distance_to_both_bars(tmp_path):
    p = make_pipeline(tmp_path, ScriptedEvaluator([]), FakeGenerator(["S1"]), 1)
    assert p._shortfall(0.3, 0.5) == pytest.approx(0.3 + 0.4)
    assert p._shortfall(0.6, 0.9) == pytest.approx(0.0)
    assert p._shortfall(0.9, 0.5) == pytest.approx(0.4)  # no credit above a bar


def test_paper_threshold_ignores_recorded_slowest_runtime_after_safe_evaluation(
    tmp_path,
):
    """Runtime is a safety cutoff in the evaluator, not a paper win bar."""
    result = _result(0.7, 1.0)
    result.max_sample_time_seconds = 120.0
    pipeline = make_pipeline(
        tmp_path,
        ScriptedEvaluator([]),
        FakeGenerator(["S1"]),
        1,
        min_accuracy=0.6,
        min_syntax_rate=0.9,
    )
    pipeline.eval_max_seconds_per_example = 90.0

    assert pipeline._meets_threshold(result) is True


def test_paper_threshold_uses_accuracy_and_syntax_not_delimiter_metadata():
    """Delimiter metadata and recorded runtime are not paper-win bars."""
    result = _result(0.7, 1.0)
    result.contains_delimiters = False
    result.max_sample_time_seconds = 120.0

    assert result.meets_threshold(
        min_accuracy=0.6,
        min_syntax_rate=0.9,
        require_delimiters=True,
        max_seconds_per_example=90.0,
    ) is True


def test_acceptance_rule_strict_decrease_with_lex_tiebreak(tmp_path):
    p = make_pipeline(tmp_path, ScriptedEvaluator([]), FakeGenerator(["S1"]), 1)
    # strict decrease accepts
    assert p._should_accept(0.3, 0.5, 0.7, 0.7, 0.3, 0.5) is True
    # increase rejects
    assert p._should_accept(0.8, 0.2, 0.6, 0.7, 0.3, 0.5) is False
    # equal shortfall + higher accuracy accepts
    assert p._should_accept(0.7, 0.4, 0.5, 0.7, 0.3, 0.6) is True
    # equal shortfall + same accuracy + higher syntax accepts
    assert p._should_accept(0.7, 0.3, 0.7, 0.7, 0.3, 0.5) is True
    # exact tie rejects
    assert p._should_accept(0.7, 0.3, 0.5, 0.7, 0.3, 0.5) is False


# ---------------------------------------------------------------------------
# Loop behavior: seeding, reject-no-B, reject-after-B, success-needs-B
# ---------------------------------------------------------------------------


def test_greedy_flow_rebases_on_incumbent_and_confirms_on_slice_b(tmp_path):
    # cand1 "S1": A=(.3,.5) below bars -> seeds incumbent, B=(.2,.5) also eval'd
    # cand2 "S2": A=(.2,.5) worse than incumbent's A -> REJECT, no B eval
    # cand3 "S3": A=(.5,.7) better on A -> B=(0.13,0.03) drags pooled below
    #             incumbent's pooled -> REJECT after B
    # cand4 "S4": A=(.6,.9) meets bars -> B=(.6,.9) confirms -> SUCCESS
    evaluator = ScriptedEvaluator(
        [
            _result(0.3, 0.5),   # S1 slice A
            _result(0.2, 0.5),   # S1 slice B (seeding)
            _result(0.2, 0.5),   # S2 slice A -> reject, no B
            _result(0.5, 0.7),   # S3 slice A -> stage-1 pass
            _result(0.13, 0.03), # S3 slice B -> pooled reject
            _result(0.6, 0.9),   # S4 slice A -> bars met
            _result(0.6, 0.9),   # S4 slice B -> confirmed
        ]
    )
    generator = FakeGenerator(["S1", "S2", "S3", "S4"])
    pipeline = make_pipeline(tmp_path, evaluator, generator, max_iterations=4)

    result = pipeline.synthesize(task_description="dummy", output_name="dummy")

    assert result.success is True
    assert result.strategy_code == "S4"

    # Exactly the scripted eval calls happened, with slice B at offset 10.
    assert [c["sample_offset"] for c in evaluator.calls] == [0, 10, 0, 0, 10, 0, 10]
    assert all(c["sample_size"] == 10 for c in evaluator.calls)

    # Every refinement was based on the incumbent S1 (never on rejected S2/S3),
    # quoting the incumbent's slice-A scores, not pooled and not the candidate's.
    assert len(generator.refine_calls) == 3
    for call in generator.refine_calls:
        assert call["previous_strategy"] == "S1"
        assert call["previous_accuracy"] == pytest.approx(0.3)
        assert call["previous_syntax_rate"] == pytest.approx(0.5)

    # Rejections are announced to the author...
    assert "REJECTED" in generator.refine_calls[1]["evaluation_feedback"]
    assert "REJECTED" in generator.refine_calls[2]["evaluation_feedback"]
    # ...but slice-B numbers stay out of every author-facing feedback string
    # (S3's B result 13%/3% is the canary).
    for call in generator.refine_calls:
        assert "13" not in call["evaluation_feedback"]


def test_eval_error_after_seeding_also_rebases_on_incumbent(tmp_path):
    # cand1 seeds the incumbent; cand2's evaluation errors out (but ran
    # examples, so it is not a harness failure) -> the refinement base must
    # still be the incumbent, not the errored candidate.
    evaluator = ScriptedEvaluator(
        [
            _result(0.3, 0.5),                                   # S1 slice A
            _result(0.2, 0.5),                                   # S1 slice B
            _result(0.0, 0.0, success=False, error="boom"),      # S2 slice A
        ]
    )
    generator = FakeGenerator(["S1", "S2", "S3"])
    pipeline = make_pipeline(tmp_path, evaluator, generator, max_iterations=2)

    with pytest.raises(SynthesisExhaustionError):
        pipeline.synthesize(task_description="dummy", output_name="dummy")

    assert generator.refine_calls, "expected refinement after the eval error"
    assert generator.refine_calls[-1]["previous_strategy"] == "S1"
    # the errored candidate was not confirmed on slice B
    assert [c["sample_offset"] for c in evaluator.calls] == [0, 10, 0]


def test_single_slice_bar_crossing_is_not_success(tmp_path):
    # cand1 meets both bars on slice A but misses them on slice B -> the run
    # must continue (cand1 may seed the incumbent, but no SUCCESS). cand2
    # clears both slices -> SUCCESS on attempt 2.
    evaluator = ScriptedEvaluator(
        [
            _result(0.6, 0.9),   # S1 slice A: bars met
            _result(0.1, 0.2),   # S1 slice B: bars missed -> not a win
            _result(0.7, 0.9),   # S2 slice A
            _result(0.6, 0.9),   # S2 slice B
        ]
    )
    generator = FakeGenerator(["S1", "S2"])
    pipeline = make_pipeline(tmp_path, evaluator, generator, max_iterations=2)

    result = pipeline.synthesize(task_description="dummy", output_name="dummy")

    assert result.success is True
    assert result.strategy_code == "S2"
    assert len(result.attempts) == 2


def test_warm_resume_restores_best_incumbent_before_scoring_seed(tmp_path):
    restored = [
        SynthesisAttempt(
            attempt_number=12,
            strategy_code="S12",
            full_dafny_code="",
            timestamp="restored",
            eval_result=_result(8 / 49, 45 / 49, n=49),
        ),
        SynthesisAttempt(
            attempt_number=15,
            strategy_code="S15",
            full_dafny_code="",
            timestamp="restored",
            eval_result=_result(3 / 49, 44 / 49, n=49),
        ),
    ]
    for attempt in restored:
        attempt.eval_result.planned_num_examples = 49
    evaluator = ScriptedEvaluator([_result(0.1, 0.9)])
    generator = FakeGenerator(["S16", "S17"])
    pipeline = make_pipeline(
        tmp_path,
        evaluator,
        generator,
        max_iterations=1,
        min_accuracy=13 / 49,
        min_syntax_rate=45 / 49,
    )

    with pytest.raises(SynthesisExhaustionError):
        pipeline.synthesize(
            task_description="dummy",
            output_name="warm",
            initial_strategy_code="S16",
            initial_attempt_offset=15,
            initial_attempts=restored,
        )

    assert [call["sample_offset"] for call in evaluator.calls] == [0]
    assert pipeline._incumbent is not None
    assert pipeline._incumbent.attempt_number == 12
    assert generator.refine_calls[-1]["previous_strategy"] == "S12"


def test_history_only_continuation_refines_restored_incumbent_without_replay(
    tmp_path,
):
    restored = []
    for attempt_number, strategy, accuracy in ((1, "S1", 0.2), (2, "S2", 0.5)):
        evaluation = _result(accuracy, 0.9, n=10)
        evaluation.planned_num_examples = 10
        restored.append(
            SynthesisAttempt(
                attempt_number=attempt_number,
                strategy_code=strategy,
                full_dafny_code="",
                timestamp="restored",
                eval_result=evaluation,
            )
        )
    candidate_results = [_result(0.1, 0.9), _result(0.1, 0.9)]
    generator = FakeGenerator(["INITIAL-MUST-NOT-RUN", "S3", "S4"])
    pipeline = make_pipeline(
        tmp_path,
        ScriptedEvaluator(candidate_results),
        generator,
        max_iterations=2,
    )

    with pytest.raises(SynthesisExhaustionError) as error:
        pipeline.synthesize(
            task_description="dummy",
            output_name="history-only",
            initial_attempt_offset=2,
            initial_attempts=restored,
            initial_failure_ledger={"next_id": 0, "modes": []},
        )

    assert generator.generate_initial_calls == []
    assert [call["previous_strategy"] for call in generator.refine_calls] == [
        "S2",
        "S2",
    ]
    assert "Required thresholds:" in generator.refine_calls[0]["evaluation_feedback"]
    assert "Accuracy: 50.0%" in generator.refine_calls[0]["evaluation_feedback"]
    assert [attempt.attempt_number for attempt in error.value.attempts] == [1, 2, 3, 4]
    assert [attempt.strategy_code for attempt in error.value.attempts[-2:]] == ["S3", "S4"]


def test_history_only_continuation_requires_restored_failure_ledger(tmp_path):
    evaluation = _result(0.2, 0.9, n=10)
    evaluation.planned_num_examples = 10
    pipeline = make_pipeline(
        tmp_path,
        ScriptedEvaluator([]),
        FakeGenerator(["INITIAL-MUST-NOT-RUN", "S2"]),
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="failure ledger"):
        pipeline.synthesize(
            task_description="dummy",
            output_name="history-without-ledger",
            initial_attempt_offset=1,
            initial_attempts=[
                SynthesisAttempt(
                    attempt_number=1,
                    strategy_code="S1",
                    full_dafny_code="",
                    timestamp="restored",
                    eval_result=evaluation,
                )
            ],
        )


def test_warm_resume_keeps_complete_timed_out_score_as_incumbent(tmp_path):
    restored = [
        SynthesisAttempt(
            attempt_number=1,
            strategy_code="timed-out-tie",
            full_dafny_code="",
            timestamp="restored",
            eval_result=_result(8 / 49, 45 / 49, n=49),
            failed_at=FailureStage.TIMEOUT,
        ),
        SynthesisAttempt(
            attempt_number=12,
            strategy_code="accepted-incumbent",
            full_dafny_code="",
            timestamp="restored",
            eval_result=_result(8 / 49, 45 / 49, n=49),
            failed_at=FailureStage.EVALUATION,
        ),
    ]
    for attempt in restored:
        attempt.eval_result.planned_num_examples = 49
    pipeline = make_pipeline(
        tmp_path,
        ScriptedEvaluator([]),
        FakeGenerator(["unused"]),
        max_iterations=1,
        min_accuracy=13 / 49,
        min_syntax_rate=45 / 49,
    )

    pipeline._restore_incumbent_from_attempts(restored)

    assert pipeline._incumbent is not None
    assert pipeline._incumbent.attempt_number == 1
    assert pipeline._incumbent.strategy_code == "timed-out-tie"


def test_warm_resume_restores_failure_mode_history_before_scoring_seed(tmp_path):
    candidate_result = _result(0.0, 0.9, n=10)
    wrong_sample = next(
        sample for sample in candidate_result.sample_outputs if not sample["is_correct"]
    )
    restored_ledger = {
        "next_id": 1,
        "modes": [
            {
                "id": "mode_A",
                "medoid": fingerprint(wrong_sample),
                "attempts": [1, 15],
            }
        ],
    }
    evaluator = ScriptedEvaluator([candidate_result])
    generator = FakeGenerator(["S16", "S17"])
    pipeline = make_pipeline(
        tmp_path,
        evaluator,
        generator,
        max_iterations=1,
        min_accuracy=13 / 49,
        min_syntax_rate=45 / 49,
    )

    with pytest.raises(SynthesisExhaustionError):
        pipeline.synthesize(
            task_description="dummy",
            output_name="warm-ledger",
            initial_strategy_code="S16",
            initial_attempt_offset=15,
            initial_failure_ledger=restored_ledger,
        )

    assert pipeline._failure_ledger["next_id"] == 1
    assert pipeline._failure_ledger["modes"][0]["id"] == "mode_A"
    assert pipeline._failure_ledger["modes"][0]["attempts"] == [1, 15, 16]
    assert "mode_A: appeared in attempt(s) 1,15,16" in (
        generator.refine_calls[-1]["evaluation_feedback"]
    )


def _table5_opus_history_through_attempt_38():
    """Return a contiguous restored prefix whose current-target best is attempt 38."""
    attempts = []
    for attempt_number in range(1, 38):
        result = _result(0 / 49, 40 / 49, n=49)
        result.planned_num_examples = 49
        attempts.append(
            SynthesisAttempt(
                attempt_number=attempt_number,
                strategy_code=f"S{attempt_number}",
                full_dafny_code="",
                timestamp="restored",
                eval_result=result,
                failed_at=FailureStage.EVALUATION,
            )
        )

    incumbent = _result(11 / 49, 45 / 49, n=49)
    incumbent.planned_num_examples = 49
    attempts.append(
        SynthesisAttempt(
            attempt_number=38,
            strategy_code="S38",
            full_dafny_code="",
            timestamp="restored",
            eval_result=incumbent,
        )
    )
    return attempts


def _fixed_table5_opus_pipeline(tmp_path, results, generator):
    pipeline = make_pipeline(
        tmp_path,
        ScriptedEvaluator(results),
        generator,
        max_iterations=2,
        min_accuracy=20 / 49,
        min_syntax_rate=47 / 49,
    )
    pipeline.eval_sample_size = 49
    return pipeline


def test_fixed_warm_continuation_refines_before_39_and_spends_both_new_iterations(
    tmp_path,
):
    attempt_39 = _result(20 / 49, 47 / 49, n=49)
    attempt_39.planned_num_examples = 49
    attempt_40 = _result(4 / 49, 42 / 49, n=49)
    attempt_40.planned_num_examples = 49
    generator = FakeGenerator(["must-not-be-replayed", "S39", "S40"])
    pipeline = _fixed_table5_opus_pipeline(
        tmp_path, [attempt_39, attempt_40], generator
    )

    result = pipeline.synthesize(
        task_description="dummy",
        output_name="opus-fixed-warm",
        initial_strategy_code="S38",
        initial_attempt_offset=38,
        initial_attempts=_table5_opus_history_through_attempt_38(),
        fixed_warm_continuation=True,
    )

    assert generator.generate_initial_calls == []
    assert len(generator.refine_calls) == 2
    assert generator.refine_calls[0]["previous_strategy"] == "S38"
    assert generator.refine_calls[0]["previous_accuracy"] == pytest.approx(11 / 49)
    assert generator.refine_calls[0]["previous_syntax_rate"] == pytest.approx(45 / 49)
    assert [attempt.attempt_number for attempt in result.attempts[-2:]] == [39, 40]
    assert [attempt.strategy_code for attempt in result.attempts[-2:]] == ["S39", "S40"]
    assert len(result.attempts) == 40
    assert result.success is True
    assert result.strategy_code == "S39"


def test_fixed_warm_continuation_returns_restored_incumbent_when_new_attempts_regress(
    tmp_path,
):
    attempt_39 = _result(5 / 49, 43 / 49, n=49)
    attempt_39.planned_num_examples = 49
    attempt_40 = _result(10 / 49, 44 / 49, n=49)
    attempt_40.planned_num_examples = 49
    generator = FakeGenerator(["must-not-be-replayed", "S39", "S40"])
    pipeline = _fixed_table5_opus_pipeline(
        tmp_path, [attempt_39, attempt_40], generator
    )

    result = pipeline.synthesize(
        task_description="dummy",
        output_name="opus-fixed-warm-regressions",
        initial_strategy_code="S38",
        initial_attempt_offset=38,
        initial_attempts=_table5_opus_history_through_attempt_38(),
        fixed_warm_continuation=True,
    )

    assert [attempt.attempt_number for attempt in result.attempts[-2:]] == [39, 40]
    assert len(result.attempts) == 40
    assert result.success is False
    assert result.strategy_code == "S38"
    assert result.full_dafny_code == "// dafny\nS38"
    assert pipeline._incumbent is not None
    assert pipeline._incumbent.attempt_number == 38


@pytest.mark.parametrize(
    ("history", "offset", "seed", "message"),
    [
        ([], 0, "S38", "history"),
        (_table5_opus_history_through_attempt_38(), 37, "S38", "offset"),
        (_table5_opus_history_through_attempt_38(), 38, "not-restored", "seed"),
    ],
)
def test_fixed_warm_continuation_rejects_invalid_restored_state(
    tmp_path, history, offset, seed, message
):
    pipeline = _fixed_table5_opus_pipeline(
        tmp_path, [], FakeGenerator(["unused", "S39", "S40"])
    )

    with pytest.raises(ValueError, match=message):
        pipeline.synthesize(
            task_description="dummy",
            output_name="invalid-fixed-warm",
            initial_strategy_code=seed,
            initial_attempt_offset=offset,
            initial_attempts=history,
            fixed_warm_continuation=True,
        )


def test_worker_hard_timeout_is_recorded_and_search_continues(tmp_path):
    worker_timeout = EvaluationResult(
        success=False,
        accuracy=0.0,
        contains_delimiters=False,
        syntax_rate=0.0,
        num_examples=0,
        num_correct=0,
        total_time_seconds=32.0,
        error="[worker-hard-timeout] worker 0 exceeded 32.00s hard response deadline",
    )
    evaluator = ScriptedEvaluator([worker_timeout, _result(0.7, 1.0)])
    generator = FakeGenerator(["S1", "S2"])
    pipeline = make_pipeline(
        tmp_path,
        evaluator,
        generator,
        max_iterations=2,
        min_accuracy=0.6,
        min_syntax_rate=0.9,
    )

    result = pipeline.synthesize(task_description="dummy", output_name="timeout")

    assert result.success is True
    assert len(result.attempts) == 2
    assert result.attempts[0].failed_at.value == "timeout"
    assert result.attempts[0].eval_result.sample_outputs == []


def test_initial_failure_ledger_loader_restores_tuple_medoids(tmp_path):
    ledger_path = tmp_path / "failure-ledger.json"
    medoid = list(fingerprint(_result(0.0, 0.9, n=1).sample_outputs[0]))
    ledger_path.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "source_report_sha256": "a" * 64,
                "included_attempts": [1, 15],
                "excluded_attempts": [6],
                "ledger": {
                    "next_id": 1,
                    "modes": [
                        {"id": "mode_A", "medoid": medoid, "attempts": [1, 15]}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    restored = _load_initial_failure_ledger(ledger_path)

    assert restored["next_id"] == 1
    assert restored["modes"][0]["medoid"] == tuple(medoid)
    assert restored["modes"][0]["attempts"] == [1, 15]


# ---------------------------------------------------------------------------
# Pass->fail flip diff: rejected candidates' feedback shows, per example that
# the incumbent got right and the candidate got wrong, both helper-call traces
# and both extracted answers — so the author can see WHAT the mutation broke.
# Slice-A only (slice B stays secret). Format contract:
#   - a "pass->fail" section listing each regressed example by index, with the
#     incumbent's and candidate's `actual` answers and helper_trace call
#     sequences (helper names in order);
#   - at most 3 regressed examples rendered in full, a "more" note for the rest;
#   - a one-line "fail->pass" count (no traces for improvements);
#   - empty string when nothing flipped either way.
# ---------------------------------------------------------------------------


def _sample(is_correct, actual, trace_helpers):
    return {
        "is_correct": is_correct,
        "is_syntax_valid": True,
        "actual": actual,
        "helper_trace": [{"helper": h, "detail": ""} for h in trace_helpers],
        "time_seconds": 0.01,
    }


def _result_with_samples(accuracy, syntax_rate, samples, success=True, error=None):
    return EvaluationResult(
        success=success,
        accuracy=accuracy,
        contains_delimiters=True,
        syntax_rate=syntax_rate,
        num_examples=len(samples),
        num_correct=sum(1 for s in samples if s["is_correct"]),
        total_time_seconds=0.1,
        max_sample_time_seconds=0.01,
        sample_outputs=samples,
        error=error,
    )


def test_flip_diff_renders_pass_to_fail_examples_with_both_traces(tmp_path):
    p = make_pipeline(tmp_path, ScriptedEvaluator([]), FakeGenerator(["S1"]), 1)
    incumbent = _result_with_samples(
        0.67, 1.0,
        [
            _sample(True, "42", ["StepA", "StepB"]),   # flips to wrong
            _sample(True, "7", ["StepA"]),             # stays right -> not listed
            _sample(False, "0", ["StepA"]),            # flips to right -> count only
        ],
    )
    candidate = _result_with_samples(
        0.67, 1.0,
        [
            _sample(False, "41", ["StepA", "StepC"]),
            _sample(True, "7", ["StepA"]),
            _sample(True, "5", ["StepA"]),
        ],
    )

    out = p._render_flip_diff(incumbent, candidate)

    assert "pass->fail" in out
    # the regressed example: both answers and both traces, paired by position
    assert "42" in out and "41" in out
    assert "StepB" in out and "StepC" in out
    # the example that stayed correct is not listed
    assert "example 1" not in out
    # improvements are a count, not a trace dump
    assert "fail->pass" in out

    # nothing flipped -> no section at all
    assert p._render_flip_diff(incumbent, incumbent) == ""


def test_flip_diff_caps_rendered_examples_at_three(tmp_path):
    p = make_pipeline(tmp_path, ScriptedEvaluator([]), FakeGenerator(["S1"]), 1)
    incumbent = _result_with_samples(
        1.0, 1.0, [_sample(True, f"g{i}", ["T"]) for i in range(5)]
    )
    candidate = _result_with_samples(
        0.0, 1.0, [_sample(False, f"c{i}", ["T"]) for i in range(5)]
    )

    out = p._render_flip_diff(incumbent, candidate)

    rendered = sum(1 for i in range(5) if f"c{i}" in out)
    assert rendered == 3
    assert "more" in out  # the 2 unrendered flips are acknowledged


def test_reject_feedback_contains_flip_diff_but_never_slice_b(tmp_path):
    # S1 seeds the incumbent (A then B). S2 is worse on slice A with example 0
    # flipping pass->fail -> stage-1 reject; its refinement feedback must carry
    # the flip diff built from SLICE-A results only.
    evaluator = ScriptedEvaluator(
        [
            _result_with_samples(
                0.5, 1.0,
                [_sample(True, "42", ["SeedTrace"]), _sample(False, "0", ["SeedTrace"])],
            ),  # S1 slice A
            _result_with_samples(
                0.5, 1.0,
                [_sample(True, "9", ["SecretB"]), _sample(False, "0", ["SecretB"])],
            ),  # S1 slice B
            _result_with_samples(
                0.0, 1.0,
                [_sample(False, "41", ["CandTrace"]), _sample(False, "0", ["CandTrace"])],
            ),  # S2 slice A -> reject, flip on example 0
        ]
    )
    generator = FakeGenerator(["S1", "S2", "S3"])
    pipeline = make_pipeline(tmp_path, evaluator, generator, max_iterations=2)

    with pytest.raises(SynthesisExhaustionError):
        pipeline.synthesize(task_description="dummy", output_name="dummy")

    feedback = generator.refine_calls[-1]["evaluation_feedback"]
    assert "pass->fail" in feedback
    assert "SeedTrace" in feedback and "CandTrace" in feedback
    assert "42" in feedback and "41" in feedback
    assert "SecretB" not in feedback


# ---------------------------------------------------------------------------
# The replaced machinery is gone (one best-tracker: the incumbent)
# ---------------------------------------------------------------------------


def test_pareto_anchor_and_restart_machinery_deleted(tmp_path):
    pipeline = make_pipeline(tmp_path, ScriptedEvaluator([]), FakeGenerator(["S1"]), 1)
    for name in (
        "_compute_pareto_best",
        "_update_anchor_state",
        "_should_restart",
        "_apply_restart_cooldown",
        "_lookup_best_so_far",
    ):
        assert not hasattr(pipeline, name), f"{name} should be deleted"
    assert not hasattr(pipeline, "restart_after_stuck_iters")
    assert not hasattr(pipeline, "restart_cooldown_iters")
