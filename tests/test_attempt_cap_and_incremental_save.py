"""Tests for two related fixes to the synthesis feedback loop
(synthesis/evaluate/feedback_loop.py, synthesis/evaluate/evaluator.py):

1. An attempt (one generate -> verify -> compile -> evaluate cycle) can
   currently run forever -- a bad strategy can loop to its max steps on
   every example and burn hours of GPU time with no ceiling. There must be
   a wall-clock cap on a single attempt; hitting it stops that attempt
   cleanly, marks it as timed out (not a crash, not a legitimate score),
   and lets the loop move on to the next attempt.

2. Per-example evaluation records for an attempt are currently written to
   disk only once, at the very end of the whole run -- a killed or crashed
   run loses everything already computed. Progress must reach disk after
   each attempt finishes, not only when the run ends.

All tests use fakes (no real model, GPU, or benchmark) and tiny
fractional-second caps/delays so the whole file runs in well under a
second.
"""

import time
from pathlib import Path

import pytest

from synthesis.evaluate.evaluator import ATTEMPT_DEADLINE_EARLY_STOP_REASON, EvaluationResult
from synthesis.evaluate.feedback_loop import FailureStage, SynthesisExhaustionError, SynthesisPipeline
from synthesis.verify.compiler import CompilationResult
from synthesis.verify.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Fakes -- stand-ins for the real generator/verifier/compiler/evaluator so
# the pipeline can be driven end to end without touching a model, Dafny, or
# a real dataset.
# ---------------------------------------------------------------------------


class FakeGenerator:
    """Always hands back the same strategy string; never touches a real model."""

    def __init__(self):
        self._model = None
        self._vllm = None
        self.generate_initial_calls = 0

    def set_synthesis_context(self, *args, **kwargs):
        pass
    def set_task_description(self, task_description):
        self.task_description = task_description

    def generate_initial(
        self, task_description, allowed_helpers=None, start_inside_constrained=False
    ):
        # Mirrors the real StrategyGenerator.generate_initial. Keep the two in
        # step: the pipeline passes start_inside_constrained by keyword, so a
        # stand-in that lacks it fails with a TypeError that has nothing to do
        # with attempt caps.
        self.generate_initial_calls += 1
        return "STRATEGY"

    def inject_strategy(self, strategy_code):
        return f"// dafny\n{strategy_code}"

    def refine_after_verification_error(self, *args, **kwargs):
        return "STRATEGY"

    def refine_after_compilation_error(self, *args, **kwargs):
        return "STRATEGY"

    def refine_after_runtime_error(self, *args, **kwargs):
        return "STRATEGY"

    def refine_after_evaluation_failure(self, *args, **kwargs):
        return "STRATEGY"


class FakeVerifier:
    def verify(self, full_code):
        return VerificationResult(success=True)


class FakeCompiler:
    """Stands in both for the pipeline's `self.compiler` (constructed with no
    args) and for the real DafnyCompiler class that synthesize() otherwise
    re-constructs internally on every run from self.compiler's attributes
    (see the autouse `_patch_dafny_compiler` fixture below) -- so its
    constructor accepts (and ignores) the same keyword arguments as the real
    DafnyCompiler.
    """

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
    """synthesize() does not call self.compiler.compile() directly -- it
    re-constructs a fresh `DafnyCompiler(...)` per run (for a per-run output
    directory) from self.compiler's attributes. Patch the class the module
    looks up so that reconstruction also produces a fake, never a real Dafny
    subprocess invocation.
    """
    import synthesis.evaluate.feedback_loop as feedback_loop_module

    monkeypatch.setattr(feedback_loop_module, "DafnyCompiler", FakeCompiler)


class FakeEvaluator:
    """Simulates the real Evaluator.evaluate_sample contract: it produces
    per-example records one at a time (a fake slow "unit of work" is a
    time.sleep between examples) and, if a `deadline` is passed and gets
    crossed before all examples finish, returns early with the partial
    records plus an early_stop_reason carrying the attempt-deadline marker
    -- exactly like the real sequential evaluation loop does.

    `seconds_per_example` and `num_examples` are fixed per instance;
    `runs` records one dict per call for assertions.
    """

    def __init__(self, seconds_per_example: float, num_examples: int = 5):
        self.sample_seed = 0
        self.seconds_per_example = seconds_per_example
        self.num_examples = num_examples
        self.runs = []
        # Attributes SynthesisPipeline.synthesize() reads directly off the
        # evaluator (for provenance/context, not for behavior).
        self.model_name = "fake-model"
        self.dataset_name = "fake_dataset"
        self.max_steps = 1
        self.step_token_budget = 1

    def split_provenance(self, bar_split_name):
        return {"bar_split_name": bar_split_name}

    def evaluate_sample(
        self,
        compiled_module_path,
        sample_size=None,
        early_stop_min_accuracy=None,
        early_stop_min_syntax_rate=None,
        early_stop_runtime_failures=None,
        min_examples_before_threshold_stop=None,
        deadline=None,
    ):
        start = time.time()
        sample_outputs = []
        early_stop_reason = None
        for i in range(self.num_examples):
            time.sleep(self.seconds_per_example)
            sample_outputs.append({"index": i, "is_correct": True})
            if deadline is not None and time.time() >= deadline:
                early_stop_reason = (
                    f"{ATTEMPT_DEADLINE_EARLY_STOP_REASON}; "
                    f"N={len(sample_outputs)} of {self.num_examples} examples completed"
                )
                break
        result = EvaluationResult(
            success=True,
            accuracy=1.0,
            contains_delimiters=True,
            syntax_rate=1.0,
            num_examples=len(sample_outputs),
            num_correct=len(sample_outputs),
            total_time_seconds=time.time() - start,
            sample_outputs=sample_outputs,
            early_stopped=early_stop_reason is not None,
            early_stop_reason=early_stop_reason,
        )
        self.runs.append({"deadline": deadline, "result": result})
        return result


def make_pipeline(tmp_path, evaluator, max_attempt_seconds, max_iterations=3):
    return SynthesisPipeline(
        evaluator=evaluator,
        generator=FakeGenerator(),
        verifier=FakeVerifier(),
        compiler=FakeCompiler(),
        max_iterations=max_iterations,
        output_dir=tmp_path,
        save_reports=True,
        min_accuracy=0.0,
        min_syntax_rate=0.0,
        require_delimiters=False,
        eval_sample_size=5,
        eval_max_seconds_per_example=None,
        min_examples_before_threshold_stop=None,
        max_attempt_seconds=max_attempt_seconds,
    )


def _progress_report_path(pipeline, run_dir_glob="*"):
    results_dirs = list(Path(pipeline.output_dir).glob(f"{run_dir_glob}/results/progress_report.json"))
    assert results_dirs, "expected a progress_report.json to have been written"
    return results_dirs[0]


# ---------------------------------------------------------------------------
# (i) An attempt that exceeds the cap is stopped and marked as timed out;
#     the loop moves on rather than dying.
# ---------------------------------------------------------------------------


def test_attempt_exceeding_cap_is_marked_timed_out_and_loop_continues(tmp_path):
    # Each fake eval takes 5 * 0.05s = 0.25s; the cap is far tighter than
    # that, so every attempt should time out. max_iterations=2 keeps the
    # test itself fast.
    evaluator = FakeEvaluator(seconds_per_example=0.05, num_examples=5)
    pipeline = make_pipeline(tmp_path, evaluator, max_attempt_seconds=0.05, max_iterations=2)

    with pytest.raises(SynthesisExhaustionError) as exc_info:
        pipeline.synthesize(task_description="dummy task", output_name="dummy")

    attempts = exc_info.value.attempts
    assert len(attempts) == 2, "loop should have moved on to a second attempt, not died"
    for attempt in attempts:
        assert attempt.failed_at == FailureStage.TIMEOUT
        assert "timed out" in attempt.error_summary.lower() or "budget" in attempt.error_summary.lower()


# ---------------------------------------------------------------------------
# (ii) An attempt that finishes within the cap is unaffected: no behavior
#      change, and it is NOT marked as timed out.
# ---------------------------------------------------------------------------


def test_attempt_within_cap_is_not_marked_timed_out(tmp_path):
    # Each fake eval takes 5 * 0.01s = 0.05s; cap is generous (10s), so the
    # attempt finishes comfortably inside its budget.
    evaluator = FakeEvaluator(seconds_per_example=0.01, num_examples=5)
    pipeline = make_pipeline(tmp_path, evaluator, max_attempt_seconds=10.0, max_iterations=1)

    # min_accuracy/min_syntax_rate are 0.0 and the fake eval always
    # "succeeds" with accuracy=1.0, so this should be a normal SUCCESS.
    result = pipeline.synthesize(task_description="dummy task", output_name="dummy")

    assert result.success is True
    assert len(result.attempts) == 1
    assert result.attempts[0].failed_at is None
    assert result.attempts[0].eval_result.early_stopped is False


# ---------------------------------------------------------------------------
# (iii) Per-example (attempt) records reach disk before the whole run
#       finishes -- not only in the final report.
# ---------------------------------------------------------------------------


class _CrashingOnSecondCallVerifier:
    """Verifies attempt 1 normally, then raises on attempt 2 -- simulating
    the process dying partway through a multi-attempt run."""

    def __init__(self):
        self.calls = 0

    def verify(self, full_code):
        self.calls += 1
        if self.calls == 1:
            return VerificationResult(success=True)
        raise RuntimeError("simulated crash mid-run")


def test_progress_is_written_to_disk_before_run_completes(tmp_path):
    # min_accuracy is impossible to meet, so attempt 1 finishes (with its
    # per-example eval records) but does not succeed -> the loop moves on to
    # attempt 2, which crashes during verification. If incremental saving
    # works, attempt 1's finished records are already on disk even though
    # synthesize() never reaches a normal return or the run-ending report.
    evaluator = FakeEvaluator(seconds_per_example=0.0, num_examples=3)

    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=FakeGenerator(),
        verifier=_CrashingOnSecondCallVerifier(),
        compiler=FakeCompiler(),
        max_iterations=3,
        output_dir=tmp_path,
        save_reports=True,
        min_accuracy=2.0,  # impossible to meet -> attempt 1 fails, loop continues
        min_syntax_rate=0.0,
        require_delimiters=False,
        eval_sample_size=3,
        eval_max_seconds_per_example=None,
        min_examples_before_threshold_stop=None,
        max_attempt_seconds=10.0,
    )

    with pytest.raises(RuntimeError, match="simulated crash mid-run"):
        pipeline.synthesize(task_description="dummy task", output_name="dummy")

    report_path = _progress_report_path(pipeline)
    with open(report_path) as f:
        import json

        data = json.load(f)

    assert data["total_attempts"] == 1
    assert len(data["attempts"]) == 1
    assert data["attempts"][0]["evaluation"]["sample_outputs"], (
        "the finished attempt's per-example eval records must already be on disk"
    )


# ---------------------------------------------------------------------------
# (iv) A timed-out attempt keeps the records it produced before the cap
#      hit -- they are not discarded.
# ---------------------------------------------------------------------------


def test_timed_out_attempt_keeps_partial_records(tmp_path):
    # 5 examples at 0.05s each = 0.25s total; cap of 0.12s should let
    # roughly 2 examples finish before the deadline trips.
    evaluator = FakeEvaluator(seconds_per_example=0.05, num_examples=5)
    pipeline = make_pipeline(tmp_path, evaluator, max_attempt_seconds=0.12, max_iterations=1)

    with pytest.raises(SynthesisExhaustionError) as exc_info:
        pipeline.synthesize(task_description="dummy task", output_name="dummy")

    attempts = exc_info.value.attempts
    assert len(attempts) == 1
    timed_out_attempt = attempts[0]
    assert timed_out_attempt.failed_at == FailureStage.TIMEOUT
    assert timed_out_attempt.eval_result is not None
    assert len(timed_out_attempt.eval_result.sample_outputs) > 0, (
        "records produced before the cap hit must not be discarded"
    )
    assert len(timed_out_attempt.eval_result.sample_outputs) < 5, (
        "sanity check: the fake really was interrupted before finishing all examples"
    )

    # And the same partial records must have reached disk (requirement iii
    # applied to a timed-out attempt specifically).
    report_path = _progress_report_path(pipeline)
    with open(report_path) as f:
        import json

        data = json.load(f)
    on_disk_samples = data["attempts"][0]["evaluation"]["sample_outputs"]
    assert len(on_disk_samples) == len(timed_out_attempt.eval_result.sample_outputs)


def test_timeout_restart_preserves_active_surface(monkeypatch, tmp_path):
    from synthesis.evaluate import feedback_loop as feedback_loop_module

    class RecordingGenerator(FakeGenerator):
        def generate_initial(self, task_description, allowed_helpers=None, start_inside_constrained=False):
            self.seen_start_inside_constrained = start_inside_constrained
            return super().generate_initial(task_description, allowed_helpers, start_inside_constrained)

    evaluator = FakeEvaluator(seconds_per_example=0.0, num_examples=1)
    pipeline = make_pipeline(tmp_path, evaluator, max_attempt_seconds=1.0, max_iterations=1)
    pipeline.generator = RecordingGenerator()
    attempt = feedback_loop_module.SynthesisAttempt(
        attempt_number=1,
        strategy_code="STRATEGY",
        full_dafny_code="// dafny",
        timestamp="now",
    )

    monkeypatch.setattr(pipeline, "_start_inside_constrained", lambda: True)
    pipeline._handle_attempt_timeout(
        attempt, [], 1.1, "dummy task", "dummy", tmp_path
    )

    assert pipeline.generator.seen_start_inside_constrained is True
