from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

import pytest


def test_evaluation_result_thresholds_and_summary():
    from evaluation.evaluator import EvaluationResult

    result = EvaluationResult(
        success=True,
        accuracy=0.8,
        format_rate=0.9,
        syntax_rate=0.95,
        num_examples=10,
        num_correct=8,
        total_time_seconds=5.0,
        sample_outputs=[
            {
                "question": "What is 5+3?",
                "expected": "8",
                "actual": "7",
                "is_correct": False,
            }
        ],
    )

    assert result.meets_threshold(0.7, 0.8, 0.9)
    assert not result.meets_threshold(0.9, 0.8, 0.9)
    summary = result.get_feedback_summary()
    assert "Accuracy: 80.0%" in summary
    assert "Sample Failures:" in summary
    as_dict = result.to_dict()
    assert as_dict["syntax_rate"] == 0.95


def test_evaluator_helpers_match_current_surface():
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(dataset_name="gsm_symbolic")
    assert evaluator._extract_constrained_content("hello <<5+3=8>> world") == ["5+3=8"]
    assert evaluator._extract_answer_gsm("hello <<5+3>> world") == "8"
    assert evaluator._extract_answer_gsm("hello <<5+3=8>> world") == "8"
    assert evaluator._check_format_validity("hello <<5+3=8>> world")
    assert evaluator._get_grammar_file().name == "gsm.lark"


def test_evaluator_gsm_expression_evaluation_uses_final_segment_only():
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(dataset_name="gsm_symbolic")

    assert evaluator._extract_answer_gsm("reasoning <<1+1>> trailing <<16 * 8.5 + 4 * 10.5 + 13>>") == "191"
    assert evaluator._extract_answer_gsm("reasoning <<5+3>> trailing <<not valid>>") is None
    assert evaluator._answers_match("191", "191.0")


def test_failure_stage_and_attempt_record_evaluation_results():
    from verification.compiler import CompilationResult
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import FailureStage, SynthesisAttempt
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="strategy",
        timestamp="2026-04-14T00:00:00",
        verification_result=VerificationResult(success=True, raw_output="ok"),
        compilation_result=CompilationResult(success=True, output_dir=Path("/tmp")),
        runtime_result=RuntimeResult(success=True, output=["x"], cost=1),
        eval_result=EvaluationResult(
            success=True,
            accuracy=0.4,
            format_rate=0.5,
            syntax_rate=0.6,
            num_examples=5,
            num_correct=2,
            total_time_seconds=1.0,
        ),
        failed_at=FailureStage.EVALUATION,
        error_summary="evaluation failed",
    )

    attempt_dict = attempt.to_dict()
    assert FailureStage.EVALUATION.value == "evaluation"
    assert attempt_dict["failed_at"] == "evaluation"
    assert attempt_dict["evaluation"]["accuracy"] == 0.4


def test_pipeline_requires_evaluator_and_stores_thresholds():
    from synthesis.feedback_loop import SynthesisPipeline

    with pytest.raises(TypeError):
        SynthesisPipeline()

    pipeline = SynthesisPipeline(
        evaluator=Mock(),
        generator=Mock(),
        verifier=Mock(),
        compiler=Mock(),
        runner=Mock(),
        min_accuracy=0.7,
        min_format_rate=0.8,
        min_syntax_rate=0.9,
        eval_sample_size=4,
        save_reports=False,
    )

    assert pipeline.min_accuracy == 0.7
    assert pipeline.min_format_rate == 0.8
    assert pipeline.min_syntax_rate == 0.9
    assert pipeline.eval_sample_size == 4


def test_build_evaluation_failure_prompt_and_generator_refinement():
    from generation.generator import StrategyGenerator
    from generation.prompts import build_evaluation_failure_prompt

    system_prompt, user_prompt = build_evaluation_failure_prompt(
        previous_strategy="old strategy",
        evaluation_feedback="Accuracy: 30%",
    )
    assert system_prompt
    assert "old strategy" in user_prompt
    assert "Accuracy: 30%" in user_prompt

    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator._generate_valid_strategy = Mock(return_value="new strategy")
    refined = generator.refine_after_evaluation_failure("old strategy", "Accuracy: 30%")
    generator._generate_valid_strategy.assert_called_once()
    assert refined == "new strategy"


@patch("synthesis.feedback_loop.DafnyCompiler")
def test_pipeline_retries_after_evaluation_failure(mock_dafny_compiler):
    from verification.compiler import CompilationResult
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import FailureStage, SynthesisPipeline
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    mock_generator = Mock()
    mock_generator.generate_initial = Mock(return_value="strategy_initial")
    mock_generator.refine_after_evaluation_failure = Mock(return_value="strategy_refined")
    mock_generator.inject_strategy = Mock(return_value="full python code")

    mock_verifier = Mock()
    mock_verifier.verify = Mock(return_value=VerificationResult(success=True, raw_output="ok"))

    mock_runner = Mock()
    mock_runner.run_python_native = Mock(return_value=RuntimeResult(success=True, output=["token"], cost=1))

    mock_evaluator = Mock()
    eval_results = [
        EvaluationResult(
            success=True,
            accuracy=0.2,
            format_rate=0.3,
            syntax_rate=0.4,
            num_examples=3,
            num_correct=1,
            total_time_seconds=1.0,
        ),
        EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=3,
            num_correct=3,
            total_time_seconds=1.0,
        ),
    ]
    mock_evaluator.evaluate_sample = Mock(side_effect=eval_results)

    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "GeneratedCSD.py"
        module_path.write_text("# mock module", encoding="utf-8")

        mock_compiler_instance = Mock()
        mock_compiler_instance.compile = Mock(
            return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=module_path,
                raw_output="compiled",
            )
        )
        mock_dafny_compiler.return_value = mock_compiler_instance

        seed_compiler = Mock()
        seed_compiler.dafny_path = "dafny"
        seed_compiler.timeout = 120
        seed_compiler.extra_args = []

        pipeline = SynthesisPipeline(
            evaluator=mock_evaluator,
            generator=mock_generator,
            verifier=mock_verifier,
            compiler=seed_compiler,
            runner=mock_runner,
            max_iterations=3,
            output_dir=Path(tmpdir),
            save_reports=False,
            min_accuracy=0.5,
            min_format_rate=0.5,
            min_syntax_rate=0.5,
        )

        result = pipeline.synthesize("test task", output_name="test_csd")

    assert result.success
    assert len(result.attempts) == 2
    assert result.attempts[0].failed_at == FailureStage.EVALUATION
    assert result.attempts[1].failed_at is None
    assert mock_generator.refine_after_evaluation_failure.call_count == 1
    assert mock_evaluator.evaluate_sample.call_count == 2
