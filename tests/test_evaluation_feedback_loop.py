"""
Tests for the evaluation feedback loop implementation.

Tests cover:
1. Safe arithmetic evaluation (Evaluator._safe_eval_arithmetic)
2. EvaluationResult - dataclass and threshold methods
3. Evaluator - main evaluator class (extract answer, format checks, etc.)
4. FailureStage.EVALUATION - enum value
5. SynthesisAttempt.eval_result - field
6. SynthesisPipeline - evaluator as required parameter
7. Prompts - evaluation failure prompt
8. Generator - refine_after_evaluation_failure method
9. Integration test - all components together
10. End-to-end evaluation loop iteration
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Test 1: Safe arithmetic evaluation (via Evaluator._safe_eval_arithmetic)
# =============================================================================

class TestSafeArithmeticEvaluation:
    """Tests for safe arithmetic expression evaluation."""

    def setup_method(self):
        from synthesis.evaluator import Evaluator
        self.evaluator = Evaluator()

    def test_simple_addition(self):
        result = self.evaluator._safe_eval_arithmetic("2 + 3")
        assert result == 5.0

    def test_simple_subtraction(self):
        result = self.evaluator._safe_eval_arithmetic("10 - 4")
        assert result == 6.0

    def test_multiplication(self):
        result = self.evaluator._safe_eval_arithmetic("6 * 7")
        assert result == 42.0

    def test_division(self):
        result = self.evaluator._safe_eval_arithmetic("15 / 3")
        assert result == 5.0

    def test_complex_expression(self):
        result = self.evaluator._safe_eval_arithmetic("(2 + 3) * 4 - 5")
        assert result == 15.0

    def test_negative_numbers(self):
        result = self.evaluator._safe_eval_arithmetic("-5 + 3")
        assert result == -2.0

    def test_floating_point(self):
        result = self.evaluator._safe_eval_arithmetic("3.14 * 2")
        assert result is not None
        assert abs(result - 6.28) < 0.001

    def test_invalid_expression(self):
        result = self.evaluator._safe_eval_arithmetic("2 + * 3")
        assert result is None

    def test_non_math_expression(self):
        result = self.evaluator._safe_eval_arithmetic("hello")
        assert result is None


# =============================================================================
# Test 2: EvaluationResult
# =============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
        )
        assert result.success is True
        assert result.accuracy == 0.8
        assert result.num_correct == 8

    def test_meets_threshold_all_pass(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
            sample_outputs=[
                {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
                {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
            ],
        )
        assert result.meets_threshold(
            min_accuracy=0.7,
            min_format_rate=0.8,
            min_syntax_rate=0.9,
        ) is True

    def test_meets_threshold_accuracy_fail(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.5,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=5,
            total_time_seconds=5.0,
            sample_outputs=[
                {"is_correct": False, "is_valid_format": True, "syntax_rate": 1.0},
            ],
        )
        assert result.meets_threshold(min_accuracy=0.7) is False

    def test_meets_threshold_format_fail(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.5,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
            sample_outputs=[
                {"is_correct": True, "is_valid_format": False, "syntax_rate": 1.0},
            ],
        )
        assert result.meets_threshold(min_format_rate=0.7) is False

    def test_get_feedback_summary(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
        )
        summary = result.get_feedback_summary()
        assert "Accuracy: 80.0%" in summary
        assert "Format Rate: 90.0%" in summary
        assert "10 examples" in summary

    def test_to_dict(self):
        from synthesis.evaluator import EvaluationResult
        result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["accuracy"] == 0.8
        assert d["num_examples"] == 10


# =============================================================================
# Test 3: GSM answer extraction (via Evaluator methods)
# =============================================================================

class TestGSMAnswerExtraction:
    """Tests for GSM arithmetic answer extraction."""

    def setup_method(self):
        from synthesis.evaluator import Evaluator
        self.evaluator = Evaluator(dataset_name="gsm_symbolic")

    def test_extract_answer_with_equals(self):
        output = "Let me calculate: <<5 + 3 = 8>>"
        answer = self.evaluator._extract_answer_gsm(output)
        assert answer == "8"

    def test_extract_numeric_expression(self):
        output = "Result: <<5 + 3>>"
        answer = self.evaluator._extract_answer_gsm(output)
        assert answer == "8"

    def test_extract_complex_expression(self):
        output = "<<(10 + 5) * 2>>"
        answer = self.evaluator._extract_answer_gsm(output)
        assert answer == "30"

    def test_no_constrained_content(self):
        output = "no delimiters here"
        answer = self.evaluator._extract_answer_gsm(output)
        assert answer is None


# =============================================================================
# Test 4: FailureStage.EVALUATION
# =============================================================================

class TestFailureStageEvaluation:
    """Tests for the EVALUATION stage in FailureStage enum."""

    def test_evaluation_stage_exists(self):
        from synthesis.feedback_loop import FailureStage
        assert hasattr(FailureStage, 'EVALUATION')
        assert FailureStage.EVALUATION.value == "evaluation"

    def test_all_stages_present(self):
        from synthesis.feedback_loop import FailureStage
        stages = [s.value for s in FailureStage]
        assert "verification" in stages
        assert "compilation" in stages
        assert "runtime" in stages
        assert "evaluation" in stages


# =============================================================================
# Test 5: SynthesisAttempt.eval_result
# =============================================================================

class TestSynthesisAttemptEvalResult:
    """Tests for the eval_result field in SynthesisAttempt."""

    def test_eval_result_field_exists(self):
        from synthesis.feedback_loop import SynthesisAttempt
        from synthesis.evaluator import EvaluationResult

        attempt = SynthesisAttempt(
            attempt_number=1,
            strategy_code="test",
            full_dafny_code="test",
            timestamp="2024-01-01",
        )
        assert hasattr(attempt, 'eval_result')
        assert attempt.eval_result is None

    def test_eval_result_can_be_set(self):
        from synthesis.feedback_loop import SynthesisAttempt
        from synthesis.evaluator import EvaluationResult

        eval_result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
        )

        attempt = SynthesisAttempt(
            attempt_number=1,
            strategy_code="test",
            full_dafny_code="test",
            timestamp="2024-01-01",
            eval_result=eval_result,
        )
        assert attempt.eval_result is not None
        assert attempt.eval_result.accuracy == 0.8

    def test_to_dict_includes_evaluation(self):
        from synthesis.feedback_loop import SynthesisAttempt
        from synthesis.evaluator import EvaluationResult

        eval_result = EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=10,
            num_correct=8,
            total_time_seconds=5.0,
        )

        attempt = SynthesisAttempt(
            attempt_number=1,
            strategy_code="test",
            full_dafny_code="test",
            timestamp="2024-01-01",
            eval_result=eval_result,
        )

        d = attempt.to_dict()
        assert "evaluation" in d
        assert d["evaluation"]["accuracy"] == 0.8


# =============================================================================
# Test 6: SynthesisPipeline requires evaluator
# =============================================================================

class TestSynthesisPipelineEvaluator:
    """Tests for SynthesisPipeline evaluator requirement."""

    def test_evaluator_is_required_parameter(self):
        from synthesis.feedback_loop import SynthesisPipeline
        import inspect

        sig = inspect.signature(SynthesisPipeline.__init__)
        params = list(sig.parameters.keys())

        # evaluator should be the first parameter after self
        assert "evaluator" in params
        # Check it's required (no default)
        evaluator_param = sig.parameters["evaluator"]
        assert evaluator_param.default is inspect.Parameter.empty

    def test_pipeline_creation_without_evaluator_fails(self):
        from synthesis.feedback_loop import SynthesisPipeline

        with pytest.raises(TypeError):
            SynthesisPipeline()  # Should fail - evaluator is required

    def test_pipeline_stores_thresholds(self):
        from synthesis.feedback_loop import SynthesisPipeline
        from synthesis.evaluator import Evaluator

        mock_evaluator = Mock(spec=Evaluator)

        pipeline = SynthesisPipeline(
            evaluator=mock_evaluator,
            min_accuracy=0.7,
            min_format_rate=0.8,
            min_syntax_rate=0.9,
            eval_sample_size=20,
        )

        assert pipeline.min_accuracy == 0.7
        assert pipeline.min_format_rate == 0.8
        assert pipeline.min_syntax_rate == 0.9
        assert pipeline.eval_sample_size == 20


# =============================================================================
# Test 7: Evaluation failure prompt
# =============================================================================

class TestEvaluationFailurePrompt:
    """Tests for the evaluation failure prompt."""

    def test_build_evaluation_failure_prompt_exists(self):
        from synthesis.prompts import build_evaluation_failure_prompt
        assert callable(build_evaluation_failure_prompt)

    def test_build_evaluation_failure_prompt_returns_tuple(self):
        from synthesis.prompts import build_evaluation_failure_prompt

        result = build_evaluation_failure_prompt(
            previous_strategy="generated := helpers.PureConstrainedGeneration(...);",
            evaluation_feedback="Accuracy: 30%, Format: 50%",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prompt_contains_strategy(self):
        from synthesis.prompts import build_evaluation_failure_prompt

        strategy = "generated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);"
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy=strategy,
            evaluation_feedback="Accuracy: 30%",
        )

        assert strategy in user_prompt

    def test_prompt_contains_feedback(self):
        from synthesis.prompts import build_evaluation_failure_prompt

        feedback = "Accuracy: 30%, Format Rate: 50%"
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy="test strategy",
            evaluation_feedback=feedback,
        )

        assert feedback in user_prompt

    def test_prompt_mentions_evaluation_context(self):
        from synthesis.prompts import build_evaluation_failure_prompt

        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy="test",
            evaluation_feedback="test",
        )

        # Should mention that it passed verification/compilation/runtime
        assert "verification" in user_prompt.lower() or "passed" in user_prompt.lower()


# =============================================================================
# Test 8: Generator.refine_after_evaluation_failure
# =============================================================================

class TestGeneratorRefineAfterEvaluationFailure:
    """Tests for the refine_after_evaluation_failure method."""

    def test_method_exists(self):
        from synthesis.generator import StrategyGenerator
        assert hasattr(StrategyGenerator, 'refine_after_evaluation_failure')

    def test_method_signature(self):
        from synthesis.generator import StrategyGenerator
        import inspect

        sig = inspect.signature(StrategyGenerator.refine_after_evaluation_failure)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "previous_strategy" in params
        assert "evaluation_feedback" in params

    def test_method_calls_internal_methods(self):
        """Test that refine_after_evaluation_failure uses the generator's internal methods."""
        from synthesis.generator import StrategyGenerator

        # Create a generator instance without loading the actual model
        generator = StrategyGenerator.__new__(StrategyGenerator)
        generator._model = None
        generator._tokenizer = None
        generator._template = "template"
        generator.model_name = "test"
        generator.device = "cpu"
        generator.torch_dtype = None
        generator.max_new_tokens = 512
        generator.temperature = 0.7
        generator.top_p = 0.9

        # Mock the internal methods
        mock_strategy = "// CSD_RATIONALE_BEGIN\n// test\n// CSD_RATIONALE_END\ngenerated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);\ncost := helpers.cost;"
        generator._generate_text = Mock(return_value=mock_strategy)
        generator._extract_strategy = Mock(return_value=mock_strategy)
        generator._ensure_rationale_block = Mock(return_value=mock_strategy)

        result = generator.refine_after_evaluation_failure(
            previous_strategy="old strategy",
            evaluation_feedback="Accuracy: 30%",
        )

        # Verify the internal methods were called
        generator._generate_text.assert_called_once()
        generator._extract_strategy.assert_called_once()
        generator._ensure_rationale_block.assert_called_once()
        assert result == mock_strategy


# =============================================================================
# Test 9: Evaluator class
# =============================================================================

class TestEvaluator:
    """Tests for the main Evaluator class."""

    def test_evaluator_creation(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator(
            dataset_name="gsm_symbolic",
            model_name="test-model",
            device="cpu",
            sample_size=5,
            max_steps=100,
        )

        assert evaluator.dataset_name == "gsm_symbolic"
        assert evaluator.sample_size == 5

    def test_evaluator_supports_gsm(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator(dataset_name="gsm_symbolic")
        assert evaluator.dataset_name == "gsm_symbolic"

    def test_evaluator_supports_folio(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator(dataset_name="folio")
        assert evaluator.dataset_name == "folio"

    def test_extract_constrained_content(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator()

        output = "Some text <<5+3=8>> more text <<10*2=20>>"
        matches = evaluator._extract_constrained_content(output)

        assert len(matches) == 2
        assert "5+3=8" in matches
        assert "10*2=20" in matches

    def test_extract_answer_gsm(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator(dataset_name="gsm_symbolic")

        output = "Let me calculate: <<5+3=8>>"
        answer = evaluator._extract_answer_gsm(output)

        assert answer == "8"

    def test_extract_answer_folio(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator(dataset_name="folio")

        assert evaluator._extract_answer_folio("The answer is True") == "True"
        assert evaluator._extract_answer_folio("This is False") == "False"
        assert evaluator._extract_answer_folio("Unknown") == "Unknown"

    def test_check_format_validity(self):
        from synthesis.evaluator import Evaluator

        evaluator = Evaluator()

        assert evaluator._check_format_validity("text <<content>> more") is True
        assert evaluator._check_format_validity("no delimiters here") is False
        assert evaluator._check_format_validity("only << opening") is False


# =============================================================================
# Test 10: Integration test
# =============================================================================

class TestIntegration:
    """Integration tests for the complete evaluation feedback loop."""

    def test_full_pipeline_components_connected(self):
        """Test that all components can be instantiated and connected."""
        from synthesis.evaluator import Evaluator, EvaluationResult
        from synthesis.feedback_loop import SynthesisPipeline, FailureStage, SynthesisAttempt
        from synthesis.prompts import build_evaluation_failure_prompt

        # Create evaluator
        evaluator = Evaluator(
            dataset_name="gsm_symbolic",
            sample_size=5,
        )

        # Create pipeline with evaluator
        pipeline = SynthesisPipeline(
            evaluator=evaluator,
            min_accuracy=0.5,
            min_format_rate=0.5,
        )

        # Create an attempt with eval result
        eval_result = EvaluationResult(
            success=True,
            accuracy=0.4,
            format_rate=0.6,
            syntax_rate=0.8,
            num_examples=5,
            num_correct=2,
            total_time_seconds=1.0,
        )

        attempt = SynthesisAttempt(
            attempt_number=1,
            strategy_code="test",
            full_dafny_code="test",
            timestamp="2024-01-01",
            eval_result=eval_result,
            failed_at=FailureStage.EVALUATION,
            error_summary=eval_result.get_feedback_summary(),
        )

        # Verify the flow
        assert attempt.failed_at == FailureStage.EVALUATION

        # Generate refinement prompt
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy=attempt.strategy_code,
            evaluation_feedback=attempt.error_summary,
        )

        assert len(user_prompt) > 0

    def test_evaluation_result_flow(self):
        """Test the flow of evaluation results through the system."""
        from synthesis.evaluator import EvaluationResult, Evaluator

        evaluator = Evaluator(dataset_name="gsm_symbolic")

        # Simulate evaluating an expression
        result = evaluator._safe_eval_arithmetic("5 + 3")
        assert result == 8.0

        # Create evaluation result
        eval_result = EvaluationResult(
            success=True,
            accuracy=1.0,
            format_rate=1.0,
            syntax_rate=1.0,
            num_examples=1,
            num_correct=1,
            total_time_seconds=0.1,
            sample_outputs=[
                {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
            ],
        )

        # Check thresholds
        assert eval_result.meets_threshold(
            min_accuracy=0.9,
            min_format_rate=0.9,
            min_syntax_rate=0.9,
        )

    def test_failure_triggers_refinement(self):
        """Test that evaluation failure would trigger refinement."""
        from synthesis.evaluator import EvaluationResult
        from synthesis.feedback_loop import FailureStage

        # Simulate a poor evaluation result
        eval_result = EvaluationResult(
            success=True,
            accuracy=0.2,
            format_rate=0.3,
            syntax_rate=0.5,
            num_examples=10,
            num_correct=2,
            total_time_seconds=5.0,
            sample_outputs=[
                {
                    "question": "What is 5+3?",
                    "expected": "8",
                    "actual": "wrong",
                    "is_correct": False,
                }
            ],
        )

        # This should fail thresholds (sample has is_correct=False)
        assert not eval_result.meets_threshold(min_accuracy=0.5)

        # The feedback summary should contain useful info
        feedback = eval_result.get_feedback_summary()
        assert "20.0%" in feedback  # accuracy
        assert "Sample Failures" in feedback


# =============================================================================
# Test 11: End-to-end evaluation loop iteration
# =============================================================================

class TestEvaluationLoopIteration:
    """
    End-to-end tests that verify the evaluation feedback loop actually iterates
    correctly when evaluation fails and succeeds on retry.
    """

    def test_loop_iterates_on_evaluation_failure(self):
        """
        Test that the synthesis loop:
        1. Generates initial strategy
        2. Passes verify/compile/runtime
        3. Fails evaluation (below threshold)
        4. Calls refine_after_evaluation_failure
        5. Generates refined strategy
        6. Passes all stages including evaluation
        7. Returns success
        """
        from synthesis.feedback_loop import SynthesisPipeline, SynthesisResult
        from synthesis.evaluator import EvaluationResult
        from synthesis.verifier import VerificationResult
        from synthesis.compiler import CompilationResult
        from synthesis.runner import RuntimeResult
        from pathlib import Path
        import tempfile

        # Create mock components
        mock_generator = Mock()
        mock_verifier = Mock()
        mock_compiler = Mock()
        mock_runner = Mock()
        mock_evaluator = Mock()

        # Track how many times generate/refine are called
        generation_calls = []

        def mock_generate_initial(task_desc, cost_contract=""):
            generation_calls.append(("initial", task_desc))
            return "// CSD_RATIONALE_BEGIN\n// Initial strategy\n// CSD_RATIONALE_END\ngenerated := helpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);\ncost := helpers.cost;"

        def mock_refine_after_evaluation_failure(prev_strategy, feedback):
            generation_calls.append(("refine_eval", feedback[:50]))
            return "// CSD_RATIONALE_BEGIN\n// Refined strategy\n// CSD_RATIONALE_END\ngenerated := helpers.HybridGeneration(lm, parser, prompt, maxSteps);\ncost := helpers.cost;"

        def mock_inject_strategy(strategy, cost_contract=""):
            return f"// Full Dafny code with: {strategy[:50]}..."

        mock_generator.generate_initial = mock_generate_initial
        mock_generator.refine_after_evaluation_failure = mock_refine_after_evaluation_failure
        mock_generator.inject_strategy = mock_inject_strategy

        # Verifier always passes
        mock_verifier.verify = Mock(return_value=VerificationResult(
            success=True,
            errors=[],
            raw_output="Verified",
        ))

        # Compiler always passes
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_module_path = Path(tmpdir) / "GeneratedCSD.py"
            mock_module_path.write_text("# mock module")

            mock_compiler.compile = Mock(return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=mock_module_path,
                raw_output="Compiled",
            ))
            mock_compiler.dafny_path = "dafny"
            mock_compiler.timeout = 120
            mock_compiler.extra_args = []

            # Runner always passes
            mock_runner.run = Mock(return_value=RuntimeResult(
                success=True,
                output=["token1", "token2"],
                cost=5,
                execution_time_ms=100.0,
            ))

            # Evaluator: fails first time, passes second time
            eval_call_count = [0]

            def mock_evaluate_sample(compiled_module_path, sample_size=None):
                eval_call_count[0] += 1
                if eval_call_count[0] == 1:
                    # First attempt: fails threshold
                    return EvaluationResult(
                        success=True,
                        accuracy=0.3,  # Below 0.5 threshold
                        format_rate=0.4,
                        syntax_rate=0.5,
                        num_examples=10,
                        num_correct=3,
                        total_time_seconds=1.0,
                        sample_outputs=[
                            {"question": "test", "expected": "8", "actual": "5", "is_correct": False,
                             "is_valid_format": False, "syntax_rate": 0.0}
                        ],
                    )
                else:
                    # Second attempt: passes threshold
                    return EvaluationResult(
                        success=True,
                        accuracy=0.8,  # Above 0.5 threshold
                        format_rate=0.9,
                        syntax_rate=0.95,
                        num_examples=10,
                        num_correct=8,
                        total_time_seconds=1.0,
                        sample_outputs=[
                            {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
                        ],
                    )

            mock_evaluator.evaluate_sample = mock_evaluate_sample

            # Create pipeline with thresholds
            pipeline = SynthesisPipeline(
                evaluator=mock_evaluator,
                generator=mock_generator,
                verifier=mock_verifier,
                compiler=mock_compiler,
                runner=mock_runner,
                max_iterations=5,
                min_accuracy=0.5,
                min_format_rate=0.5,
                eval_sample_size=10,
                save_reports=False,
            )

            # Run synthesis
            result = pipeline.synthesize(
                task_description="Test task",
                output_name="test_csd",
            )

            # Verify the loop iterated correctly
            assert result.success is True
            assert len(result.attempts) == 2  # Two attempts

            # Verify generation calls
            assert len(generation_calls) == 2
            assert generation_calls[0][0] == "initial"
            assert generation_calls[1][0] == "refine_eval"

            # Verify evaluator was called twice
            assert eval_call_count[0] == 2

            # Verify first attempt failed at evaluation
            assert result.attempts[0].failed_at is not None
            assert result.attempts[0].failed_at.value == "evaluation"

            # Verify second attempt succeeded
            assert result.attempts[1].failed_at is None
            assert result.attempts[1].eval_result.accuracy == 0.8

    @patch('synthesis.feedback_loop.DafnyCompiler')
    def test_loop_exhausts_attempts_on_repeated_failure(self, MockDafnyCompiler):
        """
        Test that the loop exhausts max_iterations when evaluation
        keeps failing, and raises SynthesisExhaustionError.
        """
        from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError
        from synthesis.evaluator import EvaluationResult
        from synthesis.verifier import VerificationResult
        from synthesis.compiler import CompilationResult
        from synthesis.runner import RuntimeResult
        from pathlib import Path
        import tempfile

        mock_generator = Mock()
        mock_verifier = Mock()
        mock_compiler = Mock()
        mock_runner = Mock()
        mock_evaluator = Mock()

        # Generator always returns same strategy
        mock_generator.generate_initial = Mock(return_value="strategy1")
        mock_generator.refine_after_evaluation_failure = Mock(return_value="strategy_refined")
        mock_generator.inject_strategy = Mock(return_value="full code")

        # All stages pass except evaluation
        mock_verifier.verify = Mock(return_value=VerificationResult(
            success=True, errors=[], raw_output="ok"
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_path = Path(tmpdir) / "module.py"
            mock_path.write_text("# mock")

            # Configure the DafnyCompiler mock that will be created in synthesize()
            mock_compiler_instance = Mock()
            mock_compiler_instance.compile = Mock(return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=mock_path,
                raw_output="ok",
            ))
            MockDafnyCompiler.return_value = mock_compiler_instance

            # Configure the original compiler for its properties
            mock_compiler.dafny_path = "dafny"
            mock_compiler.timeout = 120
            mock_compiler.extra_args = []

            mock_runner.run = Mock(return_value=RuntimeResult(
                success=True, output=["t"], cost=1, execution_time_ms=10.0
            ))

            # Evaluator always fails threshold
            mock_evaluator.evaluate_sample = Mock(return_value=EvaluationResult(
                success=True,
                accuracy=0.2,  # Always below threshold
                format_rate=0.3,
                syntax_rate=0.4,
                num_examples=10,
                num_correct=2,
                total_time_seconds=1.0,
            ))

            pipeline = SynthesisPipeline(
                evaluator=mock_evaluator,
                generator=mock_generator,
                verifier=mock_verifier,
                compiler=mock_compiler,
                runner=mock_runner,
                max_iterations=3,
                min_accuracy=0.5,
                save_reports=False,
            )

            # Should raise SynthesisExhaustionError after 3 attempts
            with pytest.raises(SynthesisExhaustionError) as exc_info:
                pipeline.synthesize("test task")

            # Verify all attempts failed at evaluation
            assert len(exc_info.value.attempts) == 3
            for attempt in exc_info.value.attempts:
                assert attempt.failed_at.value == "evaluation"

            # Verify refine was called for each retry
            assert mock_generator.refine_after_evaluation_failure.call_count == 3

    @patch('synthesis.feedback_loop.DafnyCompiler')
    def test_loop_succeeds_immediately_if_first_attempt_passes(self, MockDafnyCompiler):
        """
        Test that if the first attempt passes evaluation, the loop
        returns immediately without refinement.
        """
        from synthesis.feedback_loop import SynthesisPipeline
        from synthesis.evaluator import EvaluationResult
        from synthesis.verifier import VerificationResult
        from synthesis.compiler import CompilationResult
        from synthesis.runner import RuntimeResult
        from pathlib import Path
        import tempfile

        mock_generator = Mock()
        mock_verifier = Mock()
        mock_compiler = Mock()
        mock_runner = Mock()
        mock_evaluator = Mock()

        mock_generator.generate_initial = Mock(return_value="good_strategy")
        mock_generator.refine_after_evaluation_failure = Mock(return_value="should_not_be_called")
        mock_generator.inject_strategy = Mock(return_value="full code")

        mock_verifier.verify = Mock(return_value=VerificationResult(
            success=True, errors=[], raw_output="ok"
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_path = Path(tmpdir) / "module.py"
            mock_path.write_text("# mock")

            # Configure the DafnyCompiler mock that will be created in synthesize()
            mock_compiler_instance = Mock()
            mock_compiler_instance.compile = Mock(return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=mock_path,
                raw_output="ok",
            ))
            MockDafnyCompiler.return_value = mock_compiler_instance

            mock_compiler.dafny_path = "dafny"
            mock_compiler.timeout = 120
            mock_compiler.extra_args = []

            mock_runner.run = Mock(return_value=RuntimeResult(
                success=True, output=["t"], cost=1, execution_time_ms=10.0
            ))

            # Evaluator passes immediately
            mock_evaluator.evaluate_sample = Mock(return_value=EvaluationResult(
                success=True,
                accuracy=0.9,
                format_rate=0.95,
                syntax_rate=1.0,
                num_examples=10,
                num_correct=9,
                total_time_seconds=1.0,
                sample_outputs=[
                    {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
                ],
            ))

            pipeline = SynthesisPipeline(
                evaluator=mock_evaluator,
                generator=mock_generator,
                verifier=mock_verifier,
                compiler=mock_compiler,
                runner=mock_runner,
                max_iterations=5,
                min_accuracy=0.5,
                save_reports=False,
            )

            result = pipeline.synthesize("test task")

            # Should succeed on first attempt
            assert result.success is True
            assert len(result.attempts) == 1

            # refine_after_evaluation_failure should NOT have been called
            mock_generator.refine_after_evaluation_failure.assert_not_called()

    @patch('synthesis.feedback_loop.DafnyCompiler')
    def test_loop_handles_evaluation_error(self, MockDafnyCompiler):
        """
        Test that when evaluator.evaluate_sample returns success=False
        (e.g., environment setup failed), the loop treats it as failure
        and refines.
        """
        from synthesis.feedback_loop import SynthesisPipeline
        from synthesis.evaluator import EvaluationResult
        from synthesis.verifier import VerificationResult
        from synthesis.compiler import CompilationResult
        from synthesis.runner import RuntimeResult
        from pathlib import Path
        import tempfile

        mock_generator = Mock()
        mock_verifier = Mock()
        mock_compiler = Mock()
        mock_runner = Mock()
        mock_evaluator = Mock()

        mock_generator.generate_initial = Mock(return_value="strategy1")
        mock_generator.refine_after_evaluation_failure = Mock(return_value="strategy2")
        mock_generator.inject_strategy = Mock(return_value="full code")

        mock_verifier.verify = Mock(return_value=VerificationResult(
            success=True, errors=[], raw_output="ok"
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_path = Path(tmpdir) / "module.py"
            mock_path.write_text("# mock")

            # Configure the DafnyCompiler mock that will be created in synthesize()
            mock_compiler_instance = Mock()
            mock_compiler_instance.compile = Mock(return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=mock_path,
                raw_output="ok",
            ))
            MockDafnyCompiler.return_value = mock_compiler_instance

            mock_compiler.dafny_path = "dafny"
            mock_compiler.timeout = 120
            mock_compiler.extra_args = []

            mock_runner.run = Mock(return_value=RuntimeResult(
                success=True, output=["t"], cost=1, execution_time_ms=10.0
            ))

            # First call: evaluation error, second call: success
            eval_calls = [0]

            def mock_eval(compiled_module_path, sample_size=None):
                eval_calls[0] += 1
                if eval_calls[0] == 1:
                    return EvaluationResult(
                        success=False,  # Evaluation itself failed
                        accuracy=0.0,
                        format_rate=0.0,
                        syntax_rate=0.0,
                        num_examples=0,
                        num_correct=0,
                        total_time_seconds=0.0,
                        error="Failed to load dataset",
                    )
                else:
                    return EvaluationResult(
                        success=True,
                        accuracy=0.9,
                        format_rate=0.9,
                        syntax_rate=0.9,
                        num_examples=10,
                        num_correct=9,
                        total_time_seconds=1.0,
                        sample_outputs=[
                            {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
                        ],
                    )

            mock_evaluator.evaluate_sample = mock_eval

            pipeline = SynthesisPipeline(
                evaluator=mock_evaluator,
                generator=mock_generator,
                verifier=mock_verifier,
                compiler=mock_compiler,
                runner=mock_runner,
                max_iterations=5,
                min_accuracy=0.5,
                save_reports=False,
            )

            result = pipeline.synthesize("test task")

            assert result.success is True
            assert len(result.attempts) == 2

            # First attempt should have failed at evaluation with error
            assert result.attempts[0].failed_at.value == "evaluation"
            assert "Failed to load dataset" in result.attempts[0].error_summary

    @patch('synthesis.feedback_loop.DafnyCompiler')
    def test_mixed_failure_stages(self, MockDafnyCompiler):
        """
        Test a scenario where different attempts fail at different stages,
        including evaluation.
        """
        from synthesis.feedback_loop import SynthesisPipeline, FailureStage
        from synthesis.evaluator import EvaluationResult
        from synthesis.verifier import VerificationResult
        from synthesis.compiler import CompilationResult
        from synthesis.runner import RuntimeResult
        from pathlib import Path
        import tempfile

        mock_generator = Mock()
        mock_verifier = Mock()
        mock_compiler = Mock()
        mock_runner = Mock()
        mock_evaluator = Mock()

        # Track attempts
        attempt_count = [0]

        def mock_generate_initial(task, cost=""):
            return "strategy_initial"

        def mock_refine_verification(prev, err):
            return "strategy_after_verify_fail"

        def mock_refine_eval(prev, feedback):
            return "strategy_after_eval_fail"

        mock_generator.generate_initial = mock_generate_initial
        mock_generator.refine_after_verification_error = mock_refine_verification
        mock_generator.refine_after_evaluation_failure = mock_refine_eval
        mock_generator.inject_strategy = Mock(return_value="full code")

        # Verification: fails first, then passes
        from synthesis.verifier import VerificationError

        verify_calls = [0]

        def mock_verify(code):
            verify_calls[0] += 1
            if verify_calls[0] == 1:
                error = VerificationError(
                    file="test.dfy",
                    line=1,
                    column=1,
                    message="Test verification error",
                )
                return VerificationResult(success=False, errors=[error], raw_output="fail")
            return VerificationResult(success=True, errors=[], raw_output="ok")

        mock_verifier.verify = mock_verify

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_path = Path(tmpdir) / "module.py"
            mock_path.write_text("# mock")

            # Configure the DafnyCompiler mock that will be created in synthesize()
            mock_compiler_instance = Mock()
            mock_compiler_instance.compile = Mock(return_value=CompilationResult(
                success=True,
                output_dir=Path(tmpdir),
                main_module_path=mock_path,
                raw_output="ok",
            ))
            MockDafnyCompiler.return_value = mock_compiler_instance

            mock_compiler.dafny_path = "dafny"
            mock_compiler.timeout = 120
            mock_compiler.extra_args = []

            mock_runner.run = Mock(return_value=RuntimeResult(
                success=True, output=["t"], cost=1, execution_time_ms=10.0
            ))

            # Evaluation: fails once, then passes
            eval_calls = [0]

            def mock_eval(compiled_module_path, sample_size=None):
                eval_calls[0] += 1
                if eval_calls[0] == 1:
                    return EvaluationResult(
                        success=True,
                        accuracy=0.2,
                        format_rate=0.3,
                        syntax_rate=0.4,
                        num_examples=10,
                        num_correct=2,
                        total_time_seconds=1.0,
                    )
                return EvaluationResult(
                    success=True,
                    accuracy=0.9,
                    format_rate=0.95,
                    syntax_rate=1.0,
                    num_examples=10,
                    num_correct=9,
                    total_time_seconds=1.0,
                    sample_outputs=[
                        {"is_correct": True, "is_valid_format": True, "syntax_rate": 1.0},
                    ],
                )

            mock_evaluator.evaluate_sample = mock_eval

            pipeline = SynthesisPipeline(
                evaluator=mock_evaluator,
                generator=mock_generator,
                verifier=mock_verifier,
                compiler=mock_compiler,
                runner=mock_runner,
                max_iterations=5,
                min_accuracy=0.5,
                save_reports=False,
            )

            result = pipeline.synthesize("test task")

            assert result.success is True
            assert len(result.attempts) == 3

            # First attempt: verification failure
            assert result.attempts[0].failed_at == FailureStage.VERIFICATION

            # Second attempt: evaluation failure
            assert result.attempts[1].failed_at == FailureStage.EVALUATION

            # Third attempt: success
            assert result.attempts[2].failed_at is None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
