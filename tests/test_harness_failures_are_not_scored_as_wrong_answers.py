"""A broken harness must stop the run, not be recorded as a wrong answer.

The defect
----------
`Evaluator._evaluate_one_example` wraps the whole per-example body in one
`except Exception`. Whatever goes wrong in there, it builds a sample dict with
`is_correct: False` and `accuracy_applicable: True` and returns it. The sample
is then counted in the accuracy denominator (`num_accuracy_examples`), so a
missing Python module and a model that got the question wrong produce the
identical number, and the docstring even promises it: "It never raises".

This is the wiring gap that made three earlier fixes dead on arrival. Removing a
swallow inside `_gsm_symbolic_equivalence` or `eval_logic.is_correct` changes
nothing if the caller one level up catches the raise and turns it back into a
score. Fixing the inner function while the outer one still swallows is fixing
nothing.

The rule, and why it is a list rather than a principle
-----------------------------------------------------
Only two kinds of failure are re-raised:

  ModuleNotFoundError / ImportError   the harness itself is broken -- a file is
                                      missing, so nothing was measured
  UngradableExample                   the dataset row cannot be graded the way
                                      CRANE grades, so there is no verdict to
                                      record

Everything else -- a model that produced nonsense, a timeout, a solver that
fell over on one expression -- is a real outcome of running that example and
stays caught, because stopping the whole run over one bad example is its own
kind of lost measurement.

That is an explicit list of three names, checked structurally below so it
cannot quietly grow into "anything that smells like a setup problem".

Why UngradableExample rather than TypeError and ValueError
----------------------------------------------------------
The raises added upstream of here used plain TypeError and ValueError.
Re-raising those by type would be far wider than intended: TypeError is what
you get from any ordinary bug anywhere in generation (None where a string was
expected, a wrong argument count), and ValueError is what `float("abc")`
raises. Listing them would mean any incidental bug in the generation path
aborts the entire evaluation run. A named class says the one thing meant --
"this row cannot be graded" -- and nothing else. It subclasses ValueError so
existing `except ValueError` handlers behave exactly as before.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EVALUATOR_SOURCE = (
    Path(__file__).resolve().parents[1] / "synthesis" / "evaluate" / "evaluator.py"
)


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------


class _Boom:
    """Stands in for the `logic` module; its first call raises what we choose."""

    def __init__(self, error):
        self._error = error

    def build_dynamic_parser(self, evaluator, env, example):
        raise self._error

    def accuracy_applicable(self, aux):
        return True


def _run_one_example(error):
    """Call the real `_evaluate_one_example` with a `logic` that blows up.

    `build_dynamic_parser` is the first thing inside the try, so this reaches
    the catch under test without needing a model, a dataset or a GPU.
    """
    from synthesis.evaluate.evaluator import Evaluator

    ev = Evaluator.__new__(Evaluator)
    ev.dataset_name = "gsm_symbolic"
    ev.max_steps = 1
    ev.max_seconds_per_example = None
    ev.early_stop_on_answer = False
    # Instance attributes shadow the real methods: this test is about the
    # catch, not about prompt formatting.
    ev._format_prompt = lambda example: "prompt"
    ev._get_expected_answer = lambda example: "42"
    ev._accuracy_applicable_for_example = lambda aux: True

    return Evaluator._evaluate_one_example(
        ev,
        0,
        {"question": "q"},
        1,
        {"lm": None},
        _Boom(error),
        lambda **kwargs: None,
        {},
    )


def test_a_missing_module_stops_the_run_instead_of_scoring_zero():
    with pytest.raises(ModuleNotFoundError):
        _run_one_example(ModuleNotFoundError("No module named 'vllm_startup'"))


def test_any_import_error_stops_the_run_too():
    with pytest.raises(ImportError):
        _run_one_example(ImportError("cannot import name 'foo'"))


def test_an_ungradable_row_stops_the_run():
    from synthesis.evaluate.benchmarks.common.ungradable import UngradableExample

    with pytest.raises(UngradableExample):
        _run_one_example(UngradableExample("no variable_types on this row"))


def test_an_ordinary_failure_is_still_recorded_as_a_sample():
    """Guard against over-fixing. One example blowing up mid-generation is a
    real outcome; aborting the whole run over it loses every other example."""
    sample = _run_one_example(RuntimeError("the model produced garbage"))

    assert sample["is_correct"] is False
    assert "the model produced garbage" in sample["error"]


def test_a_timeout_is_still_recorded_as_a_sample():
    """A per-example timeout is a measurement, not a broken harness: it says
    this example did not finish inside its budget."""
    from synthesis.evaluate.evaluator import PerExampleTimeout

    sample = _run_one_example(PerExampleTimeout("out of time"))

    assert sample["timed_out"] is True
    assert sample["is_correct"] is False


def test_ungradable_is_a_value_error_so_existing_handlers_are_unaffected():
    from synthesis.evaluate.benchmarks.common.ungradable import UngradableExample

    assert issubclass(UngradableExample, ValueError)


def test_the_gsm_grader_raises_the_named_class_not_a_bare_value_error():
    """The class has to be what the graders actually raise, or the re-raise
    above catches nothing and the whole fix is decorative again."""
    from synthesis.evaluate.benchmarks.common.ungradable import UngradableExample
    from synthesis.evaluate.benchmarks.gsm_symbolic.eval_logic import is_correct

    with pytest.raises(UngradableExample):
        is_correct(
            evaluator=None,
            actual="a * 2",
            expected="a + a",
            example={"answer_parsed": "a + a", "answer": "#### 42"},
            aux=None,
            scored_output="<<a * 2>>",
        )

    with pytest.raises(UngradableExample):
        is_correct(
            evaluator=None,
            actual="a * 2",
            expected="a + a",
            example={"variable_types": "['not', 'a', 'mapping']"},
            aux=None,
            scored_output="<<a * 2>>",
        )


# --------------------------------------------------------------------------
# the list stays explicit
# --------------------------------------------------------------------------

EXPECTED_RERAISED = {"ModuleNotFoundError", "ImportError", "UngradableExample"}


def check_reraise_list(source: str) -> tuple[bool, str]:
    """Does `_evaluate_one_example` re-raise exactly the three named errors?

    A plain function so the test below can run it against the shape the file
    had before this fix and confirm it says no. A structural check that cannot
    fail is not a check.
    """
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_evaluate_one_example":
            target = node
            break
    if target is None:
        return False, "_evaluate_one_example is gone; this check needs rewriting"

    for node in ast.walk(target):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            body_is_bare_raise = (
                len(handler.body) == 1
                and isinstance(handler.body[0], ast.Raise)
                and handler.body[0].exc is None
            )
            if not body_is_bare_raise or handler.type is None:
                continue
            listed = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            names = {n.id for n in listed if isinstance(n, ast.Name)}
            if names == EXPECTED_RERAISED:
                return True, "re-raises exactly the three named errors"
            return False, (
                f"The re-raise clause names {sorted(names)}, not "
                f"{sorted(EXPECTED_RERAISED)}. The list is meant to be fixed: "
                "widening it turns ordinary per-example failures into aborted "
                "runs, and narrowing it puts a broken harness back into the "
                "accuracy denominator."
            )

    return False, (
        "No re-raise clause at all -- every failure in the per-example body is "
        "still caught and returned as a sample with is_correct: False, which "
        "is counted in the accuracy denominator."
    )


def test_the_reraise_list_is_exactly_the_three_named_errors():
    ok, detail = check_reraise_list(EVALUATOR_SOURCE.read_text())
    assert ok, detail


# The shape the file had before this fix: one blanket catch, nothing re-raised.
BLANKET_CATCH_SHAPE = '''
def _evaluate_one_example(self, i, example, dataset_len, env, logic, run, sfx):
    try:
        return logic.build_dynamic_parser(self, env, example)
    except Exception as e:
        return {"is_correct": False, "error": str(e)}
'''


def test_the_check_rejects_the_blanket_catch():
    ok, detail = check_reraise_list(BLANKET_CATCH_SHAPE)
    assert not ok, (
        "The check passed against the blanket-catch shape, so it would pass no "
        "matter what the file says and proves nothing."
    )
    assert "No re-raise clause" in detail


WIDENED_LIST_SHAPE = '''
def _evaluate_one_example(self, i, example, dataset_len, env, logic, run, sfx):
    try:
        return logic.build_dynamic_parser(self, env, example)
    except (ModuleNotFoundError, ImportError, UngradableExample, TypeError, ValueError):
        raise
    except Exception as e:
        return {"is_correct": False, "error": str(e)}
'''


def test_the_check_rejects_a_widened_list():
    """The failure this is guarding against is the list growing later, so the
    check has to notice extra names, not just an empty clause."""
    ok, detail = check_reraise_list(WIDENED_LIST_SHAPE)
    assert not ok
    assert "TypeError" in detail


# --------------------------------------------------------------------------
# the same bug, one file over
# --------------------------------------------------------------------------
#
# Re-raising past the per-example catch does not end the story. The raise lands
# in the outer catch at evaluator.py:2958, which returns
# EvaluationResult(success=False, accuracy=0.0, num_examples=0, error=<msg>).
# Whether that is honest or laundered depends entirely on whether the consumer
# looks at success/num_examples before it looks at accuracy.
#
# feedback_loop.py does: classify_eval_failure (:303-322) keys off num_examples
# and returns HARNESS, and :2038-2052 raises SynthesisExhaustionError with the
# real error text instead of sending it to the strategy model as feedback.
#
# run_reference_strategy.py did not. `_evaluate` read result.accuracy at :143
# with no check at all, and main() wrote it to --output-json and printed
# "accuracy=0.000" (:219-224). A missing module produced a saved results file
# claiming a measured zero. `error` was never carried into the payload, so
# nothing in the file said otherwise.


class _FailedResult:
    success = False
    accuracy = 0.0
    syntax_rate = 0.0
    num_correct = 0
    num_examples = 0
    contains_delimiters = False
    error = "No module named 'vllm_startup'"
    sample_outputs: list = []
    total_time_seconds = 0.1
    max_sample_time_seconds = 0.1


def test_the_reference_runner_refuses_to_report_a_harness_failure_as_zero():
    """A run that measured nothing must not be written out as accuracy 0.000."""
    from synthesis.evaluate.run_reference_strategy import _payload_from_result

    with pytest.raises(SystemExit) as caught:
        _payload_from_result(_FailedResult())

    assert "vllm_startup" in str(caught.value), (
        "Stopped, but without the underlying error, which is the one thing "
        "that says which file is missing."
    )


def test_zero_examples_is_refused_even_when_the_result_claims_success():
    """The two conditions are separate on purpose. An evaluator that reports
    success while having run nothing is the exact shape Spider's fake 0% took:
    reevaluate_compiled_csd.py:94-97 checks both, and so does this."""
    from synthesis.evaluate.run_reference_strategy import _payload_from_result

    class _EmptyButHappy(_FailedResult):
        success = True
        error = None

    with pytest.raises(SystemExit) as caught:
        _payload_from_result(_EmptyButHappy())

    assert "zero examples" in str(caught.value)


def test_a_real_result_is_still_reported_normally():
    """Guard against over-fixing: a genuine 0% from a model that got every
    question wrong is a real measurement and must still be written out."""
    from synthesis.evaluate.run_reference_strategy import _payload_from_result

    class _RealZero(_FailedResult):
        success = True
        error = None
        num_examples = 50

    payload = _payload_from_result(_RealZero())

    assert payload["accuracy"] == 0.0
    assert payload["num_examples"] == 50
