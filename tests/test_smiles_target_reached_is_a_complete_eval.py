"""Collecting CARS's 100 unique-valid molecules finishes a SMILES trial.

It must not count as an early stop: EvaluationResult.meets_threshold rejects
early-stopped results, so a strategy scoring 100/100 on train could never win
(seen 2026-09-30 on smiles_4b_chain_extenders attempt 26).
"""

from itertools import count
from types import SimpleNamespace

from synthesis.evaluate.benchmarks.smiles.eval_logic import should_stop_collected
from synthesis.evaluate.evaluator import Evaluator


def _run(monkeypatch, n_examples):
    evaluator = object.__new__(Evaluator)
    ids = count()
    monkeypatch.setattr(
        evaluator,
        "_evaluate_one_example",
        lambda *args, **kwargs: {
            "is_correct": True,
            "timed_out": False,
            "smiles_eval": {"unique_valid_candidate": True, "smiles": f"C{next(ids)}"},
        },
    )
    logic = SimpleNamespace(
        get_generation_runner=lambda: object(),
        should_stop_collected=should_stop_collected,
    )
    return evaluator._evaluate_examples_sequential_with_early_stop(
        [{"prompt": str(i)} for i in range(n_examples)], {}, logic, None, None, None, None
    )


def test_hitting_the_unique_valid_target_on_the_last_sample_is_complete(monkeypatch):
    samples, reason = _run(monkeypatch, 100)
    assert len(samples) == 100
    assert reason is None


def test_hitting_the_target_before_the_sample_cap_stops_without_an_early_stop(monkeypatch):
    samples, reason = _run(monkeypatch, 150)
    assert len(samples) == 100
    assert reason is None
