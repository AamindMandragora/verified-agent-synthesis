"""CARS baseline settings (arXiv 2510.01902, Appendix C, user-approved 2026-09-20):
temperature 1.0 / top-p 1.0 / no top-k (already the library's own default -- see
_cars_max_new_tokens's docstring neighbor _cars_sampler_steps, which never overrides
sampling), a 256/512-token cap for SMILES/Spider (was 150 for both), and for SMILES a
draw-until-target-unique loop capped at 10x the target instead of one draw per row.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

from synthesis.evaluate import run_legacy_fixed_strategy as runner
from synthesis.evaluate.run_legacy_fixed_strategy import (
    _cars_max_new_tokens,
    _cars_smiles_class_targets,
)


def test_cars_token_cap_matches_the_paper_for_smiles_and_spider():
    assert _cars_max_new_tokens("smiles", eval_max_steps=150) == 256
    assert _cars_max_new_tokens("spider", eval_max_steps=150) == 512


def test_cars_token_cap_still_honors_the_campaign_budget_for_gsm():
    assert _cars_max_new_tokens("gsm_symbolic", eval_max_steps=150) == 150
    assert _cars_max_new_tokens("gsm_symbolic", eval_max_steps=10) == 32


def test_cars_smiles_class_targets_counts_rows_per_class():
    examples = [
        {"class_name": "acrylates"},
        {"class_name": "acrylates"},
        {"class_name": "isocyanates"},
    ]
    assert _cars_smiles_class_targets(examples) == {"acrylates": 2, "isocyanates": 1}


class FakeConstrainedModel:
    def __init__(self, model_id, grammar_str, **kwargs):
        self.model_id = model_id
        self.grammar_constraint = None

    def _set_grammar_constraint(self, grammar_text):
        self.grammar_constraint = grammar_text


class FakeCARSRunner:
    """Records every get_samples call; each call "generates" the next molecule
    off a per-test-provided sequence, written to disk the way the real CARS logs a step."""

    calls: list[dict[str, Any]] = []
    molecule_sequence: list[str] = []
    _call_index = 0

    def __init__(self, model, prompt, sample_style, log_dir):
        self.log_dir = log_dir
        self.prompt = prompt

    def get_samples(self, n_samples, n_steps, stop_after, max_new_tokens):
        FakeCARSRunner.calls.append(
            {"n_steps": n_steps, "max_new_tokens": max_new_tokens, "prompt": self.prompt}
        )
        idx = FakeCARSRunner._call_index
        FakeCARSRunner._call_index += 1
        molecule = FakeCARSRunner.molecule_sequence[idx % len(FakeCARSRunner.molecule_sequence)]
        payload = {"steps": [{"tokens": list(molecule)}]}
        Path(self.log_dir, "step.json").write_text(json.dumps(payload))


def _install_fake_cars(monkeypatch, molecule_sequence: list[str]):
    FakeCARSRunner.calls = []
    FakeCARSRunner.molecule_sequence = molecule_sequence
    FakeCARSRunner._call_index = 0

    cars_pkg = types.ModuleType("cars")
    cars_pkg.__path__ = []
    cars_lib = types.ModuleType("cars.lib")
    cars_lib.ConstrainedModel = FakeConstrainedModel
    cars_cars = types.ModuleType("cars.cars")
    cars_cars.CARS = FakeCARSRunner
    monkeypatch.setitem(sys.modules, "cars", cars_pkg)
    monkeypatch.setitem(sys.modules, "cars.lib", cars_lib)
    monkeypatch.setitem(sys.modules, "cars.cars", cars_cars)


class FakeSmilesLogic:
    """SMILES scoring: correct iff the molecule was marked syntax-valid and is not
    one already accepted for its class in a prior draw (mirrors unique_valid_candidate)."""

    def __init__(self, examples, invalid_molecules):
        self._examples = examples
        self._invalid = set(invalid_molecules)

    def load_dataset_sample(self, runtime):
        return self._examples

    @staticmethod
    def expected_answer(runtime, example):
        return example.get("class_name", "")

    def extract_actual(self, runtime, output, example):
        molecule = output.strip()
        valid = molecule not in self._invalid
        return molecule, "generated", {"syntax_valid": valid}

    @staticmethod
    def is_correct(runtime, actual, expected, example, aux, output):
        return bool(aux and aux.get("syntax_valid"))


class FakeEvaluator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _check_syntax_validity(output, *, example):
        return True, []


def _run_cars_smiles(monkeypatch, tmp_path, examples, molecule_sequence, invalid_molecules=()):
    _install_fake_cars(monkeypatch, molecule_sequence)

    fake_logic = FakeSmilesLogic(examples, invalid_molecules)

    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator

    monkeypatch.setattr(registry, "get_logic", lambda dataset: fake_logic)
    monkeypatch.setattr(evaluator, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(runner, "_configure_fixed_eval_runtime", lambda *a: None)
    monkeypatch.setattr(
        runner,
        "_legacy_benchmark_prompt",
        lambda logic, rt, example, profile: example["prompt"],
    )
    monkeypatch.setattr(runner, "_baseline_row_question", lambda *a: "question")
    monkeypatch.setenv("CARS_REPO_DIR", str(tmp_path))
    monkeypatch.setenv("CSD_SMILES_ROLLING_PROMPT", "0")

    args = argparse.Namespace(
        dataset="smiles",
        eval_model="Qwen/Qwen2.5-1.5B-Instruct",
        eval_backend="vllm",
        device="cuda:0",
        eval_sample_size=1,
        eval_start_index=0,
        eval_end_index=None,
        eval_max_steps=150,
        eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        gsm_split_file=None,
        gsm_split_name="test",
        spider_split_file=None,
        spider_split_name="eval",
        smiles_classes=None,
        cars_search_steps=200,
        output_json=tmp_path / "cars_smiles.json",
    )
    rc = runner.run_cars_legacy_adapter(args)
    assert rc == 0
    return json.loads(args.output_json.read_text())


def test_smiles_cars_uses_the_256_token_cap(monkeypatch, tmp_path):
    examples = [{"class_name": "acrylates", "grammar_text": "", "prompt": "Molecule:"}]
    _run_cars_smiles(monkeypatch, tmp_path, examples, molecule_sequence=["C=C"])
    assert FakeCARSRunner.calls[0]["max_new_tokens"] == 256


def test_smiles_cars_stops_once_the_target_unique_count_is_reached(monkeypatch, tmp_path):
    # Target for "acrylates" is 2 (two rows in the sliced sample); every draw is a
    # distinct valid molecule, so exactly 2 draws should happen, not one per row
    # beyond that and not the 10x cap (20).
    examples = [
        {"class_name": "acrylates", "grammar_text": "", "prompt": "Molecule:"},
        {"class_name": "acrylates", "grammar_text": "", "prompt": "Molecule:"},
    ]
    result = _run_cars_smiles(
        monkeypatch, tmp_path, examples, molecule_sequence=["C=C", "C=CC", "C=CCC"]
    )
    assert len(FakeCARSRunner.calls) == 2
    assert len(result["answers"]) == 2


def test_smiles_cars_caps_at_10x_target_when_nothing_ever_validates(monkeypatch, tmp_path):
    examples = [{"class_name": "isocyanates", "grammar_text": "", "prompt": "Molecule:"}]
    result = _run_cars_smiles(
        monkeypatch,
        tmp_path,
        examples,
        molecule_sequence=["N=C=O"],
        invalid_molecules={"N=C=O"},
    )
    # target 1 * cap multiplier 10 = 10 draws, then give up. None of them is a
    # molecule, so the run collected none of the N it was asked for: no scored rows,
    # and the draws it spent are recorded as metadata instead.
    assert len(FakeCARSRunner.calls) == 10
    assert result["answers"] == []
    assert result["accuracy"] == 0.0
    assert result["metrics"]["cars_smiles_draws_by_class"]["isocyanates"] == {
        "target": 1,
        "draw_cap": 10,
        "draws": 10,
        "rejected_draws": 10,
        "duplicate_draws": 0,
        "scored_unique_molecules": 0,
    }
