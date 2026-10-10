import argparse
import sys
import types
from pathlib import Path

from synthesis.evaluate import run_legacy_fixed_strategy as runner
from synthesis.evaluate.run_legacy_fixed_strategy import _gcd_generation_kwargs


def test_smiles_gcd_samples_at_the_cars_paper_gcd_temperature():
    assert _gcd_generation_kwargs("smiles") == {
        "do_sample": True,
        "temperature": 1.0,
    }


def test_non_smiles_gcd_remains_greedy():
    assert _gcd_generation_kwargs("gsm_symbolic") == {"do_sample": False}
    assert _gcd_generation_kwargs("spider") == {"do_sample": False}


def test_smiles_sampling_reaches_syncode_constructor(monkeypatch, tmp_path: Path):
    constructor_kwargs = []

    class FakeSyncode:
        def __init__(self, **kwargs):
            constructor_kwargs.append(kwargs)
            from transformers import AutoTokenizer

            self.model = types.SimpleNamespace(
                tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
            )

        def infer(self, prompt, stop_words):
            return ["C=C"]

    syncode_package = types.ModuleType("syncode")
    syncode_package.__path__ = []
    syncode_infer = types.ModuleType("syncode.infer")
    syncode_infer.Syncode = FakeSyncode
    monkeypatch.setitem(sys.modules, "syncode", syncode_package)
    monkeypatch.setitem(sys.modules, "syncode.infer", syncode_infer)

    class FakeLogic:
        @staticmethod
        def load_dataset_sample(runtime):
            return [
                {
                    "class_name": "acrylates", "draw_index": 0, "draw_seed": 100000, "smiles_split_name": "train", "smiles_trial": 1,
                    "grammar_text": "start: /.+/",
                    "prompt": "Molecule:",
                }
            ]

        @staticmethod
        def expected_answer(runtime, example):
            return "expected"

        @staticmethod
        def extract_actual(runtime, output, example):
            return "C=C", "generated", {"syntax_valid": True}

        @staticmethod
        def is_correct(runtime, actual, expected, example, aux, output):
            return False

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @staticmethod
        def _check_syntax_validity(output, *, example):
            return True, []

    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator

    monkeypatch.setattr(registry, "get_logic", lambda dataset: FakeLogic())
    monkeypatch.setattr(evaluator, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_legacy_benchmark_prompt",
        lambda logic, runtime, example, mode: example["prompt"],
    )
    monkeypatch.setattr(runner, "_baseline_row_question", lambda *args: "question")
    monkeypatch.setattr(runner, "_build_minimal_json", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles",
        eval_model="Qwen/Qwen2.5-1.5B-Instruct",
        eval_backend="vllm",
        device="cuda:0",
        eval_sample_size=1,
        eval_max_steps=400,
        eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        gsm_split_file=None,
        gsm_split_name="train",
        spider_split_file=None,
        spider_split_name="train",
        smiles_classes="acrylates",
        output_json=tmp_path / "gcd.json",
    )

    assert runner.run_gcd_legacy_adapter(args) == 0
    assert constructor_kwargs[0]["do_sample"] is True
    assert constructor_kwargs[0]["temperature"] == 1.0
