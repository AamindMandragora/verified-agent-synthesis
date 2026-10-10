"""Non-CARS SMILES baseline lanes must not feed previously accepted molecules back into
the prompt they send the model.

Background: every SMILES baseline lane (GCD, CRANE, IterGen, unconstrained, CARS) used to
append each accepted molecule onto the prompt for that class's later draws (a "rolling
prompt"), gated by the ``CSD_SMILES_ROLLING_PROMPT`` env var (default on) for GCD/CRANE/
IterGen, and unconditionally for unconstrained. The unconstrained lane's docstring claimed
this "match[es] the CARS paper protocol", but the CARS authors' own sampling loop
(``~/cars/cars/cars.py::CARS.get_sample``/``get_samples``) encodes the prompt once in
``__init__`` and reuses the same ``prompt_ids`` for every draw -- no rolling suffix at all.

So for every SMILES lane except CARS, the prompt for draw N must be identical to the
prompt for draw 1, regardless of what the lane accepted on earlier draws. The CARS lane
is covered by tests/test_cars_smiles_lane_does_not_roll_its_prompt.py.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate import run_legacy_fixed_strategy as runner


class _ChatTokenizer:
    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(m["content"] for m in messages)


_SMILES_PROMPT = (
    "Generate a molecule. "
    "Your response must be a single SMILES molecule and nothing else.\n"
    "Molecule: CCO\n"
    "Molecule:"
)
_GRAMMAR = "start: /[A-Z0-9=()]+/"


def _two_examples(prompt: str = _SMILES_PROMPT, grammar: str = _GRAMMAR) -> list[dict]:
    return [
        {"class_name": "acrylates", "draw_index": 0, "draw_seed": 100000, "smiles_split_name": "train", "smiles_trial": 1, "grammar_text": grammar, "prompt": prompt},
        {"class_name": "acrylates", "draw_index": 0, "draw_seed": 100000, "smiles_split_name": "train", "smiles_trial": 1, "grammar_text": grammar, "prompt": prompt},
    ]


def _install_fake_syncode(monkeypatch, attr_name: str, fake_cls) -> None:
    package = types.ModuleType("syncode")
    package.__path__ = []
    infer_module = types.ModuleType("syncode.infer")
    setattr(infer_module, attr_name, fake_cls)
    monkeypatch.setitem(sys.modules, "syncode", package)
    monkeypatch.setitem(sys.modules, "syncode.infer", infer_module)


def _install_common_evaluator_fakes(monkeypatch, load_dataset_sample):
    class FakeLogic:
        @staticmethod
        def load_dataset_sample(runtime):
            return load_dataset_sample()

        @staticmethod
        def expected_answer(runtime, example):
            return None

        @staticmethod
        def extract_actual(runtime, scored_output, example):
            return scored_output, "completion", {"syntax_valid": True}

        @staticmethod
        def is_correct(runtime, actual, expected, example, aux, scored_output):
            return False

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def _check_syntax_validity(scored_output, *, example):
            return True, []

    import synthesis.evaluate.benchmarks.registry as registry
    import synthesis.evaluate.evaluator as evaluator

    monkeypatch.setattr(registry, "get_logic", lambda dataset: FakeLogic())
    monkeypatch.setattr(evaluator, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(runner, "_configure_fixed_eval_runtime", lambda *args: None)
    monkeypatch.setattr(
        runner, "_legacy_benchmark_prompt", lambda logic, runtime, example, mode: example["prompt"]
    )
    monkeypatch.setattr(runner, "_baseline_row_question", lambda *args: "question")


# ---------------------------------------------------------------------------
# GCD
# ---------------------------------------------------------------------------


def test_gcd_lane_sends_the_same_prompt_on_draw_two_as_draw_one(monkeypatch, tmp_path):
    prompts_seen: list[str] = []

    class FakeSyncode:
        def __init__(self, **kwargs):
            self.model = types.SimpleNamespace(tokenizer=_ChatTokenizer())

        def infer(self, prompt, stop_words):
            prompts_seen.append(prompt)
            return ["C=CC(=O)OCC"]

    _install_fake_syncode(monkeypatch, "Syncode", FakeSyncode)
    _install_common_evaluator_fakes(monkeypatch, _two_examples)
    monkeypatch.setattr(runner, "_build_minimal_json", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles", eval_model="Qwen/Qwen2.5-1.5B-Instruct", eval_backend="vllm",
        device="cuda:0", eval_sample_size=2, eval_max_steps=400, eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.3, vllm_tensor_parallel_size=1,
        gsm_split_file=None, gsm_split_name="train", spider_split_file=None, spider_split_name="train",
        smiles_classes="acrylates", output_json=tmp_path / "gcd.json",
    )

    assert runner.run_gcd_legacy_adapter(args) == 0
    assert len(prompts_seen) == 2
    assert prompts_seen[0] == prompts_seen[1]


# ---------------------------------------------------------------------------
# CRANE
# ---------------------------------------------------------------------------


def test_crane_lane_sends_the_same_prompt_on_draw_two_as_draw_one(monkeypatch, tmp_path):
    prompts_seen: list[str] = []

    class FakeAdaptiveSynCode:
        def __init__(self, **kwargs):
            self.model = types.SimpleNamespace(tokenizer=_ChatTokenizer())

        def infer(self, prompt, stop_words=None):
            prompts_seen.append(prompt)
            return ["Reasoning. <<C=CC(=O)OCC>> tail"]

    _install_fake_syncode(monkeypatch, "AdaptiveSynCode", FakeAdaptiveSynCode)
    _install_common_evaluator_fakes(monkeypatch, _two_examples)
    monkeypatch.setattr(runner, "_legacy_local_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(runner, "_build_minimal_json", lambda rows, *a, **k: None)

    args = argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles", eval_model="Qwen/Qwen2.5-Coder-7B-Instruct", eval_backend="vllm",
        device="cuda", eval_sample_size=2, eval_max_steps=64, eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.1, vllm_tensor_parallel_size=1,
        gsm_split_file=None, gsm_split_name="eval", spider_split_file=None, spider_split_name="eval",
        smiles_classes="acrylates", output_json=str(tmp_path / "crane.json"),
    )

    assert runner._crane_via_adaptive_syncode(args, "smiles") == 0
    assert len(prompts_seen) == 2
    assert prompts_seen[0] == prompts_seen[1]


# ---------------------------------------------------------------------------
# IterGen
# ---------------------------------------------------------------------------


def test_itergen_lane_sends_the_same_prompt_on_draw_two_as_draw_one(monkeypatch, tmp_path):
    pytest.importorskip("rdkit")
    from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task

    task = get_smiles_task("acrylates")
    start_prompts: list[str] = []

    class FakeIterGen:
        def __init__(self, **kwargs):
            self.tokenizer = _ChatTokenizer()

        def start(self, prompt):
            start_prompts.append(prompt)

        def forward(self, units=None, num=1):
            return ["CCCCCOC(=O)C=C"]

        def finished(self):
            return True

        def backward(self, unit=None, num=1):
            pass

    itergen_package = types.ModuleType("itergen")
    itergen_package.__path__ = []
    itergen_main = types.ModuleType("itergen.main")
    itergen_main.IterGen = FakeIterGen
    monkeypatch.setitem(sys.modules, "itergen", itergen_package)
    monkeypatch.setitem(sys.modules, "itergen.main", itergen_main)

    def _examples():
        return [
            {
                "class_name": "acrylates", "draw_index": 0, "draw_seed": 100000, "smiles_split_name": "train", "smiles_trial": 1,
                "grammar_text": task["grammar_text"],
                "prompt": task["prompt"],
                "prompt_exemplars": task["prompt_exemplars"],
            },
            {
                "class_name": "acrylates", "draw_index": 0, "draw_seed": 100000, "smiles_split_name": "train", "smiles_trial": 1,
                "grammar_text": task["grammar_text"],
                "prompt": task["prompt"],
                "prompt_exemplars": task["prompt_exemplars"],
            },
        ]

    _install_common_evaluator_fakes(monkeypatch, _examples)
    monkeypatch.setattr(runner, "_itergen_add_import_paths", lambda root: None)
    monkeypatch.setattr(runner, "_install_itergen_transformers_compat", lambda cls: None)
    monkeypatch.setattr(runner, "_legacy_local_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(runner, "_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(runner, "_build_minimal_json", lambda rows, *a, **k: None)
    monkeypatch.setattr(runner.Path, "exists", lambda self: True)

    args = argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles", eval_model="Qwen/Qwen3.5-2B", eval_backend="vllm", device="cuda",
        eval_sample_size=2, eval_max_steps=64, eval_step_token_budget=1,
        vllm_gpu_memory_utilization=0.1, vllm_tensor_parallel_size=1,
        gsm_split_file=None, gsm_split_name="eval", spider_split_file=None, spider_split_name="eval",
        smiles_classes="acrylates", output_json=str(tmp_path / "itergen.json"),
    )

    assert runner._run_itergen_legacy_adapter_inner(args) == 0
    assert len(start_prompts) == 2
    assert start_prompts[0] == start_prompts[1]


# ---------------------------------------------------------------------------
# Unconstrained
# ---------------------------------------------------------------------------


def test_unconstrained_lane_sends_the_same_prompt_on_draw_two_as_draw_one(monkeypatch, tmp_path):
    pytest.importorskip("rdkit")
    prompts_seen: list[str] = []

    class FakeCompletion:
        def __init__(self, text: str):
            self.text = text
            self.token_ids = [1, 2, 3]
            self.finish_reason = "stop"
            self.stop_reason = "\n"

    class FakeOutput:
        def __init__(self, text: str):
            self.outputs = [FakeCompletion(text)]

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return _ChatTokenizer()

        def generate(self, prompts, sampling_params):
            prompts_seen.append(prompts[0])
            return [FakeOutput("CCCCCOC(=O)C=C\n")]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(runner, "_legacy_local_cuda_device", lambda device: "cpu")
    monkeypatch.setattr(runner, "_build_minimal_json", lambda *a, **k: None)

    args = argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles", eval_backend="vllm", eval_model="fake/model", device="cpu",
        eval_sample_size=2, eval_start_index=0, eval_end_index=2, eval_max_steps=64,
        eval_step_token_budget=1, vllm_gpu_memory_utilization=0.3, vllm_tensor_parallel_size=1,
        smiles_classes="acrylates", smiles_samples_per_class=2,
        output_json=tmp_path / "unconstrained.json",
    )

    assert runner.run_unconstrained_smiles_adapter(args) == 0
    assert len(prompts_seen) == 2
    assert prompts_seen[0] == prompts_seen[1]
