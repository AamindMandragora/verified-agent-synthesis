"""The shared SMILES prompt ends with "Return exactly one line containing the SMILES
molecule" and then a bare ``Molecule: ``, so one draw is one line. The unconstrained lane
stopped at a blank line instead, which let a single draw write a whole list of molecules;
the scorer then keeps the text after the LAST ``Molecule:``, so that draw got as many
tries as it had lines. The 2026-09-20 acrylates trace shows it: draw 9 ran 352 tokens
over about fifteen molecules and was scored on ``CCCCCCOC(=O)C=C``, the last one. GCD,
which sends this very prompt, gets one molecule per draw because its grammar ends the
generation. Both backends now stop at the first newline, so every lane gets one line per
draw.

What this does not change: the model answering with an IUPAC name, or with ``0``, is the
model answering badly under an unconstrained decode. That is the lane's result, not a
harness fault.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from synthesis.evaluate import run_legacy_fixed_strategy as runner


class _FakeCompletion:
    def __init__(self, text: str):
        self.text = text
        self.token_ids = [1, 2, 3]
        self.finish_reason = "stop"
        self.stop_reason = "\n"


class _FakeOutput:
    def __init__(self, text: str):
        self.outputs = [_FakeCompletion(text)]


class _FakeChatTokenizer:
    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(m["content"] for m in messages)


class _FakeLLM:
    def __init__(self, **kwargs):
        pass

    def get_tokenizer(self):
        return _FakeChatTokenizer()

    def generate(self, prompts, sampling_params):
        return [_FakeOutput("C=CC(=O)OCCCC")]


def _args(tmp_path, backend: str) -> argparse.Namespace:
    return argparse.Namespace(
        smiles_split_name="train",
        smiles_trial=1,
        dataset="smiles",
        eval_model="Qwen/Qwen2.5-1.5B-Instruct",
        eval_backend=backend,
        device="cpu",
        eval_sample_size=1,
        eval_start_index=0,
        eval_end_index=1,
        eval_max_steps=64,
        smiles_classes="acrylates",
        smiles_samples_per_class=1,
        vllm_gpu_memory_utilization=0.3,
        output_json=tmp_path / "unconstrained_smiles.json",
    )


def test_a_vllm_draw_stops_at_the_end_of_the_line(monkeypatch, tmp_path):
    seen: list[dict] = []

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    assert runner.run_unconstrained_smiles_adapter(_args(tmp_path, "vllm")) == 0
    assert seen and seen[0]["stop"] == ["\n"]


def test_a_huggingface_draw_stops_at_the_end_of_the_line(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    seen: list[dict] = []

    class _FakeTokenizer(_FakeChatTokenizer):
        def __call__(self, text, return_tensors=None):
            self.last = text
            return self

        def to(self, _device):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return "C=CC(=O)OCCCC"

    class _FakeModel:
        device = "cpu"
        generation_config = types.SimpleNamespace(eos_token_id=0)

        def generate(self, **kwargs):
            seen.append(kwargs)
            return torch.tensor([[1, 2, 3, 4, 5]])

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeTokenizer())
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeModel()),
    )

    assert runner.run_unconstrained_smiles_adapter(_args(tmp_path, "hf")) == 0
    assert seen and seen[0]["stop_strings"] == ["\n"]
    assert seen[0].get("tokenizer") is not None, "stop_strings needs the tokenizer to work"
