"""SynCode collects both of a Qwen3.5 model's end tokens into ``_stop_token_ids`` and
opens the grammar mask to both, but only its own hand-written decode loop (opp=True)
stopped on them. GCD runs with opp=False, which hands the work to HuggingFace's
``model.generate``, and that call was never told the second end token -- so a finished
molecule kept "generating" the end token until the 400-token cap.

The 2026-09-20 acrylates GCD trace shows it: three draws read
``generated_tokens=400 ... stop_reason=end_token last_token_id=248046 raw='C=O'``. The
model wrote C=O in three tokens and spent the other ~397 on an end token nobody was
listening for. Both decode paths now stop on the same set.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
from transformers import PretrainedConfig  # noqa: E402


def _load_hugging_face_model():
    """The GCD lane puts the vendored SynCode on sys.path at run time; do the same here."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "synthesis" / "evaluate" / "syncode"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from syncode.language_model import HuggingFaceModel

    return HuggingFaceModel


HuggingFaceModel = _load_hugging_face_model()

TOKENIZER_EOS = 248046
CONFIG_EOS = 248044


class _FakeTokenizer:
    eos_token_id = TOKENIZER_EOS

    def __call__(self, batch, return_tensors=None):
        return types.SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            to=lambda _device: self._inputs(),
        )

    def _inputs(self):
        class _Inputs(dict):
            input_ids = torch.tensor([[1, 2, 3]])

        return _Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return "C=O"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.config = PretrainedConfig(eos_token_id=CONFIG_EOS)
        self.generation_config = types.SimpleNamespace(eos_token_id=CONFIG_EOS)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 4, 5, TOKENIZER_EOS]])


def _model_under_test() -> _FakeModel:
    hf = object.__new__(HuggingFaceModel)
    model = _FakeModel()
    hf.model = model
    hf.tokenizer = _FakeTokenizer()
    hf.device = "cpu"
    hf.prompt_template = ""
    hf.grammar_decoder = None
    hf.grammar_processor = None
    hf.opp = False
    hf.gen_args = {"max_new_tokens": 400, "do_sample": True, "temperature": 1.0}
    hf._stop_token_ids = {TOKENIZER_EOS, CONFIG_EOS}
    hf.generate_grammar_constrained_completion("Molecule: ", 1)
    return model


def test_the_plain_generate_path_is_told_every_end_token():
    model = _model_under_test()
    assert model.calls, "the opp=False path must call model.generate"
    passed = model.calls[0].get("eos_token_id")
    assert passed is not None, "model.generate was never told which tokens end a generation"
    assert set(passed if isinstance(passed, (list, tuple)) else [passed]) == {
        TOKENIZER_EOS,
        CONFIG_EOS,
    }


def test_the_caller_can_still_set_its_own_end_tokens():
    hf = object.__new__(HuggingFaceModel)
    model = _FakeModel()
    hf.model = model
    hf.tokenizer = _FakeTokenizer()
    hf.device = "cpu"
    hf.prompt_template = ""
    hf.grammar_decoder = None
    hf.grammar_processor = None
    hf.opp = False
    hf.gen_args = {"max_new_tokens": 400, "do_sample": True, "eos_token_id": [7]}
    hf._stop_token_ids = {TOKENIZER_EOS, CONFIG_EOS}
    hf.generate_grammar_constrained_completion("Molecule: ", 1)
    assert model.calls[0]["eos_token_id"] == [7]
