"""Both of Qwen3.5's end tokens must be recognized as terminal, not just one.

Why this file exists
---------------------
Qwen3.5 has two end tokens: the tokenizer's ``<|im_end|>`` (248046, its real
chat stop) and the model config's ``<|endoftext|>`` (248044). IterGen's own
``start()`` overwrites ``generation_config.eos_token_id`` with the
tokenizer's single id, dropping the other one. ``_itergen_generation_stop_token_ids``
is what the Spider IterGen lane uses to find where a completion's generated
text actually ends (``_itergen_generation_token_evidence``); if it only knows
about one end token, a completion that happens to end on the other one is
never trimmed at its real boundary.
"""

from __future__ import annotations

from synthesis.evaluate.run_legacy_fixed_strategy import (
    _itergen_generation_stop_token_ids,
)


class _FakeModelConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _FakeModel:
    def __init__(self, eos_token_id):
        self.config = _FakeModelConfig(eos_token_id)


class _FakeTokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _FakeIterGen:
    def __init__(self, model_eos_token_id):
        self.model = _FakeModel(model_eos_token_id)


def test_both_end_tokens_are_collected_when_they_differ():
    """The exact Qwen3.5 shape: tokenizer eos (<|im_end|>) != model config eos (<|endoftext|>)."""
    tokenizer = _FakeTokenizer(eos_token_id=248046)
    iter_gen = _FakeIterGen(model_eos_token_id=248044)

    ids = _itergen_generation_stop_token_ids(iter_gen, tokenizer)

    assert ids == {248046, 248044}, (
        f"expected both Qwen3.5 end tokens, got {ids}. Reading only the "
        "tokenizer's id (or only iter_gen.generation_config, which IterGen's "
        "start() narrows to that same single id) drops <|endoftext|>."
    )


def test_a_single_shared_end_token_is_not_duplicated_or_lost():
    """Models with one eos id (e.g. Qwen2.5) must still get exactly that one id."""
    tokenizer = _FakeTokenizer(eos_token_id=151645)
    iter_gen = _FakeIterGen(model_eos_token_id=151645)

    assert _itergen_generation_stop_token_ids(iter_gen, tokenizer) == {151645}


def test_a_list_valued_model_config_eos_is_unioned_too():
    tokenizer = _FakeTokenizer(eos_token_id=248046)
    iter_gen = _FakeIterGen(model_eos_token_id=[248044, 248046])

    assert _itergen_generation_stop_token_ids(iter_gen, tokenizer) == {248044, 248046}


def test_an_explicit_override_still_wins():
    """``generation_stop_token_ids`` set directly on iter_gen or the tokenizer is
    an intentional override and must be honored over the tokenizer/model reading."""
    tokenizer = _FakeTokenizer(eos_token_id=1)
    iter_gen = _FakeIterGen(model_eos_token_id=2)
    iter_gen.generation_stop_token_ids = {99}

    assert _itergen_generation_stop_token_ids(iter_gen, tokenizer) == {99}


class _FakeGenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_iter_gen_generation_config_end_tokens_are_included():
    """Qwen2.5: IterGen stops on generation_config.json's [151645, 151643]; the
    tokenizer and model config know only 151645. The trim must see 151643 too."""
    tokenizer = _FakeTokenizer(eos_token_id=151645)
    iter_gen = _FakeIterGen(model_eos_token_id=151645)
    iter_gen.generation_config = _FakeGenerationConfig([151645, 151643])

    assert _itergen_generation_stop_token_ids(iter_gen, tokenizer) == {151645, 151643}
