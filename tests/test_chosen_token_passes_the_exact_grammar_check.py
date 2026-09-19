"""The token the model ends up with must pass the exact grammar check, not only the fast mask.

Why: the fast token mask from the syncode library is loose on purpose. After `SC` in the
SMILES grammar it offers about 24,600 tokens and only 828 are valid
(saved-results/2026-09-17-visible-span-findings.md, section 2c). The Dafny proof assumes
every offered token keeps the text a valid prefix. Syncode's own answer is to check the
picked token with the real parser and pick again on failure. These tests pin that rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

VOCAB = ["IE", "C", "O", "<|im_end|>"]
EOS_INDEX = 3


class _LooseMaskParser:
    """The mask offers every ordinary token; the exact check rejects some of them."""

    def __init__(self, exactly_valid):
        self._exactly_valid = set(exactly_valid)
        self.exact_checks = []

    def IsCompletePrefix(self, prefix):
        return False

    def IsValidPrefix(self, prefix):
        return True

    def ValidNextToken(self, prefix, token):
        self.exact_checks.append(token)
        return token in self._exactly_valid


class _Recorder:
    def update_tensors(self, *args, **kwargs):
        pass


class _StandInLM:
    def __init__(self, logits):
        n = len(VOCAB)
        self._Tokens = list(VOCAB)
        self._token_ids = list(range(n))
        self._token_ids_tensor = torch.arange(n)
        self._logits_tensor = torch.tensor(logits, dtype=torch.float32)
        self._full_logits = None
        self._logits_dirty = False
        self._constrained_temperature = 0.0
        self._structured_prompt = None
        self.Logits = _Recorder()

    def _parser_full_mask(self, parser, prefix):
        mask = torch.ones(len(VOCAB), dtype=torch.bool)
        mask[EOS_INDEX] = False
        return mask

    def _token_indices_for_token(self, token):
        return [VOCAB.index(token)]

    def _restore_generation_transaction(self):
        pass

    _select_constrained_index = _TensorizedLMBase._select_constrained_index
    _select_exactly_valid_index = _TensorizedLMBase._select_exactly_valid_index
    MaskToken = _TensorizedLMBase.MaskToken


def _mask_then_choose(lm, parser):
    _TensorizedLMBase.MaskValidNextAndEos(lm, parser, [], "<|im_end|>")
    return _TensorizedLMBase.ChooseNextToken(lm)


def test_a_token_the_mask_offers_but_the_grammar_rejects_is_never_returned():
    lm = _StandInLM(logits=[9.0, 5.0, 1.0, 0.0])  # the model likes "IE" best
    parser = _LooseMaskParser(exactly_valid={"C", "O"})

    assert _mask_then_choose(lm, parser) == "C"
    assert parser.exact_checks == ["IE", "C"]


def test_a_valid_first_pick_costs_one_check_and_is_kept():
    lm = _StandInLM(logits=[1.0, 9.0, 5.0, 0.0])
    parser = _LooseMaskParser(exactly_valid={"C", "O"})

    assert _mask_then_choose(lm, parser) == "C"
    assert parser.exact_checks == ["C"]


def test_when_nothing_offered_is_valid_the_step_stops_instead_of_emitting_a_bad_token():
    lm = _StandInLM(logits=[9.0, 5.0, 1.0, 0.0])
    parser = _LooseMaskParser(exactly_valid=set())

    assert _mask_then_choose(lm, parser) == "<|im_end|>"


def test_a_pick_with_no_parser_mask_in_force_is_not_checked():
    """MaskTokensExcept and friends set their own masks; there is no prefix to check against."""
    lm = _StandInLM(logits=[9.0, 5.0, 1.0, 0.0])
    assert _TensorizedLMBase.ChooseNextToken(lm) == "IE"


# --------------------------------------------------------------------------
# Same rule against the real SMILES parser (the case measured on 2026-09-17)
# --------------------------------------------------------------------------

def _real_smiles_parser():
    transformers = pytest.importorskip("transformers")
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    class _VDA:
        class Parser:
            pass

    class _Dafny:
        Seq = staticmethod(lambda s: s)
        SeqWithoutIsStrInference = staticmethod(lambda items: items)

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3.5-4B tokenizer is not cached on this machine")
    vocab = [tok.decode([i]) for i in range(len(tok))]
    grammar = (REPO_ROOT / "synthesis/evaluate/grammars/smiles_chain_extenders.lark").read_text()
    return create_lark_dafny_parser(grammar, _VDA(), _Dafny(), tokenizer=tok, start="start")(vocab)


def test_real_parser_single_token_answer_is_exact_even_though_the_mask_offers_the_token():
    parser = _real_smiles_parser()
    prefix = ["S", "C"]
    mask = parser._get_accept_mask_for_prefix(prefix)
    ie = parser._token_str_to_idx["IE"]
    assert any(bool(mask[i]) for i in ie), "precondition: the loose mask offers IE after SC"
    assert not parser.is_valid_prefix("SCIE")

    assert not parser.ValidNextToken(prefix, "IE")
    assert parser.ValidNextToken(prefix, "C")
    assert not parser.GroupHasValidMember(prefix, ["IE"])
    assert parser.GroupHasValidMember(prefix, ["IE", "C"])
