"""``is_valid`` must apply the grammar-completeness check to every end token.

Why this file exists
---------------------
``SyncodeLogitsProcessor.is_valid`` decides whether a proposed next token may
be accepted. For an end token, the rule is: only accept it once the parse is
actually complete (``AcceptSequence(['$END'])`` reachable) -- otherwise a
model could stop mid-parse and the answer would be silently truncated.

The check compared the next token against ``self.tokenizer.eos_token_id``
only -- a single id. Qwen3.5 has two end tokens (the tokenizer's
``<|im_end|>`` and the model config's ``<|endoftext|>``, both collected into
``self.eos_token_ids`` at init by ``HuggingFaceModel.__init__``). Whichever
one was NOT ``tokenizer.eos_token_id`` skipped the completeness check
entirely and fell through to ``dfa_mask_store.is_valid_prefix``, which asks a
different question (is this a valid PREFIX, not a valid END). This file pins
the fix: both ids in ``self.eos_token_ids`` must get the same check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNCODE_ROOT = REPO_ROOT / "synthesis" / "evaluate" / "syncode"
for _p in (SYNCODE_ROOT, SYNCODE_ROOT / "syncode"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from syncode.grammar_decoder import SyncodeLogitsProcessor
from syncode.parse_result import AcceptSequence

TOKENIZER_EOS = 248046  # <|im_end|>
MODEL_CONFIG_EOS = 248044  # <|endoftext|>


class _FakeParseResult:
    def __init__(self, accept_sequences):
        self.accept_sequences = accept_sequences


class _FakeIncParser:
    def __init__(self, accept_sequences):
        self._result = _FakeParseResult(accept_sequences)

    def get_acceptable_next_terminals(self, partial_code):
        return self._result


class _FakeDecoder:
    """Only the state ``is_valid`` reads: both end-token ids and a parser stub."""

    def __init__(self, accept_sequences):
        self.eos_token_ids = {TOKENIZER_EOS, MODEL_CONFIG_EOS}
        self.inc_parser = _FakeIncParser(accept_sequences)

    def _get_partial_codes(self, input_ids):
        return ["irrelevant partial code"]


def _is_valid(next_token_id: int, *, parse_is_complete: bool) -> bool:
    accept_sequences = (
        [AcceptSequence(["$END"])] if parse_is_complete else [AcceptSequence(["SOME_OTHER_TERMINAL"])]
    )
    decoder = _FakeDecoder(accept_sequences)
    input_ids = torch.tensor([[1, 2, 3]])
    next_token = torch.tensor([next_token_id])
    return SyncodeLogitsProcessor.is_valid(decoder, input_ids, next_token)


@pytest.mark.parametrize("end_token_id", [TOKENIZER_EOS, MODEL_CONFIG_EOS])
def test_either_end_token_is_rejected_mid_parse(end_token_id):
    assert _is_valid(end_token_id, parse_is_complete=False) is False, (
        f"end token {end_token_id} was accepted while the parse was still "
        "incomplete -- only the tokenizer's own eos id was getting the "
        "completeness check before this fix."
    )


@pytest.mark.parametrize("end_token_id", [TOKENIZER_EOS, MODEL_CONFIG_EOS])
def test_either_end_token_is_accepted_once_the_parse_is_complete(end_token_id):
    assert _is_valid(end_token_id, parse_is_complete=True) is True
