"""The real parser wrapper applies the closer rule to its masks and to IsValidPrefix."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VOCAB = ["1", "2", "+", ">>", ">>\n", " >>", ">>>", ">", "\n", " ", "</s>"]


class _Tok:
    eos_token = "</s>"
    eos_token_id = VOCAB.index("</s>")
    name_or_path = "closer-rule-test-tokenizer"
    def decode(self, ids, **_): return "".join(VOCAB[i] for i in ids)
    def encode(self, text, **_): return [VOCAB.index(c) for c in text if c in VOCAB]
    def get_vocab(self): return {v: i for i, v in enumerate(VOCAB)}
    def convert_ids_to_tokens(self, ids): return [VOCAB[i] for i in ids]
    def __len__(self): return len(VOCAB)
    vocab_size = len(VOCAB)


@pytest.fixture(scope="module")
def parser():
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    class _VDA:
        class Parser:
            pass

    class _Dafny:
        Seq = staticmethod(lambda s: s)
        SeqWithoutIsStrInference = staticmethod(lambda lst: lst)

    grammar = (REPO / "synthesis/evaluate/grammars/gsm.lark").read_text()
    try:
        factory = create_lark_dafny_parser(grammar, _VDA(), _Dafny(), start="csd_start", tokenizer=_Tok())
    except Exception as exc:  # no syncode on this machine
        pytest.skip(f"cannot build parser: {exc}")
    return factory(VOCAB)


def _allowed(parser, prefix):
    mask = parser._get_accept_mask_for_prefix(prefix)
    return {VOCAB[i] for i in mask.nonzero().flatten().tolist()}


def test_no_closer_lookalike_is_offered_inside_a_span(parser):
    # Before the rule the over-approximate grammar mask offered " >>" here.
    for prefix in (["1"], ["1", "+", "2"], ["1", "+", "2", " "]):
        assert not ({">>", ">>\n", " >>", ">>>"} & _allowed(parser, prefix)), prefix


def test_ordinary_tokens_are_still_offered(parser):
    assert {"+", "1", "2"} <= _allowed(parser, ["1"])


def test_content_holding_a_closer_is_not_a_valid_prefix(parser):
    assert parser.IsValidPrefix(["1", "+", "2"])
    assert not parser.IsValidPrefix(["1", "+", "2", " >>"])
