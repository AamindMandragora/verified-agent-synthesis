"""The LM wrapper's free-text paths must not let delimiter text into the output outside a span."""
import torch

from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

VOCAB = ["<<", " <<", ">>", " >>", "<", ">", "x", ")<<", "<eos>", " "]


class _CharTokenizer:
    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            match = max((t for t in VOCAB if text.startswith(t, i)), key=len, default=None)
            assert match is not None, text[i:]
            ids.append(VOCAB.index(match))
            i += len(match)
        return ids


class _FakeDafny:
    @staticmethod
    def Seq(s):
        return s

    @staticmethod
    def SeqWithoutIsStrInference(items):
        return list(items)


def _lm(prefix_text="", starts_inside=False):
    lm = object.__new__(_TensorizedLMBase)
    lm.tokenizer = _CharTokenizer()
    lm._dafny = _FakeDafny()
    lm._token_id_to_str = {}
    lm._delimiter_sets = None
    lm._last_prefix_text = prefix_text
    lm._starts_inside_span = starts_inside
    return lm


def _logits(best):
    logits = torch.zeros(len(VOCAB))
    logits[VOCAB.index(best)] = 10.0
    return logits


def _argmax_text(lm, best):
    return VOCAB[int(lm._free_text_logits(_logits(best)).argmax())]


def test_stray_close_and_opener_variants_cannot_win_outside_a_span():
    for bad in [">>", " >>", " <<", ")<<"]:
        assert _argmax_text(_lm("so "), bad) != bad


def test_exact_opener_and_ordinary_tokens_still_win():
    for ok in ["<<", "x", "<", ">"]:
        assert _argmax_text(_lm("so "), ok) == ok


def test_a_token_cannot_complete_a_delimiter_across_the_boundary():
    assert _argmax_text(_lm("a >"), ">") != ">"
    assert _argmax_text(_lm("a <"), "<") != "<"


def test_nothing_is_banned_inside_a_span():
    assert _argmax_text(_lm("so <<1+2"), ">>") == ">>"
    assert _argmax_text(_lm("CCO", starts_inside=True), ">>") == ">>"


def test_the_callers_logits_are_not_modified():
    logits = _logits(">>")
    _lm("so ")._free_text_logits(logits)
    assert logits[VOCAB.index(">>")] == 10.0


def _chunk(lm, texts):
    ids = [VOCAB.index(t) for t in texts]
    tokens, on_open, on_eos, steps = lm._build_unconstrained_chunk_result(ids, "<<", "<eos>", 50)
    return "".join(tokens), on_open, on_eos, steps


def test_chunk_drops_a_stray_close_and_keeps_going():
    text, on_open, _, steps = _chunk(_lm("so "), ["x", ">>", "x", "<<", "x"])
    assert (text, on_open, steps) == ("x>x<<", True, 4)


def test_chunk_handles_a_close_split_across_tokens_and_the_boundary():
    assert _chunk(_lm("so "), ["x", ">", ">", "x"])[0] == "x>x"
    assert _chunk(_lm("so >"), [">", "x"])[0] == "x"


def test_chunk_still_stops_at_an_opener_variant():
    text, on_open, _, _ = _chunk(_lm("so"), ["x", " <<", "x"])
    assert (text, on_open) == ("x <<", True)


class _MergingTokenizer(_CharTokenizer):
    """Re-tokenizes badly: decodes "x>" pieces back with a doubled ">"."""

    def encode(self, text, add_special_tokens=False):
        return [VOCAB.index(">>") if ch == ">" else VOCAB.index(ch) for ch in text]


def test_chunk_output_is_clean_even_if_retokenizing_misbehaves():
    lm = _lm("so ")
    lm.tokenizer = _MergingTokenizer()
    text, _, _, _ = _chunk(lm, ["x", ">>", "x"])
    assert text == "x>x"
