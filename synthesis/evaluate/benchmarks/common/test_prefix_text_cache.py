"""Contract for the prefix-to-text cache in parser_utils.

Every grammar question the decoder asks -- IsValidPrefix, IsCompletePrefix,
ValidNextTokenCountUpTo, ValidNextToken, GroupHasValidMember, about five per decode
step -- needs the answer-so-far as plain text, but it is stored as a list of
token objects. Converting walks every token, so without a cache the same
conversion is redone about five times per step, and because the answer gets
longer as you generate, the total cost grows with the square of its length.
Measured on focal: at a 400-token answer that was 31 seconds per 50-example
eval, at 800 tokens it was 125 seconds.

The cache is deliberately placed on _tokens_to_text and not on _structured_text
or _complete_text. Those two build DIFFERENT strings from the same prefix --
_structured_text prepends the "<<" span opener for CRANE and _complete_text does
not -- so a cache at that level would hand back one where the other was asked
for. Both call _tokens_to_text, so caching there is safe and covers both.

These tests build the parser with tokenizer=None, which skips loading the
341 MB grammar mask store. The text conversion under test never consults the
tokenizer, and the whole file runs in about ten seconds.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_COMPILED = os.path.join(_REPO, "outputs/compiled_references/cars/ref_cars")

if _COMPILED not in sys.path:
    sys.path.insert(0, _COMPILED)

_dafny = pytest.importorskip("_dafny", reason="needs a compiled Dafny reference")
VerifiedDecoderAgent = pytest.importorskip(
    "VerifiedDecoderAgent", reason="needs a compiled Dafny reference"
)

from synthesis.evaluate.benchmarks.common import parser_utils  # noqa: E402

GRAMMAR = os.path.join(_REPO, "synthesis/evaluate/grammars/smiles_chain_extenders.lark")


def _make_parser(span_opener=None):
    with open(GRAMMAR) as fh:
        grammar_text = fh.read()
    Parser = parser_utils.create_lark_dafny_parser(
        grammar_text, VerifiedDecoderAgent, _dafny,
        start="start", tokenizer=None,
        constrained_span_opener=span_opener,
    )
    tokens = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(t) for t in ["C", "O", "N", "(", ")", "=", "1"]]
    )
    return Parser(tokens)


def _prefix(chars):
    return _dafny.SeqWithoutIsStrInference([_dafny.Seq(c) for c in chars])


class _CallCounter:
    """Counts how many single tokens get converted, so a test can tell a cache
    hit from a redone conversion."""

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self._real(*a, **kw)


# --------------------------------------------------------------------------
# Correctness must not change.
# --------------------------------------------------------------------------

def test_same_prefix_gives_same_text_every_time():
    p = _make_parser()
    pre = _prefix("CCO")
    assert p._tokens_to_text(pre) == "CCO"
    assert p._tokens_to_text(pre) == "CCO"
    assert p._tokens_to_text(pre) == "CCO"


def test_different_prefixes_of_equal_length_do_not_collide():
    """Two live prefixes of the same length must each get their own text. A
    cache keyed on length alone, or on an id that got reused, would hand the
    second prefix the first one's text."""
    p = _make_parser()
    a = _prefix("CCO")
    b = _prefix("ONC")
    assert p._tokens_to_text(a) == "CCO"
    assert p._tokens_to_text(b) == "ONC"
    assert p._tokens_to_text(a) == "CCO"
    assert p._tokens_to_text(b) == "ONC"


def test_single_token_temporary_lists_still_convert_correctly():
    """parser_utils calls _tokens_to_text([prefix[i]]) with a fresh list each
    time. Those temporaries die immediately, so their id can be handed to a
    later, unrelated list. A cache must never serve a stale hit for them."""
    p = _make_parser()
    pre = _prefix("CON")
    assert [p._tokens_to_text([pre[i]]) for i in range(3)] == ["C", "O", "N"]
    for _ in range(200):
        assert p._tokens_to_text([pre[0]]) == "C"
        assert p._tokens_to_text([pre[1]]) == "O"


def test_empty_prefix_is_empty_text():
    p = _make_parser()
    assert p._tokens_to_text(_prefix("")) == ""


# --------------------------------------------------------------------------
# The cache must actually cache.
# --------------------------------------------------------------------------

def test_repeat_conversion_of_same_prefix_does_no_extra_work():
    p = _make_parser()
    pre = _prefix("CCOCCOCCO")

    counter = _CallCounter(parser_utils.dafny_seq_to_str)
    parser_utils.dafny_seq_to_str = counter
    try:
        assert p._tokens_to_text(pre) == "CCOCCOCCO"
        after_first = counter.calls
        assert after_first >= 9, "first conversion should walk every token"
        for _ in range(20):
            assert p._tokens_to_text(pre) == "CCOCCOCCO"
        after_repeats = counter.calls
    finally:
        parser_utils.dafny_seq_to_str = counter._real

    assert after_repeats == after_first, (
        f"20 repeat conversions re-walked the tokens: {after_first} token "
        f"conversions became {after_repeats}"
    )


def test_five_predicate_calls_per_step_convert_the_prefix_once():
    """The real access pattern: one decode step asks several questions about
    the same prefix, and all of them should share one conversion."""
    p = _make_parser()
    pre = _prefix("CCOCC")

    counter = _CallCounter(parser_utils.dafny_seq_to_str)
    parser_utils.dafny_seq_to_str = counter
    try:
        p._structured_text(pre)
        baseline = counter.calls
        p._structured_text(pre)
        p._complete_text(pre)
        p._structured_text(pre)
        p._complete_text(pre)
        total = counter.calls
    finally:
        parser_utils.dafny_seq_to_str = counter._real

    assert total == baseline, (
        f"four further predicate calls on the same prefix cost "
        f"{total - baseline} extra token conversions; expected 0"
    )


# --------------------------------------------------------------------------
# Caching must not confuse the two texts.
# --------------------------------------------------------------------------

def test_structured_and_complete_text_stay_distinct_under_caching():
    p = _make_parser(span_opener="<<")
    pre = _prefix("CCO")
    assert p._structured_text(pre) == "<<CCO"
    assert p._complete_text(pre) == "CCO"

    other = _prefix("ONC")
    assert p._complete_text(other) == "ONC"
    assert p._structured_text(other) == "<<ONC"

    assert p._structured_text(pre) == "<<CCO"
    assert p._complete_text(pre) == "CCO"


def test_no_span_opener_means_both_texts_match():
    p = _make_parser(span_opener=None)
    pre = _prefix("CCO")
    assert p._structured_text(pre) == "CCO"
    assert p._complete_text(pre) == "CCO"


# --------------------------------------------------------------------------
# The cache must not grow without bound.
# --------------------------------------------------------------------------

def test_cache_is_bounded():
    """A run creates a new prefix object every decode step across thousands of
    examples. A cache that keeps them all pins that memory for the whole run."""
    p = _make_parser()
    for i in range(5000):
        p._tokens_to_text(_prefix("CO" * (i % 7 + 1)))

    cache = getattr(p, "_prefix_text_cache", None)
    assert cache is not None, (
        "expected the parser to expose its prefix->text cache as "
        "_prefix_text_cache so its size can be checked"
    )
    assert len(cache) <= 1024, (
        f"cache grew to {len(cache)} entries after 5000 distinct prefixes"
    )


def test_correctness_survives_eviction():
    """Once the cache has been churned past its limit, conversions must still be
    right. An evicted entry means redo the work, never a wrong string."""
    p = _make_parser()
    pre = _prefix("CCO")
    assert p._tokens_to_text(pre) == "CCO"
    for i in range(5000):
        p._tokens_to_text(_prefix("CO" * (i % 7 + 1)))
    assert p._tokens_to_text(pre) == "CCO"
