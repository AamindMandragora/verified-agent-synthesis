"""IntersectTokenSets / SubtractTokenSets fast paths: same answers, and
IntersectTokenSets(lm.Tokens, x) costs O(|x|) instead of O(vocabulary).

Contract being matched (synthesis/verify/library/VerifiedAgentSynthesis.dfy,
CSDHelpers.IntersectTokenSets):

    ensures forall t :: t in result ==> t in a && t in b
    ensures |result| <= |a|

The Dafny body walks `a` in order and keeps a[i] whenever a[i] is in b.
The Python fast path that production has used since it was added walks `a` in
order and keeps a[i] whenever its text is in b and that text was not already
kept. The two agree whenever `a` has no repeated texts (which is what
lm.ValidTokensIdsLogits() states for lm.Tokens). On a real Qwen vocabulary
`a` DOES repeat texts (several byte tokens decode to the same string), and
there the fast path drops the repeats. Both answers meet the ensures clauses.
These tests pin the new code to the fast path's exact output (order, repeats,
membership, and the very token objects returned), so eval results do not move.

CPU only. Needs `dafny` and the Qwen3.5 tokenizer in the HF cache; skips
otherwise.
"""
from __future__ import annotations

import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "synthesis" / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
TOKENIZER_NAME = "Qwen/Qwen3.5-2B"


def _dafny_bin():
    found = shutil.which("dafny")
    if found:
        return found
    for cand in (Path.home() / ".dotnet/tools/dafny", Path("/opt/homebrew/bin/dafny")):
        if cand.exists():
            return str(cand)
    return None


# --- the previous fast path, copied verbatim: the reference answer ----------

def _ref_seq_to_str(tok):
    try:
        return "".join(tok[i] for i in range(len(tok)))
    except Exception:
        return str(tok)


def _ref_intersect(_dafny, a, b):
    b_set = set()
    for i in range(len(b)):
        b_set.add(_ref_seq_to_str(b[i]))
    out = []
    seen_a = set()
    for i in range(len(a)):
        t = a[i]
        s = _ref_seq_to_str(t)
        if s in b_set and s not in seen_a:
            seen_a.add(s)
            out.append(t)
    return _dafny.SeqWithoutIsStrInference(out)


def _ref_subtract(_dafny, a, b):
    b_set = set()
    for i in range(len(b)):
        b_set.add(_ref_seq_to_str(b[i]))
    out = []
    for i in range(len(a)):
        t = a[i]
        if _ref_seq_to_str(t) not in b_set:
            out.append(t)
    return _dafny.SeqWithoutIsStrInference(out)


def _dedupe_by_text(seq):
    seen, out = set(), []
    for i in range(len(seq)):
        s = _ref_seq_to_str(seq[i])
        if s not in seen:
            seen.add(s)
            out.append(seq[i])
    return out


def _assert_same(got, want):
    """Same length, same order, and the very same token objects."""
    assert len(got) == len(want)
    for i in range(len(want)):
        assert got[i] is want[i], (i, _ref_seq_to_str(got[i]), _ref_seq_to_str(want[i]))


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    dafny = _dafny_bin()
    if dafny is None or not LIB.exists():
        pytest.skip("dafny or the library source is not available")
    out_dir = tmp_path_factory.mktemp("token_set_lib")
    res = subprocess.run(
        [dafny, "build", "--target:py", "--no-verify", str(LIB), "-o", str(out_dir / "lib")],
        capture_output=True, text=True,
    )
    built = out_dir / "lib-py"
    if res.returncode != 0 or not built.exists():
        pytest.skip(f"dafny build failed:\n{res.stdout}\n{res.stderr}")
    for mod in ("_dafny", "VerifiedDecoderAgent"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(built))
    import _dafny
    import VerifiedDecoderAgent as VDA
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import _attach_helper_fastpath

    dafny_intersect = VDA.CSDHelpers.IntersectTokenSets
    dafny_subtract = VDA.CSDHelpers.SubtractTokenSets
    _attach_helper_fastpath(VDA)
    yield {
        "_dafny": _dafny,
        "VDA": VDA,
        "dafny_intersect": dafny_intersect,
        "dafny_subtract": dafny_subtract,
        "intersect": VDA.CSDHelpers.IntersectTokenSets,
        "subtract": VDA.CSDHelpers.SubtractTokenSets,
    }
    sys.path.remove(str(built))


@pytest.fixture(scope="module")
def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # pragma: no cover - machine without the cache
        pytest.skip(f"tokenizer {TOKENIZER_NAME} not in the HF cache: {exc}")


def _make_lm(lib, tokenizer, tokens, ids):
    from synthesis.evaluate.benchmarks.common.model_utils import _TensorizedLMBase

    class _LM(_TensorizedLMBase, lib["VDA"].LM):
        def __init__(self):
            lib["VDA"].LM.__init__(self)
            _TensorizedLMBase.__init__(self, lib["_dafny"], tokenizer, tokens, ids, logits_device="cpu")

    return _LM()


@pytest.fixture(scope="module")
def real_lm(lib, tokenizer):
    from synthesis.evaluate.benchmarks.common.model_utils import _build_tokens_dafny

    ids = list(range(len(tokenizer)))
    tokens = _build_tokens_dafny(lib["_dafny"], tokenizer, ids)
    return _make_lm(lib, tokenizer, tokens, ids)


SMALL_VOCAB_TEXTS = ["a", "b", "�", "c", "�", "ab", "", " a", "b", "SELECT", "\n"]


@pytest.fixture(scope="module")
def small_lm(lib, tokenizer):
    _dafny = lib["_dafny"]
    tokens = _dafny.SeqWithoutIsStrInference([_dafny.Seq(s) for s in SMALL_VOCAB_TEXTS])
    return _make_lm(lib, tokenizer, tokens, list(range(len(SMALL_VOCAB_TEXTS))))


def _random_query(_dafny, rng, pool, n, extra_texts=("zz_not_in_vocab", "éé", "  SELECT ")):
    texts = [rng.choice(pool) for _ in range(n)]
    texts += [rng.choice(extra_texts) for _ in range(rng.randint(0, 3))]
    if texts and rng.random() < 0.5:
        texts += rng.sample(texts, k=min(len(texts), 3))  # repeats in b
    rng.shuffle(texts)  # b in non-vocab order
    return _dafny.SeqWithoutIsStrInference([_dafny.Seq(t) for t in texts])


# --- small vocabulary with repeated texts --------------------------------------

def test_small_vocab_intersect_matches_previous_fastpath(lib, small_lm):
    _dafny = lib["_dafny"]
    rng = random.Random(1)
    a = small_lm.Tokens
    for trial in range(400):
        b = _random_query(_dafny, rng, SMALL_VOCAB_TEXTS, rng.randint(0, 8))
        got = lib["intersect"](a, b)
        _assert_same(got, _ref_intersect(_dafny, a, b))
        # Against the Dafny body: identical once the body's repeats of `a` are dropped.
        _assert_same(got, _dedupe_by_text(lib["dafny_intersect"](a, b)))


def test_empty_inputs(lib, small_lm):
    _dafny = lib["_dafny"]
    empty = _dafny.SeqWithoutIsStrInference([])
    assert len(lib["intersect"](small_lm.Tokens, empty)) == 0
    assert len(lib["intersect"](empty, small_lm.Tokens)) == 0
    assert len(lib["intersect"](empty, empty)) == 0
    _assert_same(lib["subtract"](small_lm.Tokens, empty), _ref_subtract(_dafny, small_lm.Tokens, empty))
    assert len(lib["subtract"](empty, small_lm.Tokens)) == 0


def test_tokens_built_by_slicing_or_concatenation(lib, small_lm):
    """Tokens whose storage is a lazy slice or concat, not a plain list."""
    _dafny = lib["_dafny"]
    S = _dafny.Seq
    odd = [S("abc")[0:2], S("xa")[1:2], S("a") + S("b"), S(" ") + S("a"), S("SELECT")[0:0], S("zz") + S("q")]
    b = _dafny.SeqWithoutIsStrInference(odd)
    for a in (small_lm.Tokens, _dafny.SeqWithoutIsStrInference(odd + [S("b"), S("ab")])):
        _assert_same(lib["intersect"](a, b), _ref_intersect(_dafny, a, b))
        _assert_same(lib["subtract"](a, b), _ref_subtract(_dafny, a, b))
        _assert_same(lib["intersect"](b, a), _ref_intersect(_dafny, b, a))


def test_general_path_with_plain_sequences(lib):
    """Neither argument is an LM vocabulary: plain sequences with repeats."""
    _dafny = lib["_dafny"]
    rng = random.Random(2)
    pool = ["x", "y", "z", "xy", "", "�", "q"]
    for _ in range(300):
        a = _random_query(_dafny, rng, pool, rng.randint(0, 10))
        b = _random_query(_dafny, rng, pool, rng.randint(0, 10))
        _assert_same(lib["intersect"](a, b), _ref_intersect(_dafny, a, b))
        _assert_same(lib["subtract"](a, b), _ref_subtract(_dafny, a, b))
        _assert_same(lib["intersect"](a, b), _dedupe_by_text(lib["dafny_intersect"](a, b)))
        _assert_same(lib["subtract"](a, b), list(lib["dafny_subtract"](a, b).Elements))


def test_small_vocab_as_second_argument_and_subtract(lib, small_lm):
    _dafny = lib["_dafny"]
    rng = random.Random(3)
    v = small_lm.Tokens
    for _ in range(300):
        x = _random_query(_dafny, rng, SMALL_VOCAB_TEXTS, rng.randint(0, 8))
        _assert_same(lib["intersect"](x, v), _ref_intersect(_dafny, x, v))
        _assert_same(lib["subtract"](x, v), _ref_subtract(_dafny, x, v))
        _assert_same(lib["subtract"](v, x), _ref_subtract(_dafny, v, x))


# --- real Qwen3.5 vocabulary ---------------------------------------------------

def _vocab_texts(lm):
    toks = lm.Tokens
    return [_ref_seq_to_str(toks[i]) for i in range(len(toks))]


def test_real_vocab_intersect_matches_previous_fastpath(lib, real_lm):
    _dafny = lib["_dafny"]
    rng = random.Random(4)
    texts = _vocab_texts(real_lm)
    counts = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    repeated = [t for t, c in counts.items() if c > 1]
    for n in (0, 1, 13, 1000, 5000):
        b = _random_query(_dafny, rng, texts + repeated * 50, n)
        _assert_same(lib["intersect"](real_lm.Tokens, b), _ref_intersect(_dafny, real_lm.Tokens, b))


def test_real_vocab_intersect_matches_dafny_body(lib, real_lm):
    """One call of the compiled Dafny body on the full vocab (about 5 s)."""
    _dafny = lib["_dafny"]
    rng = random.Random(5)
    b = _random_query(_dafny, rng, _vocab_texts(real_lm) + ["�"] * 20, 40)
    got = lib["intersect"](real_lm.Tokens, b)
    _assert_same(got, _dedupe_by_text(lib["dafny_intersect"](real_lm.Tokens, b)))


def test_real_vocab_second_argument_and_subtract(lib, real_lm):
    _dafny = lib["_dafny"]
    rng = random.Random(6)
    texts = _vocab_texts(real_lm)
    x = _random_query(_dafny, rng, texts + ["�"] * 10, 200)
    v = real_lm.Tokens
    _assert_same(lib["intersect"](x, v), _ref_intersect(_dafny, x, v))
    _assert_same(lib["subtract"](x, v), _ref_subtract(_dafny, x, v))
    _assert_same(lib["subtract"](v, x), _ref_subtract(_dafny, v, x))


def test_full_vocab_intersect_is_fast(lib, real_lm):
    """IntersectTokenSets(lm.Tokens, x) with |x| <= 1000 must take < 5 ms after warm-up.

    The previous fast path scanned all 248k vocabulary entries per call (~0.74 s).
    """
    _dafny = lib["_dafny"]
    rng = random.Random(7)
    texts = _vocab_texts(real_lm)
    b = _random_query(_dafny, rng, texts, 990)
    assert len(b) <= 1000
    fn = lib["intersect"]
    fn(real_lm.Tokens, b)  # warm-up
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        fn(real_lm.Tokens, b)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    assert med < 0.005, f"median {med * 1000:.1f} ms per call"
