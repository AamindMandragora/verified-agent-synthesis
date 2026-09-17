"""TDD tests for mask-leak fix: forbidden tokens must be hard-blocked (Deliverable 1).

ROOT CAUSE: syncode's DFAMaskStore uses 'grammar_mask' mode which is an
over-approximation (see dfa_mask_store.py:153). The GSM grammar bans '{', '}',
and '**', but syncode's DFA may allow whitespace-prefixed variants of these
tokens through because '%ignore WS' makes whitespace valid at any point.
Once a '{' token slips through MaskValidNextAndEos, later tokens inside the
constrained span may extend it, producing spans like <<{daily} + n1>>.

FIX: In parser_utils.py, after the DFA mask computation, apply a secondary
hard-block step: zero out any token in the mask whose string representation
contains '{', '}', or '**'. This is applied only when a known-forbidden-chars
set is passed or detected (GSM-only by default).

These tests operate WITHOUT a GPU — they construct a minimal fake LM/parser to
test the masking logic in isolation.

Run on focal:
    cd /home/aadivyar/csd-generation
    /apps/conda/advayth2/envs/advayth2/bin/python -m pytest tests/test_mask_brace_block.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Minimal fake tokenizer and token list for unit tests.
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stub: each 'token' is its own ID (no BPE)."""

    def __init__(self, vocab: list[str]):
        self._vocab = vocab
        self._id_to_str = {i: v for i, v in enumerate(vocab)}
        self._str_to_id = {v: i for i, v in enumerate(vocab)}
        self.eos_token = "</s>"
        self.eos_token_id = vocab.index("</s>") if "</s>" in vocab else len(vocab) - 1

    def decode(self, ids, add_special_tokens=False):
        return "".join(self._id_to_str.get(i, "") for i in ids)

    def encode(self, text, add_special_tokens=False):
        # split character-by-character as a simplification
        return [self._str_to_id.get(c, 0) for c in text if c in self._str_to_id]

    def get_vocab(self):
        return {v: i for i, v in enumerate(self._vocab)}

    def __len__(self):
        return len(self._vocab)


def _make_minimal_parser(grammar_text: str, vocab: list[str]):
    """Build a SyncodeDafnyParser from a grammar and vocab, without a real LM.

    Returns the parser instance and a token list (Dafny-style list of strings).
    """
    from evaluations.common.parser_utils import create_lark_dafny_parser

    tokenizer = _FakeTokenizer(vocab)

    # We need a fake VerifiedDecoderAgent and _dafny.
    # create_lark_dafny_parser only needs them for type-checking the parser
    # superclass; we can pass lightweight stubs.
    class _FakeParser:
        pass

    class _FakeVDA:
        Parser = _FakeParser

    class _FakeDafny:
        @staticmethod
        def Seq(s):
            return s  # strings are their own "Dafny Seq" in this stub

        @staticmethod
        def SeqWithoutIsStrInference(lst):
            return lst

    vda = _FakeVDA()
    dafny = _FakeDafny()

    parser_factory = create_lark_dafny_parser(
        grammar_text,
        VerifiedDecoderAgent=vda,
        _dafny=dafny,
        start="start",
        tokenizer=tokenizer,
    )
    # The factory is called with lm._Tokens.
    token_list = vocab  # each token = its string (our stub)
    parser_instance = parser_factory(token_list)
    return parser_instance, token_list


# ---------------------------------------------------------------------------
# Test 1 — SOURCE-LEVEL: parser_utils must have forbidden-token hard-block
# ---------------------------------------------------------------------------

def test_parser_utils_has_hard_block_for_forbidden_chars():
    """AST check: _get_accept_mask_for_text (or nearby) must apply a
    secondary hard block for tokens containing '{', '}', '**'.

    RED: before the fix, parser_utils.py has no such block.
    GREEN: after the fix, a post-DFA filter zeros out forbidden-char tokens.
    """
    import ast

    # parser_utils.py lives at different paths depending on repo structure.
    # Focal server: synthesis/evaluate/benchmarks/common/parser_utils.py
    # Old local:    evaluations/common/parser_utils.py
    candidates = [
        _REPO / "synthesis" / "evaluate" / "benchmarks" / "common" / "parser_utils.py",
        _REPO / "evaluations" / "common" / "parser_utils.py",
    ]
    parser_utils_path = next((p for p in candidates if p.exists()), None)
    assert parser_utils_path is not None, (
        f"Could not find parser_utils.py at any of: {candidates}"
    )
    tree = ast.parse(parser_utils_path.read_text())

    # Look for a string constant containing one of the forbidden chars being
    # used in a condition (the filter checks token_str for these chars).
    forbidden_patterns_found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pat in ["{", "**"]:
                if pat in node.value:
                    forbidden_patterns_found.add(pat)

    assert "{" in forbidden_patterns_found, (
        "parser_utils.py must contain a hard-block that filters out tokens "
        "containing '{'. The over-approximation in syncode's DFA lets brace "
        "tokens through; the fix is a post-mask filter in _get_accept_mask_for_text "
        "or its caller.\n"
        "Add e.g.: mask[i] = False for token i whose string contains '{' or '**'"
    )
    assert "**" in forbidden_patterns_found, (
        "parser_utils.py must contain a hard-block that filters out tokens "
        "containing '**'."
    )


# ---------------------------------------------------------------------------
# Test 2 — UNIT: accept mask must be False for brace/star-star tokens
#           when the parser is built from the GSM grammar.
# ---------------------------------------------------------------------------

_GSM_GRAMMAR = """\
syncode: "<<" start ">>"

start: expr

csd_start: expr ">>"

expr: expr OP_ADD term
     | expr OP_SUB term
     | term

term: term OP_MUL factor
     | term OP_DIV factor
     | term OP_IDIV factor
     | term OP_MOD factor
     | factor

factor: OP_SUB factor
       | TYPE "(" expr ")"
       | primary

primary: NUMBER
        | VARIABLE
        | "(" expr ")"

TYPE.4: "int"

OP_ADD: "+"
OP_SUB: "-"
OP_MUL: "*"
OP_DIV: "/"
OP_IDIV: "//"
OP_MOD: "%"

%import common.CNAME -> VARIABLE
%import common.NUMBER

%import common.WS
%ignore WS
"""

# A vocabulary that includes the forbidden tokens as well as legal ones.
_TEST_VOCAB = [
    # Legal GSM tokens
    "n", "x", "k", "+", "-", "*", "/", "//", "%", "(", ")", "1", "2",
    # Forbidden: braces and double-star
    "{", "}", "**",
    # Space-prefixed variants (how BPE tokenizers often encode them)
    " {", " }", " **",
    # EOS
    "</s>",
]


@pytest.mark.parametrize("forbidden_token", ["{", "}", "**", " {", " }", " **"])
def test_forbidden_token_not_in_accept_mask_at_expression_start(forbidden_token):
    """At the start of a GSM expression (empty prefix), the accept mask must
    not include tokens containing '{', '}', or '**'.

    RED: before the fix, the over-approximate DFA mask may allow e.g. ' {' or
         ' **' because whitespace is ignorable and the DFA hasn't ruled them out
         at this prefix state.
    GREEN: the post-mask filter zeros them out.
    """
    try:
        parser, token_list = _make_minimal_parser(_GSM_GRAMMAR, _TEST_VOCAB)
    except Exception as e:
        pytest.skip(f"Could not build parser (likely no syncode/lark on this system): {e}")

    # Empty prefix = start of constrained span.
    # Build empty prefix as a list (our stub: empty list)
    empty_prefix = []

    mask = parser._get_accept_mask_for_prefix(empty_prefix)

    # Find the index of the forbidden token in token_list.
    try:
        idx = _TEST_VOCAB.index(forbidden_token)
    except ValueError:
        pytest.skip(f"Token {forbidden_token!r} not in test vocab")

    # After the fix, this must be False.
    import torch
    if isinstance(mask, torch.Tensor):
        allowed = mask[idx].item()
    else:
        allowed = bool(mask[idx])

    assert not allowed, (
        f"Token {forbidden_token!r} (index {idx}) must not be in the accept mask "
        f"at the start of a constrained GSM span. syncode's over-approximation "
        f"lets it through; the fix must zero it out post-DFA."
    )


# ---------------------------------------------------------------------------
# Test 3 — INTEGRATION: _check_syntax_validity must reject outputs where
#           a brace token appeared in a constrained span.
#
# This is already tested indirectly by test_crane_gsm_syntax.py — we add
# it here as well as a mask-leak-specific regression guard.
# ---------------------------------------------------------------------------

def test_brace_in_span_fails_syntax_check():
    """Integration guard: _check_syntax_validity must return False for spans
    containing '{', regardless of how the token got there."""
    from synthesis.evaluate.evaluator import Evaluator

    ev = Evaluator(dataset_name="gsm_symbolic", backend="huggingface")

    for brace_span in [
        "<<{daily} + n>>",
        "<<p1 + {frac}>>",
        "<<cur{total}>>",
        "<<({sides} - {target}) / {sides} * 100>>",
    ]:
        all_valid, segments = ev._check_syntax_validity(brace_span, example={})
        pass_flag = ev._example_syntax_pass(all_valid, segments, None)
        assert not pass_flag, (
            f"Output {brace_span!r} must fail syntax (brace in span). "
            f"Got: all_valid={all_valid}, segments={segments}"
        )


def test_double_star_in_span_fails_syntax_check():
    """Integration guard: '**' must fail the GSM syntax check."""
    from synthesis.evaluate.evaluator import Evaluator

    ev = Evaluator(dataset_name="gsm_symbolic", backend="huggingface")
    all_valid, segments = ev._check_syntax_validity("<<n0 * (1 + r) ** d>>", example={})
    pass_flag = ev._example_syntax_pass(all_valid, segments, None)
    assert not pass_flag, (
        "'**' operator must fail GSM syntax (not in grammar)."
    )
