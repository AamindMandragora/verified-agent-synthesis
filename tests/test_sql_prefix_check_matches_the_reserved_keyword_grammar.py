"""The SQL prefix check must agree with the grammar as syncode lexes it: keywords
are reserved words, and a finished word that cannot be parsed is an error even when
it is the last word typed.

Measured 2026-09-19 by brute force (saved-results/2026-09-18-span-cheat-audit.md):
before this rule 112 impossible prefixes were accepted, e.g. ``SELECT AND AND``.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sql_parser():
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
    grammar = (REPO_ROOT / "synthesis/evaluate/grammars/sql.lark").read_text()
    return create_lark_dafny_parser(grammar, _VDA(), _Dafny(), tokenizer=tok, dfa_mode="grammar_mask")(vocab)


@pytest.mark.parametrize("text", [
    "SELECT AND AND",
    "SELECT 1 WHERE ",
    "SELECT a FROM t WHERE WHERE ",
    "SELECT FROM FROM",
    "SELECT NULL AS F , s d-",  # a -- comment here could never be followed by FROM
    "SELECT a FROM t --x",
])
def test_a_prefix_no_query_can_start_with_is_rejected(sql_parser, text):
    assert not sql_parser._is_valid_prefix(text)


@pytest.mark.parametrize("text", [
    "SELECT",
    "SELECT AND",  # may still grow into the name ANDy
    "SELECT a FROM t WHERE a > 1",
    "SELECT count(*) FROM singer",
    "SELECT ( ( a",
    "SELECT 3.",  # two finished words (3 and .) that are really the start of 3.5
    "SELECT a FROM t WHERE a > 3.",
])
def test_a_prefix_some_query_starts_with_is_accepted(sql_parser, text):
    assert sql_parser._is_valid_prefix(text)


def test_a_keyword_is_not_a_column_name(sql_parser):
    assert not sql_parser._is_complete("SELECT ( AND ) FROM t")
    assert sql_parser._is_complete("SELECT ( ANDy ) FROM t")
