"""Every Spider grammar we control must accept double-quoted string literals.

About 20% of Spider's dev_gold.sql gold queries write string literals with double
quotes (``WHERE Airline = "JetBlue Airways"``) instead of single quotes. Our own
``synthesis/evaluate/grammars/sql.lark`` -- loaded by our method's eval, the CARS
baseline, the GCD baseline, and the CRANE baseline (all four call sites read this
same file; see ``synthesis/evaluate/run_legacy_fixed_strategy.py``) -- only defines
``STRING: /'[^']*'/``, so it rejects every one of those golds outright.

The IterGen authors' own grammar (pinned unmodified at
``synthesis/evaluate/baselines/itergen_authors/sql_grammar.lark``, see that
directory's README for the exact upstream commit) is checked here too, but is not
changed: it already accepts double-quoted text end-to-end, just via its ``name``
rule's ``ESCAPED_STRING`` alternative (the standard-SQL double-quotes-are-identifiers
reading) rather than a real ``literal`` alternative. Acceptance is what this project's
grammars are graded on (this file and the coverage-regression test in
``test_gold_answers_pass_the_prefix_and_completeness_checks.py`` both check
"does it parse", not "what node type did it parse to"), so the authors' grammar
already satisfies the requirement and does not need or get an edit. This test's
IterGen cases pin that down as a fact, not an assumption.

Uses the same construction as the gold-answers test: a real
``SyncodeDafnyParser`` via ``create_lark_dafny_parser`` (tokenizer=None, so it is
pure Lark and does not touch the DFA mask store) for ``_is_valid_prefix``/
``_is_complete``, and a plain ``Lark(..., parser="lalr", lexer="basic")`` parser as
the independent "does the grammar cover this string at all" ground truth used to
count gold coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from lark import Lark

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesis.evaluate.benchmarks.common.parser_utils import (  # noqa: E402
    create_lark_dafny_parser,
)
from synthesis.project_defaults import default_spider_data_dir  # noqa: E402

OURS_GRAMMAR_PATH = REPO_ROOT / "synthesis" / "evaluate" / "grammars" / "sql.lark"
ITERGEN_GRAMMAR_PATH = (
    REPO_ROOT
    / "synthesis"
    / "evaluate"
    / "baselines"
    / "itergen_authors"
    / "sql_grammar.lark"
)


class _FakeParserBase:
    pass


class _FakeVerifiedDecoderAgent:
    Parser = _FakeParserBase


class _FakeDafny:
    Seq = staticmethod(lambda s: s)
    SeqWithoutIsStrInference = staticmethod(lambda items: items)


def _make_parser(grammar_path: Path, **kwargs):
    grammar_text = grammar_path.read_text()
    factory = create_lark_dafny_parser(
        grammar_text,
        _FakeVerifiedDecoderAgent(),
        _FakeDafny(),
        tokenizer=None,
        **kwargs,
    )
    return factory([])


def _plain_lark_full_parser(grammar_path: Path) -> Lark:
    return Lark(grammar_path.read_text(), start="start", parser="lalr", lexer="basic")


def _spider_gold_queries() -> list[str]:
    spider_dir = default_spider_data_dir()
    gold_path = spider_dir / "dev_gold.sql"
    if not gold_path.exists():
        pytest.skip(f"Spider gold data not found at {gold_path}")
    queries = []
    for line in gold_path.read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        gold_sql, _, _gold_db_id = line.partition("\t")
        gold_sql = gold_sql.strip()
        if gold_sql:
            queries.append(gold_sql)
    return queries


def _count_covered(lark_parser: Lark, texts: list[str]) -> int:
    covered = 0
    for text in texts:
        try:
            lark_parser.parse(text)
            covered += 1
        except Exception:
            pass
    return covered


# A handful of real double-quoted and single-quoted golds from dev_gold.sql, picked
# because each grammar's NAME/CNAME rule accepts every identifier in them, so a
# rejection can only come from the STRING/literal handling this test targets.
OURS_DOUBLE_QUOTED_GOLDS = [
    'SELECT Country FROM AIRLINES WHERE Airline  =  "JetBlue Airways"',
    'SELECT Airline ,  Abbreviation FROM AIRLINES WHERE Country  =  "USA"',
    'SELECT AirportCode ,  AirportName FROM AIRPORTS WHERE city  =  "Anthony"',
]
OURS_SINGLE_QUOTED_GOLDS = [
    "SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'",
    "SELECT T2.petid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.Lname  =  'Smith'",
]

ITERGEN_DOUBLE_QUOTED_GOLDS = list(OURS_DOUBLE_QUOTED_GOLDS)
ITERGEN_SINGLE_QUOTED_GOLDS = list(OURS_SINGLE_QUOTED_GOLDS)

# Pinned counts of dev_gold.sql queries (1,034 total, duplicates included -- same
# loader as test_gold_answers_pass_the_prefix_and_completeness_checks.py) each
# grammar's plain-Lark parser accepts outright. Measured 2026-09-20.
OURS_GOLD_COVERAGE_BEFORE_FIX = 726
OURS_GOLD_COVERAGE_AFTER_FIX = 915
ITERGEN_GOLD_COVERAGE = 1026  # unchanged: the authors' grammar is not edited.


@pytest.fixture(scope="module")
def spider_gold_queries() -> list[str]:
    return _spider_gold_queries()


@pytest.fixture(scope="module")
def ours_dafny_parser():
    return _make_parser(OURS_GRAMMAR_PATH, dfa_mode="grammar_mask")


@pytest.fixture(scope="module")
def ours_plain_lark() -> Lark:
    return _plain_lark_full_parser(OURS_GRAMMAR_PATH)


@pytest.fixture(scope="module")
def itergen_dafny_parser():
    return _make_parser(ITERGEN_GRAMMAR_PATH)


@pytest.fixture(scope="module")
def itergen_plain_lark() -> Lark:
    return _plain_lark_full_parser(ITERGEN_GRAMMAR_PATH)


# ---------------------------------------------------------------------------
# Our grammar (also used, unmodified, by the CARS, GCD, and CRANE Spider lanes).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold", OURS_DOUBLE_QUOTED_GOLDS)
def test_ours_double_quoted_gold_parses_completely(ours_dafny_parser, gold):
    assert ours_dafny_parser._is_valid_prefix(gold) is True
    assert ours_dafny_parser._is_complete(gold) is True


@pytest.mark.parametrize("gold", OURS_SINGLE_QUOTED_GOLDS)
def test_ours_single_quoted_gold_still_parses_completely(ours_dafny_parser, gold):
    assert ours_dafny_parser._is_valid_prefix(gold) is True
    assert ours_dafny_parser._is_complete(gold) is True


def test_ours_unterminated_double_quote_is_a_valid_prefix_but_not_complete(ours_dafny_parser):
    text = 'SELECT Country FROM AIRLINES WHERE Airline  =  "JetBlue'
    assert ours_dafny_parser._is_valid_prefix(text) is True
    assert ours_dafny_parser._is_complete(text) is False


def test_ours_gold_coverage_did_not_regress_and_improved(ours_plain_lark, spider_gold_queries):
    covered = _count_covered(ours_plain_lark, spider_gold_queries)
    assert covered >= OURS_GOLD_COVERAGE_BEFORE_FIX, (
        f"ours(sql.lark) now covers {covered} gold queries, fewer than the "
        f"pre-fix {OURS_GOLD_COVERAGE_BEFORE_FIX} -- coverage regressed"
    )
    assert covered == OURS_GOLD_COVERAGE_AFTER_FIX, (
        f"ours(sql.lark) covers {covered} gold queries, expected exactly "
        f"{OURS_GOLD_COVERAGE_AFTER_FIX} after the double-quote fix"
    )


# ---------------------------------------------------------------------------
# IterGen authors' own grammar: pinned to show it is NOT edited, because it
# already accepts double-quoted text (see module docstring).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold", ITERGEN_DOUBLE_QUOTED_GOLDS)
def test_itergen_authors_grammar_already_parses_double_quoted_gold(itergen_dafny_parser, gold):
    assert itergen_dafny_parser._is_valid_prefix(gold) is True
    assert itergen_dafny_parser._is_complete(gold) is True


@pytest.mark.parametrize("gold", ITERGEN_SINGLE_QUOTED_GOLDS)
def test_itergen_authors_grammar_single_quoted_gold_still_parses_completely(itergen_dafny_parser, gold):
    assert itergen_dafny_parser._is_valid_prefix(gold) is True
    assert itergen_dafny_parser._is_complete(gold) is True


def test_itergen_authors_grammar_unterminated_double_quote_is_a_valid_prefix_but_not_complete(itergen_dafny_parser):
    text = 'SELECT Country FROM AIRLINES WHERE Airline  =  "JetBlue'
    assert itergen_dafny_parser._is_valid_prefix(text) is True
    assert itergen_dafny_parser._is_complete(text) is False


def test_itergen_authors_grammar_coverage_is_unchanged(itergen_plain_lark, spider_gold_queries):
    covered = _count_covered(itergen_plain_lark, spider_gold_queries)
    assert covered >= ITERGEN_GOLD_COVERAGE, (
        f"itergen_authors/sql_grammar.lark now covers {covered} gold queries, "
        f"fewer than the {ITERGEN_GOLD_COVERAGE} measured before this change -- "
        "coverage regressed"
    )
    assert covered == ITERGEN_GOLD_COVERAGE, (
        f"itergen_authors/sql_grammar.lark covers {covered} gold queries, "
        f"expected exactly {ITERGEN_GOLD_COVERAGE} (this file is not edited)"
    )
