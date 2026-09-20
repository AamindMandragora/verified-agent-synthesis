"""The GCD baseline must be able to write the end of its answer.

SynCode parses from the rule named ``start`` (``create_base_parser`` hard-codes
it). ``grammars/sql.lark`` keeps the statement terminator in ``csd_start``, the
rule the CSD span parser uses, so GCD's view of the grammar was a query with no
way to finish: ";" was masked at every step, the lane's ";" stop word could
never fire, and $END was only ever reachable through the model's own end token,
which ranks far below carrying on. ``_gcd_gsm_grammar`` already makes the same
repair for GSM.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "synthesis/evaluate/syncode", REPO / "synthesis/evaluate/syncode/syncode"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synthesis.evaluate.run_legacy_fixed_strategy import _gcd_spider_grammar

COMPLETE_QUERY = " SELECT COUNT( * ) FROM singer"
INCOMPLETE_QUERY = " SELECT COUNT( * ) FROM"


@pytest.fixture(scope="module")
def gcd_spider_grammar() -> str:
    return _gcd_spider_grammar((REPO / "synthesis/evaluate/grammars/sql.lark").read_text())


def _parse(grammar_text: str, code: str):
    from syncode.parsers import create_parser
    from syncode.parsers.grammars.grammar import Grammar

    parser = create_parser(Grammar(grammar_text), parser="lalr", ignore_whitespace=True)
    parser.reset()
    return parser.get_acceptable_next_terminals(code)


def test_end_is_reachable_once_the_query_is_terminated(gcd_spider_grammar: str) -> None:
    assert _parse(gcd_spider_grammar, COMPLETE_QUERY + " ;").function_end is True


def test_end_is_not_reachable_while_the_query_is_incomplete(gcd_spider_grammar: str) -> None:
    assert _parse(gcd_spider_grammar, INCOMPLETE_QUERY).function_end is False
    assert _parse(gcd_spider_grammar, INCOMPLETE_QUERY + " ;").function_end is False
