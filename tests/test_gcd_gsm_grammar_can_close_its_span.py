"""GSM GCD's prompt ends with "<<", so the grammar it decodes under must let the
model write the closing ">>". SynCode parses from the rule named ``start``."""
from pathlib import Path

import lark
import pytest

from synthesis.evaluate import run_legacy_fixed_strategy as legacy

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = [{"variable_types": {"x": "int", "k": "int"}}]


def _parser() -> lark.Lark:
    base = legacy._legacy_gsm_symbolic_grammar_base(REPO, EXAMPLES)
    return lark.Lark(legacy._gcd_gsm_grammar(base), start="start", parser="lalr")


def test_closed_span_is_accepted():
    _parser().parse("x * k / 100>>")


def test_unclosed_expression_is_not_a_complete_answer():
    with pytest.raises(lark.exceptions.LarkError):
        _parser().parse("x * k / 100")


def test_opening_marker_is_not_repeated():
    with pytest.raises(lark.exceptions.LarkError):
        _parser().parse("<<x * k>>")
