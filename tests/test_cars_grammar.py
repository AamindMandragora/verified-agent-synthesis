"""CARS must compile the pinned SQL grammar without editing its source."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
import llguidance
from synthesis.evaluate import run_legacy_fixed_strategy as adapter

ROOT = Path(__file__).resolve().parents[1]

class Compiler:
    def __init__(self):
        self.inputs = []

    def _set_grammar_constraint(self, text):
        self.inputs.append(text)
        error = llguidance.LLMatcher.validate_grammar(llguidance.grammar_from('grammar', text))
        if error:
            raise ValueError(error)
        self.grammar_constraint = SimpleNamespace(text=text)

def test_pinned_sql_compiles_through_cars_loading_path():
    path = ROOT / 'synthesis/evaluate/grammars/sql.lark'
    source = path.read_text()
    model, cache = Compiler(), {}
    adapter._cars_set_cached_grammar(model, source, cache)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '405cb737863b63bb0fb8d832268443bcb5a89089f6691406b37d95e052a1a2ac'
    expected = source.replace('select_expr.0:', 'select_expr:').replace('JOIN_EXPR.5:', 'JOIN_EXPR:').replace('SELECT_CONSTRAINT.9:', 'SELECT_CONSTRAINT:').replace('AGGREGATION.8:', 'AGGREGATION:')
    assert model.inputs == [expected]
    original = model.grammar_constraint
    adapter._cars_set_cached_grammar(model, source, cache)
    assert model.grammar_constraint is original
    assert len(model.inputs) == 1

def test_priority_like_literal_comment_and_regex_unchanged():
    source = 'start.3: "item.5:" TOKEN\nTOKEN.-2: /name[.]7:/\n// other.8: unchanged\n'
    model, cache = Compiler(), {}
    adapter._cars_set_cached_grammar(model, source, cache)
    assert model.inputs == [source.replace('start.3:', 'start:').replace('TOKEN.-2:', 'TOKEN:')]

def test_no_priority_is_byte_identical():
    source = 'start: "hello"\n'
    model = Compiler()
    adapter._cars_set_cached_grammar(model, source, {})
    assert model.inputs == [source]

def test_other_grammar_errors_are_not_hidden_or_cached():
    cache = {}
    with pytest.raises(Exception):
        adapter._cars_set_cached_grammar(Compiler(), 'start: missing_rule\n', cache)
    assert cache == {}
