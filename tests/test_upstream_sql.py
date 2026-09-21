from pathlib import Path
import hashlib
import inspect
from lark import Lark

ROOT = Path(__file__).resolve().parents[1]

def test_sql_is_exact_pinned_upstream():
    content = (ROOT / 'synthesis/evaluate/grammars/sql.lark').read_bytes()
    assert hashlib.sha256(content).hexdigest() == '405cb737863b63bb0fb8d832268443bcb5a89089f6691406b37d95e052a1a2ac'

def test_sql_feature_and_boundary_examples():
    import pytest
    parser = Lark((ROOT / 'synthesis/evaluate/grammars/sql.lark').read_text(), parser='lalr', lexer='basic')
    # Upstream's basic lexer treats int as CNAME, not TYPENAME; exact adoption
    # must preserve that limitation rather than silently fixing the grammar.
    for text in ['SELECT COALESCE(a,b) FROM t', 'SELECT "employee name" FROM t;', 'SELECT a FROM t -- comment\n']:
        parser.parse(text)
    with pytest.raises(Exception):
        parser.parse('SELECT * FROMurator')
    with pytest.raises(Exception):
        parser.parse('SELECT CAST(a AS int) FROM t')

def test_sql_runtime_uses_existing_upstream_start_not_custom_rule():
    from synthesis.evaluate.benchmarks.sql_spider.environment import setup_dafny_environment
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import setup_dafny_environment as gsm_setup
    assert inspect.signature(setup_dafny_environment).parameters['parser_start'].default == 'start'
    assert inspect.signature(gsm_setup).parameters['parser_start'].default == 'csd_start'

def test_scoring_uses_same_lexer_as_generation():
    from types import SimpleNamespace
    from synthesis.evaluate.benchmarks.sql_spider.eval_logic import get_syntax_parser
    grammar = (ROOT / 'synthesis/evaluate/grammars/sql.lark').read_text()
    parser = get_syntax_parser(SimpleNamespace(_get_grammar_text=lambda: grammar), None)
    assert parser.options.lexer == 'basic'
