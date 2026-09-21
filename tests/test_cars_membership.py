"""Original CARS class definitions must reach baseline and synthesis scoring."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from synthesis.evaluate.benchmarks.smiles import metrics, eval_logic

@pytest.mark.parametrize('class_name,smiles,expected', [
    ('acrylates', 'C=CC(=O)O', False),
    ('acrylates', 'C=CC(=O)OC', True),
    ('isocyanates', 'N=C=O', False),
    ('isocyanates', 'CN=C=O', True),
    ('chain_extenders', 'C[OH]', True),
    ('chain', 'C[OH]', True),
    ('chain_extenders', 'CC', False),
    ('acrylates', 'not a molecule', False),
])
def test_original_cars_membership(class_name, smiles, expected):
    assert metrics.target_class_membership(class_name, smiles) is expected

def test_missing_rdkit_is_an_error_not_a_different_score(monkeypatch):
    monkeypatch.setattr(metrics, 'Chem', None)
    with pytest.raises(RuntimeError, match='RDKit'):
        metrics.target_class_membership('acrylates', 'C=CC(=O)OC')

def test_unknown_class_fails_like_original_cars():
    with pytest.raises(ValueError, match='Invalid monomer class'):
        metrics.target_class_membership('unknown', 'CC')

@pytest.mark.parametrize('molecule,expected', [('C=CC(=O)O', False), ('C=CC(=O)OC', True)])
def test_synthesis_extract_and_verdict_use_cars(molecule, expected):
    from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
    example = dict(get_smiles_task('acrylates'))
    example.update(class_name='acrylates', prompt_exemplars=[])
    actual, _, aux = eval_logic.extract_actual(None, f'<<{molecule}>>', example)
    assert aux['class_membership'] is expected
    assert eval_logic.is_correct(None, actual, '', example, aux, molecule) is expected

def test_spider_synthesis_checker_uses_upstream_grammar():
    import hashlib
    from synthesis.evaluate.evaluator import Evaluator
    evaluator = object.__new__(Evaluator)
    evaluator.dataset_name = 'spider'
    evaluator.grammars_dir = None
    evaluator._grammar_file = None
    evaluator._base_grammar_text = None
    evaluator._syntax_parser = None
    grammar = evaluator._get_grammar_text()
    assert hashlib.sha256(grammar.encode()).hexdigest() == '405cb737863b63bb0fb8d832268443bcb5a89089f6691406b37d95e052a1a2ac'
    from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql
    parser = sql.get_syntax_parser(evaluator, None)
    assert parser.options.lexer == 'basic'
    parser.parse('SELECT COALESCE(a,b) FROM t')
    actual, _, aux = sql.extract_actual(evaluator, '<<SELECT COALESCE(a,b) FROM t>>', {})
    assert actual == 'SELECT COALESCE(a,b) FROM t'
    assert sql.example_syntax_pass(True, [], aux)
    _, _, aux = sql.extract_actual(evaluator, '<<SELECT * FROMurator>>', {})
    assert not sql.example_syntax_pass(True, [], aux)
