import ast
import inspect
import json

from synthesis.evaluate import run_legacy_fixed_strategy as runner


def test_duplicate_correct_molecule_counts_once_per_class():
    seen = {}
    score = runner._unique_smiles_correct
    assert score('acrylates', 'CCO', True, True, seen)
    assert not score('acrylates', 'CCO', True, True, seen)
    assert score('chain_extenders', 'CCO', True, True, seen)
    assert not score('acrylates', 'CCC', False, True, seen)
    assert not score('acrylates', '', True, True, seen)
    assert not score('acrylates', 'broken', True, False, seen)


def test_fresh_run_does_not_reuse_seen_molecules():
    assert runner._unique_smiles_correct('a', 'CCO', True, True, {})
    assert runner._unique_smiles_correct('a', 'CCO', True, True, {})


def test_both_adapters_apply_rule_only_to_smiles():
    for adapter in (runner.run_gcd_legacy_adapter, runner._crane_via_adaptive_syncode):
        tree = ast.parse(inspect.getsource(adapter))
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "dataset == 'smiles'"
                   and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id == '_unique_smiles_correct' for call in ast.walk(node))]
        assert guarded, adapter.__name__


def test_saved_rows_keep_syntax_and_denominator(tmp_path):
    seen = {}
    rows = [dict(question='a', llm_response='CCO', syntax_valid=True,
                 correct=runner._unique_smiles_correct('a', 'CCO', True, True, seen),
                 generation_seconds=2) for _ in range(2)]
    target = tmp_path / 'result.json'
    runner._build_minimal_json(rows, target, run_wall_time_seconds=20)
    result = json.loads(target.read_text())
    assert result['accuracy'] == 0.5
    assert result['syntax_rate'] == 1.0
    assert result['metrics']['num_examples'] == 2
    assert result['metrics']['total_generation_seconds'] == 4
    assert result['metrics']['run_wall_time_seconds'] == 20
    definitions = result['metrics']['timing_definitions']
    assert 'retained' in definitions['total_generation_seconds']
    assert 'rejected' in definitions['run_wall_time_seconds']
