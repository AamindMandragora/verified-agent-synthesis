import json
from types import SimpleNamespace
from lark import Lark
from scripts.analysis.upstream_sql.rescore import rescore_file

def test_rescore_preserves_source_and_rejects_ambiguous_alignment(tmp_path):
    source = tmp_path / 'output.json'
    payload = {'accuracy': 1.0, 'syntax_rate': 1.0, 'answers': [
        {'question': 'q', 'generated_answer': 'SELECT a FROM t'},
        {'question': 'ambiguous', 'generated_answer': 'SELECT a FROM t'}]}
    source.write_text(json.dumps(payload))
    original = source.read_bytes()
    parser = Lark('start: "SELECT a FROM t"', parser='lalr')
    examples = {'q': [{'question': 'q', 'query': 'SELECT a FROM t', 'db_id': 'x'}],
                'ambiguous': [{'db_id': 'a'}, {'db_id': 'b'}]}
    result = rescore_file(source, parser, parser, examples, lambda sql, ex: True)
    assert source.read_bytes() == original
    assert result['rows'][0]['new_correct'] is True
    assert result['rows'][1]['alignment'] == 'ambiguous'
    assert result['new_accuracy'] is None
    assert result['new_syntax_rate'] == 1.0
    assert result['stored_accuracy'] == 1.0

def test_newly_accepted_output_gets_execution_check(tmp_path):
    source = tmp_path / 'output.json'
    source.write_text(json.dumps({'answers': [{'question': 'q', 'generated_answer': 'SELECT b FROM t'}]}))
    old = Lark('start: "SELECT a FROM t"', parser='lalr')
    new = Lark('start: "SELECT b FROM t"', parser='lalr')
    calls = []
    def execute(sql, example):
        calls.append(sql)
        return False
    result = rescore_file(source, old, new, {'q': [{'db_id': 'x'}]}, execute)
    assert result['old_syntax_rate'] == 0
    assert result['new_syntax_rate'] == 1
    assert result['new_accuracy'] == 0
    assert calls == ['SELECT b FROM t']

def test_question_alignment_normalizes_typographic_apostrophe(tmp_path):
    from scripts.analysis.upstream_sql.rescore import question_key
    assert question_key('people’s  age') == question_key("people's age")

def test_raw_itergen_keeps_unknown_accuracy_without_question_alignment(tmp_path):
    source = tmp_path / 'raw.jsonl'
    source.write_text(json.dumps({'completion': 'SELECT a FROM t', 'exec': True}) + '\n')
    parser = Lark('start: "SELECT a FROM t"', parser='lalr')
    result = rescore_file(source, parser, parser, {}, lambda sql, ex: True)
    assert result['new_syntax_rate'] == 1
    assert result['new_accuracy'] is None
