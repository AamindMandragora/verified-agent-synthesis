"""Rescore saved Spider answers without modifying source artifacts or generating text."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from lark import Lark
from synthesis.evaluate.benchmarks.sql_spider.eval_logic import extract_actual

LOG = logging.getLogger('spider-grammar-rescore')

def question_key(text):
    # HF and local Spider render apostrophes differently. Never normalize SQL.
    return ' '.join(text.replace('\u2019', "'").replace('\u2018', "'").split())

def rescore_file(path, old_parser, new_parser, examples, execute):
    raw = path.read_bytes()
    if path.suffix == '.jsonl':
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        payload = {'answers': [dict(r, generated_answer=r['completion']) for r in records]}
    else:
        payload = json.loads(raw)
    body = payload.get('evaluation_result', payload)
    answers = body.get('answers', body.get('sample_outputs'))
    if not isinstance(answers, list) or not answers:
        raise ValueError('No nonempty saved answers/sample_outputs list')
    LOG.info('[sql-rescore] input=%s rows=%d', path, len(answers))
    evaluators = [SimpleNamespace(_get_syntax_parser=lambda ex, p=p: p) for p in (old_parser, new_parser)]
    rows = []
    for index, answer in enumerate(answers):
        text = answer.get('generated_answer', answer.get('full_output'))
        if not isinstance(text, str):
            raise ValueError(f'row {index} has no saved output text')
        matches = examples.get(question_key(answer.get('question', '')), [])
        if answer.get('db_id'):
            matches = [x for x in matches if x['db_id'] == answer['db_id']]
        aligned = len(matches) == 1
        ex = matches[0] if aligned else {}
        row = {'index': index, 'alignment': 'matched' if aligned else ('ambiguous' if matches else 'missing')}
        for label, ev in zip(('old', 'new'), evaluators):
            actual, _, aux = extract_actual(ev, text, ex)
            row[label + '_syntax'] = bool(aux['syntax_valid'])
            row[label + '_reason'] = aux.get('output_rejection_reason')
            row[label + '_correct'] = (bool(execute(actual, ex)) if actual else False) if aligned else None
        rows.append(row)
    if path.read_bytes() != raw:
        raise RuntimeError(f'Source changed during scoring: {path}')
    result = {'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'n': len(rows), 'stored_accuracy': body.get('accuracy'),
              'stored_syntax_rate': body.get('syntax_rate'), 'rows': rows}
    for label in ('old', 'new'):
        result[label + '_syntax_rate'] = sum(r[label + '_syntax'] for r in rows) / len(rows)
        correct = [r[label + '_correct'] for r in rows]
        result[label + '_accuracy'] = sum(correct) / len(correct) if all(x is not None for x in correct) else None
    LOG.info('[sql-rescore] output=%s old_syntax=%s new_syntax=%s', path, result['old_syntax_rate'], result['new_syntax_rate'])
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-grammar', type=Path, required=True)
    p.add_argument('--new-grammar', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True, help='JSON array of source file paths')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Exclusive create prevents overwriting an earlier derived score report.
    with a.output.open('x') as out:
        from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider
        from synthesis.evaluate.benchmarks.sql_spider.executor import prediction_matches_gold
        examples = defaultdict(list)
        for ex in load_spider(source='local'):
            examples[question_key(ex['question'])].append(ex)
        @lru_cache(maxsize=50000)
        def cached_execution(sql, question, db_id, gold):
            return prediction_matches_gold(sql, {'question': question, 'db_id': db_id, 'query': gold})
        def execute(sql, ex):
            return cached_execution(sql, ex['question'], ex['db_id'], ex['query'])
        old = Lark(a.old_grammar.read_text(), parser='lalr', lexer='contextual')
        new = Lark(a.new_grammar.read_text(), parser='lalr', lexer='basic')
        results = {'old_grammar_sha256': hashlib.sha256(a.old_grammar.read_bytes()).hexdigest(),
                   'new_grammar_sha256': hashlib.sha256(a.new_grammar.read_bytes()).hexdigest(),
                   'old_lexer': 'contextual', 'new_lexer': 'basic', 'files': [], 'errors': []}
        for name in json.loads(a.inputs.read_text()):
            try:
                results['files'].append(rescore_file(Path(name), old, new, examples, execute))
            except Exception as e:
                LOG.exception('[sql-rescore] failed input=%s', name)
                results['errors'].append({'source': name, 'error': str(e)})
        json.dump(results, out, indent=2)
        LOG.info('[sql-rescore] saved=%s files=%d errors=%d', a.output, len(results['files']), len(results['errors']))
    return bool(results['errors'])

if __name__ == '__main__':
    raise SystemExit(main())
