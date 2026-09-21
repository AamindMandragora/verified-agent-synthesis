from pathlib import Path
import hashlib
import json
import pytest
from scripts.runtime import run_table5_8_queue as queue


def test_remaining_queue_evidence_is_local():
    root = Path(queue.__file__).resolve().parents[2] / 'outputs/paper-evidence-20260921'
    assert Path(queue.BAR_BINDINGS['spider']['source_path']).parent == root / 'spider-contract'
    for row in queue.IMPORTED_GPT_BASE['artifacts'].values():
        assert Path(row['source_path']).parent == root / 'historical-gpt'


def test_generated_sql_manifest_only_replaces_three_verified_inputs():
    root = Path(queue.__file__).resolve().parents[2] / 'outputs/paper-evidence-20260921/remaining-inputs'
    if not root.exists():
        pytest.skip('Historical artifact integration requires the preserved focal evidence')
    before = json.loads((root / 'original-all-inputs.json').read_text())
    after = json.loads((root / 'patched-all-inputs.json').read_text())
    records = json.loads((root / 'preservation.json').read_text())
    assert len(before) == len(after) == 114
    changed = [(a,b) for a,b in zip(before,after) if a != b]
    assert len(changed) == 3
    for old,new in changed:
        record = next(row for row in records if row['source'] == old)
        assert new == record['destination']
        assert hashlib.sha256(Path(new).read_bytes()).hexdigest() == record['before_sha256']
