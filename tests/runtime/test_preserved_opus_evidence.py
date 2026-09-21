from pathlib import Path
from scripts.runtime import run_table5_8_queue as queue


def test_default_historical_evidence_survives_old_worktree_removal():
    root = Path(queue.__file__).resolve().parents[2] / 'outputs/paper-evidence-20260921/historical-opus'
    items = [queue.BAR_BINDINGS['gsm_symbolic'], queue.BAR_BINDINGS['smiles'],
             *queue.IMPORTED_OPUS_BASE['artifacts'].values()]
    assert len(items) == 6
    for item in items:
        assert Path(item['source_path']).parent == root
        assert len(item.get('sha256', item.get('source_sha256'))) == 64
