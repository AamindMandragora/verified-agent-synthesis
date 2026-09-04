from pathlib import Path

from scripts.runtime import run_table5_8_queue as queue


def test_shared_gpu_headroom_boundary():
    row = queue.build_scope(Path('/repo'))[0]
    row['gpu_scope'] = [3]
    for free, expected in [(21979, None), (21980, 3), (22041, 3)]:
        snapshot = {3: {'total_mib': 40442, 'free_mib': free}}
        assert queue.choose_gpu(row, snapshot, {}, snapshot, (3,)) == expected
        assert queue.choose_gpus(row, snapshot, {}, snapshot, (3,)) == ((3,) if expected else None)
    assert queue.choose_gpus(row, snapshot, {3: 20480}, snapshot, (3,)) is None
