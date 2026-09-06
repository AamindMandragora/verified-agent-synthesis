from pathlib import Path

from scripts.runtime import run_table5_8_queue as queue
from synthesis.scripts import eval_worker_pool


def test_two_synthesis_workers_fit_only_with_fraction_based_reservation():
    row = queue.build_scope(Path('/repo'))[0]
    row['gpu_scope'] = [3]
    assert row['synthesis_workers_per_gpu'] == 2
    assert row['synthesis_worker_overhead_mib'] == 2048
    assert queue._demand(row, 40442) == 36450
    for free, expected in [(37949, None), (37950, 3), (40442, 3)]:
        snapshot = {3: {'total_mib': 40442, 'free_mib': free}}
        assert queue.choose_gpu(row, snapshot, {}, snapshot, (3,)) == expected
        assert queue.choose_gpus(row, snapshot, {}, snapshot, (3,)) == ((3,) if expected else None)
    assert queue.choose_gpus(row, snapshot, {3: 36450}, snapshot, (3,)) is None


def test_two_20480_reservations_would_not_fit_the_physical_card():
    row = queue.build_scope(Path('/repo'))[0]
    assert 2 * row['memory_reservation_mib'] + queue.GPU_SAFETY_MIB > 40442


def test_only_synthesis_sets_two_worker_gpu_slots(tmp_path, monkeypatch):
    row = queue.build_scope(Path('/repo'))[0]
    synthesis = queue.synthesis_environment(row, (3,), {}, tmp_path)
    heldout = queue.heldout_environment(row, (3,), {}, tmp_path)

    assert synthesis['CSD_EVAL_GPU_SLOTS'] == '3,3'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', synthesis['CUDA_VISIBLE_DEVICES'])
    monkeypatch.setenv('CSD_EVAL_GPU_SLOTS', synthesis['CSD_EVAL_GPU_SLOTS'])
    assert eval_worker_pool._queue_gpu_slots() == [3, 3]
    assert 'CSD_EVAL_GPU_SLOTS' not in heldout
