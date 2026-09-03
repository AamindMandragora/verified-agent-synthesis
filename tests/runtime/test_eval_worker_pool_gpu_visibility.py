import io
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from synthesis.scripts import eval_worker_pool
from synthesis.scripts import sharded_eval_core


def _four_idle_gpus(*_args, **_kwargs):
    return SimpleNamespace(
        stdout=(
            "0, 10, 40960, 0\n"
            "1, 10, 40960, 0\n"
            "2, 10, 40960, 0\n"
            "3, 10, 40960, 0\n"
        )
    )


def test_gpu_slot_detection_respects_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(sharded_eval_core.subprocess, "run", _four_idle_gpus)

    slots = sharded_eval_core.detect_gpu_slots(
        workers_per_gpu=1,
        idle_util_threshold=30,
        min_free_mb=8000,
    )

    assert slots == [3]


def test_worker_pool_fallback_stays_on_the_assigned_visible_gpu(monkeypatch):
    created_gpus = []

    class FakeWorker:
        def __init__(self, worker_id, gpu):
            created_gpus.append((worker_id, gpu))

        def configure(self, _config):
            pass

        def shutdown(self):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("CSD_EVAL_POOL_SIZE", raising=False)
    monkeypatch.setattr(eval_worker_pool, "detect_gpu_slots", lambda *_args: [])
    monkeypatch.setattr(eval_worker_pool, "_Worker", FakeWorker)

    pool = eval_worker_pool.EvalWorkerPool({})

    assert created_gpus == [(0, 2)]
    pool.shutdown()


def test_worker_pool_uses_explicit_queue_gpu_bundle_without_global_detection(monkeypatch):
    created_gpus = []

    class FakeWorker:
        def __init__(self, worker_id, gpu):
            created_gpus.append((worker_id, gpu))

        def configure(self, _config):
            pass

        def shutdown(self):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setenv("CSD_EVAL_GPU_SLOTS", "3,1")
    monkeypatch.delenv("CSD_EVAL_POOL_SIZE", raising=False)
    monkeypatch.setattr(
        eval_worker_pool,
        "detect_gpu_slots",
        lambda *_args: (_ for _ in ()).throw(AssertionError("global scan used")),
    )
    monkeypatch.setattr(eval_worker_pool, "_Worker", FakeWorker)

    pool = eval_worker_pool.EvalWorkerPool({})

    assert created_gpus == [(0, 3), (1, 1)]
    pool.shutdown()


def test_dispatch_starts_all_worker_shards_before_waiting_for_results():
    entered: list[int] = []
    entered_lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    class FakeWorker:
        alive = True

        def __init__(self, worker_id):
            self.worker_id = worker_id

        def evaluate(
            self,
            _module,
            examples,
            _start_index,
            _dataset_len,
            *,
            timeout_seconds,
        ):
            assert timeout_seconds == 7200.0
            with entered_lock:
                entered.append(self.worker_id)
                if len(entered) == 2:
                    both_started.set()
            assert release.wait(timeout=2)
            return [{"example": example} for example in examples]

    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    workers = [FakeWorker(0), FakeWorker(1)]
    outcome = {}

    def run_dispatch():
        try:
                outcome["results"] = pool._dispatch(
                    SimpleNamespace(max_seconds_per_example=0.0),
                "/tmp/compiled.py",
                list(enumerate(["a", "b", "c", "d"])),
                workers,
            )
        except BaseException as exc:
            outcome["error"] = exc

    dispatch_thread = threading.Thread(target=run_dispatch)
    dispatch_thread.start()
    try:
        assert both_started.wait(timeout=0.5), (
            "the second worker did not start while the first shard was running"
        )
    finally:
        release.set()
        dispatch_thread.join(timeout=3)

    assert not dispatch_thread.is_alive()
    assert "error" not in outcome
    assert outcome["results"] == [
        {"example": "a"},
        {"example": "b"},
        {"example": "c"},
        {"example": "d"},
    ]


def test_worker_request_timeout_stops_the_worker(monkeypatch):
    worker = object.__new__(eval_worker_pool._Worker)
    worker.worker_id = 7
    worker.gpu = 3
    worker.alive = True
    worker.first_request_done = False
    worker.proc = SimpleNamespace(stdin=object())
    worker.resp_stream = object()
    stopped = []

    monkeypatch.setattr(eval_worker_pool, "send_msg", lambda *_args: None)
    monkeypatch.setattr(
        eval_worker_pool,
        "wait_for_response",
        lambda *_args, **_kwargs: False,
        raising=False,
    )
    worker.abort = lambda: stopped.append(True)

    with pytest.raises(eval_worker_pool.WorkerRequestTimeout, match="worker 7.*0.01s"):
        worker.evaluate(
            "/tmp/compiled.py",
            ["example"],
            0,
            1,
            timeout_seconds=0.01,
        )

    assert stopped == [True]
    assert worker.alive is False


def test_worker_abort_terminates_only_its_process_group(monkeypatch):
    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("worker", timeout)

    class FakeResponse:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    worker = object.__new__(eval_worker_pool._Worker)
    worker.worker_id = 3
    worker.gpu = 3
    worker.alive = True
    worker.proc = FakeProcess()
    worker.resp_stream = FakeResponse()
    signals = []
    monkeypatch.setattr(
        eval_worker_pool.os,
        "killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )
    monkeypatch.setattr(eval_worker_pool, "WORKER_TERMINATION_GRACE_SECONDS", 0.0)

    worker.abort()

    assert (4321, signal.SIGTERM) in signals
    assert (4321, signal.SIGKILL) in signals
    assert worker.resp_stream.closed is True
    assert worker.alive is False


def test_worker_abort_kills_children_after_leader_exits_on_term():
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(\"ready\", flush=True); time.sleep(30)'], "
                "stdout=subprocess.PIPE, text=True); "
                "child.stdout.readline(); "
                "print(child.pid, flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    worker = object.__new__(eval_worker_pool._Worker)
    worker.worker_id = 3
    worker.gpu = 3
    worker.alive = True
    worker.proc = leader
    worker.resp_stream = leader.stdout

    try:
        worker.abort()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and os.path.exists(f"/proc/{child_pid}"):
            time.sleep(0.01)
        assert not os.path.exists(f"/proc/{child_pid}")
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=2)


def test_worker_request_returns_existing_result_before_deadline(monkeypatch):
    payload = pickle.dumps(
        {"ok": True, "results": [{"example": "ok"}]},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    response = io.BytesIO(struct.pack(">Q", len(payload)) + payload)
    worker = object.__new__(eval_worker_pool._Worker)
    worker.worker_id = 1
    worker.gpu = 3
    worker.alive = True
    worker.first_request_done = False
    worker.proc = SimpleNamespace(stdin=object())
    worker.resp_stream = response

    monkeypatch.setattr(eval_worker_pool, "send_msg", lambda *_args: None)
    monkeypatch.setattr(
        eval_worker_pool,
        "wait_for_response",
        lambda stream, _seconds: stream is response,
        raising=False,
    )

    assert worker.evaluate(
        "/tmp/compiled.py",
        ["example"],
        0,
        1,
        timeout_seconds=1.0,
    ) == [{"example": "ok"}]


def test_dispatch_sends_one_example_per_worker_request():
    class FakeWorker:
        alive = True

        def __init__(self, worker_id):
            self.worker_id = worker_id

        def evaluate(
            self,
            _module,
            examples,
            start_index,
            _dataset_len,
            *,
            timeout_seconds,
        ):
            assert len(examples) == 1
            assert timeout_seconds == 7200.0
            return [{"example": examples[0], "index": start_index}]

    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    evaluator = SimpleNamespace(max_seconds_per_example=2.0)
    results = pool._dispatch(
        evaluator,
        "/tmp/compiled.py",
        list(enumerate(["a", "b", "c", "d"])),
        [FakeWorker(0), FakeWorker(1)],
    )

    assert results == [
        {"example": "a", "index": 0},
        {"example": "b", "index": 1},
        {"example": "c", "index": 2},
        {"example": "d", "index": 3},
    ]


def test_dispatch_does_not_retry_a_hard_worker_timeout_in_process():
    class TimedOutWorker:
        alive = True
        worker_id = 0

        def evaluate(self, *_args, **_kwargs):
            raise eval_worker_pool.WorkerRequestTimeout("hard timeout")

        def abort(self):
            self.alive = False

    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    evaluator = SimpleNamespace(max_seconds_per_example=2.0)
    worker = TimedOutWorker()

    with pytest.raises(eval_worker_pool.WorkerRequestTimeout, match="hard timeout"):
        pool._dispatch(
            evaluator,
            "/tmp/compiled.py",
            [(0, "example")],
            [worker],
        )

    assert worker.alive is False


def test_dispatch_aborts_every_worker_and_returns_promptly_on_hard_timeout():
    released = threading.Event()

    class TimedOutWorker:
        alive = True
        worker_id = 0

        def evaluate(self, *_args, **_kwargs):
            raise eval_worker_pool.WorkerRequestTimeout("hard timeout")

        def abort(self):
            self.alive = False

    class BlockingWorker:
        alive = True
        worker_id = 1

        def evaluate(self, *_args, **_kwargs):
            released.wait(timeout=1.0)
            return [{"example": "late"}]

        def abort(self):
            self.alive = False
            released.set()

    workers = [TimedOutWorker(), BlockingWorker()]
    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    evaluator = SimpleNamespace(max_seconds_per_example=2.0)

    started = time.monotonic()
    with pytest.raises(eval_worker_pool.WorkerRequestTimeout, match="hard timeout"):
        pool._dispatch(
            evaluator,
            "/tmp/compiled.py",
            list(enumerate(["timeout", "blocked"])),
            workers,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert [worker.alive for worker in workers] == [False, False]


def test_dispatch_preserves_an_explicit_no_timeout_configuration():
    class FakeWorker:
        alive = True
        worker_id = 0

        def evaluate(self, *_args, timeout_seconds, **_kwargs):
            assert timeout_seconds is None
            return [{"example": "ok"}]

    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    results = pool._dispatch(
        SimpleNamespace(max_seconds_per_example=None),
        "/tmp/compiled.py",
        [(0, "example")],
        [FakeWorker()],
    )

    assert results == [{"example": "ok"}]


def test_failed_respawn_after_hard_timeout_never_falls_back_to_parent(monkeypatch):
    class TimedOutWorker:
        alive = True
        worker_id = 0
        gpu = 3

        def evaluate(self, *_args, **_kwargs):
            raise eval_worker_pool.WorkerRequestTimeout("hard timeout")

        def abort(self):
            self.alive = False

    worker = TimedOutWorker()
    pool = object.__new__(eval_worker_pool.EvalWorkerPool)
    pool.config = {}
    pool.workers = [worker]
    evaluator = SimpleNamespace(max_seconds_per_example=2.0)

    with pytest.raises(eval_worker_pool.WorkerRequestTimeout, match="hard timeout"):
        pool.evaluate_examples(evaluator, "/tmp/compiled.py", ["first"])

    monkeypatch.setattr(
        eval_worker_pool,
        "_Worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("respawn failed")),
    )
    parent_calls = []
    monkeypatch.setattr(
        eval_worker_pool,
        "_evaluate_in_process",
        lambda *_args, **_kwargs: parent_calls.append(True) or ["unsafe"],
    )

    with pytest.raises(
        eval_worker_pool.WorkerRequestTimeout,
        match="no clean evaluation worker could respawn",
    ):
        pool.evaluate_examples(evaluator, "/tmp/compiled.py", ["second"])

    assert parent_calls == []
