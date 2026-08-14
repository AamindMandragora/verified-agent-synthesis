"""Persistent data-parallel vLLM worker pool for per-iteration synthesis eval.

Problem this solves: the synthesis loop's per-iteration Evaluator.evaluate_sample
call re-evaluates the same compiled CSD on a fresh sample every iteration, and a
vLLM engine takes ~24s to load (measured 2026-07-18, Qwen2.5-1.5B-Instruct,
gpu_memory_utilization=0.4, enforce_eager=True, idle A100 -- see
saved-results/... for the raw log). A subprocess-per-call sharding approach
(synthesis/scripts/sharded_eval_core.py) pays that reload cost every iteration.
This module instead spawns one worker subprocess per idle GPU slot ONCE, at
the start of a synthesis run, each loading its engine ONCE and then serving
every iteration's eval requests for the rest of the run -- keep-B-only, this
pool is now THE eval path for GSM and Spider (see evaluator.py
POOLABLE_DATASETS). SMILES keeps its original single-process path unchanged:
it carries rolling-prompt state between examples that a data-parallel split
cannot reproduce (see evaluator.py's POOLABLE_DATASETS docstring).

IPC: each worker is a `synthesis.scripts.eval_worker_main` subprocess reading
length-prefixed pickle messages from stdin (requests) and writing them to a
DEDICATED response pipe fd, not stdout (pickle, not JSON, because dataset
examples are arbitrary Python objects -- GSM examples are a dataclass, Spider
examples are dicts). Responses use a separate fd, passed to the child via
`pass_fds` and an env var, because vLLM writes plain, unstructured text to
stdout during engine load that would otherwise corrupt the framed protocol
(observed directly: a log line got read as an 8-byte length header, producing
a garbage length and a MemoryError). Requests stay on stdin since nothing
else writes to it. Requests/responses are one round-trip per iteration per
worker; the worker's Evaluator and vLLM engine live for the whole synthesis
run.

Fault handling: if a worker dies or errors mid-request, its shard's examples
are redistributed across the surviving workers and retried once. Before the
next call runs, evaluate_examples tries to respawn any dead worker on the
same GPU with the same configuration, so a single dead worker does not
permanently shrink the pool. Only if every worker is dead AND respawning
every one of them also fails does the pool fall back to evaluating the whole
batch in-process (single-threaded, no subprocess) via
Evaluator._evaluate_one_example directly -- slower, but never crashes the
synthesis run.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from synthesis.scripts.sharded_eval_core import (
    detect_gpu_slots,
    plan_shards,
    visible_physical_gpu_ids,
)

LOG = "[sharded-eval]"

# Hard cap on pool size regardless of how many GPU slots look idle -- focal
# has 4 A100s; leave one slot of headroom for the generator model / other
# jobs sharing the box. Raise this only if you also re-check the box's real
# GPU count.
MAX_POOL_WORKERS = 4

# Config knobs for GPU-slot detection, matching sharded_eval_core's defaults
# (same "idle" definition used by the standalone reevaluate_sharded CLI).
_WORKERS_PER_GPU = 1
_IDLE_UTIL_THRESHOLD = 30
_MIN_FREE_MB = 8000

_HEADER = struct.Struct(">Q")


def _queue_gpu_slots() -> list[int] | None:
    """Return the queue-owned physical GPU bundle without a global GPU scan."""
    raw = os.environ.get("CSD_EVAL_GPU_SLOTS", "").strip()
    if not raw:
        return None
    try:
        slots = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise RuntimeError(
            f"CSD_EVAL_GPU_SLOTS must contain numeric physical GPU IDs: {raw!r}"
        ) from exc
    if not slots or any(gpu < 0 for gpu in slots):
        raise RuntimeError(f"invalid CSD_EVAL_GPU_SLOTS bundle: {raw!r}")
    if len(slots) > MAX_POOL_WORKERS:
        raise RuntimeError(
            f"CSD_EVAL_GPU_SLOTS requests {len(slots)} workers; cap is {MAX_POOL_WORKERS}"
        )
    visible = visible_physical_gpu_ids()
    if visible is not None and any(gpu not in visible for gpu in slots):
        raise RuntimeError(
            f"queue GPU bundle {slots} is outside CUDA_VISIBLE_DEVICES={visible}"
        )
    return slots


def send_msg(stream, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def recv_msg(stream) -> Optional[Any]:
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    data = _read_exact(stream, length)
    if data is None:
        return None
    return pickle.loads(data)


def _read_exact(stream, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _evaluator_worker_config(evaluator) -> dict:
    """The subset of Evaluator constructor kwargs a worker needs to build an
    identically-configured Evaluator of its own. Keep in sync with
    Evaluator.__init__ -- deliberately explicit (no **vars(evaluator)) so a
    new Evaluator attribute doesn't silently leak into worker config."""
    return dict(
        dataset_name=evaluator.dataset_name,
        model_name=evaluator.model_name,
        backend=evaluator.backend,
        device=evaluator.device,
        sample_size=evaluator.sample_size,
        max_steps=evaluator.max_steps,
        load_in_4bit=evaluator.load_in_4bit,
        load_in_8bit=evaluator.load_in_8bit,
        vllm_tensor_parallel_size=evaluator.vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=evaluator.vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=evaluator.vllm_gpu_memory_utilization,
        vllm_max_model_len=evaluator.vllm_max_model_len,
        vllm_enforce_eager=evaluator.vllm_enforce_eager,
        sample_seed=evaluator.sample_seed,
        max_seconds_per_example=evaluator.max_seconds_per_example,
        step_token_budget=evaluator.step_token_budget,
        gsm_source_dir=evaluator.gsm_source_dir,
        gsm_split_file=evaluator.gsm_split_file,
        gsm_split_name=evaluator.gsm_split_name,
        spider_split_file=evaluator.spider_split_file,
        spider_split_name=evaluator.spider_split_name,
        smiles_classes=evaluator.smiles_classes,
        grammars_dir=evaluator.grammars_dir,
        early_stop_on_answer=evaluator.early_stop_on_answer,
    )


class _Worker:
    def __init__(self, worker_id: int, gpu: int):
        self.worker_id = worker_id
        self.gpu = gpu
        self.alive = True
        self.first_request_done = False
        # Responses come back over a DEDICATED pipe fd, not stdout. vLLM (and
        # its own child EngineCore process) writes plain, unstructured text to
        # stdout during engine load (e.g. "Loading safetensors checkpoint
        # shards: ..."), which corrupted the framed pickle protocol when it
        # shared the channel with our replies -- observed directly: recv_msg
        # read one of those log lines as if it were an 8-byte length header,
        # producing a garbage huge length and a MemoryError on the next read.
        # Requests stay on stdin: nothing else writes to the worker's stdin,
        # so that channel was never at risk.
        resp_r, resp_w = os.pipe()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CSD_EVAL_RESP_FD": str(resp_w)}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "synthesis.scripts.eval_worker_main"],
            stdin=subprocess.PIPE,
            stdout=None,  # inherit -- worker prints/vLLM logs go straight to the run's console/log, unused for IPC
            stderr=None,  # inherit -- same
            env=env,
            pass_fds=(resp_w,),
        )
        os.close(resp_w)  # parent's copy; the child has its own after fork+exec
        self.resp_stream = os.fdopen(resp_r, "rb")

    def configure(self, config: dict) -> None:
        send_msg(self.proc.stdin, {"cmd": "configure", "config": config})
        reply = recv_msg(self.resp_stream)
        if reply is None or not reply.get("ok"):
            raise RuntimeError(f"worker {self.worker_id} failed to configure: {reply}")

    def evaluate(self, compiled_module_path: str, examples: list, start_index: int, dataset_len: int) -> list:
        t0 = time.time()
        send_msg(
            self.proc.stdin,
            {
                "cmd": "evaluate",
                "compiled_module_path": compiled_module_path,
                "examples": examples,
                "start_index": start_index,
                "dataset_len": dataset_len,
            },
        )
        reply = recv_msg(self.resp_stream)
        if reply is None:
            self.alive = False
            raise RuntimeError(f"worker {self.worker_id} died (no reply, pipe closed)")
        if not reply.get("ok"):
            raise RuntimeError(f"worker {self.worker_id} error: {reply.get('error')}")
        elapsed = time.time() - t0
        if not self.first_request_done:
            self.first_request_done = True
            print(
                f"{LOG} worker {self.worker_id} (GPU {self.gpu}) first request done in "
                f"{elapsed:.1f}s (includes one-time engine load)",
                flush=True,
            )
        else:
            print(f"{LOG} worker {self.worker_id} (GPU {self.gpu}) request done in {elapsed:.1f}s", flush=True)
        return reply["results"]

    def shutdown(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            send_msg(self.proc.stdin, {"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        try:
            self.resp_stream.close()
        except Exception:
            pass


class EvalWorkerPool:
    """One pool per synthesis run. Workers are spawned once (see
    get_synthesis_eval_pool) and reused for every subsequent evaluate_examples
    call for the rest of the run's lifetime."""

    def __init__(self, config: dict):
        # Keep the run's config around so a worker that dies mid-run can be
        # replaced with one configured exactly the same way (same model,
        # same everything) -- see _respawn_dead_workers.
        self.config = config
        queue_slots = _queue_gpu_slots()
        gpu_slots = queue_slots or detect_gpu_slots(
            _WORKERS_PER_GPU, _IDLE_UTIL_THRESHOLD, _MIN_FREE_MB
        )
        visible_gpus = visible_physical_gpu_ids()
        fallback_gpus = visible_gpus if visible_gpus is not None else [0]
        if not fallback_gpus:
            raise RuntimeError("no CUDA devices are visible to the evaluation pool")
        # Test-only override: pin an exact worker count instead of reading
        # live nvidia-smi utilization, so the identity test (pool of 1 vs 2
        # vs 3) is reproducible regardless of what else is running on the
        # box. Production runs never set this.
        size_override = os.environ.get("CSD_EVAL_POOL_SIZE")
        if queue_slots is not None:
            n_workers = len(gpu_slots)
        elif size_override:
            n_workers = int(size_override)
            gpu_slots = (gpu_slots or fallback_gpus)[:n_workers]
        else:
            n_workers = min(len(gpu_slots), MAX_POOL_WORKERS) or 1
            gpu_slots = gpu_slots[:n_workers] if gpu_slots else [fallback_gpus[0]]
        print(
            f"{LOG} pool startup: {len(gpu_slots)} worker(s) on GPU(s) {gpu_slots} "
            f"(idle slots found: {len(gpu_slots)}, cap {MAX_POOL_WORKERS})",
            flush=True,
        )
        self.workers: list[_Worker] = []
        for i, gpu in enumerate(gpu_slots):
            worker = _Worker(i, gpu)
            worker.configure(config)
            self.workers.append(worker)
        atexit.register(self.shutdown)

    def _alive_workers(self) -> list[_Worker]:
        return [w for w in self.workers if w.alive]

    def _respawn_dead_workers(self) -> None:
        """Replace any dead worker with a fresh one on the same worker id and
        GPU, configured the same way as the original. This runs once per
        evaluate_examples call -- if a replacement dies again immediately, we
        do not retry it again in this same call, only on the next one, so a
        worker that can never come up cannot spin us in a loop here."""
        for i, worker in enumerate(self.workers):
            if worker.alive:
                continue
            print(
                f"{LOG} worker {worker.worker_id} (GPU {worker.gpu}) is dead; "
                "attempting respawn",
                flush=True,
            )
            try:
                replacement = _Worker(worker.worker_id, worker.gpu)
                replacement.configure(self.config)
            except Exception as exc:
                print(
                    f"{LOG} respawn of worker {worker.worker_id} (GPU {worker.gpu}) "
                    f"failed: {exc!r}",
                    flush=True,
                )
                continue
            print(
                f"{LOG} worker {worker.worker_id} (GPU {worker.gpu}) respawned",
                flush=True,
            )
            self.workers[i] = replacement

    def evaluate_examples(self, evaluator, compiled_module_path: Path, dataset: list) -> list[dict]:
        """Evaluate every example in `dataset` (no early stop -- see
        evaluator.py's _posthoc_early_stop for why that's safe) split across
        the pool's surviving workers, and return results in original order."""
        self._respawn_dead_workers()
        alive = self._alive_workers()
        if not alive:
            print(f"{LOG} no surviving workers; falling back to in-process eval", flush=True)
            return _evaluate_in_process(evaluator, compiled_module_path, dataset)

        return self._dispatch(evaluator, compiled_module_path, list(enumerate(dataset)), alive)

    def _dispatch(
        self, evaluator, compiled_module_path: Path, indexed_examples: list[tuple[int, Any]], workers: list[_Worker]
    ) -> list[dict]:
        n = len(indexed_examples)
        dataset_len = n  # for pathological-guard message parity; posthoc re-derives the true total anyway
        shard_bounds = plan_shards(n, len(workers))
        print(
            f"{LOG} dispatch: {n} example(s) across {len(shard_bounds)} worker(s), "
            f"shard sizes {[hi - lo for lo, hi in shard_bounds]}",
            flush=True,
        )
        results: list[Optional[dict]] = [None] * n
        # Position within THIS call's results array for a given original
        # dataset index -- not the same as the original index once this is a
        # retry over a redistributed subset of failed examples.
        position_of = {idx: pos for pos, (idx, _) in enumerate(indexed_examples)}
        failed_indices: list[int] = []
        shard_calls = []
        for worker, (lo, hi) in zip(workers, shard_bounds):
            shard = indexed_examples[lo:hi]
            if not shard:
                continue
            local_indices = [idx for idx, _ in shard]
            examples = [ex for _, ex in shard]
            shard_calls.append((worker, local_indices, examples, lo))
        if shard_calls:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(shard_calls)
            ) as executor:
                future_to_shard = {
                    executor.submit(
                        worker.evaluate,
                        str(compiled_module_path),
                        examples,
                        lo,
                        dataset_len,
                    ): (worker, local_indices)
                    for worker, local_indices, examples, lo in shard_calls
                }
                print(
                    f"{LOG} parallel dispatch: {len(future_to_shard)} worker shard(s) started",
                    flush=True,
                )
                for future in concurrent.futures.as_completed(future_to_shard):
                    worker, local_indices = future_to_shard[future]
                    try:
                        shard_results = future.result()
                        for idx, result in zip(local_indices, shard_results):
                            results[position_of[idx]] = result
                    except Exception as exc:
                        print(
                            f"{LOG} worker {worker.worker_id} FAILED mid-shard: {exc!r}",
                            flush=True,
                        )
                        worker.alive = False
                        failed_indices.extend(local_indices)

        if failed_indices:
            survivors = self._alive_workers()
            index_to_example = {idx: ex for idx, ex in indexed_examples}
            failed_pairs = [(idx, index_to_example[idx]) for idx in failed_indices]
            print(
                f"{LOG} redistributing {len(failed_pairs)} example(s) from dead worker(s) "
                f"to {len(survivors)} surviving worker(s)",
                flush=True,
            )
            if survivors:
                retried = self._dispatch(evaluator, compiled_module_path, failed_pairs, survivors)
            else:
                print(f"{LOG} all workers dead; falling back to in-process eval for the remainder", flush=True)
                retried = _evaluate_in_process(evaluator, compiled_module_path, [ex for _, ex in failed_pairs])
            for (idx, _), r in zip(failed_pairs, retried):
                results[position_of[idx]] = r

        assert all(r is not None for r in results), "merge produced a gap -- every index must be filled"
        print(f"{LOG} merge: {len(results)} example(s) reassembled in original order", flush=True)
        return results  # type: ignore[return-value]

    def shutdown(self) -> None:
        for w in self.workers:
            w.shutdown()


def _evaluate_in_process(evaluator, compiled_module_path: Path, examples: list) -> list[dict]:
    """Disaster fallback: no surviving workers, evaluate this batch directly
    in the calling process (no subprocess, no parallelism)."""
    env = evaluator._setup_environment(Path(compiled_module_path))
    logic = evaluator._benchmark_logic()
    run_crane_csd = logic.get_generation_runner()
    smiles_suffix: dict = {}
    return [
        evaluator._evaluate_one_example(i, ex, len(examples), env, logic, run_crane_csd, smiles_suffix)
        for i, ex in enumerate(examples)
    ]


# One pool per (dataset_name, model_name, backend) config for the life of the
# process -- a synthesis run only ever evaluates one dataset/model, so this is
# a run-scoped singleton in practice, keyed defensively in case a process ever
# evaluates more than one config (e.g. tests).
_POOLS: dict[tuple, EvalWorkerPool] = {}


def get_synthesis_eval_pool(evaluator) -> EvalWorkerPool:
    config = _evaluator_worker_config(evaluator)
    key = tuple(sorted((k, str(v)) for k, v in config.items()))
    pool = _POOLS.get(key)
    if pool is None:
        pool = EvalWorkerPool(config)
        _POOLS[key] = pool
    return pool
