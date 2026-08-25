"""Shared shard/launch/merge core for data-parallel vLLM re-evaluation.

Extracted from synthesis.scripts.reevaluate_sharded so the shard-plan, GPU-slot
detection, and merge-math live in exactly one place (keep-B-only: no second
copy of this logic). Both the standalone re-eval CLI
(synthesis/scripts/reevaluate_sharded.py) and, where wired in, the synthesis
loop's per-iteration eval call into this module.

Mechanism (unchanged from the pre-extraction version, validated 40/40
byte-identical to a sequential run): split the example-index list into
contiguous shards, launch one `synthesis.scripts.reevaluate_compiled_csd`
subprocess per shard on its own GPU (staggered starts — concurrent vLLM
engine init races on memory profiling, 2026-07-17 mini test), wait for all
shards, then merge the per-shard answer lists and recompute accuracy /
syntax_rate over the combined set. Each example is still generated one at a
time inside its shard's subprocess (batch-of-1) with the same masking and
decoding as a sequential run — sharding only decides which GPU runs which
example.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOG = "[sharded-eval]"

# dataset -> split-name -> key in the split JSON holding example indices.
# One vocabulary per dataset, no aliases: Spider's held-out side is "test"
# (the "eval" alias was removed 2026-07-17 — aliases hid a bar/eval split
# mixup). GSM and Spider only — SMILES selects by class, not index split.
INDICES_KEYS = {
    "spider": {"train": "train_indices", "test": "test_indices"},
    "gsm_symbolic": {"train": "train_indices", "test": "test_indices"},
}


def visible_physical_gpu_ids() -> list[int] | None:
    """Physical GPU indices allowed by CUDA_VISIBLE_DEVICES, or None when unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    visible: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must use numeric physical GPU indices for "
                f"sharded evaluation, got {token!r}"
            ) from exc
        if index >= 0 and index not in visible:
            visible.append(index)
    return visible


def indices_key_for(dataset: str, split_name: str) -> str:
    if dataset not in INDICES_KEYS:
        raise ValueError(
            f"dataset {dataset!r} not shardable (no index-based split file); "
            f"supported: {sorted(INDICES_KEYS)}"
        )
    keys = INDICES_KEYS[dataset]
    if split_name not in keys:
        raise ValueError(f"split name {split_name!r} invalid for {dataset}: {sorted(keys)}")
    return keys[split_name]


def plan_shards(n_examples: int, n_workers: int) -> list[tuple[int, int]]:
    """Contiguous (lo, hi) slices covering n_examples, sizes differing by <=1."""
    n_workers = min(n_workers, n_examples)
    base, extra = divmod(n_examples, n_workers)
    shards, lo = [], 0
    for i in range(n_workers):
        hi = lo + base + (1 if i < extra else 0)
        shards.append((lo, hi))
        lo = hi
    return shards


def slice_split(split: dict, indices_key: str, lo: int, hi: int) -> dict:
    out = dict(split)
    out[indices_key] = split[indices_key][lo:hi]
    size_key = indices_key.replace("_indices", "_size")
    if size_key in out:
        out[size_key] = hi - lo
    return out


def _row_source_index(row: dict[str, Any]) -> int:
    aliases = ("source_index", "spider_source_index", "crane_source_index")
    present: list[tuple[str, int]] = []
    for key in aliases:
        if key not in row:
            continue
        value = row[key]
        # None has historically meant that this optional alias is absent.
        if value is None:
            continue
        if type(value) is not int:
            raise ValueError(f"source alias {key} must be an integer")
        present.append((key, value))
    if not present:
        raise ValueError("row has no source index alias")
    resolved = present[0][1]
    if any(value != resolved for _, value in present[1:]):
        raise ValueError("source index aliases disagree")
    return resolved


_IMMUTABLE_PROVENANCE_FIELDS = (
    "compiled_csd_path",
    "compiled_csd_sha256",
    "eval_model",
    "cell_id",
    "manifest_commit",
    "dataset",
    "max_steps",
    "step_token_budget",
    "smiles_class",
)


def merge_results(
    parts: list[dict],
    *,
    dataset: str,
    split_file: str,
    split_name: str,
    planned_slices: list[list[int]],
) -> dict:
    """Merge rows while preserving generic outputs without U8 provenance.

    Every shard still must return a prefix of its planned slice, with aligned
    source/outcome rows.  When every shard has reevaluation provenance, the
    strict U8 identity checks and merged provenance are retained.  When no
    shard has it, the generic answer/evidence/eval_split shape is preserved
    without fabricating a provenance block; mixed presence is rejected.
    """
    if not parts:
        raise ValueError("no shard results to merge")
    if len(parts) != len(planned_slices):
        raise ValueError(
            "shard result count does not match the planned shard slices"
        )
    if any(not isinstance(shard_slice, list) for shard_slice in planned_slices):
        raise ValueError("planned shard slices must be lists")
    if any(
        type(source_index) is not int
        for shard_slice in planned_slices
        for source_index in shard_slice
    ):
        raise ValueError("planned shard source indices must be strict integers")
    canonical_indices = [
        source_index
        for shard_slice in planned_slices
        for source_index in shard_slice
    ]
    if len(set(canonical_indices)) != len(canonical_indices):
        raise ValueError("canonical split contains duplicate source indices")
    canonical_set = set(canonical_indices)
    answers: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    actual_source_indices: list[int] = []
    seen_source_indices: set[int] = set()
    split_name_key = f"{dataset.split('_')[0]}_split_name"
    canonical_split_file = str(Path(split_file).resolve())
    first_eval_split: dict[str, Any] | None = None
    first_provenance: dict[str, Any] | None = None
    first_identity: tuple[Any, ...] | None = None
    provenance_presence: bool | None = None

    for shard_index, (part, assigned_slice) in enumerate(
        zip(parts, planned_slices)
    ):
        shard_answers = part.get("answers")
        shard_evidence = part.get("reevaluation_sample_evidence")
        if not isinstance(shard_answers, list):
            raise ValueError(f"shard {shard_index} answers must be a list")
        if not isinstance(shard_evidence, list):
            raise ValueError(f"shard {shard_index} evidence must be a list")
        if len(shard_answers) != len(shard_evidence):
            raise ValueError(
                f"shard {shard_index} evidence count does not match answers count"
            )

        eval_split = part.get("eval_split")
        if not isinstance(eval_split, dict):
            raise ValueError(f"shard {shard_index} is missing eval_split provenance")
        if first_eval_split is None:
            first_eval_split = dict(eval_split)
        elif eval_split.get("bar_split_name") != first_eval_split.get("bar_split_name"):
            raise ValueError("shard split provenance bar_split_name mismatch")
        if eval_split.get(split_name_key) != split_name:
            raise ValueError(
                f"shard {shard_index} split name does not match canonical {split_name!r}"
            )

        provenance = part.get("reevaluation_provenance")
        has_provenance = provenance is not None
        if provenance_presence is None:
            provenance_presence = has_provenance
        elif has_provenance != provenance_presence:
            raise ValueError("shard reevaluation provenance presence is mixed")
        if has_provenance:
            if not isinstance(provenance, dict):
                raise ValueError(
                    f"shard {shard_index} reevaluation provenance is invalid"
                )
            missing_fields = [
                field
                for field in (*_IMMUTABLE_PROVENANCE_FIELDS, split_name_key)
                if field not in provenance
            ]
            if missing_fields:
                raise ValueError(
                    f"shard {shard_index} provenance missing immutable fields"
                )
            if provenance["dataset"] != dataset:
                raise ValueError(f"shard {shard_index} dataset provenance mismatch")
            if provenance[split_name_key] != split_name:
                raise ValueError(f"shard {shard_index} split provenance mismatch")
            identity = tuple(
                provenance[field] for field in _IMMUTABLE_PROVENANCE_FIELDS
            )
            if first_identity is None:
                first_identity = identity
                first_provenance = dict(provenance)
            elif identity != first_identity:
                raise ValueError(
                    f"shard {shard_index} immutable reevaluation provenance mismatch"
                )
            declared_sample_size = provenance.get("sample_size")
            if type(declared_sample_size) is not int or declared_sample_size != len(
                assigned_slice
            ):
                raise ValueError(
                    f"shard {shard_index} provenance sample_size does not match "
                    "its planned slice"
                )

        row_source_indices: list[int] = []
        for local_index, (answer, evidence) in enumerate(
            zip(shard_answers, shard_evidence)
        ):
            if not isinstance(answer, dict) or not isinstance(evidence, dict):
                raise ValueError(f"shard {shard_index} rows must be objects")
            for outcome_key in ("is_correct", "is_syntax_valid"):
                answer_outcome = answer.get(outcome_key)
                evidence_outcome = evidence.get(outcome_key)
                if type(answer_outcome) is not bool or type(evidence_outcome) is not bool:
                    raise ValueError(
                        f"shard {shard_index} row {local_index} {outcome_key} outcomes must be bool"
                    )
                if answer_outcome != evidence_outcome:
                    raise ValueError(
                        f"shard {shard_index} row {local_index} answer/evidence {outcome_key} mismatch"
                    )
            answer_source = _row_source_index(answer)
            evidence_source = _row_source_index(evidence)
            if answer_source is None or evidence_source is None:
                raise ValueError(
                    f"shard {shard_index} row {local_index} lacks source index"
                )
            if answer_source != evidence_source:
                raise ValueError(
                    f"shard {shard_index} row {local_index} answer/evidence source mismatch"
                )
            if answer_source not in canonical_set:
                raise ValueError(
                    f"shard {shard_index} row {local_index} source index is outside canonical split"
                )
            if answer_source in seen_source_indices:
                raise ValueError(
                    f"duplicate evaluated source index {answer_source} across shards"
                )
            seen_source_indices.add(answer_source)
            row_source_indices.append(answer_source)
            global_index = len(answers)
            answer_copy = dict(answer)
            answer_copy["example_index"] = global_index
            evidence_copy = dict(evidence)
            evidence_copy["evaluated_index"] = global_index
            answers.append(answer_copy)
            evidence_rows.append(evidence_copy)
            actual_source_indices.append(answer_source)

        expected_prefix = assigned_slice[: len(row_source_indices)]
        if row_source_indices != expected_prefix:
            raise ValueError(
                f"shard {shard_index} returned source indices that are not a prefix "
                "of its assigned shard slice"
            )
        if provenance_presence:
            if not isinstance(provenance, dict):
                raise ValueError(
                    f"shard {shard_index} reevaluation provenance is invalid"
                )
            declared = provenance.get("evaluated_source_indices")
            if not isinstance(declared, list):
                raise ValueError(
                    f"shard {shard_index} provenance evaluated_source_indices is missing"
                )
            if any(type(value) is not int for value in declared):
                raise ValueError(
                    f"shard {shard_index} provenance source indices must be strict integers"
                )
            declared_indices = list(declared)
            if declared_indices != row_source_indices:
                raise ValueError(
                    f"shard {shard_index} provenance source indices do not match rows"
                )

    if not answers:
        raise ValueError("no evaluated shard answers")
    if first_eval_split is None:
        raise ValueError("missing canonical split provenance")
    canonical_eval_split = dict(first_eval_split)
    canonical_eval_split["gsm_split_file"] = (
        canonical_split_file if dataset == "gsm_symbolic" else None
    )
    canonical_eval_split["gsm_split_name"] = split_name if dataset == "gsm_symbolic" else None
    canonical_eval_split["spider_split_file"] = (
        canonical_split_file if dataset == "spider" else None
    )
    canonical_eval_split["spider_split_name"] = split_name if dataset == "spider" else None

    planned_sample_size = len(canonical_indices)
    evaluated_count = len(answers)
    merged_provenance: dict[str, Any] | None = None
    if provenance_presence:
        if first_provenance is None:
            raise ValueError("missing canonical reevaluation provenance")
        merged_provenance = dict(first_provenance)
        merged_provenance["dataset"] = dataset
        merged_provenance["evaluated_source_indices"] = actual_source_indices
        merged_provenance["sample_size"] = planned_sample_size
        merged_provenance["planned_sample_size"] = planned_sample_size
        merged_provenance["evaluated_count"] = evaluated_count
        merged_provenance["sample_offset"] = 0
        merged_provenance["gsm_split_file"] = (
            canonical_split_file if dataset == "gsm_symbolic" else None
        )
        merged_provenance["gsm_split_name"] = (
            split_name if dataset == "gsm_symbolic" else None
        )
        merged_provenance["spider_split_file"] = (
            canonical_split_file if dataset == "spider" else None
        )
        merged_provenance["spider_split_name"] = (
            split_name if dataset == "spider" else None
        )

    merged_metrics = {
        "num_shards": len(parts),
        "shard_sizes": [len(p["answers"]) for p in parts],
        "planned_sample_size": planned_sample_size,
        "evaluated_count": evaluated_count,
    }
    print(
        f"{LOG} merge: {len(parts)} shard(s), sizes {merged_metrics['shard_sizes']} "
        f"-> {evaluated_count} example(s) total",
        flush=True,
    )
    merged = {
        "accuracy": sum(a["is_correct"] for a in answers) / evaluated_count,
        "syntax_rate": sum(a["is_syntax_valid"] for a in answers) / evaluated_count,
        "metrics": merged_metrics,
        "answers": answers,
        "eval_split": canonical_eval_split,
        "reevaluation_sample_evidence": evidence_rows,
    }
    if merged_provenance is not None:
        merged["reevaluation_provenance"] = merged_provenance
    return merged

def detect_gpu_slots(workers_per_gpu: int, idle_util_threshold: int, min_free_mb: int) -> list[int]:
    """Return a list of GPU indices, one entry per worker slot ("anything idle":
    a GPU qualifies if its utilization is under the threshold and it has enough
    free memory for at least one worker)."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpu_rows: dict[int, tuple[int, int, int]] = {}
    for line in q.stdout.strip().splitlines():
        idx, used, total, util = [int(x.strip()) for x in line.split(",")]
        gpu_rows[idx] = (used, total, util)
    visible = visible_physical_gpu_ids()
    candidate_ids = visible if visible is not None else list(gpu_rows)
    slots: list[int] = []
    for idx in candidate_ids:
        if idx not in gpu_rows:
            print(f"{LOG} GPU {idx}: not reported by nvidia-smi, skipping", flush=True)
            continue
        used, total, util = gpu_rows[idx]
        free = total - used
        if util > idle_util_threshold:
            print(f"{LOG} GPU {idx}: busy (util {util}% > {idle_util_threshold}%), skipping", flush=True)
            continue
        n_fit = min(workers_per_gpu, free // min_free_mb)
        print(f"{LOG} GPU {idx}: util {util}%, free {free} MiB -> {n_fit} worker slot(s)", flush=True)
        slots.extend([idx] * n_fit)
    return slots


def run_sharded_reevaluation(
    *,
    csd_path: str,
    dataset: str,
    sample_size: int,
    output_json: str,
    split_file: str,
    split_name: str,
    passthrough: list[str],
    workers_per_gpu: int = 2,
    stagger_seconds: int = 45,
    gpu_util: float = 0.18,
    idle_util_threshold: int = 30,
    min_free_mb: int = 8000,
) -> int:
    """Shard `sample_size` examples of `split_file`/`split_name` across idle GPU
    slots, running one `reevaluate_compiled_csd` subprocess per shard, then
    merge. Returns a process-style exit code (0 = success)."""
    indices_key = indices_key_for(dataset, split_name)

    split = json.loads(Path(split_file).read_text())
    split_indices = split.get(indices_key)
    if not isinstance(split_indices, list):
        raise ValueError(f"split {indices_key} must be a list")
    if any(type(source_index) is not int for source_index in split_indices):
        raise ValueError(f"split {indices_key} must contain strict integers")
    n = min(sample_size, len(split_indices))

    slots = detect_gpu_slots(workers_per_gpu, idle_util_threshold, min_free_mb)
    if not slots:
        print(f"{LOG} ERROR: no idle GPU capacity found", flush=True)
        return 2
    shards = plan_shards(n, len(slots))
    print(f"{LOG} worker count chosen: {len(shards)} (idle GPU slots: {len(slots)}, "
          f"examples: {n}) on GPU slots {slots[:len(shards)]}", flush=True)

    out_base = Path(output_json)
    work_dir = out_base.parent / f"{out_base.stem}_shards"
    work_dir.mkdir(parents=True, exist_ok=True)

    split_flag = "--spider-split-file" if dataset == "spider" else "--gsm-split-file"
    name_flag = "--spider-split-name" if dataset == "spider" else "--gsm-split-name"

    procs = []
    for i, ((lo, hi), gpu) in enumerate(zip(shards, slots)):
        shard_split_path = work_dir / f"split_shard{i}.json"
        shard_split_path.write_text(json.dumps(slice_split(split, indices_key, lo, hi)))
        part_json = work_dir / f"part{i}.json"
        log_path = work_dir / f"part{i}.log"
        cmd = [
            sys.executable, "-m", "synthesis.scripts.reevaluate_compiled_csd", csd_path,
            "--dataset", dataset, "--sample-size", str(hi - lo),
            split_flag, str(shard_split_path), name_flag, split_name,
            "--vllm-gpu-memory-utilization", str(gpu_util),
            "--output-json", str(part_json), *passthrough,
        ]
        if i > 0:
            print(f"{LOG} stagger {stagger_seconds}s before worker {i}", flush=True)
            time.sleep(stagger_seconds)
        print(f"{LOG} worker {i}: examples [{lo},{hi}) on GPU {gpu}, log {log_path}", flush=True)
        with open(log_path, "w") as lf:
            procs.append((i, part_json, subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )))

    failures = []
    parts = []
    for i, part_json, proc in procs:
        rc = proc.wait()
        if rc != 0 or not part_json.exists():
            failures.append(i)
            print(f"{LOG} worker {i} FAILED (rc={rc}); log: {work_dir}/part{i}.log", flush=True)
        else:
            parts.append(json.loads(part_json.read_text()))
            print(f"{LOG} worker {i} done", flush=True)

    if failures:
        print(f"{LOG} {len(failures)} shard(s) failed: {failures}. Part files kept in {work_dir}; "
              f"NOT writing merged output.", flush=True)
        return 1

    planned_slices = [
        list(split[indices_key][lo:hi]) for lo, hi in shards
    ]
    merged = merge_results(
        parts,
        dataset=dataset,
        split_file=split_file,
        split_name=split_name,
        planned_slices=planned_slices,
    )
    out_base.write_text(json.dumps(merged, indent=1))
    print(f"{LOG} merged {len(merged['answers'])} examples -> {out_base} "
          f"acc={merged['accuracy']:.4f} syn={merged['syntax_rate']:.4f}", flush=True)
    return 0
