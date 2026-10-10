#!/usr/bin/env python3
"""Resume one approved recovery row through the isolated Claude Code account."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime.claude_recovery.checkpoint_from_log import (
    _ATTEMPT_START,
    _evaluated_record,
    _strategy,
)
from scripts.runtime.run_cold_synthesis_queue import parse_gpu_list


LOGGER = logging.getLogger("claude-recovery-queue")


class NoRemainingAttempts(ValueError):
    """Raised when a row has already evaluated its approved attempt cap."""


def requires_delimiter_line(dataset: str) -> bool:
    """Only GSM wraps expressions in << >>, so whitelist the one dataset that
    has the feature instead of blacklisting each one that does not as we
    discover it the hard way."""
    return dataset == "gsm_symbolic"


def rebuild_replay_checkpoint(
    *,
    log_path: Path,
    base_history_path: Path,
    base_seed_path: Path,
    base_replay_attempt: int,
    total_cap: int,
    num_examples: int,
    require_delimiters: bool = True,
) -> tuple[list[dict[str, Any]], str, int]:
    """Replay the latest evaluated strategy and retain all earlier evaluations."""
    prior = json.loads(base_history_path.read_text(encoding="utf-8"))
    if not isinstance(prior, list):
        raise ValueError(f"history must be a list: {base_history_path}")

    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    evaluated: dict[int, dict[str, Any]] = {}
    evaluated_strategies: dict[int, str] = {}
    matches = list(_ATTEMPT_START.finditer(log_text))
    attempts_in_range = 0
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        attempt = int(match.group("number"))
        if attempt < base_replay_attempt or attempt > total_cap:
            continue
        attempts_in_range += 1
        block = log_text[match.start():end]
        try:
            evaluated[attempt] = _evaluated_record(
                block,
                attempt,
                num_examples,
                require_delimiters=require_delimiters,
            )
            evaluated_strategies[attempt] = _strategy(block, attempt)
        except ValueError:
            continue

    if not evaluated:
        if attempts_in_range:
            # The log has attempt blocks in the range we care about, but every
            # single one failed to parse. Silently returning "empty" here
            # used to look identical to a fresh run with no attempts yet --
            # the caller would happily start over and throw away whatever
            # compute produced those attempts. Refuse instead so the bad log
            # gets looked at rather than quietly discarded.
            raise ValueError(
                f"found {attempts_in_range} attempt block(s) between "
                f"{base_replay_attempt} and {total_cap} in {log_path}, but none "
                "of them parsed; refusing to silently discard them and restart "
                "from the base checkpoint"
            )
        # No attempt blocks in range at all -- this is genuinely a fresh run,
        # so falling back to the base seed and history is correct.
        return (
            prior,
            base_seed_path.read_text(encoding="utf-8").rstrip(),
            base_replay_attempt,
        )

    replay_attempt = max(evaluated)
    if replay_attempt >= total_cap:
        raise NoRemainingAttempts(
            f"attempt cap already evaluated: {replay_attempt}/{total_cap}"
        )

    replaced = {
        attempt for attempt in evaluated if attempt < replay_attempt
    }
    history = [
        row for row in prior if int(row["attempt_number"]) not in replaced
    ]
    history.extend(evaluated[attempt] for attempt in sorted(replaced))
    history.sort(key=lambda row: int(row["attempt_number"]))
    numbers = [int(row["attempt_number"]) for row in history]
    if numbers != sorted(set(numbers)) or any(number >= replay_attempt for number in numbers):
        raise ValueError(
            "recovery history must be unique, increasing, and older than the replay; "
            f"replay={replay_attempt} history={numbers}"
        )
    return history, evaluated_strategies[replay_attempt], replay_attempt


def _resolved(repo: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else repo / path)


def build_synthesis_command(
    job: dict[str, Any],
    *,
    repo: Path,
    python: Path,
    seed_path: Path,
    history_path: Path,
    replay_attempt: int,
    claude_executable: Path,
    claude_config_dir: Path,
    expected_account: str,
) -> list[str]:
    """Build the frozen row command with Claude Code as the only author provider."""
    remaining = int(job["total_cap"]) - replay_attempt + 1
    command = [
        str(python),
        "-m",
        "synthesis.run_synthesis",
        "--task",
        str(job["task"]),
        "--generation-model",
        "claude-opus-5-5",
        "--generation-backend",
        "claude",
        "--claude-executable",
        str(claude_executable),
        "--claude-config-dir",
        str(claude_config_dir),
        "--claude-expected-account",
        expected_account,
        "--claude-timeout-seconds",
        "900",
        "--claude-idle-timeout-seconds",
        "900",
        "--claude-emergency-timeout-seconds",
        "7200",
        "--claude-max-retries",
        "2",
        "--claude-retry-delay-seconds",
        "30",
        "--claude-telemetry-dir",
        str(repo / ".context" / "claude_stream_telemetry"),
        "--claude-author-lock-file",
        str(repo / ".context" / "claude_author.lock"),
        "--eval-model",
        str(job["eval_model"]),
        "--eval-backend",
        "vllm",
        "--max-iterations",
        str(remaining),
        "--initial-attempt-offset",
        str(replay_attempt - 1),
        "--initial-strategy-file",
        str(seed_path),
        "--initial-attempt-history-file",
        str(history_path),
        "--output-name",
        str(job["output_name"]),
        "--output-dir",
        f"outputs/generated/{job['output_name']}",
        "--min-accuracy",
        str(job["min_accuracy"]),
        "--min-syntax-rate",
        str(job["min_syntax_rate"]),
        "--eval-sample-size",
        str(job["train_sample_size"]),
        "--eval-max-steps",
        str(job["eval_max_steps"]),
        "--eval-step-token-budget",
        "1",
        "--eval-max-seconds-per-example",
        str(job["eval_max_seconds"]),
        "--eval-min-examples-before-threshold-stop",
        str(job["train_sample_size"]),
        "--max-tokens",
        "32768",
        "--vllm-gpu-memory-utilization",
        str(job["gpu_mem_util"]),
        "--vllm-max-model-len",
        "16384",
        "--device",
        "auto",
        "--adaptive-helper-mask",
        "--helper-selection-policy",
        "bandit",
        "--refinement-beam-size",
        "2",
        "--vllm-tensor-parallel-size",
        "1",
        "--dataset",
        str(job["dataset"]),
    ]
    dataset = str(job["dataset"])
    if dataset == "gsm_symbolic":
        command.extend(
            [
                "--gsm-split-file",
                _resolved(repo, str(job["train_split_file"])),
                "--gsm-split-name",
                str(job["train_split_name"]),
            ]
        )
    elif dataset == "spider":
        command.extend(
            [
                "--spider-split-file",
                _resolved(repo, str(job["train_split_file"])),
                "--spider-split-name",
                str(job["train_split_name"]),
            ]
        )
    elif dataset == "smiles":
        command.extend(
            [
                "--smiles-classes",
                str(job["smiles_class"]),
                "--smiles-samples-per-class",
                str(job["train_sample_size"]),
            ]
        )
    else:
        raise ValueError(f"unsupported recovery dataset: {dataset}")
    return command


def author_environment(
    inherited: dict[str, str], gpu: int | tuple[int, ...]
) -> dict[str, str]:
    """Remove paid API credentials and select the assigned evaluator GPU(s).

    Accepts either a single GPU id or a tuple of them. CUDA_VISIBLE_DEVICES
    and CSD_EVAL_GPU_SLOTS are set to the same comma-joined list (matching
    synthesis_environment in run_cold_synthesis_queue.py), because
    _queue_gpu_slots() in eval_worker_pool.py rejects any GPU id that is not
    already in CUDA_VISIBLE_DEVICES -- setting only one of the two crashes
    the run.
    """
    gpu_list = ",".join(str(g) for g in gpu) if isinstance(gpu, tuple) else str(gpu)
    clean = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("AWS_", "BEDROCK_"))
        and not key.endswith("_API_KEY")
    }
    clean.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_list,
            "CSD_EVAL_GPU_SLOTS": gpu_list,
            "HF_HOME": "/home/aadivyar/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/home/aadivyar/.cache/huggingface",
            "PYTHONUNBUFFERED": "1",
            "LD_LIBRARY_PATH": (
                "/apps/conda/aadivyar/envs/csd/lib:"
                + clean.get("LD_LIBRARY_PATH", "")
            ),
        }
    )
    return clean


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_job(manifest: Path, cell: str) -> dict[str, Any]:
    jobs = json.loads(manifest.read_text(encoding="utf-8"))
    matches = [job for job in jobs if str(job.get("cell_id")) == cell]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest row for {cell}; found {len(matches)}")
    return matches[0]


def run_cell(args: argparse.Namespace) -> int:
    job = _load_job(args.manifest, args.cell)
    lock_path = Path(str(job["lock_file"]))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOGGER.warning("[claude-recovery] duplicate blocked cell=%s lock=%s", args.cell, lock_path)
            return 75

        checkpoint_dir = args.checkpoint_root / args.cell
        try:
            history, seed, replay = rebuild_replay_checkpoint(
                log_path=Path(str(job["log_file"])),
                base_history_path=Path(str(job["history_file"])),
                base_seed_path=Path(str(job["seed_file"])),
                base_replay_attempt=int(job["base_replay_attempt"]),
                total_cap=int(job["total_cap"]),
                num_examples=int(job.get("checkpoint_sample_size", job["train_sample_size"])),
                require_delimiters=requires_delimiter_line(str(job["dataset"])),
            )
        except NoRemainingAttempts as exc:
            terminal = Path(str(job["terminal_state_file"]))
            _atomic_text(
                terminal,
                json.dumps(
                    {
                        "phase": "complete_failure",
                        "reason": "attempt_cap_evaluated",
                        "detail": str(exc),
                        "evaluated_attempt": int(job["total_cap"]),
                        "total_cap": int(job["total_cap"]),
                    },
                    indent=2,
                )
                + "\n",
            )
            LOGGER.warning(
                "[claude-recovery] terminal cap cell=%s marker=%s detail=%s",
                args.cell,
                terminal,
                exc,
            )
            return 75
        history_path = checkpoint_dir / f"before{replay}.json"
        seed_path = checkpoint_dir / f"attempt{replay}.dfy"
        _atomic_text(history_path, json.dumps(history, indent=2) + "\n")
        _atomic_text(seed_path, seed.rstrip() + "\n")
        command = build_synthesis_command(
            job,
            repo=args.repo,
            python=args.python,
            seed_path=seed_path,
            history_path=history_path,
            replay_attempt=replay,
            claude_executable=args.claude_executable,
            claude_config_dir=args.claude_config_dir,
            expected_account=args.expected_account,
        )
        LOGGER.warning(
            "[claude-recovery] start cell=%s gpu=%s replay=%d next=%d cap=%d "
            "history=%d provider=claude account=%s",
            args.cell,
            args.gpu,
            replay,
            replay + 1,
            int(job["total_cap"]),
            len(history),
            args.expected_account,
        )
        log_path = Path(str(job["log_file"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=author_environment(os.environ, args.gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            status = process.wait()
        LOGGER.warning("[claude-recovery] finish cell=%s status=%d", args.cell, status)
        return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    # A single id ("0") or a comma-separated pair ("0,3") -- a poolable cell
    # (GSM/Spider) needs two GPUs so its eval workers can shard; parse_gpu_list
    # is the same parser run_cold_synthesis_queue.py uses for its --gpus flag.
    parser.add_argument("--gpu", type=parse_gpu_list, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--claude-executable", type=Path, required=True)
    parser.add_argument("--claude-config-dir", type=Path, required=True)
    parser.add_argument("--expected-account", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    try:
        return run_cell(args)
    except NoRemainingAttempts as exc:
        LOGGER.warning("[claude-recovery] terminal cap cell=%s detail=%s", args.cell, exc)
        return 75
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.exception("[claude-recovery] failed cell=%s detail=%s", args.cell, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
