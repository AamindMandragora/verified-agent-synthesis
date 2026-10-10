import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.runtime.run_warm_task_recovery_queue import ConfigError, dispatch

from scripts.runtime.claude_recovery.launch_queue_cell import (
    NoRemainingAttempts,
    author_environment,
    build_synthesis_command,
    rebuild_replay_checkpoint,
)
from scripts.runtime.claude_recovery.prepare_manifest import (
    ordered_jobs,
    validate_approved_cells,
)
from scripts.runtime.run_warm_task_recovery_queue import worker_environment
from scripts.runtime.supervise_warm_task_recovery import (
    COMPLETE_FAILURE,
    SYNTHESIS_REQUIRED,
    apply_conditional_extensions,
    job_phase,
)


def _attempt(number: int, accuracy: float, syntax: float, label: str) -> str:
    delimiter = "yes" if syntax else "no"
    return f"""Attempt {number}/40
Strategy: // {label}
method Main() {{
  // attempt {number}
}}

[1/4] Verifying with Dafny...
  ✓ Verification passed
  ✗ Evaluation below threshold:
    Accuracy: {accuracy:.1f}% (min: 55.1%)
    Contains << >>: {delimiter} (required: yes)
    Syntax: {syntax:.1f}% (min: 91.8%)
"""


def test_rebuild_checkpoint_replays_latest_evaluated_attempt(tmp_path: Path):
    history_path = tmp_path / "before3.json"
    history_path.write_text(
        json.dumps(
            [
                {"attempt_number": 1, "strategy_code": "// one", "accuracy": 0.1},
                {"attempt_number": 2, "strategy_code": "// two", "accuracy": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    seed_path = tmp_path / "attempt3.dfy"
    seed_path.write_text("// base three\nmethod Main() {}\n", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(
        _attempt(3, 26.5, 77.6, "stale three")
        + _attempt(3, 28.6, 79.6, "current three")
        + _attempt(4, 32.7, 81.6, "four")
        + "Attempt 5/40\nStrategy: // incomplete five\nmethod Main() {}\n",
        encoding="utf-8",
    )

    history, seed, replay = rebuild_replay_checkpoint(
        log_path=log_path,
        base_history_path=history_path,
        base_seed_path=seed_path,
        base_replay_attempt=3,
        total_cap=40,
        num_examples=49,
    )

    assert replay == 4
    assert [row["attempt_number"] for row in history] == [1, 2, 3]
    assert history[-1]["num_correct"] == 14
    assert "current three" in history[-1]["strategy_code"]
    assert "attempt 4" in seed
    assert "incomplete five" not in seed


def test_rebuild_checkpoint_uses_base_seed_before_any_completed_replay(tmp_path: Path):
    history_path = tmp_path / "history.json"
    history_path.write_text("[]\n", encoding="utf-8")
    seed_path = tmp_path / "seed.dfy"
    seed_path.write_text("// original seed\nmethod Main() {}\n", encoding="utf-8")
    log_path = tmp_path / "empty.log"
    log_path.write_text("", encoding="utf-8")

    history, seed, replay = rebuild_replay_checkpoint(
        log_path=log_path,
        base_history_path=history_path,
        base_seed_path=seed_path,
        base_replay_attempt=8,
        total_cap=40,
        num_examples=300,
    )

    assert history == []
    assert seed.startswith("// original seed")
    assert replay == 8


def test_incomplete_duplicate_does_not_hide_completed_attempt(tmp_path: Path):
    history_path = tmp_path / "before3.json"
    history_path.write_text("[]\n", encoding="utf-8")
    seed_path = tmp_path / "attempt3.dfy"
    seed_path.write_text("method Main() {}\n", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(
        _attempt(3, 20.0, 80.0, "three")
        + _attempt(4, 30.0, 90.0, "completed four")
        + "Attempt 4/40\nStrategy: // partial duplicate four\nmethod Main() {}\n",
        encoding="utf-8",
    )

    history, seed, replay = rebuild_replay_checkpoint(
        log_path=log_path,
        base_history_path=history_path,
        base_seed_path=seed_path,
        base_replay_attempt=3,
        total_cap=40,
        num_examples=49,
    )

    assert replay == 4
    assert [row["attempt_number"] for row in history] == [3]
    assert "completed four" in seed
    assert "partial duplicate" not in seed


def test_smiles_checkpoint_accepts_summary_without_delimiter_line(tmp_path: Path):
    history_path = tmp_path / "before1.json"
    history_path.write_text("[]\n", encoding="utf-8")
    seed_path = tmp_path / "attempt1.dfy"
    seed_path.write_text("method Main() {}\n", encoding="utf-8")
    log_path = tmp_path / "run.log"
    block = _attempt(2, 18.0, 100.0, "smiles two").replace(
        "    Contains << >>: yes (required: yes)\n", ""
    )
    log_path.write_text(block, encoding="utf-8")

    history, seed, replay = rebuild_replay_checkpoint(
        log_path=log_path,
        base_history_path=history_path,
        base_seed_path=seed_path,
        base_replay_attempt=1,
        total_cap=40,
        num_examples=50,
        require_delimiters=False,
    )

    assert replay == 2
    assert history == []
    assert "smiles two" in seed


def test_rebuild_checkpoint_refuses_a_consumed_attempt_cap(tmp_path: Path):
    history_path = tmp_path / "history.json"
    history_path.write_text("[]\n", encoding="utf-8")
    seed_path = tmp_path / "seed.dfy"
    seed_path.write_text("method Main() {}\n", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(_attempt(40, 10.0, 90.0, "terminal"), encoding="utf-8")

    with pytest.raises(NoRemainingAttempts):
        rebuild_replay_checkpoint(
            log_path=log_path,
            base_history_path=history_path,
            base_seed_path=seed_path,
            base_replay_attempt=39,
            total_cap=40,
            num_examples=300,
        )


def test_command_uses_claude_account_and_preserves_row_settings(tmp_path: Path):
    job = {
        "cell_id": "gsm-qwen35-9b",
        "task": "Solve the task.",
        "eval_model": "Qwen/Qwen3.5-9B",
        "dataset": "gsm_symbolic",
        "train_sample_size": 49,
        "min_accuracy": 0.551020408163,
        "min_syntax_rate": 0.918367346939,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "gpu_mem_util": 0.60,
        "output_name": "warmfix_gsm-qwen35-9b_0714_r2",
        "total_cap": 40,
        "train_split_file": "environment/benchmark_splits/gsm.json",
        "train_split_name": "train",
    }
    command = build_synthesis_command(
        job,
        repo=tmp_path,
        python=Path("/test/python"),
        seed_path=tmp_path / "seed.dfy",
        history_path=tmp_path / "history.json",
        replay_attempt=21,
        claude_executable=Path("/test/claude"),
        claude_config_dir=Path("/test/claude-config"),
        expected_account="aadivya@fermi.ai",
    )
    rendered = " ".join(map(str, command))

    assert "--generation-backend claude" in rendered
    assert "--generation-model claude-opus-5-5" in rendered
    assert "--claude-expected-account aadivya@fermi.ai" in rendered
    assert "--claude-idle-timeout-seconds 900" in rendered
    assert "--claude-emergency-timeout-seconds 7200" in rendered
    assert "--claude-max-retries 2" in rendered
    assert "--claude-telemetry-dir" in rendered
    assert "--claude-author-lock-file" in rendered
    assert "--initial-attempt-offset 20" in rendered
    assert "--max-iterations 20" in rendered
    assert "--min-accuracy 0.551020408163" in rendered
    assert "--min-syntax-rate 0.918367346939" in rendered
    assert "--eval-sample-size 49" in rendered
    assert "--eval-max-steps 900" in rendered
    assert "bedrock" not in rendered.lower()
    assert "anthropic-thinking" not in rendered


def test_manifest_orders_smaller_gpu_reservations_first():
    jobs = [
        {"cell_id": "nine", "memory_reservation_mib": 25000, "gpu_mem_util": 0.6},
        {"cell_id": "two", "memory_reservation_mib": 16384, "gpu_mem_util": 0.4},
        {"cell_id": "four", "memory_reservation_mib": 19000, "gpu_mem_util": 0.45},
    ]

    assert [job["cell_id"] for job in ordered_jobs(jobs, gpu_total_mib=40960)] == [
        "two",
        "four",
        "nine",
    ]


def test_manifest_can_reserve_one_gpu_for_a_large_priority_row_first():
    jobs = [
        {"cell_id": "small", "memory_reservation_mib": 16000, "gpu_mem_util": 0.4},
        {
            "cell_id": "gsm-qwen25-14b",
            "memory_reservation_mib": 34000,
            "gpu_mem_util": 0.81,
            "dispatch_priority": 0,
        },
    ]

    assert [job["cell_id"] for job in ordered_jobs(jobs, gpu_total_mib=40960)] == [
        "gsm-qwen25-14b",
        "small",
    ]

def test_manifest_explicitly_rejects_protected_gsm14b():
    with pytest.raises(ValueError, match="unapproved recovery cell"):
        validate_approved_cells([{"cell_id": "gsm14b"}])


def test_terminal_marker_excludes_consumed_cap_from_supervisor(tmp_path: Path):
    terminal = tmp_path / "terminal" / "spider.json"
    terminal.parent.mkdir()
    terminal.write_text(
        json.dumps({"phase": "complete_failure", "reason": "attempt_cap_evaluated"}),
        encoding="utf-8",
    )
    job = {
        "cell_id": "spider-qwen35-4b",
        "output_name": "spider-output",
        "heldout_output_json": str(tmp_path / "heldout.json"),
        "terminal_state_file": str(terminal),
    }

    assert job_phase(tmp_path, job) == COMPLETE_FAILURE


def test_queue_cell_script_can_run_as_a_direct_entrypoint():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "runtime" / "claude_recovery" / "launch_queue_cell.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Resume one approved recovery row" in result.stdout


def test_author_environment_streams_logs_and_removes_paid_api_credentials():
    environment = author_environment(
        {
            "PATH": "/bin",
            "AWS_BEARER_TOKEN_BEDROCK": "secret",
            "ANTHROPIC_API_KEY": "secret",
        },
        gpu=2,
    )

    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert "AWS_BEARER_TOKEN_BEDROCK" not in environment
    assert "ANTHROPIC_API_KEY" not in environment


def test_recovery_manifest_preserves_campaign_thresholds_and_sample_sizes():
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (repo / "saved-results/2026-07-15-claude-helper-recovery-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = {job["cell_id"]: job for job in manifest}

    for cell in ("gsm-qwen35-2b", "gsm-qwen35-4b", "gsm-qwen35-9b"):
        assert jobs[cell]["min_syntax_rate"] == 0.90

    smiles = jobs["smiles-qwen35-9b-isocyanates"]
    assert smiles["train_sample_size"] == 100
    assert smiles["heldout_sample_size"] == 100


def test_manifest_contains_exactly_the_seven_user_approved_rows():
    repo = Path(__file__).resolve().parents[3]
    queue_jobs = json.loads(
        (repo / "saved-results/2026-07-15-claude-helper-recovery-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert {job["cell_id"] for job in queue_jobs} == {
        "gsm-qwen35-2b",
        "gsm-qwen35-4b",
        "gsm-qwen35-9b",
        "gsm-qwen25-14b",
        "smiles-qwen35-9b-isocyanates",
        "spider-qwen35-4b",
        "spider-qwen25-7b",
    }
    gsm14 = next(job for job in queue_jobs if job["cell_id"] == "gsm-qwen25-14b")
    assert gsm14["min_syntax_rate"] == 0.90
    assert gsm14["heldout_sample_size"] == 49


def test_transient_provider_exit_requeues_the_same_row_once():
    calls = []
    statuses = iter([76, 0])
    job = {
        "cell_id": "gsm-qwen35-2b",
        "memory_reservation_mib": 1_000,
        "gpu_mem_util": 0.1,
    }

    def worker(current, gpu):
        calls.append((current["cell_id"], gpu))
        return next(statuses)

    dispatch(
        [job],
        snapshot=lambda: {0: {"used_mib": 0, "total_mib": 40_960}},
        worker=worker,
        poll_seconds=0,
        transient_retry_delay_seconds=0,
        transient_max_requeues=1,
    )

    assert calls == [("gsm-qwen35-2b", 0), ("gsm-qwen35-2b", 0)]


def test_exhausted_transient_provider_exit_fails_controller_for_supervisor_retry():
    job = {
        "cell_id": "gsm-qwen35-2b",
        "memory_reservation_mib": 1_000,
        "gpu_mem_util": 0.1,
    }

    with pytest.raises(ConfigError, match="gsm-qwen35-2b:76"):
        dispatch(
            [job],
            snapshot=lambda: {0: {"used_mib": 0, "total_mib": 40_960}},
            worker=lambda _job, _gpu: 76,
            poll_seconds=0,
            transient_retry_delay_seconds=0,
            transient_max_requeues=1,
        )


def test_permanent_worker_failure_fails_controller_instead_of_succeeding():
    job = {
        "cell_id": "gsm-qwen35-4b",
        "memory_reservation_mib": 1_000,
        "gpu_mem_util": 0.1,
    }

    with pytest.raises(ConfigError, match="gsm-qwen35-4b:1"):
        dispatch(
            [job],
            snapshot=lambda: {0: {"used_mib": 0, "total_mib": 40_960}},
            worker=lambda _job, _gpu: 1,
            poll_seconds=0,
        )


def test_transient_exit_without_scientific_report_remains_retryable(tmp_path: Path):
    output_name = "transient-timeout"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    (run_dir / "results").mkdir()
    (run_dir / "run.log").write_text(
        "TimeoutError: Claude Code generation timed out after 1800s\n",
        encoding="utf-8",
    )
    job = {
        "cell_id": "gsm-qwen35-2b",
        "output_name": output_name,
        "heldout_output_json": str(tmp_path / "heldout.json"),
    }

    assert job_phase(tmp_path, job) == SYNTHESIS_REQUIRED


def test_failure_report_below_an_extended_cap_is_not_terminal(tmp_path: Path):
    output_name = "extended"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "run-1"
    results = run_dir / "results"
    results.mkdir(parents=True)
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    (results / "failure_report.json").write_text(
        json.dumps({"attempts": [{"attempt_number": 40}]}),
        encoding="utf-8",
    )
    job = {
        "cell_id": "spider-qwen35-9b",
        "output_name": output_name,
        "total_cap": 80,
        "heldout_output_json": str(tmp_path / "heldout.json"),
    }

    assert job_phase(tmp_path, job) == SYNTHESIS_REQUIRED


def test_gsm_4b_and_9b_extend_only_after_both_fail(tmp_path: Path):
    def job(cell: str) -> dict:
        return {
            "cell_id": cell,
            "output_name": cell,
            "total_cap": 40,
            "extension_total_cap": 80,
            "extension_group": "gsm-qwen35-4b-9b",
            "heldout_output_json": str(tmp_path / f"{cell}-heldout.json"),
        }

    jobs = [job("gsm-qwen35-4b"), job("gsm-qwen35-9b")]
    for cell in ("gsm-qwen35-4b",):
        run_dir = tmp_path / "outputs" / "generated" / cell / "run-1"
        (run_dir / "results").mkdir(parents=True)
        (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
        (run_dir / "results" / "failure_report.json").write_text(
            json.dumps({"attempts": [{"attempt_number": 40}]}), encoding="utf-8"
        )

    assert [item["total_cap"] for item in apply_conditional_extensions(tmp_path, jobs)] == [40, 40]

    cell = "gsm-qwen35-9b"
    run_dir = tmp_path / "outputs" / "generated" / cell / "run-1"
    (run_dir / "results").mkdir(parents=True)
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    (run_dir / "results" / "failure_report.json").write_text(
        json.dumps({"attempts": [{"attempt_number": 40}]}), encoding="utf-8"
    )

    assert [item["total_cap"] for item in apply_conditional_extensions(tmp_path, jobs)] == [80, 80]


def test_all_recovery_rows_use_one_shared_retry_and_heldout_controller():
    repo = Path(__file__).resolve().parents[3]
    unit = (
        repo / "deploy/focal/systemd/csd-claude-recovery-queue.service"
    ).read_text(encoding="utf-8")

    assert "2026-07-15-claude-helper-recovery-manifest.json" in unit
    assert "supervise_warm_task_recovery.py" in unit
    assert "run_warm_task_recovery_queue.py" in unit
    assert "--retry-seconds 3600" in unit
    assert "Restart=on-failure" in unit
    assert "Conflicts=csd-gsm14b-claude-durable.service" in unit
    monitor = (repo / "deploy/focal/systemd/csd-codex-incident-monitor.service").read_text(
        encoding="utf-8"
    )
    assert "--recovery-service csd-cold-synthesis-queue.service" in monitor
    assert "--recovery-service csd-gsm14b-claude-durable.service" not in monitor


def test_worker_routes_custom_manifest_and_checkpoint_root_to_shared_launcher():
    job = {
        "last_clean_attempt": 55,
        "total_cap": 80,
        "output_name": "gsm14",
        "source_log": "/source.log",
        "history_file": "/history.json",
        "log_file": "/run.log",
        "launcher_manifest": "/gsm14-manifest.json",
        "checkpoint_root": "/gsm14-checkpoints",
    }

    environment = worker_environment(job, 2, {})

    assert environment["CLAUDE_RECOVERY_MANIFEST"] == "/gsm14-manifest.json"
    assert environment["CLAUDE_RECOVERY_CHECKPOINT_ROOT"] == "/gsm14-checkpoints"
