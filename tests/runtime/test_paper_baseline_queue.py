import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.runtime import run_paper_baseline_queue as queue


def test_direct_script_entrypoint_supports_help():
    script = Path(__file__).parents[2] / "scripts" / "runtime" / "run_paper_baseline_queue.py"
    result = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "Restart-safe held-out baseline queue" in result.stdout


def test_manifest_scope_is_exactly_the_requested_heldout_cells():
    rows = queue.build_scope(Path("/repo"))
    assert len(rows) == 38
    assert sum(row["dataset"] == "gsm_symbolic" for row in rows) == 6
    assert sum(row["dataset"] == "spider" for row in rows) == 2
    assert sum(row["dataset"] == "smiles" for row in rows) == 30
    assert {row["strategy"] for row in rows} == {"unconstrained", "gcd", "crane", "itergen", "cars"}
    assert all(row["heldout_split_name"] == "test" for row in rows)
    assert all("strategy" in row and "output_json" in row for row in rows)


def test_live_scope_binds_the_exact_clean_crane_checkout():
    repo = Path(__file__).parents[2]
    rows = queue.build_scope(repo)
    assert {row["crane_commit"] for row in rows} == {"616379ce33ac6245933c16e6264b41f7d5800183"}


def test_manifest_generation_requires_external_exact_meta_bindings(tmp_path: Path):
    with pytest.raises(ValueError, match="metadecode-bindings"):
        queue.write_manifest(tmp_path, tmp_path / "manifest.json")


def test_fixed_baseline_command_binds_strategy_model_split_sample_and_output():
    row = queue.build_scope(Path("/repo"))[0]
    command = queue.fixed_baseline_command(row, Path("/env/python"))
    assert command[:3] == ["/env/python", "-m", "synthesis.evaluate.run_legacy_fixed_strategy"]
    assert command[command.index("--strategy") + 1] == row["strategy"]
    assert command[command.index("--eval-model") + 1] == row["eval_model"]
    assert command[command.index("--eval-sample-size") + 1] == str(row["sample_size"])
    assert command[command.index("--output-json") + 1] == row["output_json"]
    assert command[command.index("--gsm-split-file") + 1] == row["split_file"]
    assert command[command.index("--gsm-split-name") + 1] == "test"


def test_every_fixed_command_uses_only_real_evaluator_options():
    help_result = subprocess.run(
        [sys.executable, "-m", "synthesis.evaluate.run_legacy_fixed_strategy", "--help"],
        cwd=Path(__file__).parents[2], text=True, capture_output=True, check=True,
    )
    for row in queue.build_scope(Path(__file__).parents[2]):
        command = queue.fixed_baseline_command(row, Path(sys.executable))
        assert all(argument in help_result.stdout for argument in command if argument.startswith("--"))


def test_rerun_command_is_one_cold_cycle_without_warm_start():
    row = queue.build_scope(Path("/repo"))[0]
    command = queue.rerun_command(row, Path("/env/python"))
    assert command[command.index("--max-iterations") + 1] == "40"
    assert "--initial-strategy-file" not in command
    assert command[command.index("--generation-model") + 1] == "claude-opus-5"
    assert "--output-name" not in command
    assert "--gsm-split-file" not in command
    assert "--spider-split-file" not in command


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider", "smiles"])
def test_rerun_command_reaches_real_parser_for_each_dataset(dataset):
    row = next(row for row in queue.build_scope(Path("/repo")) if row["dataset"] == dataset)
    command = queue.rerun_command(row, Path(sys.executable))
    parser_probe = [*command, "--generation-backend", "not-a-real-backend"]
    result = subprocess.run(parser_probe, cwd=Path(__file__).parents[2], text=True, capture_output=True)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "unrecognized arguments" not in result.stderr


def test_rerun_thresholds_keep_the_cold_queue_syntax_gate():
    thresholds = queue.rerun_thresholds({"accuracy": 0.3, "syntax_rate": 0.1}, 49)
    assert thresholds["min_accuracy"] == 16 / 49
    assert thresholds["min_syntax_rate"] == 0.90


def test_exhausted_cold_rerun_selects_best_compiled_attempt(monkeypatch, tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    row["thresholds"] = {"min_accuracy": 0.5, "min_syntax_rate": 0.9}
    selected = tmp_path / "GeneratedCSD.py"
    selected.write_text("# best exhausted attempt\n", encoding="utf-8")
    monkeypatch.setattr(queue.cold_queue, "synthesis_was_exhausted", lambda repo, name, job: True)
    monkeypatch.setattr(queue.cold_queue, "compiled_csd", lambda repo, name, min_accuracy, min_syntax_rate, job: selected)
    assert queue._select_rerun_csd(row, tmp_path, 1) == selected


def _started_rerun_fixture(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    identity = queue.affected_row_id(row)
    row.update({"job_kind": "rerun", "cell_id": f"rerun-{identity}", "rerun_identity": identity,
                "rerun_command": ["true"], "assigned_gpu": 2, "claim_status": "running",
                "thresholds": {"min_accuracy": 0.5, "min_syntax_rate": 0.9}})
    claims = tmp_path / "claims"
    queue.claim_rerun(claims, identity, "m" * 64, {"command": ["true"], "status": "running"})
    state = tmp_path / "state"
    (state / f"rerun-{identity}.json").parent.mkdir(parents=True)
    (state / f"rerun-{identity}.json").write_text(json.dumps({
        "cell_id": identity, "status": "running", "phase": "synthesis", "pid": 999999,
    }), encoding="utf-8")
    return row, identity, claims, state


def test_restart_after_synthesis_child_death_recovers_csd_and_runs_heldout(monkeypatch, tmp_path: Path):
    row, identity, claims, state = _started_rerun_fixture(tmp_path)
    selected = tmp_path / "GeneratedCSD.py"
    selected.write_text("# recovered\n", encoding="utf-8")
    monkeypatch.setattr(queue, "_child_matches", lambda payload: False)
    monkeypatch.setattr(queue, "_select_rerun_csd", lambda row, repo, code: selected if code == 0 else None)
    monkeypatch.setattr(queue, "_run_heldout_rerun", lambda *args, **kwargs: 0)
    result = queue.run_row(row, repo=tmp_path, python=Path("/env/python"), claims_dir=claims,
                           manifest_sha256="m" * 64, state_dir=state,
                           runner=lambda *args, **kwargs: pytest.fail("must not restart synthesis"))
    assert result["status"] == "finished"
    assert result["reattached"] is True


def test_dead_synthesis_child_without_report_fails_durably(monkeypatch, tmp_path: Path):
    row, identity, claims, state = _started_rerun_fixture(tmp_path)
    monkeypatch.setattr(queue, "_child_matches", lambda payload: False)
    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("/env/python"),
        claims_dir=claims,
        manifest_sha256="m" * 64,
        state_dir=state,
        runner=lambda *args, **kwargs: pytest.fail("must not start a replacement author cycle"),
    )
    assert result["status"] == "failed"
    claim = next(claims.glob("*/rerun.json"))
    assert json.loads(claim.read_text(encoding="utf-8"))["status"] == "failed"


def test_restart_during_heldout_waits_and_validates_without_author_retry(monkeypatch, tmp_path: Path):
    row, identity, claims, state = _started_rerun_fixture(tmp_path)
    row["claim_status"] = "running"
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# heldout\n", encoding="utf-8")
    state_path = state / f"rerun-{identity}.json"
    state_path.write_text(json.dumps({"cell_id": identity, "status": "running", "phase": "heldout",
                                      "pid": 999999, "csd_path": str(csd), "csd_sha256": queue.sha256_file(csd)}), encoding="utf-8")
    monkeypatch.setattr(queue, "_child_matches", lambda payload: False)
    monkeypatch.setattr(queue.cold_queue, "heldout_is_complete", lambda path, job: False)
    monkeypatch.setattr(queue, "_run_heldout_rerun", lambda *args, **kwargs: 0)
    result = queue.run_row(row, repo=tmp_path, python=Path("/env/python"), claims_dir=claims,
                           manifest_sha256="m" * 64, state_dir=state,
                           runner=lambda *args, **kwargs: pytest.fail("must not retry author"))
    assert result["status"] == "finished"
    assert result["reattached"] is True


def test_real_row_receives_assigned_gpu_and_memory_cap(monkeypatch, tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    row["output_json"] = str(tmp_path / "output.json")
    row["assigned_gpu"] = 3
    seen = {}

    def runner(_command, *, cwd, env, check):
        seen.update(env)
        return SimpleNamespace(returncode=1)

    result = queue.run_row(row, repo=tmp_path, python=Path("/env/python"), claims_dir=tmp_path / "claims", manifest_sha256="m" * 64, runner=runner)
    assert result["status"] == "failed"
    assert seen["CUDA_VISIBLE_DEVICES"] == "3"
    assert seen["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] == str(row["gpu_mem_util"])


def test_rerun_env_uses_the_configured_claude_account(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    row.update({"job_kind": "rerun", "cell_id": "rerun-gsm-qwen35-2b", "rerun_identity": queue.affected_row_id(row), "rerun_command": ["true"], "assigned_gpu": 2})
    claim_root = tmp_path / "claims"
    queue.claim_rerun(claim_root, row["rerun_identity"], "m" * 64, {"command": ["true"]})
    seen = {}

    def runner(_command, *, cwd, env, check):
        seen.update(env)
        return SimpleNamespace(returncode=1)

    result = queue.run_row(row, repo=tmp_path, python=Path("/env/python"), claims_dir=claim_root,
                           manifest_sha256="m" * 64, state_dir=tmp_path / "state", runner=runner)
    assert result["status"] == "failed"
    assert seen["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert seen["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "ssdear@gmail.com"
    assert seen["CSD_OUTPUT_NAME"] == queue.rerun_output_name(row)


def test_dispatch_fails_closed_without_frozen_same_row_metadecode(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    with pytest.raises(ValueError, match="frozen metaDecode"):
        queue.dispatch([row], repo=tmp_path, python=Path("/env/python"), claims_dir=tmp_path / "claims", state_dir=tmp_path / "state", manifest_sha256="m" * 64, allowed=(0,), snapshot=lambda: {})


def test_dispatch_rejects_cli_gpu_scope_with_no_intersection(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "accuracy": 0.0, "syntax_rate": 0.0, "metrics": {"num_examples": 49}, "answers": [{} for _ in range(49)],
        "reevaluation_provenance": {"cell_id": queue.affected_row_id(row), "dataset": row["dataset"], "eval_model": row["eval_model"], "sample_size": 49, "max_steps": row["eval_max_steps"], "step_token_budget": 1, "smiles_class": None},
        "eval_split": {"gsm_split_file": "gsm_symbolic_crane_proportional_49x49_seed123.json", "gsm_split_name": "test"},
    }), encoding="utf-8")
    row.update({"metadecode_json": str(meta), "metadecode_sha256": queue.sha256_file(meta), "gpu_scope": [0]})
    with pytest.raises(ValueError, match="does not intersect"):
        queue.dispatch([row], repo=tmp_path, python=Path("/env/python"), claims_dir=tmp_path / "claims", state_dir=tmp_path / "state", manifest_sha256="m" * 64, allowed=(1,), snapshot=lambda: {})


def test_baseline_win_schedules_and_runs_one_claimed_cold_rerun(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    row["git_commit"] = "a" * 40
    row["output_json"] = str(tmp_path / "baseline.json")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "accuracy": 0.0, "syntax_rate": 0.0,
        "metrics": {"num_examples": 49}, "answers": [{} for _ in range(49)],
        "reevaluation_provenance": {
            "cell_id": "gsm-qwen35-2b", "dataset": "gsm_symbolic", "eval_model": row["eval_model"],
            "sample_size": 49, "max_steps": row["eval_max_steps"], "step_token_budget": 1, "smiles_class": None,
        },
        "eval_split": {"gsm_split_file": "gsm_symbolic_crane_proportional_49x49_seed123.json", "gsm_split_name": "test"},
    }), encoding="utf-8")
    row["metadecode_json"] = str(meta)
    row["metadecode_sha256"] = queue.sha256_file(meta)
    calls = []

    def runner(command, *, cwd, env, check, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            output = Path(command[command.index("--output-json") + 1])
            payload = {
                "accuracy": 1.0, "syntax_rate": 1.0,
                "metrics": {"num_examples": row["sample_size"]},
                "answers": [{} for _ in range(row["sample_size"])],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
        elif len(calls) == 2:
            compiled_dir = tmp_path / "compiled"
            compiled_dir.mkdir()
            (compiled_dir / "GeneratedCSD.py").write_text("# csd\n", encoding="utf-8")
            latest = tmp_path / "outputs" / "generated" / queue.rerun_output_name(row) / "latest_run.txt"
            latest.parent.mkdir(parents=True, exist_ok=True)
            run_dir = tmp_path / "run"
            (run_dir / "results").mkdir(parents=True)
            thresholds = queue.rerun_thresholds({"accuracy": 1.0, "syntax_rate": 1.0}, 49)
            (run_dir / "results" / "success_report.json").write_text(json.dumps({
                "compiled_dir": str(compiled_dir), "total_attempts": 40,
                "run_configuration": {
                    "task_description": row["task"], "output_name": queue.rerun_output_name(row), "git_commit": "a" * 40,
                    "max_iterations": 40, "thresholds": thresholds,
                    "author_model": {"backend": "claude", "model": "claude-opus-5", "max_new_tokens": 8192, "reasoning_budget_tokens": 4096, "anthropic_thinking": "always-on", "anthropic_effort": "high", "anthropic_thinking_display": "summarized"},
                    "evaluation": {"dataset": row["dataset"], "eval_model": row["eval_model"], "eval_sample_size": 49, "eval_max_steps": 900, "eval_step_token_budget": 1, "eval_max_seconds_per_example": 600, "split_provenance": {"bar_split_name": "train", "gsm_split_name": "train", "gsm_split_file": Path(row["split_file"]).name}},
                },
            }), encoding="utf-8")
            latest.write_text(str(run_dir), encoding="utf-8")
        elif len(calls) == 3:
            output = tmp_path / "outputs" / "reeval" / "paper_baseline_reruns" / f"{queue.affected_row_id(row)}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            compiled = tmp_path / "compiled" / "GeneratedCSD.py"
            payload = {"accuracy": 0.0, "syntax_rate": 0.0, "metrics": {"num_examples": 49}, "answers": [{} for _ in range(49)], "compiled_csd_path": str(compiled), "answers": [{} for _ in range(49)], "reevaluation_provenance": {"cell_id": queue.affected_row_id(row), "manifest_commit": "a" * 40, "dataset": "gsm_symbolic", "eval_model": row["eval_model"], "compiled_csd_path": str(compiled), "compiled_csd_sha256": queue.sha256_file(compiled), "sample_size": 49, "max_steps": 900, "step_token_budget": 1, "smiles_class": None}, "eval_split": {"gsm_split_file": str(row["split_file"]), "gsm_split_name": "test"}}
            output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    results = queue.dispatch([row], repo=tmp_path, python=Path("/env/python"), claims_dir=tmp_path / "claims", state_dir=tmp_path / "state", manifest_sha256="m" * 64, allowed=(0,), snapshot=lambda: {0: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960}}, poll_seconds=0, runner=runner)
    assert len(calls) == 3
    assert calls[1][calls[1].index("--max-iterations") + 1] == "40"
    assert results[-1]["status"] == "finished"


def test_artifact_validation_rejects_unbound_or_incomplete_output(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"accuracy": 0.5, "syntax_rate": 1.0}), encoding="utf-8")
    assert queue.validate_terminal_artifact(row, output) is None
    payload = {
        "accuracy": 0.5,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": row["sample_size"]},
        "answers": [{} for _ in range(row["sample_size"])],
        "paper_baseline_provenance": {
            "cell_id": row["cell_id"], "dataset": row["dataset"], "strategy": row["strategy"],
            "eval_model": row["eval_model"], "split_file": row["split_file"],
            "split_sha256": row["split_sha256"], "sample_size": row["sample_size"],
            "source_model": row["source_model"], "source_strategy": row["source_strategy"],
            "source_sample_size": row["source_sample_size"], "source_output_json": row["source_output_json"],
        },
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    validated = queue.validate_terminal_artifact(row, output)
    assert validated is not None
    assert validated["sha256"] == queue.sha256_file(output)


def test_manifest_rejects_a_row_with_changed_model_or_output_path(tmp_path: Path):
    rows = queue.build_scope(tmp_path)
    row = rows[0]
    row["eval_model"] = "Qwen/tampered"
    with pytest.raises(ValueError, match="immutable field eval_model"):
        queue._require_scope(rows, tmp_path)


@pytest.mark.parametrize(
    ("dataset", "baseline", "meta", "expected"),
    [
        ("gsm_symbolic", {"accuracy": 0.3, "syntax_rate": 0.9}, {"accuracy": 0.3, "syntax_rate": 0.8}, True),
        ("gsm_symbolic", {"accuracy": 0.3, "syntax_rate": 0.8}, {"accuracy": 0.3, "syntax_rate": 0.8}, False),
        ("spider", {"accuracy": 0.4, "syntax_rate": 0.9}, {"accuracy": 0.5, "syntax_rate": 0.8}, True),
        ("smiles", {"unique_valid_rate": 0.2}, {"unique_valid_rate": 0.2}, False),
        ("smiles", {"unique_valid_rate": 0.21}, {"unique_valid_rate": 0.2}, True),
    ],
)
def test_baseline_win_rule_uses_requested_strict_metrics(dataset, baseline, meta, expected):
    assert queue.baseline_beats_metadecode(dataset, baseline, meta) is expected


def test_claim_is_atomic_and_never_released_after_failure(tmp_path: Path):
    first = queue.claim_rerun(tmp_path, "gsm-qwen35-2b-unconstrained", "m" * 64, {"cold": True, "max_iterations": 40, "warm_start": False})
    second = queue.claim_rerun(tmp_path, "gsm-qwen35-2b-unconstrained", "m" * 64)
    assert first is True
    assert second is False
    claim = next(tmp_path.glob("*/claim.json"))
    claim_payload = json.loads(claim.read_text(encoding="utf-8"))
    assert claim_payload["cell_id"] == "gsm-qwen35-2b-unconstrained"
    assert claim_payload["started_at"]
    assert json.loads((claim.parent / "rerun.json").read_text(encoding="utf-8"))["warm_start"] is False


def test_rerun_claim_identity_ignores_baseline_strategy():
    rows = [row for row in queue.build_scope(Path("/repo")) if row["dataset"] == "gsm_symbolic" and row["eval_model"] == "Qwen/Qwen3.5-2B"]
    assert len({queue.affected_row_id(row) for row in rows}) == 1


def test_gpu_admission_counts_all_worker_demand_margin_and_scope():
    row = queue.build_scope(Path("/repo"))[0]
    row["worker_count"] = 2
    row["worker_memory_mib"] = 13_000
    snapshots = {
        0: {"used_mib": 15_000, "free_mib": 25_960, "total_mib": 40_960},
        1: {"used_mib": 0, "free_mib": 40_960, "total_mib": 40_960},
    }
    assert queue.choose_gpu(row, snapshots, {}, {0: dict(snapshots[0]), 1: dict(snapshots[1])}, (0,)) is None
    row["worker_count"] = 1
    assert queue.choose_gpu(row, snapshots, {}, {0: dict(snapshots[0]), 1: dict(snapshots[1])}, (1,)) == 1


def test_worker_demand_uses_the_larger_of_floor_and_live_fraction():
    for _slug, model in queue.MODELS:
        row = next(item for item in queue.build_scope(Path("/repo")) if item["eval_model"] == model)
        expected = max(row["worker_memory_mib"], int(__import__("math").ceil(row["gpu_mem_util"] * 40960)))
        assert queue.worker_demand_mib(row, 40960) == expected


def test_dry_run_does_not_create_claim(tmp_path: Path):
    row = queue.build_scope(tmp_path)[0]
    claim_root = tmp_path / "claims"
    queue.run_row(row, repo=tmp_path, python=Path("/env/python"), claims_dir=claim_root,
                  manifest_sha256="m" * 64, dry_run=True,
                  runner=lambda *args, **kwargs: pytest.fail("dry run called runner"))
    assert not claim_root.exists()
