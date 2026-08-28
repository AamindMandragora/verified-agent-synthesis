import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from scripts.runtime import run_table5_8_queue as queue


def test_exact_table5_to_table8_scope():
    rows = queue.build_scope(Path("/repo"))
    assert len(rows) == 31
    assert sum(row["table"] == 5 for row in rows) == 15
    assert sum(row["table"] == 6 for row in rows) == 6
    assert sum(row["table"] == 7 for row in rows) == 6
    assert sum(row["table"] == 8 for row in rows) == 4
    assert all(row["eval_model"] == "Qwen/Qwen3.5-2B" for row in rows)


def test_table5_backend_profiles_are_exact():
    rows = [row for row in queue.build_scope(Path("/repo")) if row["table"] == 5]
    assert {(row["profile"], row["generation_backend"], row["generation_model"]) for row in rows} == {
        ("gpt5.6-sol", "codex", "gpt-5.6-sol"),
        ("gemini3.1-pro", "vertex", "gemini-3.1-pro-preview"),
        ("opus5", "claude", "claude-opus-5"),
    }
    assert {row["benchmark"] for row in rows} == {"gsm_symbolic", "spider", "smiles"}


def test_ablation_scope_has_exact_single_variable_settings():
    rows = queue.build_scope(Path("/repo"))
    token = [row for row in rows if row["table"] == 6]
    assert {(row["token_budget"], row["beam_size"], row["adaptive_helper_mask"], row["helper_selection_policy"]) for row in token} == {(1, 2, True, "bandit"), (2, 2, True, "bandit"), (4, 2, True, "bandit")}
    beam = [row for row in rows if row["table"] == 7]
    assert {(row["token_budget"], row["beam_size"], row["adaptive_helper_mask"], row["helper_selection_policy"]) for row in beam} == {(1, 1, True, "bandit"), (1, 2, True, "bandit"), (1, 4, True, "bandit")}
    mask = [row for row in rows if row["table"] == 8]
    assert {(row["adaptive_helper_mask"], row["beam_size"], row["token_budget"], row["helper_selection_policy"]) for row in mask} == {(False, 2, 1, "bandit"), (True, 2, 1, "bandit")}


def test_commands_bind_canonical_splits_and_no_warm_start():
    for row in queue.build_scope(Path("/repo")):
        command = queue.synthesis_command(row, Path("/env/python"))
        assert command[command.index("--eval-model") + 1] == "Qwen/Qwen3.5-2B"
        assert command[command.index("--max-iterations") + 1] == "40"
        assert "--initial-strategy-file" not in command
        assert command[command.index("--generation-backend") + 1] == row["generation_backend"]
        assert command[command.index("--generation-model") + 1] == row["generation_model"]
        assert command[command.index("--eval-step-token-budget") + 1] == str(row["token_budget"])
        if row["table"] in (6, 7, 8):
            assert command[command.index("--refinement-beam-size") + 1] == str(row["beam_size"])
            assert command[command.index("--helper-selection-policy") + 1] == "bandit"


def test_table5_smiles_export_is_sample_count_weighted():
    values = [{"cell_id": "a", "unique_valid_rate": 0.2, "sample_count": 100}, {"cell_id": "b", "unique_valid_rate": 0.8, "sample_count": 300}]
    assert queue.weighted_smiles_rate(values) == pytest.approx(0.65)


def test_provider_preflight_is_local_and_secret_free(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    statuses = queue.provider_preflight()
    assert {item["profile"] for item in statuses} == {"gpt5.6-sol", "gemini3.1-pro", "opus5"}
    assert all("secret" not in json.dumps(item).lower() for item in statuses)


def test_gpu_admission_uses_cold_queue_memory_contract():
    row = queue.build_scope(Path("/repo"))[0]
    snapshot = {2: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960}}
    assert queue.choose_gpu(row, snapshot, {}, snapshot, (2,)) == 2
    row["gpu_scope"] = [1]
    assert queue.choose_gpu(row, snapshot, {}, snapshot, (2,)) is None


def test_multi_gpu_rows_get_only_scoped_safe_pair():
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "spider")
    snapshot = {gpu: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960} for gpu in (0, 1, 2)}
    assert queue.choose_gpus(row, snapshot, {}, snapshot, (1, 2)) == (1, 2)
    row["gpu_scope"] = [0]
    assert queue.choose_gpus(row, snapshot, {}, snapshot, (1, 2)) is None


def test_manifest_is_immutable_and_records_every_execution_dependency(tmp_path, monkeypatch):
    paths = list(queue.SOURCE_PATHS)
    for rel in paths:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rel, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    crane = tmp_path / "legacy" / "CRANE"
    crane.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=crane, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=crane, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=crane, check=True)
    (crane / "README").write_text("crane", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=crane, check=True)
    subprocess.run(["git", "commit", "-qm", "crane"], cwd=crane, check=True)
    monkeypatch.setattr(queue, "CANONICAL_CRANE_COMMIT", subprocess.run(["git", "rev-parse", "HEAD"], cwd=crane, check=True, capture_output=True, text=True).stdout.strip())
    bar_path = tmp_path / "frozen-bars.json"
    bar_path.write_text("{}", encoding="utf-8")
    bar_sha = __import__("hashlib").sha256(bar_path.read_bytes()).hexdigest()
    monkeypatch.setattr(queue, "BAR_BINDINGS", {
        "gsm_symbolic": {"min_accuracy": 13 / 49, "min_syntax_rate": 0.9, "source_path": str(bar_path), "source_sha256": bar_sha},
        "spider": {"min_accuracy": 59 / 300, "min_syntax_rate": 0.9, "source_path": str(bar_path), "source_sha256": bar_sha},
        "smiles": {"acrylates": {"min_accuracy": 0.14, "min_syntax_rate": 0.9}, "chain_extenders": {"min_accuracy": 0.20, "min_syntax_rate": 0.9}, "isocyanates": {"min_accuracy": 0.30, "min_syntax_rate": 0.9}, "source_path": str(bar_path), "source_sha256": bar_sha},
    })
    payload = queue.manifest_payload(tmp_path, queue.build_scope(tmp_path))
    assert payload["crane_commit"] == queue.CANONICAL_CRANE_COMMIT
    assert set(payload["source_sha256"]) == set(paths)
    (tmp_path / paths[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(queue.ConfigError):
        queue.manifest_payload(tmp_path, queue.build_scope(tmp_path))


def test_state_round_trip_records_phase_and_surviving_child(tmp_path):
    path = tmp_path / "state.json"
    queue.write_state(path, {"status": "running", "phase": "synthesis", "pid": 123, "pid_start": "abc"})
    assert queue.read_state(path)["phase"] == "synthesis"
    assert queue.read_state(path)["pid_start"] == "abc"


def test_environment_binds_selected_gpu_cap_and_opus_account(monkeypatch, tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5")
    env = queue.synthesis_environment(row, (3,), {"PATH": "/bin"}, tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] == str(row["gpu_mem_util"])
    assert env["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert env["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "ssdear@gmail.com"


def test_heldout_command_uses_test_split_and_provenance(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "spider")
    cmd = queue.heldout_command(row, Path("python"), tmp_path / "compiled.py")
    assert "--spider-split-name" in cmd and cmd[cmd.index("--spider-split-name") + 1] == "test"
    assert "--provenance-cell-id" in cmd


def test_artifact_guard_rejects_unchanged_preexisting_output(tmp_path):
    output = tmp_path / "heldout.json"
    output.write_text("old", encoding="utf-8")
    before = queue.artifact_fingerprint(output)
    assert not queue.artifact_is_new_or_replaced(output, before)
    output.write_text("new", encoding="utf-8")
    assert queue.artifact_is_new_or_replaced(output, before)


def test_frozen_common_bars_and_author_token_budget_are_bound():
    rows = queue.build_scope(Path("/repo"))
    expected = {
        "gsm_symbolic": (13 / 49, 0.9),
        "spider": (59 / 300, 0.9),
        "smiles": {"acrylates": (0.14, 0.9), "chain_extenders": (0.20, 0.9), "isocyanates": (0.30, 0.9)},
    }
    for row in rows:
        if row["benchmark"] == "smiles":
            assert (row["min_accuracy"], row["min_syntax_rate"]) == expected["smiles"][row["smiles_class"]]
        else:
            assert (row["min_accuracy"], row["min_syntax_rate"]) == expected[row["benchmark"]]
        assert row["synthesis_max_tokens"] == 32768


def test_profile_environment_is_forced_and_smiles_temperature_is_exported(tmp_path):
    opus = next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5")
    env = queue.synthesis_environment(opus, (2, 3), {"CSD_CLAUDE_CONFIG_DIR": "wrong", "CSD_CLAUDE_EXPECTED_ACCOUNT": "wrong"}, tmp_path)
    assert env["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert env["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "ssdear@gmail.com"
    smiles = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "smiles")
    assert queue.synthesis_environment(smiles, (2,), {}, tmp_path)["CSD_CONSTRAINED_TEMPERATURE"] == "0.7"


def test_profile_gate_rejects_wrong_opus_and_vertex_fallbacks(tmp_path):
    opus = [next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5")]
    with pytest.raises(queue.ConfigError):
        queue.validate_profile_gates(opus, {"CSD_CLAUDE_CONFIG_DIR": "wrong", "CSD_CLAUDE_EXPECTED_ACCOUNT": "wrong"})
    vertex = [next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gemini3.1-pro")]
    adc = tmp_path / "adc.json"
    adc.write_text("{}", encoding="utf-8")
    with pytest.raises(queue.ConfigError):
        queue.validate_profile_gates(vertex, {"GOOGLE_APPLICATION_CREDENTIALS": str(adc), "GOOGLE_CLOUD_PROJECT": "p", "GOOGLE_CLOUD_LOCATION": "global", "GOOGLE_API_KEY": "fallback"})
    env = queue.synthesis_environment(vertex[0], (2,), {"GOOGLE_API_KEY": "bad", "GOOGLE_CLOUD_LOCATION": "us"}, tmp_path)
    assert "GOOGLE_API_KEY" not in env
    assert env["GOOGLE_CLOUD_LOCATION"] == "global"


def test_heldout_budget_and_controller_cli_contract():
    row = next(r for r in queue.build_scope(Path("/repo")) if r["table"] == 6 and r["token_budget"] == 4)
    cmd = queue.heldout_command(row, Path("python"), Path("compiled.py"))
    assert cmd[cmd.index("--step-token-budget") + 1] == "4"
    parser = queue.controller_parser()
    args = parser.parse_args(["--manifest", "manifest.json", "--gpus", "1,2", "--state-dir", "state", "--log", "queue.log", "--poll-seconds", "5"])
    assert args.gpus == (1, 2)
    assert args.poll_seconds == 5


def test_controller_does_not_overwrite_input_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("immutable", encoding="utf-8")
    with pytest.raises(queue.ConfigError):
        queue.controller_manifest_path(manifest, tmp_path / "manifest.json")


def test_export_uses_validated_reevaluation_syntax_rate_as_cw(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "gsm_symbolic")
    payload = {
        "cell_id": row["cell_id"],
        "accuracy": 0.4,
        "syntax_rate": 0.87,
        "metrics": {"num_examples": row["heldout_sample_size"]},
        "answers": [{} for _ in range(row["heldout_sample_size"])],
        "reevaluation_sample_evidence": [{} for _ in range(row["heldout_sample_size"])],
    }
    queue.export_results([row], [payload], tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())["cells"][0]["cw"] == 0.87


def test_export_accepts_production_spider_and_smiles_artifact_shapes(tmp_path):
    spider = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "spider")
    spider_payload = {
        "cell_id": spider["cell_id"],
        "accuracy": 0.4,
        "syntax_rate": 0.91,
        "metrics": {"num_examples": spider["heldout_sample_size"]},
        "answers": [{} for _ in range(spider["heldout_sample_size"])],
        "reevaluation_sample_evidence": [{} for _ in range(spider["heldout_sample_size"])],
    }
    queue.export_results([spider], [spider_payload], tmp_path / "spider.json")
    spider_cell = json.loads((tmp_path / "spider.json").read_text())["cells"][0]
    assert spider_cell["accuracy"] == 0.4
    assert spider_cell["cw"] == 0.91

    smiles_rows = [r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gpt5.6-sol" and r["benchmark"] == "smiles"]
    values = []
    for row, count, unique in zip(smiles_rows, (100, 100, 100), (10, 20, 30)):
        values.append({
            "cell_id": row["cell_id"],
            "smiles_paper_trial": {"sample_count": count, "unique_valid_count": unique},
            "metrics": {"num_examples": count},
            "answers": [{} for _ in range(count)],
            "reevaluation_sample_evidence": [{} for _ in range(count)],
        })
    queue.export_results(smiles_rows, values, tmp_path / "smiles.json")
    smiles_cell = json.loads((tmp_path / "smiles.json").read_text())["cells"][0]
    assert smiles_cell["unique_valid_rate"] == pytest.approx(0.2)


def test_pending_row_never_reuses_preexisting_deterministic_synthesis_output(tmp_path, monkeypatch):
    row = queue.build_scope(tmp_path)[0]
    old_compiled = tmp_path / "old" / "GeneratedCSD.py"
    old_compiled.parent.mkdir(parents=True)
    old_compiled.write_text("old", encoding="utf-8")
    monkeypatch.setattr(queue, "_compiled_output", lambda repo, candidate: old_compiled)
    calls = []

    class Process:
        pid = 123
        returncode = 1

        def communicate(self):
            return b"", b""

    def runner(argv, **kwargs):
        calls.append(argv)
        return Process()

    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        gpus=(0, 1),
        runner=runner,
    )
    assert calls
    assert "synthesis.run_synthesis" in calls[0]
    assert result["status"] == "failed"


def test_invalid_codex_auth_blocks_codex_without_blocking_ready_opus(monkeypatch, tmp_path):
    rows = [
        next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gpt5.6-sol"),
        next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5"),
    ]
    commit = "a" * 40
    rows = [dict(row, git_commit=commit) for row in rows]
    pilot_evidence = tmp_path / "opus-pilot.json"
    pilot_evidence.write_text('{"status":"verified"}\n', encoding="utf-8")
    monkeypatch.setattr(
        queue,
        "codex_auth_probe",
        lambda: {"returncode": 0, "stdout": "", "stderr": "invalid_refresh_token"},
    )
    ready, blocked = queue.partition_profile_readiness(
        rows,
        {
            "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
            "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com",
        },
        provider_pilots={
            "opus5": {
                "profile": "opus5",
                "status": "ready",
                "git_commit": commit,
                "backend": "claude",
                "config_dir": "/home/aadivyar/.claude-csd-synthesis",
                "expected_account": "ssdear@gmail.com",
                "model": "claude-opus-5",
                "output_sha256": queue.sha256_text("OPUS_ROUTE_OK"),
                "attempt_count": 1,
                "synthesis_status": "success",
                "verification_status": "success",
                "evaluation_status": "success",
                "response_sha256": "b" * 64,
                "evidence_path": str(pilot_evidence),
                "evidence_sha256": queue.hash_file(pilot_evidence),
            }
        },
    )
    assert [r["profile"] for r in ready] == ["opus5"]
    assert blocked[0]["status"] == "pending"
    assert "codex" in blocked[0]["reason"]


def test_profile_readiness_probes_each_provider_profile_once_and_requires_opus_pilot(monkeypatch):
    rows = [
        row for row in queue.build_scope(Path("/repo"))
        if row["profile"] in {"gpt5.6-sol", "opus5"}
    ]
    calls = []

    def probe():
        calls.append(True)
        return {"returncode": 0, "status": "ready", "stdout": "", "stderr": ""}

    monkeypatch.setattr(queue, "codex_auth_probe", probe)
    ready, blocked = queue.partition_profile_readiness(
        rows,
        {
            "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
            "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com",
        },
        provider_pilots={},
    )
    assert len(calls) == 1
    assert not ready
    assert {row["profile"] for row in blocked} == {"gpt5.6-sol", "opus5"}
    assert all("pilot" in row["reason"] for row in blocked if row["profile"] == "opus5")


def test_controller_validates_export_separation_before_dispatch(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    row = queue.build_scope(tmp_path)[0]
    monkeypatch.setattr(queue, "validate_manifest", lambda repo, payload: [row])
    monkeypatch.setattr(queue, "partition_profile_readiness", lambda *args, **kwargs: ([row], []))
    monkeypatch.setattr(queue, "dispatch", lambda *args, **kwargs: pytest.fail("dispatch occurred before path validation"))
    args = queue.controller_parser().parse_args([
        "--manifest", str(manifest), "--state-dir", str(tmp_path / "state"),
        "--log", str(tmp_path / "run.log"),
    ])
    with pytest.raises(queue.ConfigError, match="--export"):
        queue.controller_main(args)


def test_vertex_environment_clears_all_inherited_fallbacks(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gemini3.1-pro")
    row["vertex_project"] = "approved-project"
    inherited = {
        "VERTEX_AI_PROJECT": "wrong-project",
        "VERTEX_AI_LOCATION": "us-central1",
        "VERTEX_AI_BASE_URL": "https://wrong.example",
        "VERTEX_AI_API_KEY": "wrong-key",
        "VERTEX_AI_ACCESS_TOKEN": "wrong-token",
        "GOOGLE_CLOUD_PROJECT": "wrong-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GOOGLE_VERTEX_LOCATION": "us-central1",
        "GOOGLE_API_KEY": "wrong-key",
        "GEMINI_API_KEY": "wrong-key",
        "GOOGLE_GENAI_USE_VERTEXAI": "0",
    }
    env = queue.synthesis_environment(row, (2,), inherited, tmp_path)
    assert env["GOOGLE_CLOUD_LOCATION"] == "global"
    assert env["CSD_GEMINI_BACKEND"] == "vertex"
    assert env["CSD_GEMINI_MODEL"] == "gemini-3.1-pro-preview"
    assert env["VERTEX_AI_PROJECT"] == "approved-project"
    assert env["GOOGLE_CLOUD_PROJECT"] == "approved-project"
    assert env["VERTEX_AI_LOCATION"] == "global"
    assert env["GOOGLE_VERTEX_LOCATION"] == "global"
    assert env["VERTEX_AI_BASE_URL"] == "https://aiplatform.googleapis.com/v1"
    assert not any(key in env for key in inherited if key.endswith(("API_KEY", "ACCESS_TOKEN")) or key == "GOOGLE_GENAI_USE_VERTEXAI")


def test_exhausted_failure_report_best_compiled_candidate_is_recoverable(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic")
    run_dir = tmp_path / "outputs" / "generated" / row["output_name"] / "run"
    (run_dir / "results").mkdir(parents=True)
    compiled = run_dir / "compiled" / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("best", encoding="utf-8")
    report = {
        "total_attempts": 40,
        "attempts": [{
            "attempt_number": 40,
            "compilation": {"success": True, "output_dir": str(compiled.parent)},
            "evaluation": {"num_examples": row["eval_sample_size"], "accuracy": row["min_accuracy"], "syntax_rate": row["min_syntax_rate"]},
        }],
    }
    (run_dir / "results" / "failure_report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "cold_compiled_csd", lambda repo, output_name, **kwargs: compiled)
    assert queue._compiled_output(tmp_path, row) == compiled


def test_synthesis_exhaustion_with_best_candidate_continues_to_heldout(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic")
    latest = tmp_path / "outputs" / "generated" / row["output_name"] / "latest_run.txt"
    compiled = tmp_path / "compiled" / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("best", encoding="utf-8")
    monkeypatch.setattr(queue, "_compiled_output", lambda repo, candidate: compiled)
    monkeypatch.setattr(queue, "heldout_artifact_is_valid", lambda path, candidate: path.is_file())
    calls = []

    class Process:
        def __init__(self, code):
            self.pid = 123 + len(calls)
            self.returncode = code

        def communicate(self):
            if len(calls) == 1:
                latest.parent.mkdir(parents=True, exist_ok=True)
                latest.write_text(str(tmp_path / "run"), encoding="utf-8")
            else:
                output = Path(calls[-1][calls[-1].index("--output-json") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}", encoding="utf-8")
            return "", ""

    def runner(argv, **kwargs):
        calls.append(argv)
        return Process(1 if len(calls) == 1 else 0)

    result = queue.run_row(row, repo=tmp_path, python=Path("python"), state_dir=tmp_path / "state", gpus=(0,), runner=runner)
    assert result["status"] == "complete"
    assert len(calls) == 2


def test_heldout_validator_requires_bound_nonempty_unique_source_indices(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "gsm_symbolic")
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    row["compiled_csd_path"] = str(compiled)
    row["manifest_commit"] = "manifest-1"
    row["compiled_sha256"] = queue.hash_file(compiled)
    payload = {
        "accuracy": 0.2,
        "syntax_rate": 0.9,
        "metrics": {"num_examples": row["heldout_sample_size"]},
        "answers": [{"generated_answer": "x", "source_index": 0} for _ in range(row["heldout_sample_size"])],
        "reevaluation_provenance": {
            "cell_id": row["cell_id"], "dataset": row["dataset"], "eval_model": row["eval_model"],
            "sample_size": row["heldout_sample_size"], "max_steps": row["eval_max_steps"],
            "step_token_budget": row["token_budget"], "compiled_csd_path": str(compiled),
            "compiled_csd_sha256": queue.hash_file(compiled), "manifest_commit": "manifest-1",
            "evaluated_source_indices": [0] * row["heldout_sample_size"],
        },
        "eval_split": {"gsm_split_name": "test", "gsm_split_file": row["heldout_split_file"]},
    }
    assert not queue.heldout_artifact_is_valid(tmp_path / "missing.json", row)
    artifact = tmp_path / "heldout.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)


def _bound_gsm_artifact(row, compiled, *, manifest="manifest-1", indices=None):
    indices = list(indices if indices is not None else queue.expected_heldout_indices(row))
    answers = [{"generated_answer": "x", "source_index": index, "is_correct": False, "is_syntax_valid": True} for index in indices]
    return {
        "accuracy": 0.0,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": len(answers)},
        "answers": answers,
        "reevaluation_provenance": {
            "cell_id": row["cell_id"], "dataset": row["dataset"], "eval_model": row["eval_model"],
            "sample_size": len(answers), "max_steps": row["eval_max_steps"],
            "step_token_budget": row["token_budget"], "compiled_csd_path": str(compiled),
            "compiled_csd_sha256": queue.hash_file(compiled), "manifest_commit": manifest,
            "evaluated_source_indices": indices,
        },
        "eval_split": {"gsm_split_name": "test", "gsm_split_file": row["heldout_split_file"]},
    }


def test_heldout_validator_rejects_wrong_binding_metrics_and_accepts_valid_gsm(tmp_path):
    row = next(r for r in queue.build_scope(Path.cwd()) if r["benchmark"] == "gsm_symbolic")
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    row.update({"compiled_csd_path": str(compiled), "compiled_sha256": queue.hash_file(compiled), "manifest_commit": "manifest-1"})
    expected = queue.expected_heldout_indices(row)
    assert expected
    valid = _bound_gsm_artifact(row, compiled)
    artifact = tmp_path / "valid.json"
    artifact.write_text(json.dumps(valid), encoding="utf-8")
    assert queue.heldout_artifact_is_valid(artifact, row)

    cases = []
    wrong_order = list(expected)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    cases.append(_bound_gsm_artifact(row, compiled, indices=wrong_order))
    cases.append(_bound_gsm_artifact(row, compiled, manifest="different"))
    bad_path = dict(valid)
    bad_path["reevaluation_provenance"] = dict(valid["reevaluation_provenance"], compiled_csd_path=str(tmp_path / "other.py"))
    cases.append(bad_path)
    bad_hash = dict(valid)
    bad_hash["reevaluation_provenance"] = dict(valid["reevaluation_provenance"], compiled_csd_sha256="0" * 64)
    cases.append(bad_hash)
    bad_metric = dict(valid, accuracy=1.0)
    cases.append(bad_metric)
    for number, payload in enumerate(cases):
        candidate = tmp_path / f"invalid-{number}.json"
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        assert not queue.heldout_artifact_is_valid(candidate, row)


def test_heldout_validator_checks_smiles_trial_counts_and_blank_answers(tmp_path):
    row = next(r for r in queue.build_scope(Path.cwd()) if r["benchmark"] == "smiles")
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    indices = list(range(row["heldout_sample_size"]))
    answers = [{"generated_answer": "C", "source_index": index, "is_correct": False, "is_syntax_valid": True} for index in indices]
    payload = {
        "accuracy": 0.0, "syntax_rate": 1.0, "metrics": {"num_examples": len(answers)}, "answers": answers,
        "smiles_paper_trial": {"sample_count": len(answers), "unique_valid_count": len(answers) + 1},
        "reevaluation_provenance": {"cell_id": row["cell_id"], "dataset": row["dataset"], "eval_model": row["eval_model"], "smiles_class": row["smiles_class"], "sample_size": len(answers), "max_steps": row["eval_max_steps"], "step_token_budget": row["token_budget"], "compiled_csd_path": str(compiled), "compiled_csd_sha256": queue.hash_file(compiled), "evaluated_source_indices": indices},
    }
    artifact = tmp_path / "smiles-invalid.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)
    payload["smiles_paper_trial"]["unique_valid_count"] = 1
    payload["answers"][0]["generated_answer"] = ""
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)


def test_dispatch_runs_independent_admitted_rows_concurrently(tmp_path, monkeypatch):
    rows = [r for r in queue.build_scope(tmp_path) if r["benchmark"] == "smiles"][:2]
    snapshot = {
        0: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960},
        1: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960},
    }
    active = 0
    max_active = 0
    guard = threading.Lock()

    def fake_run_row(row, **kwargs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return {"cell_id": row["cell_id"], "status": "complete"}

    monkeypatch.setattr(queue, "run_row", fake_run_row)
    results = queue.dispatch(
        rows,
        repo=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        allowed=(0, 1),
        snapshot=lambda: snapshot,
    )
    assert len(results) == 2
    assert max_active == 2


def test_command_parser_accepts_all_table_controls():
    row = next(r for r in queue.build_scope(Path("/repo")) if r["table"] == 8 and not r["adaptive_helper_mask"])
    cmd = queue.synthesis_command(row, Path("/opt/anaconda/bin/python"))
    result = subprocess.run(cmd[:3] + ["--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_generated_option_is_in_real_synthesis_parser_help():
    help_result = subprocess.run(
        ["/opt/anaconda/bin/python", "-m", "synthesis.run_synthesis", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    for row in queue.build_scope(Path("/repo")):
        command = queue.synthesis_command(row, Path("python"))
        flags = [part for part in command if part.startswith("--")]
        assert all(flag in help_result.stdout for flag in flags), (row["cell_id"], flags)


def test_provider_pilot_hash_is_canonical_and_tamper_evident():
    pilots = {
        "opus5": {"status": "ready", "model": "claude-opus-5"},
        "gpt5.6-sol": {"status": "blocked", "model": "gpt-5.6-sol"},
    }
    expected = queue.sha256_text(
        json.dumps(pilots, sort_keys=True, separators=(",", ":"))
    )
    assert queue.provider_pilots_sha256(pilots) == expected
    changed = json.loads(json.dumps(pilots))
    changed["opus5"]["status"] = "blocked"
    assert queue.provider_pilots_sha256(changed) != expected


def test_provider_pilot_requires_exact_commit_and_hashed_evidence(tmp_path):
    evidence = tmp_path / "opus-pilot.json"
    evidence.write_text('{"result":"normal rejection"}\n', encoding="utf-8")
    pilot = {
        "status": "ready",
        "backend": "claude",
        "model": "claude-opus-5",
        "git_commit": "a" * 40,
        "config_dir": "/home/aadivyar/.claude-csd-synthesis",
        "expected_account": "ssdear@gmail.com",
        "output_sha256": queue.sha256_text("OPUS_ROUTE_OK"),
        "attempt_count": 1,
        "synthesis_status": "success",
        "verification_status": "rejected",
        "evaluation_status": "not_run",
        "response_sha256": "b" * 64,
        "evidence_path": str(evidence),
        "evidence_sha256": queue.hash_file(evidence),
    }
    assert queue.validate_provider_pilot("opus5", pilot, "a" * 40) is None
    assert "different code commit" in queue.validate_provider_pilot(
        "opus5", pilot, "c" * 40
    )
    evidence.write_text("changed\n", encoding="utf-8")
    assert "evidence" in queue.validate_provider_pilot(
        "opus5", pilot, "a" * 40
    )


@pytest.mark.parametrize("profile", ["gpt5.6-sol", "gemini3.1-pro", "opus5"])
def test_compiled_output_uses_strict_cold_report_validation_for_every_profile(
    tmp_path, monkeypatch, profile
):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == profile)
    compiled = tmp_path / profile / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("compiled", encoding="utf-8")
    calls = []

    def strict_compiled(repo, output_name, **kwargs):
        calls.append((repo, output_name, kwargs["job"]["generation_backend"]))
        return compiled

    monkeypatch.setattr(queue, "cold_compiled_csd", strict_compiled)
    assert queue._compiled_output(tmp_path, row) == compiled
    assert calls == [(tmp_path, row["output_name"], row["generation_backend"])]


def test_heldout_validator_rejects_answer_source_mismatch_and_zero_zero_collapse(
    tmp_path,
):
    row = next(r for r in queue.build_scope(Path.cwd()) if r["benchmark"] == "gsm_symbolic")
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    row.update(
        {
            "compiled_csd_path": str(compiled),
            "compiled_sha256": queue.hash_file(compiled),
            "manifest_commit": "manifest-1",
        }
    )
    payload = _bound_gsm_artifact(row, compiled)
    payload["answers"][0]["source_index"] = payload["answers"][1]["source_index"]
    artifact = tmp_path / "wrong-source.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)

    payload = _bound_gsm_artifact(row, compiled)
    payload["syntax_rate"] = 0.0
    for answer in payload["answers"]:
        answer["generated_answer"] = "same malformed output"
        answer["is_syntax_valid"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)


def test_smiles_validator_binds_accuracy_to_unique_valid_count_not_membership_flags(
    tmp_path,
):
    row = next(r for r in queue.build_scope(Path.cwd()) if r["benchmark"] == "smiles")
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    row.update(
        {
            "compiled_csd_path": str(compiled),
            "compiled_sha256": queue.hash_file(compiled),
            "manifest_commit": "manifest-1",
        }
    )
    count = row["heldout_sample_size"]
    indices = list(range(count))
    answers = [
        {
            "generated_answer": f"C{index}",
            "source_index": index,
            "is_correct": True,
            "is_syntax_valid": True,
        }
        for index in indices
    ]
    payload = {
        "accuracy": 0.14,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": count},
        "answers": answers,
        "smiles_paper_trial": {"sample_count": count, "unique_valid_count": 14},
        "reevaluation_provenance": {
            "cell_id": row["cell_id"],
            "dataset": row["dataset"],
            "eval_model": row["eval_model"],
            "smiles_class": row["smiles_class"],
            "sample_size": count,
            "max_steps": row["eval_max_steps"],
            "step_token_budget": row["token_budget"],
            "compiled_csd_path": str(compiled),
            "compiled_csd_sha256": queue.hash_file(compiled),
            "manifest_commit": "manifest-1",
            "evaluated_source_indices": indices,
        },
    }
    artifact = tmp_path / "smiles-valid.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert queue.heldout_artifact_is_valid(artifact, row)


def test_controller_rejects_duplicate_or_out_of_scope_gpus(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    for gpu_text in ("0,0", "4", ""):
        args = queue.controller_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--gpus",
                gpu_text,
                "--state-dir",
                str(tmp_path / "state"),
                "--log",
                str(tmp_path / "run.log"),
                "--export",
                str(tmp_path / "export.json"),
            ]
        )
        with pytest.raises(queue.ConfigError, match="GPU"):
            queue.validate_controller_paths(args)


def test_controller_lock_is_single_owner_and_does_not_block_state_lock(tmp_path):
    with queue.controller_lock(tmp_path):
        with pytest.raises(queue.ConfigError, match="already running"):
            with queue.controller_lock(tmp_path):
                pass
        with queue.state_lock(tmp_path):
            queue.write_state(tmp_path / "row.json", {"status": "pending"})


def test_dispatch_polls_surviving_child_without_readmitting_it(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["benchmark"] == "smiles")
    snapshot = {0: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960}}
    state_reads = [
        {"status": "running", "pid": 123, "pid_start": "one"},
        None,
    ]
    run_calls = []
    sleeps = []
    monkeypatch.setattr(queue, "read_state", lambda path: state_reads.pop(0))
    monkeypatch.setattr(queue, "child_is_same_process", lambda state: True)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        queue,
        "run_row",
        lambda candidate, **kwargs: run_calls.append(candidate["cell_id"])
        or {"cell_id": candidate["cell_id"], "status": "complete"},
    )
    results = queue.dispatch(
        [row],
        repo=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        allowed=(0,),
        snapshot=lambda: snapshot,
        poll_seconds=0.1,
    )
    assert run_calls == [row["cell_id"]]
    assert sleeps == [0.1]
    assert results[0]["status"] == "complete"
