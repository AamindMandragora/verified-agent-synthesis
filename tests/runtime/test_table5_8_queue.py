import json
import subprocess
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


def test_export_requires_complete_row_level_cw_evidence(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "gsm_symbolic")
    with pytest.raises(queue.ConfigError):
        queue.export_results([row], [{"cell_id": row["cell_id"], "accuracy": 0.4}], tmp_path / "out.json")
    payload = {"cell_id": row["cell_id"], "accuracy": 0.4, "helper_calls": [{"helper": "h", "used": False}, {"helper": "k", "used": True}, {"helper": "m", "used": False}, {"helper": "n", "used": False}, {"helper": "p", "used": False}]}
    queue.export_results([row], [payload], tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())["cells"][0]["cw"] == 0.2


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
