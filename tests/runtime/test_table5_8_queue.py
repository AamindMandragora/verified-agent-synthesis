import json
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.runtime import run_table5_8_queue as queue


def _test_python_runtime():
    return {
        "executable": str(queue.CANONICAL_PYTHON),
        "python_version": "3.11.15",
        "implementation": "CPython",
        "package_count": 100,
        "packages_sha256": "3" * 64,
    }


def _fixture_row(
    benchmark: str,
    repo: Path = Path("/repo"),
    *,
    profile: str = "opus5",
    cell_suffix: str = "one",
):
    if benchmark == "gsm_symbolic":
        return next(
            row
            for row in queue.build_scope(repo)
            if row["profile"] == profile and row["table"] == 5
        )
    smiles_class = "acrylates" if benchmark == "smiles" else None
    return queue._row(
        f"fixture-{benchmark}-{profile}-{cell_suffix}",
        5,
        benchmark,
        profile,
        smiles_class=smiles_class,
    )


def test_exact_table5_to_table8_scope():
    rows = queue.build_scope(Path("/repo"))
    assert len(rows) == 8
    assert sum(row["table"] == 5 for row in rows) == 3
    assert sum(row["table"] == 6 for row in rows) == 2
    assert sum(row["table"] == 7 for row in rows) == 2
    assert sum(row["table"] == 8 for row in rows) == 1
    assert all(row["eval_model"] == "Qwen/Qwen3.5-2B" for row in rows)
    assert {row["benchmark"] for row in rows} == {"gsm_symbolic"}
    assert all(row["gpu_count"] == 1 for row in rows)
    assert all(row["memory_reservation_mib"] == 20_480 for row in rows)


def test_direct_dry_run_prints_all_eight_physical_runs():
    repo = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/runtime/run_table5_8_queue.py"), "--dry-run"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    command_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("t5-", "t6-", "t7-", "t8-"))
    ]
    assert len(command_lines) == 8


def test_main_installs_canonical_provider_routes_before_cli_handoff(
    monkeypatch, capsys
):
    canonical = {
        "CSD_PI_NODE_EXECUTABLE": str(queue.CANONICAL_PI_NODE_EXECUTABLE),
        "CSD_PI_BRIDGE_PATH": str(queue.CANONICAL_PI_BRIDGE_PATH),
        "CSD_PI_AUTH_PATH": str(queue.CANONICAL_PI_AUTH_PATH),
        "CSD_CLAUDE_CONFIG_DIR": str(queue.CANONICAL_CLAUDE_CONFIG_DIR),
        "CSD_CLAUDE_EXPECTED_ACCOUNT": queue.CANONICAL_CLAUDE_EXPECTED_ACCOUNT,
    }
    for name in canonical:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(sys, "argv", ["run_table5_8_queue.py", "--dry-run"])

    assert queue.main() == 0
    capsys.readouterr()
    assert {name: queue.os.environ.get(name) for name in canonical} == canonical
    queue.validate_profile_gates(
        queue.build_scope(Path("/repo")), dict(queue.os.environ)
    )


def test_table5_backend_profiles_are_exact():
    rows = [row for row in queue.build_scope(Path("/repo")) if row["table"] == 5]
    assert {(row["profile"], row["generation_backend"], row["generation_model"]) for row in rows} == {
        ("gpt5.6-sol", "codex", "gpt-5.6-sol"),
        ("gemini3.7-flash", "gemini", "gemini-3.7-flash"),
        ("opus5", "claude", "claude-opus-5"),
    }
    assert {row["benchmark"] for row in rows} == {"gsm_symbolic"}


def test_ablation_scope_has_exact_single_variable_settings():
    rows = queue.build_scope(Path("/repo"))
    token = [row for row in rows if row["table"] == 6]
    assert {(row["token_budget"], row["beam_size"], row["adaptive_helper_mask"], row["helper_selection_policy"]) for row in token} == {(2, 2, True, "bandit"), (4, 2, True, "bandit")}
    beam = [row for row in rows if row["table"] == 7]
    assert {(row["token_budget"], row["beam_size"], row["adaptive_helper_mask"], row["helper_selection_policy"]) for row in beam} == {(1, 1, True, "bandit"), (1, 4, True, "bandit")}
    mask = [row for row in rows if row["table"] == 8]
    assert {(row["adaptive_helper_mask"], row["beam_size"], row["token_budget"], row["helper_selection_policy"]) for row in mask} == {(False, 2, 1, "bandit")}

    control = next(row for row in rows if row["cell_id"] == "t5-opus5-gsm_symbolic")
    assert control["paper_cells"] == [
        {"table": 5, "table_cell_id": "table5-opus5-gsm_symbolic"},
        {"table": 6, "table_cell_id": "t6-opus5-gsm_symbolic-b1-B2-m1"},
        {"table": 7, "table_cell_id": "t7-opus5-gsm_symbolic-b1-B2-m1"},
        {"table": 8, "table_cell_id": "t8-opus5-gsm_symbolic-b1-B2-m1"},
    ]


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
    monkeypatch.setenv("GEMINI_API_KEY", "private-key")
    statuses = queue.provider_preflight()
    assert {item["profile"] for item in statuses} == {"gpt5.6-sol", "gemini3.7-flash", "opus5"}
    gemini = next(item for item in statuses if item["profile"] == "gemini3.7-flash")
    assert gemini == {
        "profile": "gemini3.7-flash",
        "backend": "gemini",
        "status": "api_key_present",
        "api_key_sha256": queue.sha256_text("private-key"),
    }
    assert "private-key" not in json.dumps(statuses)
    assert all("secret" not in json.dumps(item).lower() for item in statuses)


def test_gpu_admission_uses_cold_queue_memory_contract():
    row = queue.build_scope(Path("/repo"))[0]
    snapshot = {2: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960}}
    assert queue.choose_gpu(row, snapshot, {}, snapshot, (2,)) == 2
    row["gpu_scope"] = [1]
    assert queue.choose_gpu(row, snapshot, {}, snapshot, (2,)) is None


def test_one_gpu_rows_fill_distinct_gpu_lanes_without_sharing():
    row = queue.build_scope(Path("/repo"))[0]
    snapshot = {gpu: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960} for gpu in (0, 1, 2)}
    demand = queue._demand(row, 40960)
    assert queue.choose_gpus(row, snapshot, {}, snapshot, (1, 2)) == (1,)
    assert queue.choose_gpus(row, snapshot, {1: demand}, snapshot, (1, 2)) == (2,)
    assert queue.choose_gpus(
        row, snapshot, {1: demand, 2: demand}, snapshot, (1, 2)
    ) is None


def test_manifest_is_immutable_and_records_every_execution_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        queue,
        "expected_author_route",
        lambda profile, environment: {
            "auth_mode": profile,
            "account_verified": True,
        },
    )
    monkeypatch.setattr(
        queue,
        "python_runtime_fingerprint",
        lambda python, repo: _test_python_runtime(),
    )
    external_runtime = {
        "eval_model": {
            "model": queue.EVAL_MODEL,
            "revision": "1" * 40,
            "snapshot_path": str(tmp_path / "snapshot"),
        },
        "spider_data": {
            "path": str(tmp_path / "spider"),
            "file_count": 2,
            "sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(
        queue, "external_runtime_binding", lambda environment: external_runtime
    )
    monkeypatch.setattr(
        queue,
        "validate_external_runtime_binding",
        lambda binding, environment: None,
    )
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
    assert payload["external_runtime"] == external_runtime
    assert payload["python_runtime"] == _test_python_runtime()
    assert set(payload["source_sha256"]) == set(paths)
    assert len(queue.validate_manifest(tmp_path, payload)) == 8
    wrong_version = json.loads(json.dumps(payload))
    wrong_version["version"] = 2
    with pytest.raises(queue.ConfigError, match="version"):
        queue.validate_manifest(tmp_path, wrong_version)
    missing_field = json.loads(json.dumps(payload))
    del missing_field["jobs"][0]["launch_commit"]
    with pytest.raises(queue.ConfigError, match="unknown or missing fields"):
        queue.validate_manifest(tmp_path, missing_field)
    wrong_launch = json.loads(json.dumps(payload))
    wrong_launch["jobs"][0]["launch_commit"] = "f" * 40
    with pytest.raises(queue.ConfigError, match="launch commit"):
        queue.validate_manifest(tmp_path, wrong_launch)
    changed_limits = json.loads(json.dumps(payload))
    changed_limits["jobs"][0]["effective_output_tokens"] = 1
    with pytest.raises(queue.ConfigError, match="effective_output_tokens"):
        queue.validate_manifest(tmp_path, changed_limits)
    (tmp_path / paths[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(queue.ConfigError):
        queue.manifest_payload(tmp_path, queue.build_scope(tmp_path))


def test_manifest_rejects_provider_pilot_from_different_source_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        queue,
        "expected_author_route",
        lambda profile, environment: {
            "auth_mode": profile,
            "account_verified": True,
        },
    )
    monkeypatch.setattr(queue, "execution_source_paths", lambda repo: ())
    monkeypatch.setattr(queue, "execution_source_hashes", lambda repo: {})
    monkeypatch.setattr(
        queue,
        "execution_source_sha256",
        lambda repo: queue.sha256_text("{}"),
    )
    monkeypatch.setattr(queue, "validate_crane_checkout", lambda repo: None)
    monkeypatch.setattr(queue, "crane_source_hashes", lambda repo: {})
    monkeypatch.setattr(
        queue,
        "materialize_frozen_bar_sources",
        lambda repo: {
            "gsm_symbolic": "bars/gsm.json",
            "spider": "bars/spider.json",
            "smiles": "bars/smiles.json",
        },
    )
    def git_run(argv, **kwargs):
        return types.SimpleNamespace(
            stdout="" if "status" in argv else "a" * 40 + "\n"
        )

    monkeypatch.setattr(queue.subprocess, "run", git_run)
    pilot = {"execution_source_sha256": "0" * 64}

    with pytest.raises(queue.ConfigError, match="current source bytes"):
        queue.manifest_payload(
            tmp_path,
            queue.build_scope(tmp_path),
            provider_pilots={"opus5": pilot},
        )


def test_crane_source_hashes_bind_a_tracked_symlink_without_following_it(
    tmp_path, monkeypatch
):
    crane = tmp_path / "legacy" / "CRANE"
    target = crane / "src" / "ladr" / "util"
    target.mkdir(parents=True)
    regular = crane / "README"
    regular.write_text("crane\n", encoding="utf-8")
    link = crane / "src" / "mace4.src" / "util"
    link.parent.mkdir(parents=True)
    link.symlink_to("../ladr/util", target_is_directory=True)
    monkeypatch.setattr(
        queue.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout=b"README\0src/mace4.src/util\0"
        ),
    )

    hashes = queue.crane_source_hashes(tmp_path)

    assert hashes["legacy/CRANE/README"] == queue.hash_file(regular)
    assert hashes["legacy/CRANE/src/mace4.src/util"] == queue.sha256_text(
        "symlink\0../ladr/util"
    )


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
    assert env["CSD_OUTPUT_DIR"] == str(
        tmp_path / "outputs/generated" / row["output_name"]
    )
    assert env["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert env["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "ssdear@gmail.com"
    assert env["CSD_REDACT_SENSITIVE_LOGS"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_external_runtime_binding_pins_qwen_revision_and_spider_tree(tmp_path):
    login_home = tmp_path / "home"
    hf_home = login_home / ".cache" / "huggingface"
    model_root = hf_home / "hub" / "models--Qwen--Qwen3.5-2B"
    revision = "1" * 40
    (model_root / "refs").mkdir(parents=True)
    (model_root / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    model_file = snapshot / "model.safetensors"
    model_file.write_bytes(b"model weights")
    spider = login_home / "spider_data" / "spider_data"
    (spider / "database" / "concert_singer").mkdir(parents=True)
    (spider / "dev.json").write_text("[]\n", encoding="utf-8")
    database = spider / "database" / "concert_singer" / "concert_singer.sqlite"
    database.write_bytes(b"sqlite bytes")

    binding = queue.external_runtime_binding({"HOME": str(login_home)})

    assert binding["eval_model"]["model"] == queue.EVAL_MODEL
    assert binding["eval_model"]["revision"] == revision
    assert binding["eval_model"]["snapshot_path"] == str(snapshot.resolve())
    assert binding["eval_model"]["snapshot_file_count"] == 1
    assert len(binding["eval_model"]["snapshot_sha256"]) == 64
    assert binding["spider_data"]["path"] == str(spider.resolve())
    assert binding["spider_data"]["file_count"] == 2
    assert len(binding["spider_data"]["sha256"]) == 64
    queue.validate_external_runtime_binding(binding, {"HOME": str(login_home)})

    model_file.write_bytes(b"changed model weights")
    with pytest.raises(queue.ConfigError, match="Qwen model bytes"):
        queue.validate_external_runtime_binding(binding, {"HOME": str(login_home)})
    model_file.write_bytes(b"model weights")

    database.write_bytes(b"different sqlite bytes")
    with pytest.raises(queue.ConfigError, match="Spider data bytes"):
        queue.validate_external_runtime_binding(binding, {"HOME": str(login_home)})


def test_heldout_command_uses_test_split_and_provenance(tmp_path):
    row = _fixture_row("spider")
    row.update(
        eval_model_revision="1" * 40,
        eval_model_snapshot_path="/cache/snapshots/" + "1" * 40,
        eval_model_snapshot_sha256="3" * 64,
        eval_model_snapshot_file_count=10,
        spider_data_path="/data/spider",
        spider_data_sha256="2" * 64,
        spider_data_file_count=922,
    )
    cmd = queue.heldout_command(row, Path("python"), tmp_path / "compiled.py")
    assert "--spider-split-name" in cmd and cmd[cmd.index("--spider-split-name") + 1] == "test"
    assert "--provenance-cell-id" in cmd
    assert cmd[cmd.index("--provenance-eval-model-revision") + 1] == "1" * 40
    assert cmd[cmd.index("--provenance-eval-model-snapshot-path") + 1] == row["eval_model_snapshot_path"]
    assert cmd[cmd.index("--provenance-spider-data-sha256") + 1] == "2" * 64
    assert cmd[cmd.index("--provenance-spider-data-file-count") + 1] == "922"


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
    smiles = _fixture_row("smiles")
    assert queue.synthesis_environment(smiles, (2,), {}, tmp_path)["CSD_CONSTRAINED_TEMPERATURE"] == "0.7"


def test_gpt_profile_environment_uses_only_pi_oauth_runtime(tmp_path):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "gpt5.6-sol"
    )
    node = tmp_path / "node"
    bridge = tmp_path / "bridge.mjs"
    auth = tmp_path / "auth.json"
    inherited = {
        "PATH": "/bin",
        "CSD_PI_NODE_EXECUTABLE": str(node),
        "CSD_PI_BRIDGE_PATH": str(bridge),
        "CSD_PI_AUTH_PATH": str(auth),
        "OPENAI_API_KEY": "must-not-reach-child",
        "CODEX_HOME": "/must/not/reach/child",
        "CSD_CODEX_EXECUTABLE": "/must/not/reach/child/codex",
    }

    environment = queue.synthesis_environment(row, (2,), inherited, tmp_path)

    assert environment["CSD_PI_NODE_EXECUTABLE"] == str(node)
    assert environment["CSD_PI_BRIDGE_PATH"] == str(bridge)
    assert environment["CSD_PI_AUTH_PATH"] == str(auth)
    assert "OPENAI_API_KEY" not in environment
    assert "CODEX_HOME" not in environment
    assert "CSD_CODEX_EXECUTABLE" not in environment


def test_gpt_profile_gate_allows_an_unrelated_parent_openai_api_key():
    row = {"profile": "gpt5.6-sol"}
    environment = {
        "CSD_PI_NODE_EXECUTABLE": str(queue.CANONICAL_PI_NODE_EXECUTABLE),
        "CSD_PI_BRIDGE_PATH": str(queue.CANONICAL_PI_BRIDGE_PATH),
        "CSD_PI_AUTH_PATH": str(queue.CANONICAL_PI_AUTH_PATH),
        "OPENAI_API_KEY": "parent-key-is-scrubbed-before-the-child",
    }

    queue.validate_profile_gates([row], environment)


def test_gpt_expected_route_comes_from_stored_pi_oauth_binding(monkeypatch):
    route = {
        "auth_mode": "chatgpt_codex_oauth",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "account_id_sha256": "a" * 64,
        "account_verified": True,
        "harness": "pi-provider-only",
        "pi_version": "0.84.4",
        "request_contract": "system-instructions-single-user-no-tools-v1",
        "node_executable": "/bound/node",
        "node_version": "v24.5.0",
        "node_sha256": "b" * 64,
        "bridge_path": "/bound/bridge.mjs",
        "bridge_sha256": "c" * 64,
        "package_lock_sha256": "d" * 64,
        "pi_install_file_count": 123,
        "pi_install_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        queue,
        "stored_pi_oauth_route",
        lambda **_kwargs: route,
    )

    assert queue.expected_author_route("gpt5.6-sol", {}) == route


def test_pi_provider_files_are_part_of_execution_source_binding():
    for path in (
        "synthesis/generate/pi_oauth/__init__.py",
        "synthesis/generate/pi_oauth/contract.py",
        "synthesis/generate/pi_oauth/provider/bridge.mjs",
        "synthesis/generate/pi_oauth/provider/package.json",
        "synthesis/generate/pi_oauth/provider/package-lock.json",
    ):
        assert path in queue.SOURCE_PATHS


def test_profile_and_heldout_environments_isolate_author_credentials(tmp_path):
    gemini = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "gemini3.7-flash"
    )
    opus = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "opus5"
    )
    inherited = {
        "PATH": "/bin",
        "PYTHONPATH": "/untracked/python",
        "HF_HOME": "/models",
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "GEMINI_API_KEY": "gemini-secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/adc.json",
        "VERTEX_AI_ACCESS_TOKEN": "vertex-secret",
        "CLAUDE_CONFIG_DIR": "/secret/claude",
        "CODEX_HOME": "/secret/codex",
        "CSD_RATIONALE_SUMMARY_API_KEY": "summary-secret",
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-secret",
        "SPIDER_TOKEN0_CONSTRAINED": "0",
        "CSD_PARITY_SEED": "123",
        "CSD_UNCONSTRAINED_TEMPERATURE": "0.9",
    }

    gemini_synthesis = queue.synthesis_environment(
        gemini, (2,), inherited, tmp_path
    )
    opus_synthesis = queue.synthesis_environment(opus, (2,), inherited, tmp_path)
    heldout = queue.heldout_environment(gemini, (2,), inherited, tmp_path)

    assert gemini_synthesis["GEMINI_API_KEY"] == "gemini-secret"
    assert gemini_synthesis["CSD_GEMINI_BACKEND"] == "gemini"
    assert gemini_synthesis["CSD_GEMINI_MODEL"] == "gemini-3.7-flash"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in gemini_synthesis
    assert "VERTEX_AI_ACCESS_TOKEN" not in gemini_synthesis
    assert "OPENAI_API_KEY" not in gemini_synthesis
    assert "OPENAI_API_KEY" not in opus_synthesis
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in opus_synthesis
    for key in (
        "CSD_RATIONALE_SUMMARY_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "SPIDER_TOKEN0_CONSTRAINED",
        "CSD_PARITY_SEED",
        "CSD_UNCONSTRAINED_TEMPERATURE",
    ):
        assert key not in gemini_synthesis
        assert key not in opus_synthesis
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_AI_ACCESS_TOKEN",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "CSD_GEMINI_BACKEND",
        "CSD_RATIONALE_SUMMARY_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "SPIDER_TOKEN0_CONSTRAINED",
        "CSD_PARITY_SEED",
        "CSD_UNCONSTRAINED_TEMPERATURE",
    ):
        assert key not in heldout
    assert "PYTHONPATH" not in gemini_synthesis
    assert "PYTHONPATH" not in opus_synthesis
    assert "PYTHONPATH" not in heldout
    assert heldout["CUDA_VISIBLE_DEVICES"] == "2"
    assert heldout["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] == str(
        gemini["gpu_mem_util"]
    )
    assert heldout["HF_HOME"] == "/models"


def test_isolated_home_pins_real_model_cache_and_spider_data_defaults(tmp_path):
    login_home = tmp_path / "login-home"
    hf_home = login_home / ".cache/huggingface"
    spider_data = login_home / "spider_data/spider_data"
    hf_home.mkdir(parents=True)
    spider_data.mkdir(parents=True)
    inherited = {"HOME": str(login_home), "PATH": "/bin"}
    row = _fixture_row("spider", tmp_path)

    queue.validate_runtime_data_paths(inherited)
    synthesis = queue.synthesis_environment(row, (2,), inherited, tmp_path)
    heldout = queue.heldout_environment(row, (2,), inherited, tmp_path)

    assert synthesis["HOME"] != str(login_home)
    assert heldout["HOME"] != str(login_home)
    for environment in (synthesis, heldout):
        assert environment["HF_HOME"] == str(hf_home)
        assert environment["TRANSFORMERS_CACHE"] == str(hf_home)
        assert environment["SPIDER_DATA_DIR"] == str(spider_data)


def test_profile_gate_rejects_wrong_opus_and_conflicting_gemini_routes(tmp_path):
    opus = [next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5")]
    with pytest.raises(queue.ConfigError):
        queue.validate_profile_gates(opus, {"CSD_CLAUDE_CONFIG_DIR": "wrong", "CSD_CLAUDE_EXPECTED_ACCOUNT": "wrong"})
    gemini = [next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gemini3.7-flash")]
    adc = tmp_path / "adc.json"
    adc.write_text("{}", encoding="utf-8")
    with pytest.raises(queue.ConfigError, match="exactly one direct Gemini API key"):
        queue.validate_profile_gates(gemini, {"GEMINI_API_KEY": "good", "GOOGLE_APPLICATION_CREDENTIALS": str(adc)})
    with pytest.raises(queue.ConfigError, match="GEMINI_API_KEY"):
        queue.validate_profile_gates(gemini, {})
    queue.validate_profile_gates(gemini, {"GEMINI_API_KEY": "good"})


def test_heldout_budget_and_controller_cli_contract():
    row = next(r for r in queue.build_scope(Path("/repo")) if r["table"] == 6 and r["token_budget"] == 4)
    cmd = queue.heldout_command(row, Path("python"), Path("compiled.py"))
    assert cmd[cmd.index("--step-token-budget") + 1] == "4"
    parser = queue.controller_parser()
    args = parser.parse_args(["--manifest", "manifest.json", "--gpus", "1,2", "--state-dir", "state", "--log", "queue.log", "--poll-seconds", "5"])
    assert args.gpus == (1, 2)
    assert args.poll_seconds == 5
    assert args.python == queue.CANONICAL_PYTHON


def test_disk_preflight_is_sized_to_unresolved_campaign_rows(tmp_path, monkeypatch):
    required = (
        queue.DISK_FIXED_SAFETY_BYTES
        + 11 * queue.DISK_BYTES_PER_UNRESOLVED_ROW
    )
    monkeypatch.setattr(
        queue.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(total=required, used=1, free=required - 1),
    )
    with pytest.raises(queue.ConfigError, match="insufficient disk space"):
        queue.disk_space_preflight(tmp_path, unresolved_rows=11)
    monkeypatch.setattr(
        queue.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(total=required, used=0, free=required),
    )
    queue.disk_space_preflight(tmp_path, unresolved_rows=11)


def test_controller_does_not_overwrite_input_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("immutable", encoding="utf-8")
    with pytest.raises(queue.ConfigError):
        queue.controller_manifest_path(manifest, tmp_path / "manifest.json")


def _bind_export_case(row, payload, tmp_path):
    row = dict(row, manifest_sha256="a" * 64, git_commit="b" * 40)
    evidence_dir = tmp_path / row["cell_id"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    compiled = evidence_dir / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    report = evidence_dir / "success_report.json"
    report.write_text("sealed report", encoding="utf-8")
    artifact = evidence_dir / "heldout.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    payload = dict(payload)
    payload["paper_artifact_path"] = str(artifact)
    payload["paper_artifact_sha256"] = queue.hash_file(artifact)
    payload["reevaluation_provenance"] = dict(
        payload.get("reevaluation_provenance") or {},
        compiled_csd_path=str(compiled),
        compiled_csd_sha256=queue.hash_file(compiled),
    )
    payload["synthesis_report_path"] = str(report)
    payload["synthesis_report_sha256"] = queue.hash_file(report)
    payload["winning_attempt"] = 1
    payload.setdefault("synthesis_attempts", 1)
    payload.setdefault("synthesis_terminal_status", "accepted")
    payload.setdefault(
        "runtime",
        {
            "row_started_at": "2026-09-01T00:00:00Z",
            "synthesis_started_at": "2026-09-01T00:00:00Z",
            "synthesis_finished_at": "2026-09-01T00:10:00Z",
            "synthesis_wall_time_seconds": 600.0,
            "heldout_started_at": "2026-09-01T00:10:00Z",
            "heldout_finished_at": "2026-09-01T00:12:00Z",
            "heldout_wall_time_seconds": 120.0,
            "row_finished_at": "2026-09-01T00:12:00Z",
            "total_wall_time_seconds": 720.0,
            "phase_timing_coverage": "all_phases",
            "attempt_evaluation_times_seconds": [11.5],
            "attempt_timing_coverage": "winning_attempt_only",
            "heldout_evaluator_total_time_seconds": 90.0,
            "heldout_recorded_run_wall_time_seconds": 120.0,
        },
    )
    return row, payload


def _fake_compiled_selection(tmp_path, compiled, *, winning_attempt=1):
    report = tmp_path / f"selection-{compiled.parent.name}.json"
    report.write_text(
        json.dumps(
            {
                "total_attempts": winning_attempt,
                "evaluation_result": {"total_time_seconds": 11.5},
            }
        ),
        encoding="utf-8",
    )
    return {
        "compiled_csd_path": compiled,
        "report_path": report,
        "report_sha256": queue.hash_file(report),
        "winning_attempt": winning_attempt,
    }


def test_export_uses_validated_mean_constrained_work_as_cw(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["benchmark"] == "gsm_symbolic")
    payload = {
        "cell_id": row["cell_id"],
        "accuracy": 0.4,
        "syntax_rate": 0.87,
        "metrics": {"num_examples": row["heldout_sample_size"], "mean_constrained_work": 17.25},
        "answers": [{} for _ in range(row["heldout_sample_size"])],
        "reevaluation_sample_evidence": [{} for _ in range(row["heldout_sample_size"])],
    }
    row, payload = _bind_export_case(row, payload, tmp_path)
    queue.export_results([row], [payload], tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())["cells"][0]["cw"] == 17.25


def test_export_records_accuracy_syntax_attempts_and_terminal_status(tmp_path):
    row = queue.build_scope(Path("/repo"))[0]
    payload = {
        "cell_id": row["cell_id"],
        "accuracy": 15 / 49,
        "syntax_rate": 47 / 49,
        "synthesis_attempts": 17,
        "synthesis_terminal_status": "accepted",
        "metrics": {"mean_constrained_work": 12.5},
    }
    row, payload = _bind_export_case(row, payload, tmp_path)

    queue.export_results([row], [payload], tmp_path / "out.json")

    cell = json.loads((tmp_path / "out.json").read_text())["cells"][0]
    assert cell["accuracy"] == 15 / 49
    assert cell["syntax_rate"] == 47 / 49
    assert cell["synthesis_attempts"] == 17
    assert cell["synthesis_terminal_status"] == "accepted"


def test_export_reuses_one_opus_control_for_tables_6_to_8(tmp_path):
    rows = queue.build_scope(Path("/repo"))
    bound_rows = []
    values = []
    for row in rows:
        payload = {
            "cell_id": row["cell_id"],
            "accuracy": 0.5,
            "syntax_rate": 0.95,
            "metrics": {"mean_constrained_work": 12.0},
        }
        bound_row, bound_payload = _bind_export_case(row, payload, tmp_path)
        bound_rows.append(bound_row)
        values.append(bound_payload)

    queue.export_results(bound_rows, values, tmp_path / "out.json")

    cells = json.loads((tmp_path / "out.json").read_text())["cells"]
    assert len(cells) == 11
    control_ids = {
        "table5-opus5-gsm_symbolic",
        "t6-opus5-gsm_symbolic-b1-B2-m1",
        "t7-opus5-gsm_symbolic-b1-B2-m1",
        "t8-opus5-gsm_symbolic-b1-B2-m1",
    }
    controls = [cell for cell in cells if cell["table_cell_id"] in control_ids]
    assert len(controls) == 4
    assert {cell["sources"][0]["cell_id"] for cell in controls} == {
        "t5-opus5-gsm_symbolic"
    }
    assert len({cell["sources"][0]["heldout_artifact_sha256"] for cell in controls}) == 1


def test_export_records_phase_and_attempt_runtimes(tmp_path):
    row = queue.build_scope(Path("/repo"))[0]
    runtime = {
        "row_started_at": "2026-09-01T00:00:00Z",
        "synthesis_started_at": "2026-09-01T00:00:00Z",
        "synthesis_finished_at": "2026-09-01T00:10:00Z",
        "synthesis_wall_time_seconds": 600.0,
        "heldout_started_at": "2026-09-01T00:10:00Z",
        "heldout_finished_at": "2026-09-01T00:12:00Z",
        "heldout_wall_time_seconds": 120.0,
        "row_finished_at": "2026-09-01T00:12:00Z",
        "total_wall_time_seconds": 720.0,
        "phase_timing_coverage": "all_phases",
        "attempt_evaluation_times_seconds": [31.25, 28.5],
        "attempt_timing_coverage": "all_attempts",
        "heldout_evaluator_total_time_seconds": 90.0,
        "heldout_recorded_run_wall_time_seconds": 119.5,
    }
    payload = {
        "cell_id": row["cell_id"],
        "accuracy": 0.5,
        "syntax_rate": 0.95,
        "metrics": {"mean_constrained_work": 12.0},
        "runtime": runtime,
    }
    row, payload = _bind_export_case(row, payload, tmp_path)

    queue.export_results([row], [payload], tmp_path / "out.json")

    assert json.loads((tmp_path / "out.json").read_text())["cells"][0][
        "runtime"
    ] == runtime


def test_export_is_bound_to_manifest_commit_and_terminal_artifact(tmp_path):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["benchmark"] == "gsm_symbolic"
    )
    row.update(manifest_sha256="a" * 64, git_commit="b" * 40)
    compiled = tmp_path / "compiled/GeneratedCSD.py"
    compiled.parent.mkdir()
    compiled.write_text("compiled", encoding="utf-8")
    artifact = tmp_path / "heldout.json"
    artifact.write_text("sealed heldout bytes", encoding="utf-8")
    report = tmp_path / "success_report.json"
    report.write_text("sealed report bytes", encoding="utf-8")
    value = {
        "cell_id": row["cell_id"],
        "accuracy": 0.4,
        "syntax_rate": 0.87,
        "synthesis_attempts": 1,
        "synthesis_terminal_status": "accepted",
        "metrics": {"mean_constrained_work": 19.0},
        "paper_artifact_path": str(artifact),
        "paper_artifact_sha256": queue.hash_file(artifact),
        "synthesis_report_path": str(report),
        "synthesis_report_sha256": queue.hash_file(report),
        "winning_attempt": 1,
        "runtime": {
            "row_started_at": "2026-09-01T00:00:00Z",
            "synthesis_started_at": "2026-09-01T00:00:00Z",
            "synthesis_finished_at": "2026-09-01T00:10:00Z",
            "synthesis_wall_time_seconds": 600.0,
            "heldout_started_at": "2026-09-01T00:10:00Z",
            "heldout_finished_at": "2026-09-01T00:12:00Z",
            "heldout_wall_time_seconds": 120.0,
            "row_finished_at": "2026-09-01T00:12:00Z",
            "total_wall_time_seconds": 720.0,
            "phase_timing_coverage": "all_phases",
            "attempt_evaluation_times_seconds": [11.5],
            "attempt_timing_coverage": "winning_attempt_only",
            "heldout_evaluator_total_time_seconds": 90.0,
            "heldout_recorded_run_wall_time_seconds": 120.0,
        },
        "reevaluation_provenance": {
            "compiled_csd_path": str(compiled),
            "compiled_csd_sha256": queue.hash_file(compiled),
        },
    }

    queue.export_results([row], [value], tmp_path / "export.json")

    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["manifest_sha256"] == "a" * 64
    assert exported["git_commit"] == "b" * 40
    assert exported["cells"][0]["sources"] == [
        {
            "cell_id": row["cell_id"],
            "profile": row["profile"],
            "generation_backend": row["generation_backend"],
            "generation_model": row["generation_model"],
            "heldout_artifact_path": str(artifact),
            "heldout_artifact_sha256": queue.hash_file(artifact),
            "compiled_csd_path": str(compiled),
            "compiled_csd_sha256": queue.hash_file(compiled),
            "synthesis_report_path": str(report),
            "synthesis_report_sha256": queue.hash_file(report),
            "winning_attempt": 1,
                "eval_model_revision": None,
                "eval_model_snapshot_path": None,
                "eval_model_snapshot_sha256": None,
                "eval_model_snapshot_file_count": None,
        }
    ]


def test_export_accepts_production_spider_and_smiles_artifact_shapes(tmp_path):
    spider = _fixture_row("spider")
    spider_payload = {
        "cell_id": spider["cell_id"],
        "accuracy": 0.4,
        "syntax_rate": 0.91,
        "metrics": {"num_examples": spider["heldout_sample_size"], "mean_constrained_work": 23.5},
        "answers": [{} for _ in range(spider["heldout_sample_size"])],
        "reevaluation_sample_evidence": [{} for _ in range(spider["heldout_sample_size"])],
    }
    spider, spider_payload = _bind_export_case(spider, spider_payload, tmp_path)
    queue.export_results([spider], [spider_payload], tmp_path / "spider.json")
    spider_cell = json.loads((tmp_path / "spider.json").read_text())["cells"][0]
    assert spider_cell["accuracy"] == 0.4
    assert spider_cell["cw"] == 23.5

    smiles_rows = [
        queue._row(
            f"fixture-smiles-gpt5.6-sol-{smiles_class}",
            5,
            "smiles",
            "gpt5.6-sol",
            smiles_class=smiles_class,
            table_cell_id="fixture-smiles-gpt5.6-sol",
        )
        for smiles_class in queue.SMILES_CLASSES
    ]
    bound_smiles_rows = []
    values = []
    for row, count, unique in zip(smiles_rows, (100, 100, 100), (10, 20, 30)):
        payload = {
            "cell_id": row["cell_id"],
            "smiles_paper_trial": {"sample_count": count, "unique_valid_count": unique},
            "metrics": {"num_examples": count},
            "answers": [{} for _ in range(count)],
            "reevaluation_sample_evidence": [{} for _ in range(count)],
        }
        bound_row, bound_payload = _bind_export_case(row, payload, tmp_path)
        bound_smiles_rows.append(bound_row)
        values.append(bound_payload)
    queue.export_results(bound_smiles_rows, values, tmp_path / "smiles.json")
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
    monkeypatch.setattr(queue.time, "time", lambda: 1787949000.0)
    rows = [
        next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gpt5.6-sol"),
        next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "opus5"),
    ]
    commit = "a" * 40
    rows = [dict(row, git_commit=commit) for row in rows]
    pilot_evidence = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(pilot_evidence, "opus5", commit)
    opus_pilot = queue.provider_pilot_from_report(
        pilot_evidence, profile="opus5", git_commit=commit, environment={}
    )
    monkeypatch.setattr(
        queue,
        "codex_auth_probe",
        lambda environment: {"returncode": 0, "stdout": "", "stderr": "invalid_refresh_token"},
    )
    monkeypatch.setattr(
        queue,
        "claude_auth_probe",
        lambda environment: {
            "status": "ready",
            "account": "ssdear@gmail.com",
            "config_dir": "/home/aadivyar/.claude-csd-synthesis",
        },
    )
    ready, blocked = queue.partition_profile_readiness(
        rows,
        {
            "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
            "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com",
        },
        repo=tmp_path,
        provider_pilots={"opus5": opus_pilot},
    )
    assert [r["profile"] for r in ready] == ["opus5"]
    assert blocked[0]["status"] == "pending"
    assert "ChatGPT/Codex OAuth" in blocked[0]["reason"]


def test_profile_readiness_probes_each_provider_profile_once_and_requires_opus_pilot(monkeypatch):
    rows = [
        row for row in queue.build_scope(Path("/repo"))
        if row["profile"] in {"gpt5.6-sol", "opus5"}
    ]
    calls = []

    def probe(environment):
        calls.append(True)
        return {"returncode": 0, "status": "ready", "stdout": "", "stderr": ""}

    monkeypatch.setattr(queue, "codex_auth_probe", probe)
    ready, blocked = queue.partition_profile_readiness(
        rows,
        {
            "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
            "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com",
        },
        repo=Path("/repo"),
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


def test_gemini_environment_keeps_only_the_direct_ai_studio_key(tmp_path):
    row = next(r for r in queue.build_scope(Path("/repo")) if r["profile"] == "gemini3.7-flash")
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
        "GEMINI_API_KEY": "approved-key",
        "GEMINI_API_KEY_BACKUP_1": "unbound-backup",
        "GOOGLE_GENAI_USE_VERTEXAI": "0",
    }
    env = queue.synthesis_environment(row, (2,), inherited, tmp_path)
    assert env["GEMINI_API_KEY"] == "approved-key"
    assert env["CSD_GEMINI_BACKEND"] == "gemini"
    assert env["CSD_GEMINI_MODEL"] == "gemini-3.7-flash"
    assert "GEMINI_API_KEY_BACKUP_1" not in env
    assert not any(
        key in env
        for key in (
            "VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION", "VERTEX_AI_BASE_URL",
            "VERTEX_AI_API_KEY", "VERTEX_AI_ACCESS_TOKEN",
            "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_VERTEX_LOCATION", "GOOGLE_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
        )
    )


def test_exhausted_failure_report_best_compiled_candidate_is_recoverable(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic")
    run_dir = tmp_path / "outputs" / "generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
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
    selection = _fake_compiled_selection(tmp_path, compiled, winning_attempt=40)
    monkeypatch.setattr(queue, "_compiled_selection", lambda repo, candidate: selection)
    monkeypatch.setattr(
        queue, "_report_binding_is_valid", lambda state, candidate, repo: True
    )
    assert queue._compiled_output(tmp_path, row) == compiled


def test_synthesis_exhaustion_with_best_candidate_continues_to_heldout(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic")
    latest = tmp_path / "outputs" / "generated" / row["output_name"] / "latest_run.txt"
    compiled = tmp_path / "compiled" / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("best", encoding="utf-8")
    selection = _fake_compiled_selection(tmp_path, compiled, winning_attempt=40)
    monkeypatch.setattr(queue, "_compiled_selection", lambda repo, candidate: selection)
    monkeypatch.setattr(
        queue, "_report_binding_is_valid", lambda state, candidate, repo: True
    )
    monkeypatch.setattr(queue, "heldout_artifact_is_valid", lambda path, candidate: path.is_file())
    calls = []
    child_environments = []

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
        child_environments.append(kwargs["env"])
        return Process(1 if len(calls) == 1 else 0)

    source_checks = []
    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        gpus=(0,),
        runner=runner,
        admission_check=lambda candidate, **kwargs: source_checks.append(kwargs),
    )
    assert result["status"] == "complete"
    assert len(calls) == 2
    assert source_checks == [{"require_provider": False}]
    assert child_environments[0]["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-synthesis"
    assert "CSD_CLAUDE_CONFIG_DIR" not in child_environments[1]


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

    payload = _bound_gsm_artifact(row, compiled)
    payload["syntax_rate"] = 0.0
    for answer in payload["answers"]:
        answer["generated_answer"] = "same malformed output"
        answer["is_syntax_valid"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)


def _bound_gsm_artifact(row, compiled, *, manifest="manifest-1", indices=None):
    indices = list(indices if indices is not None else queue.expected_heldout_indices(row))
    answers = [{"generated_answer": "x", "source_index": index, "is_correct": False, "is_syntax_valid": True, "constrained_work": 2} for index in indices]
    return {
        "accuracy": 0.0,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": len(answers), "total_constrained_work": 2 * len(answers), "mean_constrained_work": 2.0, "examples_with_constrained_work": len(answers)},
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

    forged_work = json.loads(json.dumps(valid))
    forged_work["answers"][0]["constrained_work"] = 1
    forged_work["answers"][1]["constrained_work"] = 3
    forged_work["metrics"]["total_constrained_work"] = 1000
    forged_work["metrics"]["mean_constrained_work"] = 500.0
    artifact.write_text(json.dumps(forged_work), encoding="utf-8")
    assert not queue.heldout_artifact_is_valid(artifact, row)

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


def test_heldout_validator_accepts_real_writer_four_decimal_work_mean(tmp_path):
    from synthesis.evaluate.baseline_store import build_minimal_baseline_record
    from synthesis.evaluate.evaluator import EvaluationResult

    row = next(
        candidate
        for candidate in queue.build_scope(Path.cwd())
        if candidate["benchmark"] == "gsm_symbolic"
    )
    compiled = tmp_path / "GeneratedCSD.py"
    compiled.write_text("compiled", encoding="utf-8")
    row.update(
        compiled_csd_path=str(compiled),
        compiled_sha256=queue.hash_file(compiled),
        manifest_commit="manifest-1",
    )
    indices = queue.expected_heldout_indices(row)
    samples = [
        {
            "question": f"question {index}",
            "full_output": f"answer {index}",
            "source_index": index,
            "is_correct": False,
            "is_syntax_valid": True,
            "constrained_work": 1 if position == 0 else 0,
        }
        for position, index in enumerate(indices)
    ]
    payload = build_minimal_baseline_record(
        EvaluationResult(
            success=True,
            accuracy=0.0,
            contains_delimiters=False,
            syntax_rate=1.0,
            num_examples=len(samples),
            num_correct=0,
            total_time_seconds=0.0,
            sample_outputs=samples,
        )
    )
    payload["reevaluation_provenance"] = {
        "cell_id": row["cell_id"],
        "dataset": row["dataset"],
        "eval_model": row["eval_model"],
        "sample_size": len(samples),
        "max_steps": row["eval_max_steps"],
        "step_token_budget": row["token_budget"],
        "compiled_csd_path": str(compiled),
        "compiled_csd_sha256": queue.hash_file(compiled),
        "manifest_commit": "manifest-1",
        "evaluated_source_indices": indices,
        "smiles_class": None,
    }
    payload["eval_split"] = {
        "gsm_split_name": "test",
        "gsm_split_file": row["heldout_split_file"],
    }
    artifact = tmp_path / "writer-rounded.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert payload["metrics"]["total_constrained_work"] == 1
    assert payload["metrics"]["mean_constrained_work"] == 0.0204
    assert queue.heldout_artifact_is_valid(artifact, row)


def test_heldout_validator_checks_smiles_trial_counts_and_blank_answers(tmp_path):
    row = _fixture_row("smiles", Path.cwd())
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
    rows = [
        _fixture_row("smiles", tmp_path, cell_suffix="one"),
        _fixture_row("smiles", tmp_path, cell_suffix="two"),
    ]
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


def test_controller_rejects_state_and_export_collisions_with_row_artifacts(tmp_path):
    row = queue.build_scope(tmp_path)[0]
    heldout = tmp_path / row["heldout_output_json"]
    args = queue.controller_parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--state-dir",
            str(heldout.parent),
            "--log",
            str(tmp_path / "controller.log"),
            "--export",
            str(tmp_path / "export.json"),
        ]
    )
    with pytest.raises(queue.ConfigError, match="artifact path collision"):
        queue.validate_controller_artifact_paths(args, [row], tmp_path)

    args.state_dir = tmp_path / "state"
    args.export = heldout
    with pytest.raises(queue.ConfigError, match="artifact path collision"):
        queue.validate_controller_artifact_paths(args, [row], tmp_path)


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


def test_provider_pilot_requires_exact_commit_and_hashed_evidence(tmp_path, monkeypatch):
    commit = "a" * 40
    evidence = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(evidence, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        evidence, profile="opus5", git_commit=commit, environment={}
    )
    monkeypatch.setattr(queue.time, "time", lambda: 1787949000.0)
    assert queue.validate_provider_pilot(
        "opus5", pilot, commit, repo=tmp_path, environment={}
    ) is None
    assert "different code commit" in queue.validate_provider_pilot(
        "opus5", pilot, "c" * 40, repo=tmp_path, environment={}
    )
    evidence.write_text("changed\n", encoding="utf-8")
    assert "evidence" in queue.validate_provider_pilot(
        "opus5", pilot, commit, repo=tmp_path, environment={}
    )


def test_controller_startup_rejects_provider_pilot_that_is_already_stale(
    tmp_path, monkeypatch
):
    commit = "a" * 40
    row = next(
        dict(candidate, git_commit=commit)
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "opus5"
    )
    evidence = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(evidence, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        evidence, profile="opus5", git_commit=commit, environment={}
    )
    monkeypatch.setattr(queue.time, "time", lambda: 1787949000.0 + 25 * 60 * 60)

    with pytest.raises(queue.ConfigError, match="provider pilot is stale"):
        queue.validate_startup_provider_pilots(
            [row],
            {"opus5": pilot},
            repo=tmp_path,
            environment={},
        )


def test_admission_guard_does_not_age_out_startup_validated_immutable_pilot(
    tmp_path, monkeypatch
):
    commit = "a" * 40
    row = next(
        dict(candidate, git_commit=commit)
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "opus5"
    )
    evidence = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(evidence, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        evidence, profile="opus5", git_commit=commit, environment={}
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"provider_pilots": {"opus5": pilot}}), encoding="utf-8"
    )
    environment = {
        "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
        "CSD_CLAUDE_EXPECTED_ACCOUNT": "ssdear@gmail.com",
    }
    now = [1787949000.0]
    monkeypatch.setattr(queue.time, "time", lambda: now[0])
    queue.validate_startup_provider_pilots(
        [row], {"opus5": pilot}, repo=tmp_path, environment=environment
    )
    now[0] += 25 * 60 * 60
    monkeypatch.setattr(queue, "validate_manifest", lambda repo, payload: [row])
    monkeypatch.setattr(
        queue,
        "claude_auth_probe",
        lambda checked: {
            "status": "ready",
            "account": "ssdear@gmail.com",
            "config_dir": "/home/aadivyar/.claude-csd-synthesis",
        },
    )
    guard = queue.make_admission_guard(
        repo=tmp_path,
        manifest_path=manifest,
        expected_manifest_sha256=queue.hash_file(manifest),
        environment=environment,
        clock=lambda: now[0],
    )

    guard(row)


@pytest.mark.parametrize("profile", ["gpt5.6-sol", "gemini3.7-flash", "opus5"])
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
        return _fake_compiled_selection(tmp_path, compiled)

    monkeypatch.setattr(queue, "_validated_compiled_selection", strict_compiled)
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


def test_terminal_loader_rejects_artifact_bound_to_different_compiled_strategy(
    tmp_path
):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["benchmark"] == "gsm_symbolic"
    )
    row.update(manifest_sha256="a" * 64, manifest_commit="a" * 64)
    selected = tmp_path / "selected/GeneratedCSD.py"
    selected.parent.mkdir()
    selected.write_text("selected", encoding="utf-8")
    different = tmp_path / "different/GeneratedCSD.py"
    different.parent.mkdir()
    different.write_text("different", encoding="utf-8")
    artifact = tmp_path / row["heldout_output_json"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = _bound_gsm_artifact(row, different, manifest="a" * 64)
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / f"{row['cell_id']}.json",
        {
            "cell_id": row["cell_id"],
            "status": "complete",
            "manifest_sha256": "a" * 64,
            "manifest_commit": "a" * 64,
            "compiled_csd_path": str(selected),
            "compiled_sha256": queue.hash_file(selected),
            "heldout_output_json": str(artifact),
            "heldout_sha256": queue.hash_file(artifact),
        },
    )

    with pytest.raises(queue.ConfigError, match="incomplete or unbound"):
        queue.load_terminal_results(tmp_path, [row], state_dir)


def test_terminal_loader_records_synthesis_attempts_and_status(
    tmp_path, monkeypatch
):
    row = queue.build_scope(tmp_path)[0]
    row.update(manifest_sha256="a" * 64, manifest_commit="b" * 40)
    runtime = {
        "row_started_at": "2026-09-01T00:00:00Z",
        "synthesis_started_at": "2026-09-01T00:00:00Z",
        "synthesis_finished_at": "2026-09-01T00:10:00Z",
        "synthesis_wall_time_seconds": 600.0,
        "heldout_started_at": "2026-09-01T00:10:00Z",
        "heldout_finished_at": "2026-09-01T00:12:00Z",
        "heldout_wall_time_seconds": 120.0,
        "row_finished_at": "2026-09-01T00:12:00Z",
        "total_wall_time_seconds": 720.0,
        "phase_timing_coverage": "all_phases",
        "attempt_evaluation_times_seconds": [31.25],
        "attempt_timing_coverage": "winning_attempt_only",
        "heldout_evaluator_total_time_seconds": 90.0,
        "heldout_recorded_run_wall_time_seconds": 119.5,
    }
    artifact = tmp_path / row["heldout_output_json"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "metrics": {
                    "evaluator_total_time_seconds": 90.0,
                    "run_wall_time_seconds": 119.5,
                },
                "controller_runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "success_report.json"
    report.write_text(
        json.dumps(
            {
                "total_attempts": 7,
                "evaluation_result": {"total_time_seconds": 31.25},
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / f"{row['cell_id']}.json",
        {
            "cell_id": row["cell_id"],
            "status": "complete",
            "manifest_sha256": "a" * 64,
            "manifest_commit": "b" * 40,
            "compiled_csd_path": str(tmp_path / "GeneratedCSD.py"),
            "compiled_sha256": "c" * 64,
            "heldout_output_json": str(artifact),
            "heldout_sha256": queue.hash_file(artifact),
            "synthesis_report_path": str(report),
            "synthesis_report_sha256": queue.hash_file(report),
            "winning_attempt": 7,
            "row_started_at": "2026-09-01T00:00:00Z",
            "synthesis_started_at": "2026-09-01T00:00:00Z",
            "synthesis_finished_at": "2026-09-01T00:10:00Z",
            "synthesis_wall_time_seconds": 600.0,
            "heldout_started_at": "2026-09-01T00:10:00Z",
            "heldout_finished_at": "2026-09-01T00:12:00Z",
            "heldout_wall_time_seconds": 120.0,
            "row_finished_at": "2026-09-01T00:12:00Z",
            "total_wall_time_seconds": 720.0,
            "phase_timing_coverage": "all_phases",
        },
    )
    monkeypatch.setattr(queue, "_report_binding_is_valid", lambda *args: True)
    monkeypatch.setattr(queue, "heldout_artifact_is_valid", lambda *args: True)

    values = queue.load_terminal_results(tmp_path, [row], state_dir)

    assert values[0]["synthesis_attempts"] == 7
    assert values[0]["synthesis_terminal_status"] == "accepted"
    assert values[0]["runtime"] == runtime


def test_terminal_loader_rejects_selected_synthesis_report_after_mutation(tmp_path):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["benchmark"] == "gsm_symbolic"
    )
    row.update(manifest_sha256="a" * 64, manifest_commit="b" * 40)
    compiled = tmp_path / "selected/GeneratedCSD.py"
    compiled.parent.mkdir()
    compiled.write_text("selected", encoding="utf-8")
    artifact = tmp_path / row["heldout_output_json"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(_bound_gsm_artifact(row, compiled, manifest="b" * 40)),
        encoding="utf-8",
    )
    report = tmp_path / "success_report.json"
    report.write_text("sealed report", encoding="utf-8")
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / f"{row['cell_id']}.json",
        {
            "cell_id": row["cell_id"],
            "status": "complete",
            "phase": "heldout",
            "manifest_sha256": "a" * 64,
            "manifest_commit": "b" * 40,
            "compiled_csd_path": str(compiled),
            "compiled_sha256": queue.hash_file(compiled),
            "synthesis_report_path": str(report),
            "synthesis_report_sha256": queue.hash_file(report),
            "winning_attempt": 1,
            "heldout_output_json": str(artifact),
            "heldout_sha256": queue.hash_file(artifact),
        },
    )
    report.write_text("mutated report", encoding="utf-8")

    with pytest.raises(queue.ConfigError, match="incomplete or unbound"):
        queue.load_terminal_results(tmp_path, [row], state_dir)


def test_report_binding_rejects_hash_valid_but_semantically_invalid_report(tmp_path):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "opus5"
        and candidate["benchmark"] == "gsm_symbolic"
    )
    report = tmp_path / "success_report.json"
    report.write_text("{}", encoding="utf-8")
    state = {
        "synthesis_report_path": str(report),
        "synthesis_report_sha256": queue.hash_file(report),
        "winning_attempt": 999,
    }

    assert not queue._report_binding_is_valid(state, row, tmp_path)


def test_smiles_validator_binds_accuracy_to_unique_valid_count_not_membership_flags(
    tmp_path,
):
    row = _fixture_row("smiles", Path.cwd())
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
            "constrained_work": 1,
        }
        for index in indices
    ]
    payload = {
        "accuracy": 0.14,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": count, "total_constrained_work": count, "mean_constrained_work": 1.0, "examples_with_constrained_work": count},
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
    assert queue.controller_lock_path(tmp_path) == (
        tmp_path / ".context/table5_8/table5_8.controller.lock"
    )
    with queue.controller_lock(tmp_path):
        with pytest.raises(queue.ConfigError, match="already running"):
            with queue.controller_lock(tmp_path):
                pass
        with queue.state_lock(tmp_path):
            queue.write_state(tmp_path / "row.json", {"status": "pending"})


def test_dispatch_polls_surviving_child_without_readmitting_it(tmp_path, monkeypatch):
    row = _fixture_row("smiles", tmp_path)
    snapshot = {0: {"used_mib": 0, "free_mib": 40960, "total_mib": 40960}}
    state_reads = [
        {
            "cell_id": row["cell_id"],
            "status": "running",
            "phase": "synthesis",
            "pid": 123,
            "pid_start": "one",
            "assigned_gpus": [0],
            "reservation_mib": 16384,
        },
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


def test_dispatch_reserves_surviving_child_before_any_new_admission(
    tmp_path, monkeypatch
):
    survivor, pending = (
        _fixture_row("smiles", tmp_path, cell_suffix="survivor"),
        _fixture_row("smiles", tmp_path, cell_suffix="pending"),
    )
    survivor = dict(survivor, cell_id="survivor")
    pending = dict(pending, cell_id="pending")
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / "survivor.json",
        {
            "cell_id": "survivor",
            "status": "running",
            "phase": "synthesis",
            "pid": 123,
            "pid_start": "alive",
            "assigned_gpus": [0],
            "reservation_mib": 16384,
        },
    )
    monkeypatch.setattr(
        queue,
        "child_is_same_process",
        lambda state: state.get("cell_id") == "survivor",
    )
    launches = []
    monkeypatch.setattr(
        queue,
        "run_row",
        lambda row, **kwargs: launches.append(row["cell_id"])
        or {"cell_id": row["cell_id"], "status": "complete"},
    )
    monkeypatch.setattr(
        queue.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(RuntimeError("stop after first poll")),
    )
    snapshot = {0: {"total_mib": 40960, "free_mib": 33000}}
    with pytest.raises(RuntimeError, match="first poll"):
        queue.dispatch(
            [pending, survivor],
            repo=tmp_path,
            python=Path("python"),
            state_dir=state_dir,
            allowed=(0,),
            snapshot=lambda: snapshot,
            poll_seconds=0.1,
        )
    assert launches == []


def test_dispatch_rechecks_admission_after_gpu_fit_before_launch(
    tmp_path, monkeypatch
):
    row = _fixture_row("smiles", tmp_path)
    launches = []
    provider_checks = []
    monkeypatch.setattr(
        queue,
        "run_row",
        lambda candidate, **kwargs: launches.append(candidate["cell_id"]),
    )

    def block(candidate, *, require_provider):
        provider_checks.append(require_provider)
        raise queue.ConfigError(f"fresh admission blocked: {candidate['cell_id']}")

    monkeypatch.setattr(
        queue.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(RuntimeError("stop after blocked poll")),
    )
    snapshot = {0: {"total_mib": 40960, "free_mib": 40960}}
    with pytest.raises(RuntimeError, match="stop after blocked poll"):
        queue.dispatch(
            [row],
            repo=tmp_path,
            python=Path("python"),
            state_dir=tmp_path / "state",
            allowed=(0,),
            snapshot=lambda: snapshot,
            admission_check=block,
        )
    assert launches == []
    assert provider_checks == [True]


def test_dispatch_rejects_unknown_row_state_before_any_admission_or_launch(
    tmp_path, monkeypatch
):
    row = _fixture_row("smiles", tmp_path)
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / f"{row['cell_id']}.json",
        {
            "cell_id": row["cell_id"],
            "status": "unexpected",
            "phase": "synthesis",
        },
    )
    admissions = []
    launches = []
    monkeypatch.setattr(
        queue,
        "run_row",
        lambda candidate, **kwargs: launches.append(candidate["cell_id"]),
    )

    with pytest.raises(queue.ConfigError, match="invalid queue state"):
        queue.dispatch(
            [row],
            repo=tmp_path,
            python=Path("python"),
            state_dir=state_dir,
            allowed=(0,),
            snapshot=lambda: {
                0: {"total_mib": 40960, "free_mib": 40960}
            },
            admission_check=lambda candidate, **kwargs: admissions.append(kwargs),
        )

    assert admissions == []
    assert launches == []


def test_dispatch_revalidates_terminal_state_without_gpu_or_provider_admission(
    tmp_path, monkeypatch
):
    row = _fixture_row("smiles", tmp_path)
    state_dir = tmp_path / "state"
    queue.write_state(
        state_dir / f"{row['cell_id']}.json",
        {"cell_id": row["cell_id"], "status": "complete", "phase": "heldout"},
    )
    calls = []
    monkeypatch.setattr(
        queue,
        "run_row",
        lambda candidate, **kwargs: calls.append(kwargs)
        or {"cell_id": candidate["cell_id"], "status": "complete"},
    )
    monkeypatch.setattr(
        queue.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(
            AssertionError("terminal state waited for a GPU")
        ),
    )

    results = queue.dispatch(
        [row],
        repo=tmp_path,
        python=Path("python"),
        state_dir=state_dir,
        allowed=(0,),
        snapshot=lambda: {},
        admission_check=lambda candidate, **kwargs: pytest.fail(
            "terminal state must not require provider admission"
        ),
    )

    assert results == [{"cell_id": row["cell_id"], "status": "complete"}]
    assert calls[0]["gpus"] == ()


def test_controller_passes_running_and_blocked_rows_to_dispatch_without_overwrite(
    tmp_path, monkeypatch
):
    from scripts.runtime import run_cold_synthesis_queue as cold_queue

    rows = queue.build_scope(tmp_path)[:2]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
                    {
                        "provider_pilots": {},
                        "provider_pilot_sha256": queue.provider_pilots_sha256({}),
                        "execution_source_sha256": "c" * 64,
                        "git_commit": "b" * 40,
                        "external_runtime": {
                                "eval_model": {
                                    "revision": "d" * 40,
                                    "snapshot_path": "/cache/snapshot",
                                    "snapshot_sha256": "f" * 64,
                                    "snapshot_file_count": 10,
                            },
                            "spider_data": {
                                "path": "/data/spider",
                                "sha256": "e" * 64,
                                "file_count": 922,
                            },
                        },
                    }
        ),
        encoding="utf-8",
    )
    manifest_sha = queue.hash_file(manifest)
    state_dir = tmp_path / "state"
    running = {
        "cell_id": rows[0]["cell_id"],
        "status": "running",
        "phase": "synthesis",
        "phase": "synthesis",
        "pid": 123,
        "pid_start": "alive",
        "assigned_gpus": [0],
        "reservation_mib": 16384,
    }
    queue.write_state(state_dir / f"{rows[0]['cell_id']}.json", running)
    queue.write_state(
        state_dir / "controller.json",
        {
            "status": "validated",
            "manifest_sha256": manifest_sha,
            "provider_pilot_sha256": queue.provider_pilots_sha256({}),
        },
    )
    guard_calls = []
    pilot_freshness_checks = []
    captured_rows = []
    gpu_snapshot_calls = []
    monkeypatch.setattr(queue, "validate_manifest", lambda repo, payload: rows)
    monkeypatch.setattr(
        queue,
        "make_admission_guard",
        lambda **kwargs: lambda row, **guard_kwargs: guard_calls.append(row["cell_id"]),
    )
    monkeypatch.setattr(
        queue,
        "validate_startup_provider_pilots",
        lambda *args, **kwargs: pilot_freshness_checks.append(
            kwargs["require_freshness"]
        ),
    )
    monkeypatch.setattr(
        queue,
        "dispatch",
        lambda candidates, **kwargs: (
            kwargs["snapshot"](),
            captured_rows.extend(candidates),
            [dict(row, status="complete") for row in candidates],
        )[-1],
    )
    monkeypatch.setattr(
        cold_queue,
        "gpu_memory_snapshot",
        lambda executable: gpu_snapshot_calls.append(executable) or {},
    )
    monkeypatch.setattr(
        queue, "load_terminal_results", lambda repo, candidates, state_dir: []
    )
    monkeypatch.setattr(queue, "export_results", lambda rows, values, output: None)
    args = queue.controller_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--state-dir",
            str(state_dir),
            "--log",
            str(tmp_path / "controller.log"),
            "--export",
            str(tmp_path / "export.json"),
        ]
    )

    assert queue.controller_main(args) == 0
    assert [row["cell_id"] for row in captured_rows] == [
        row["cell_id"] for row in rows
    ]
    assert guard_calls == []
    assert pilot_freshness_checks == [False]
    assert gpu_snapshot_calls == ["nvidia-smi"]
    assert queue.read_state(state_dir / f"{rows[0]['cell_id']}.json") == running


def test_admission_guard_revalidates_manifest_each_time_and_caches_live_auth(
    tmp_path, monkeypatch
):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5")
    row["git_commit"] = "a" * 40
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fixed":true}', encoding="utf-8")
    expected_sha = queue.hash_file(manifest)
    validations = []
    auth_calls = []
    gate_calls = []
    monkeypatch.setattr(
        queue,
        "validate_manifest",
        lambda repo, payload: validations.append(payload) or [row],
    )
    monkeypatch.setattr(
        queue,
        "claude_auth_probe",
        lambda environment: auth_calls.append(True)
        or {
            "status": "ready",
            "account": "ssdear@gmail.com",
            "config_dir": "/home/aadivyar/.claude-csd-synthesis",
        },
    )
    monkeypatch.setattr(
        queue,
        "profile_block_reason",
        lambda candidate, environment, **kwargs: gate_calls.append(kwargs) or None,
    )
    guard = queue.make_admission_guard(
        repo=tmp_path,
        manifest_path=manifest,
        expected_manifest_sha256=expected_sha,
        environment={},
        auth_ttl_seconds=300,
        clock=lambda: 1000.0,
    )
    guard(row)
    guard(row)
    assert len(validations) == 2
    assert len(gate_calls) == 2
    assert len(auth_calls) == 1
    manifest.write_text('{"fixed":false}', encoding="utf-8")
    with pytest.raises(queue.ConfigError, match="manifest changed"):
        guard(row)


def test_logged_child_uses_append_file_and_its_own_process_group(
    tmp_path, monkeypatch
):
    captured = {}

    class Process:
        pid = 123
        returncode = 0

        def communicate(self):
            return None, None

    def popen(argv, **kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write("child output\n")
        kwargs["stdout"].flush()
        return Process()

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    log = tmp_path / "logs/row.log"
    process = queue.start_logged_child(
        ["python", "child.py"], cwd=tmp_path, env={}, log_path=log
    )
    queue.wait_logged_child(process)
    assert captured["stdout"] is not subprocess.PIPE
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["start_new_session"] is True
    assert log.read_text(encoding="utf-8") == "child output\n"


def test_child_start_failure_becomes_durable_failed_state(tmp_path):
    row = _fixture_row("smiles", tmp_path)

    def runner(argv, **kwargs):
        raise OSError("cannot start")

    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("python"),
        state_dir=tmp_path / "state",
        gpus=(0,),
        reservation_mib=16384,
        runner=runner,
    )
    assert result["status"] == "failed"
    assert "failed to start" in result["reason"]
    assert queue.read_state(
        tmp_path / "state" / f"{row['cell_id']}.json"
    )["status"] == "failed"


def test_provider_pilot_is_parsed_from_one_attempt_report_and_requires_fresh_binding(tmp_path, monkeypatch):
    commit = "a" * 40
    path = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    report = _write_real_pilot_report(path, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        path, profile="opus5", git_commit=commit, environment={}
    )
    assert pilot["attempt_count"] == 1
    assert pilot["evidence_sha256"] == queue.hash_file(path)
    report["timestamp"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(queue.time, "time", lambda: 1787949000.0)
    assert queue.validate_provider_pilot(
        "opus5", pilot, commit, repo=tmp_path, environment={}
    )


def test_manifest_source_closure_hashes_all_tracked_synthesis_files(tmp_path):
    paths = queue.execution_source_paths(Path.cwd())
    assert "synthesis/run_synthesis.py" in paths
    assert "synthesis/generate/generator.py" in paths
    assert any(path.startswith("synthesis/") for path in paths)


def test_exhausted_candidate_requires_exact_finite_evaluation_and_attempt_number(tmp_path, monkeypatch):
    row = next(r for r in queue.build_scope(tmp_path) if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic")
    run_dir = tmp_path / "outputs" / "generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    compiled = run_dir / "compiled" / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("bad", encoding="utf-8")
    report = {"total_attempts": 40, "attempts": [{"attempt_number": 0, "compilation": {"success": True, "output_dir": str(compiled.parent)}, "evaluation": {"num_examples": 1, "accuracy": 2.0, "syntax_rate": -1.0}}]}
    (run_dir / "results" / "failure_report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *args, **kwargs: True)
    assert queue._validated_compiled_output(tmp_path, row["output_name"], min_accuracy=0.1, min_syntax_rate=0.1, job=row) is None


def test_synthesis_report_must_match_manifest_execution_source_snapshot():
    row = next(
        candidate
        for candidate in queue.build_scope(Path("/repo"))
        if candidate["profile"] == "opus5"
        and candidate["benchmark"] == "gsm_symbolic"
    )
    row.update(git_commit="a" * 40, execution_source_sha256="1" * 64)
    report = {
        "total_attempts": 1,
        "run_configuration": {
            "task_description": row["task"],
            "output_name": row["output_name"],
            "git_commit": row["git_commit"],
            "execution_source_sha256": "0" * 64,
            "max_iterations": row["max_iterations"],
            "author_model": {
                "backend": row["generation_backend"],
                "model": row["generation_model"],
                "max_new_tokens": row["synthesis_max_tokens"],
                "reasoning_budget_tokens": row["synthesis_reasoning_budget"],
            },
            "evaluation": {
                "dataset": row["dataset"],
                "eval_model": row["eval_model"],
                "eval_sample_size": row["eval_sample_size"],
                "eval_max_steps": row["eval_max_steps"],
                "eval_step_token_budget": row["token_budget"],
                "eval_max_seconds_per_example": row["eval_max_seconds"],
                "min_examples_before_threshold_stop": row["eval_sample_size"],
                "split_provenance": {
                    "bar_split_name": "train",
                    "gsm_split_name": "train",
                    "gsm_split_file": row["heldout_split_file"],
                },
            },
            "synthesis_controls": {
                "adaptive_helper_mask": row["adaptive_helper_mask"],
                "helper_selection_policy": row["helper_selection_policy"],
                "refinement_beam_size": row["beam_size"],
            },
            "thresholds": {
                "min_accuracy": row["min_accuracy"],
                "min_syntax_rate": row["min_syntax_rate"],
            },
        },
    }

    assert not queue._report_matches_row(report, row, require_exhausted=False)
    report["run_configuration"]["execution_source_sha256"] = "1" * 64
    assert queue._report_matches_row(report, row, require_exhausted=False)


def test_compiled_output_rejects_latest_run_outside_or_symlinked_outside_row_root(
    tmp_path
):
    row = next(
        candidate
        for candidate in queue.build_scope(tmp_path)
        if candidate["profile"] == "opus5"
    )
    output_root = tmp_path / "outputs/generated" / row["output_name"]
    output_root.mkdir(parents=True)
    latest = output_root / "latest_run.txt"
    outside = tmp_path / f"{row['output_name']}_20260829_120000_abcdef"
    outside.mkdir()
    latest.write_text(str(outside), encoding="utf-8")

    assert queue._compiled_output(tmp_path, row) is None

    escaped_link = output_root / f"{row['output_name']}_20260829_120001_abcdef"
    escaped_link.symlink_to(outside, target_is_directory=True)
    latest.write_text(str(escaped_link), encoding="utf-8")
    assert queue._compiled_output(tmp_path, row) is None


def test_scope_records_effective_provider_limits_separately_from_requested_budget():
    rows = queue.build_scope(Path("/repo"))
    assert {row["effective_output_tokens"] for row in rows if row["profile"] == "opus5"} == {64000}
    assert {row["effective_thinking_tokens"] for row in rows if row["profile"] == "opus5"} == {48000}
    assert all(row["effective_output_tokens"] is None for row in rows if row["profile"] == "gpt5.6-sol")


def test_missing_compiled_fingerprint_becomes_failed_state(tmp_path, monkeypatch):
    row = _fixture_row("smiles", tmp_path)
    queue.write_state(tmp_path / "state" / f"{row['cell_id']}.json", {"cell_id": row["cell_id"], "status": "running", "phase": "heldout", "compiled_csd_path": str(tmp_path / "missing.py"), "compiled_sha256": "0" * 64})
    monkeypatch.setattr(queue, "heldout_command", lambda *args: [])
    result = queue.run_row(row, repo=tmp_path, python=Path("python"), state_dir=tmp_path / "state", gpus=(0,), runner=lambda *args, **kwargs: pytest.fail("must not launch"))
    assert result["status"] == "failed"


def test_dry_run_does_not_probe_provider_readiness(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    row = queue.build_scope(tmp_path)[0]
    monkeypatch.setattr(queue, "validate_manifest", lambda repo, payload: [row])
    monkeypatch.setattr(queue, "partition_profile_readiness", lambda *args, **kwargs: pytest.fail("provider readiness was called"))
    args = queue.controller_parser().parse_args(["--manifest", str(manifest), "--state-dir", str(tmp_path / "state"), "--log", str(tmp_path / "run.log"), "--dry-run"])
    assert queue.controller_main(args) == 0


def _real_pilot_report(profile, commit, *, output_name="pilot", compiled_dir="/tmp/compiled"):
    backend, model = {
        "opus5": ("claude", "claude-opus-5"),
        "gpt5.6-sol": ("codex", "gpt-5.6-sol"),
        "gemini3.7-flash": ("gemini", "gemini-3.7-flash"),
    }[profile]
    author_route = {
        "opus5": {
            "auth_mode": "claude_code_max",
            "config_dir": "/home/aadivyar/.claude-csd-synthesis",
            "expected_account": "ssdear@gmail.com",
            "account_verified": True,
        },
        "gpt5.6-sol": {
            "auth_mode": "chatgpt",
            "account_verified": True,
        },
        "gemini3.7-flash": {
            "auth_mode": "gemini_api_key",
            "api_key_sha256": queue.sha256_text("gemini-key"),
        },
    }[profile]
    return {
        "timestamp": "2026-08-28T20:25:50.576728+00:00",
        "total_attempts": 1,
        "run_configuration": {
            "git_commit": commit,
            "execution_source_sha256": "d" * 64,
            "python_runtime": _test_python_runtime(),
            "output_name": output_name,
            "max_iterations": 1,
            "task_description": queue.TASKS["gsm_symbolic"],
            "author_model": {
                "backend": backend,
                "model": model,
                "max_new_tokens": 32768,
                "reasoning_budget_tokens": 4096,
                "route": author_route,
            },
            "evaluation": {
                "dataset": "gsm_symbolic",
                "eval_model": queue.EVAL_MODEL,
                "eval_sample_size": 1,
                "eval_max_steps": 900,
                "eval_step_token_budget": 1,
                "eval_max_seconds_per_example": 600.0,
                "min_examples_before_threshold_stop": 1,
            },
            "synthesis_controls": {
                "adaptive_helper_mask": True,
                "helper_selection_policy": "bandit",
                "refinement_beam_size": 2,
            },
        },
        "attempts": [
            {
                "attempt_number": 1,
                "strategy_code": "generated := generatedPrefix; cost := 1;",
                "verification": {"success": True, "error_count": 0},
                "compilation": {"success": True, "output_dir": str(compiled_dir)},
                "evaluation": {
                    "success": True,
                    "early_stopped": False,
                    "num_examples": 1,
                    "num_correct": 0,
                    "accuracy_denominator": 1,
                    "accuracy": 0.0,
                    "syntax_rate": 1.0,
                    "sample_outputs": [
                        {
                            "actual": "42",
                            "is_correct": False,
                            "is_syntax_valid": True,
                        }
                    ],
                },
            }
        ],
    }


def _write_real_pilot_report(path, profile, commit):
    output_name = path.parents[1].name
    compiled_dir = path.parents[1] / "python" / output_name
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "GeneratedCSD.py").write_text("compiled", encoding="utf-8")
    report = _real_pilot_report(
        profile,
        commit,
        output_name=output_name,
        compiled_dir=compiled_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_real_pilot_parser_uses_nested_run_report_and_requires_verifier_eval(tmp_path):
    commit = "a" * 40
    report_path = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    report = _write_real_pilot_report(report_path, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        report_path, profile="opus5", git_commit=commit, environment={}
    )
    assert pilot["backend"] == "claude"
    assert pilot["model"] == "claude-opus-5"
    assert pilot["verification_status"] == "success"
    assert pilot["evaluation_status"] == "success"
    assert pilot["expected_account"] == "ssdear@gmail.com"
    report["attempts"][0]["evaluation"] = None
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(queue.ConfigError, match="evaluation"):
        queue.provider_pilot_from_report(
            report_path, profile="opus5", git_commit=commit, environment={}
        )


def test_real_pilot_parser_accepts_the_production_success_report_shape(tmp_path):
    commit = "a" * 40
    report_path = tmp_path / "outputs/generated/pilot/results/success_report.json"
    report = _write_real_pilot_report(report_path, "opus5", commit)
    attempt = report.pop("attempts")[0]
    run_root = report_path.parents[1]
    dafny_file = run_root / "dafny" / "pilot.dfy"
    dafny_file.parent.mkdir(parents=True, exist_ok=True)
    dafny_file.write_text("method Pilot() {}\n", encoding="utf-8")
    report.update(
        {
            "strategy_code": attempt["strategy_code"],
            "compiled_dir": attempt["compilation"]["output_dir"],
            "dafny_file": str(dafny_file),
            "dafny_file_canonical": str(dafny_file.resolve()),
            "evaluation_result": attempt["evaluation"],
            "sample_outputs": attempt["evaluation"]["sample_outputs"],
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    pilot = queue.provider_pilot_from_report(
        report_path, profile="opus5", git_commit=commit, environment={}
    )

    assert pilot["attempt_count"] == 1
    assert pilot["verification_status"] == "success"
    assert pilot["evaluation_status"] == "success"


def test_provider_pilot_rejects_report_route_identity_relabel(tmp_path):
    commit = "a" * 40
    report_path = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    report = _write_real_pilot_report(report_path, "opus5", commit)
    report["run_configuration"]["author_model"]["route"][
        "expected_account"
    ] = "another-account@example.com"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(queue.ConfigError, match="author route identity"):
        queue.provider_pilot_from_report(
            report_path, profile="opus5", git_commit=commit, environment={}
        )


def test_gemini37_pilot_binds_the_active_ai_studio_key(tmp_path):
    commit = "a" * 40
    report_path = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(report_path, "gemini3.7-flash", commit)
    pilot = queue.provider_pilot_from_report(
        report_path,
        profile="gemini3.7-flash",
        git_commit=commit,
        environment={"GEMINI_API_KEY": "gemini-key"},
    )
    assert pilot["backend"] == "gemini"
    assert pilot["model"] == "gemini-3.7-flash"
    assert pilot["api_key_sha256"] == queue.sha256_text("gemini-key")
    assert "gemini-key" not in json.dumps(pilot)


def test_provider_pilot_validation_reparses_report_and_rejects_fabricated_fields(
    tmp_path, monkeypatch
):
    commit = "a" * 40
    repo = tmp_path
    report_path = repo / "outputs/generated/pilot/results/failure_report.json"
    _write_real_pilot_report(report_path, "opus5", commit)
    pilot = queue.provider_pilot_from_report(
        report_path, profile="opus5", git_commit=commit, environment={}
    )
    monkeypatch.setattr(queue.time, "time", lambda: 1787949000.0)
    assert queue.validate_provider_pilot(
        "opus5", pilot, commit, repo=repo, environment={}
    ) is None
    forged = dict(pilot, evaluation_status="rejected")
    assert queue.validate_provider_pilot(
        "opus5", forged, commit, repo=repo, environment={}
    ) is not None


def test_claude_auth_probe_parses_exact_first_party_max_json(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "loggedIn": True,
                        "email": "ssdear@gmail.com",
                        "authMethod": "claude.ai",
                        "apiProvider": "firstParty",
                        "subscriptionType": "max",
                    }
                ),
                "stderr": "",
            },
        )()

    monkeypatch.setattr(queue.subprocess, "run", run)
    result = queue.claude_auth_probe({})
    assert result == {
        "status": "ready",
        "account": "ssdear@gmail.com",
        "config_dir": "/home/aadivyar/.claude-csd-synthesis",
    }
    assert calls[0][0][-2:] == ["status", "--json"]


def test_codex_probe_must_complete_the_sentinel_not_only_report_login(tmp_path, monkeypatch):
    row = next(
        row
        for row in queue.build_scope(tmp_path)
        if row["profile"] == "gpt5.6-sol"
    )
    row["git_commit"] = "a" * 40
    monkeypatch.setattr(queue, "validate_provider_pilot", lambda *args, **kwargs: None)
    reason = queue.profile_block_reason(
        row,
        {},
        repo=tmp_path,
        provider_pilots={},
        cached_probes={
            "gpt5.6-sol": {
                "status": "blocked",
                "returncode": 0,
                "stdout": "Logged in using ChatGPT",
                "stderr": "",
            }
        },
    )
    assert reason == "Pi ChatGPT/Codex OAuth is unavailable or invalid"


def test_gemini_api_key_probe_lists_the_exact_model_without_exposing_key(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"models": [{"name": "models/gemini-3.7-flash"}]}
            ).encode("utf-8")

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(queue.urllib.request, "urlopen", urlopen)
    result = queue.gemini_api_key_probe({"GEMINI_API_KEY": "gemini-key"})
    assert result == {
        "status": "ready",
        "model": "gemini-3.7-flash",
        "api_key_sha256": queue.sha256_text("gemini-key"),
    }
    assert "gemini-key" not in captured["url"]
    assert captured["headers"]["X-goog-api-key"] == "gemini-key"
    assert captured["timeout"] == 30


def test_campaign_environment_adds_canonical_nonsecret_provider_routes(tmp_path):
    private_env = tmp_path / ".env"
    private_env.write_text(
        "GEMINI_API_KEY='gemini-key'\nOPENAI_API_KEY=must-not-load\n",
        encoding="utf-8",
    )
    original = {"PATH": "/bin"}
    loaded = queue.campaign_environment(original, credential_file=private_env)
    assert loaded == {
        "PATH": "/bin",
        "GEMINI_API_KEY": "gemini-key",
        "CSD_PI_NODE_EXECUTABLE": str(queue.CANONICAL_PI_NODE_EXECUTABLE),
        "CSD_PI_BRIDGE_PATH": str(queue.CANONICAL_PI_BRIDGE_PATH),
        "CSD_PI_AUTH_PATH": str(queue.CANONICAL_PI_AUTH_PATH),
        "CSD_CLAUDE_CONFIG_DIR": str(queue.CANONICAL_CLAUDE_CONFIG_DIR),
        "CSD_CLAUDE_EXPECTED_ACCOUNT": queue.CANONICAL_CLAUDE_EXPECTED_ACCOUNT,
    }
    queue.validate_profile_gates(queue.build_scope(Path("/repo")), loaded)
    assert original == {"PATH": "/bin"}


def test_gemini_author_route_binds_only_key_fingerprint():
    route = queue.expected_author_route(
        "gemini3.7-flash", {"GEMINI_API_KEY": "gemini-key"}
    )
    assert route == {
        "auth_mode": "gemini_api_key",
        "api_key_sha256": queue.sha256_text("gemini-key"),
    }
    assert "gemini-key" not in json.dumps(route)


def test_pilot_report_requires_its_real_compiled_artifact(tmp_path):
    commit = "a" * 40
    report_path = tmp_path / "outputs/generated/pilot/results/failure_report.json"
    report = _write_real_pilot_report(report_path, "opus5", commit)
    compiled = Path(report["attempts"][0]["compilation"]["output_dir"]) / "GeneratedCSD.py"
    compiled.unlink()
    with pytest.raises(queue.ConfigError, match="compiled artifact"):
        queue.provider_pilot_from_report(
            report_path, profile="opus5", git_commit=commit, environment={}
        )


def test_pilot_report_accepts_real_timestamped_run_directory_layout(tmp_path):
    commit = "a" * 40
    run_root = tmp_path / "outputs/generated/pilot_20260828_123456_deadbeef"
    report_path = run_root / "results/failure_report.json"
    report = _write_real_pilot_report(report_path, "opus5", commit)
    old_compiled = Path(report["attempts"][0]["compilation"]["output_dir"])
    compiled = run_root / "python/pilot"
    compiled.mkdir(parents=True)
    (compiled / "GeneratedCSD.py").write_text("compiled", encoding="utf-8")
    report["run_configuration"]["output_name"] = "pilot"
    report["attempts"][0]["compilation"]["output_dir"] = str(compiled)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert old_compiled != compiled
    pilot = queue.provider_pilot_from_report(
        report_path, profile="opus5", git_commit=commit, environment={}
    )
    assert pilot["compiled_csd_sha256"] == queue.hash_file(
        compiled / "GeneratedCSD.py"
    )


def test_exhaustion_rejects_duplicate_attempt_numbers(tmp_path, monkeypatch):
    row = next(
        r
        for r in queue.build_scope(tmp_path)
        if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic"
    )
    row["git_commit"] = "a" * 40
    run_dir = tmp_path / "outputs/generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    compiled = run_dir / "compiled/GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("compiled", encoding="utf-8")
    attempt = {
        "attempt_number": 1,
        "compilation": {"success": True, "output_dir": str(compiled.parent)},
        "evaluation": {
            "num_examples": row["eval_sample_size"],
            "accuracy": 0.5,
            "syntax_rate": 0.9,
        },
    }
    report = {"total_attempts": 40, "attempts": [attempt, dict(attempt)]}
    (run_dir / "results/failure_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *a, **k: True)
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=0.1,
        min_syntax_rate=0.1,
        job=row,
    ) is None


def test_exhaustion_requires_all_declared_attempt_records(tmp_path, monkeypatch):
    row = next(
        r
        for r in queue.build_scope(tmp_path)
        if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic"
    )
    row["git_commit"] = "a" * 40
    run_dir = tmp_path / "outputs/generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    compiled = run_dir / "compiled/GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("compiled", encoding="utf-8")
    report = {
        "total_attempts": row["max_iterations"],
        "attempts": [
            {
                "attempt_number": 1,
                "compilation": {
                    "success": True,
                    "output_dir": str(compiled.parent),
                },
                "evaluation": {
                    "num_examples": row["eval_sample_size"],
                    "accuracy": 0.5,
                    "syntax_rate": 0.9,
                },
            }
        ],
    }
    (run_dir / "results/failure_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *a, **k: True)
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=0.1,
        min_syntax_rate=0.1,
        job=row,
    ) is None


def test_success_report_requires_full_metrics_and_run_local_compiled_path(
    tmp_path, monkeypatch
):
    row = next(
        r
        for r in queue.build_scope(tmp_path)
        if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic"
    )
    run_dir = tmp_path / "outputs/generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    compiled = run_dir / "python" / row["output_name"] / "GeneratedCSD.py"
    compiled.parent.mkdir(parents=True)
    compiled.write_text("compiled", encoding="utf-8")
    report = {
        "total_attempts": 1,
        "compiled_dir": str(compiled.parent),
        "evaluation_result": {
            "success": True,
            "early_stopped": False,
            "num_examples": row["eval_sample_size"],
            "num_correct": row["eval_sample_size"],
            "accuracy_denominator": row["eval_sample_size"],
            "accuracy": 1.0,
            "syntax_rate": 1.0,
            "sample_outputs": [
                {"is_correct": True, "is_syntax_valid": True}
                for _ in range(row["eval_sample_size"])
            ],
        },
    }
    report_path = run_dir / "results/success_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *a, **k: True)
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) == compiled

    selection = queue._validated_compiled_selection(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    )
    assert selection == {
        "compiled_csd_path": compiled,
        "report_path": report_path,
        "report_sha256": queue.hash_file(report_path),
        "winning_attempt": 1,
    }

    report["evaluation_result"]["accuracy"] = 0.5
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) is None
    report["evaluation_result"]["accuracy"] = 1.0

    report["evaluation_result"]["success"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) is None
    report["evaluation_result"]["success"] = True

    outside = tmp_path / "outside/GeneratedCSD.py"
    outside.parent.mkdir()
    outside.write_text("outside", encoding="utf-8")
    report["compiled_dir"] = str(outside.parent)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) is None


def test_failure_report_rejects_compiled_candidate_outside_current_run(
    tmp_path, monkeypatch
):
    row = next(
        r
        for r in queue.build_scope(tmp_path)
        if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic"
    )
    run_dir = tmp_path / "outputs/generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    outside = tmp_path / "outside/GeneratedCSD.py"
    outside.parent.mkdir()
    outside.write_text("outside", encoding="utf-8")
    attempts = []
    for number in range(1, row["max_iterations"] + 1):
        attempts.append(
            {
                "attempt_number": number,
                "compilation": {
                    "success": number == 1,
                    "output_dir": str(outside.parent) if number == 1 else None,
                },
                "evaluation": {
                    "num_examples": row["eval_sample_size"],
                    "accuracy": 0.0,
                    "syntax_rate": 0.0,
                },
            }
        )
    (run_dir / "results/failure_report.json").write_text(
        json.dumps({"total_attempts": row["max_iterations"], "attempts": attempts}),
        encoding="utf-8",
    )
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *a, **k: True)
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) is None


def test_failure_report_accepts_compiler_generated_suffixed_output_dir(
    tmp_path, monkeypatch
):
    row = next(
        r
        for r in queue.build_scope(tmp_path)
        if r["profile"] == "opus5" and r["benchmark"] == "gsm_symbolic"
    )
    run_dir = tmp_path / "outputs/generated" / row["output_name"] / f"{row['output_name']}_20260829_120000_abcdef"
    (run_dir / "results").mkdir(parents=True)
    compiled = (
        run_dir
        / "python"
        / f"{row['output_name']}_20260828_123456_deadbe"
        / "GeneratedCSD.py"
    )
    compiled.parent.mkdir(parents=True)
    compiled.write_text("compiled", encoding="utf-8")
    attempts = []
    for number in range(1, row["max_iterations"] + 1):
        attempts.append(
            {
                "attempt_number": number,
                "compilation": {
                    "success": number == 1,
                    "output_dir": str(compiled.parent) if number == 1 else None,
                },
                "verification": {"success": number == 1},
                "evaluation": {
                    "success": number == 1,
                    "early_stopped": False,
                    "num_examples": row["eval_sample_size"],
                    "num_correct": 0,
                    "accuracy_denominator": row["eval_sample_size"],
                    "accuracy": 0.0,
                    "syntax_rate": 0.0,
                    "sample_outputs": (
                        [
                            {"is_correct": False, "is_syntax_valid": False}
                            for _ in range(row["eval_sample_size"])
                        ]
                        if number == 1
                        else []
                    ),
                },
            }
        )
    (run_dir / "results/failure_report.json").write_text(
        json.dumps({"total_attempts": row["max_iterations"], "attempts": attempts}),
        encoding="utf-8",
    )
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    monkeypatch.setattr(queue, "_report_matches_row", lambda *a, **k: True)
    assert queue._validated_compiled_output(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    ) == compiled
    selection = queue._validated_compiled_selection(
        tmp_path,
        row["output_name"],
        min_accuracy=row["min_accuracy"],
        min_syntax_rate=row["min_syntax_rate"],
        job=row,
    )
    report_path = run_dir / "results/failure_report.json"
    assert selection["report_path"] == report_path
    assert selection["report_sha256"] == queue.hash_file(report_path)
    assert selection["winning_attempt"] == 1


def test_restart_recovery_hash_pins_compiled_before_heldout_launch(
    tmp_path, monkeypatch
):
    row = _fixture_row("smiles", tmp_path)
    row.update(manifest_sha256="manifest", manifest_commit="manifest")
    state_dir = tmp_path / "state"
    state_path = state_dir / f"{row['cell_id']}.json"
    latest = tmp_path / "outputs/generated" / row["output_name"] / "latest_run.txt"
    latest.parent.mkdir(parents=True)
    latest.write_text("fresh", encoding="utf-8")
    compiled = tmp_path / "compiled/GeneratedCSD.py"
    compiled.parent.mkdir()
    compiled.write_text("compiled", encoding="utf-8")
    queue.write_state(
        state_path,
        {
            "cell_id": row["cell_id"],
            "status": "running",
            "phase": "synthesis",
            "manifest_sha256": "manifest",
            "manifest_commit": "manifest",
            "pid": 123,
            "pid_start": "old",
            "output_before": None,
        },
    )
    monkeypatch.setattr(queue, "child_is_same_process", lambda state: False)
    selection = _fake_compiled_selection(tmp_path, compiled)
    monkeypatch.setattr(queue, "_compiled_selection", lambda repo, candidate: selection)
    monkeypatch.setattr(
        queue, "_report_binding_is_valid", lambda state, candidate, repo: True
    )
    monkeypatch.setattr(
        queue, "heldout_artifact_is_valid", lambda path, candidate: path.is_file()
    )

    class Process:
        pid = 456
        returncode = 0

        def communicate(self):
            output = Path(command[command.index("--output-json") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
            return "", ""

    command = []

    def runner(argv, **kwargs):
        command[:] = argv
        recovered = queue.read_state(state_path)
        assert recovered["compiled_sha256"] == queue.hash_file(compiled)
        return Process()

    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("python"),
        state_dir=state_dir,
        gpus=(0,),
        runner=runner,
    )
    assert result["status"] == "complete"


def test_pre_timing_heldout_recovery_anchors_unknown_phase_times(
    tmp_path, monkeypatch
):
    row = _fixture_row("smiles", tmp_path)
    row.update(manifest_sha256="manifest", manifest_commit="manifest")
    state_dir = tmp_path / "state"
    state_path = state_dir / f"{row['cell_id']}.json"
    compiled = tmp_path / "compiled/GeneratedCSD.py"
    compiled.parent.mkdir()
    compiled.write_text("compiled", encoding="utf-8")
    selection = _fake_compiled_selection(tmp_path, compiled)
    queue.write_state(
        state_path,
        {
            "cell_id": row["cell_id"],
            "status": "running",
            "phase": "heldout",
            "manifest_sha256": "manifest",
            "manifest_commit": "manifest",
            "compiled_csd_path": str(compiled),
            "compiled_sha256": queue.hash_file(compiled),
            **queue._selection_state(selection),
        },
    )
    monkeypatch.setattr(queue, "child_is_same_process", lambda state: False)
    monkeypatch.setattr(
        queue, "_report_binding_is_valid", lambda state, candidate, repo: True
    )
    monkeypatch.setattr(
        queue, "heldout_artifact_is_valid", lambda path, candidate: path.is_file()
    )
    command = []

    class Process:
        pid = 456
        returncode = 0

        def communicate(self):
            output = Path(command[command.index("--output-json") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
            return "", ""

    def runner(argv, **kwargs):
        command[:] = argv
        return Process()

    result = queue.run_row(
        row,
        repo=tmp_path,
        python=Path("python"),
        state_dir=state_dir,
        gpus=(0,),
        runner=runner,
    )

    assert result["status"] == "complete"
    artifact = json.loads(Path(result["heldout_output_json"]).read_text())
    runtime = artifact["controller_runtime"]
    assert runtime["phase_timing_coverage"] == "recovery_anchor"
    assert runtime["synthesis_wall_time_seconds"] == 0.0
    assert runtime["row_started_at"] == runtime["synthesis_started_at"]
    assert runtime["synthesis_started_at"] == runtime["synthesis_finished_at"]


def test_constrained_window_rate_distinguishes_equal_syntax_by_work():
    low = {"syntax_rate": 0.87, "metrics": {"mean_constrained_work": 10.0}}
    high = {"syntax_rate": 0.87, "metrics": {"mean_constrained_work": 20.0}}

    assert queue.constrained_window_rate(low) == 10.0
    assert queue.constrained_window_rate(high) == 20.0
