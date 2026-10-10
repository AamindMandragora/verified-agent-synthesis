import json
import os
import subprocess
from pathlib import Path

from scripts.runtime.claude_recovery.checkpoint_from_log import build_checkpoint


_REPO = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO / ".recovery" / "claude-code-gsm14b" / "launch_resume_from55.sh"
_SERVICE = _REPO / "deploy" / "focal" / "systemd" / "csd-gsm14b-claude-resume.service"
_HELPER_LAUNCHER = (
    _REPO / ".recovery" / "claude-code-gsm14b" / "launch_resume_from55_helpers.sh"
)
_HELPER_SERVICE = (
    _REPO / "deploy" / "focal" / "systemd" / "csd-gsm14b-claude-helper-resume.service"
)


def test_build_checkpoint_preserves_finished_attempts_and_seeds_active_attempt(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "attempt_number": 45,
                    "strategy_code": "method Main() {\n  // old\n}",
                    "accuracy": 0.42,
                    "contains_delimiters": True,
                    "syntax_rate": 0.88,
                    "num_examples": 49,
                    "num_correct": 21,
                }
            ]
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """============================================================
Attempt 46/80
============================================================
Strategy: // CSD_RATIONALE_BEGIN
// evaluated
// CSD_RATIONALE_END
method Main() {
  // attempt 46
}

[1/4] Verifying with Dafny...
  ✓ Verification passed
  ✗ Evaluation below threshold:
    Accuracy: 4.1% (min: 59.2%)
    Contains << >>: yes (required: yes)
    Syntax: 98.0% (min: 85.0%)

============================================================
Attempt 47/80
============================================================
Strategy: // CSD_RATIONALE_BEGIN
// active
// CSD_RATIONALE_END
method Main() {
  // attempt 47
}

[1/4] Verifying with Dafny...
  ✓ Verification passed
""",
        encoding="utf-8",
    )

    history, seed = build_checkpoint(
        log_path=log_path,
        prior_history_path=history_path,
        first_finished_attempt=46,
        last_finished_attempt=46,
        active_attempt=47,
        num_examples=49,
    )

    assert [row["attempt_number"] for row in history] == [45, 46]
    assert history[-1]["accuracy"] == 2 / 49
    assert history[-1]["syntax_rate"] == 48 / 49
    assert history[-1]["num_correct"] == 2
    assert history[-1]["strategy_code"].endswith("  // attempt 46\n}")
    assert seed.endswith("  // attempt 47\n}")


def test_build_checkpoint_uses_the_latest_duplicate_attempt_block(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text("[]", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """Attempt 46/80
Strategy: method Main() { // stale }
[1/4] Verifying with Dafny...
  ✗ Evaluation below threshold:
    Accuracy: 0.0% (min: 59.2%)
    Contains << >>: no (required: yes)
    Syntax: 0.0% (min: 85.0%)
Attempt 46/80
Strategy: method Main() { // current }
[1/4] Verifying with Dafny...
  ✗ Evaluation below threshold:
    Accuracy: 6.1% (min: 59.2%)
    Contains << >>: yes (required: yes)
    Syntax: 93.9% (min: 85.0%)
Attempt 47/80
Strategy: method Main() { // seed }
[1/4] Verifying with Dafny...
""",
        encoding="utf-8",
    )

    history, seed = build_checkpoint(
        log_path=log_path,
        prior_history_path=history_path,
        first_finished_attempt=46,
        last_finished_attempt=46,
        active_attempt=47,
        num_examples=49,
    )

    assert history[0]["num_correct"] == 3
    assert history[0]["syntax_rate"] == 46 / 49
    assert "current" in history[0]["strategy_code"]
    assert "seed" in seed


def test_build_checkpoint_ignores_an_incomplete_later_duplicate(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text("[]", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """Attempt 46/80
Strategy: method Main() { // complete }
[1/4] Verifying with Dafny...
  ✗ Evaluation below threshold:
    Accuracy: 6.1% (min: 59.2%)
    Contains << >>: yes (required: yes)
    Syntax: 93.9% (min: 85.0%)
Attempt 46/80
Strategy: method Main() { // partial duplicate }
[1/4] Verifying with Dafny...
Attempt 47/80
Strategy: method Main() { // seed }
[1/4] Verifying with Dafny...
""",
        encoding="utf-8",
    )

    history, seed = build_checkpoint(
        log_path=log_path,
        prior_history_path=history_path,
        first_finished_attempt=46,
        last_finished_attempt=46,
        active_attempt=47,
        num_examples=49,
    )

    assert history[0]["num_correct"] == 3
    assert "complete" in history[0]["strategy_code"]
    assert "partial duplicate" not in history[0]["strategy_code"]
    assert "seed" in seed


def test_claude_resume_launcher_preserves_attempt_boundary_and_has_no_api_fallback(tmp_path):
    context = tmp_path / ".context" / "claude_code_resume_0715"
    context.mkdir(parents=True)
    (context / "gsm14b_attempt55.dfy").write_text("cost := 0;\n", encoding="utf-8")
    (context / "gsm14b_before55.json").write_text("[]\n", encoding="utf-8")
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "REPO": str(tmp_path),
        "PY": "/fake/python",
        "CLAUDE_EXECUTABLE": "/fake/claude",
        "CLAUDE_CONFIG_DIR": "/fake/claude-config",
    }

    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    command = result.stdout
    assert "--generation-backend claude" in command
    assert "--generation-model claude-opus-5-5" in command
    assert "--initial-attempt-offset 54" in command
    assert "--max-iterations 26" in command
    assert "gsm14b_attempt55.dfy" in command
    assert "gsm14b_before55.json" in command
    assert "--claude-expected-account aadivya@fermi.ai" in command
    assert "--claude-idle-timeout-seconds 900" in command
    assert "--claude-emergency-timeout-seconds 7200" in command
    assert "--claude-max-retries 2" in command
    assert "--claude-telemetry-dir" in command
    assert "--claude-author-lock-file" in command
    assert "bedrock" not in command.lower()
    assert "anthropic-thinking" not in command


def test_claude_resume_launcher_keeps_scientific_settings_frozen():
    text = _LAUNCHER.read_text(encoding="utf-8")

    required = (
        "--eval-model Qwen/Qwen2.5-14B-Instruct",
        "--min-accuracy 0.5918",
        "--min-syntax-rate 0.85",
        "--eval-sample-size 49",
        "--eval-max-steps 900",
        "--eval-step-token-budget 1",
        "--vllm-gpu-memory-utilization 0.81",
        "--vllm-max-model-len 16384",
        "--adaptive-helper-mask",
        "--helper-selection-policy bandit",
        "--refinement-beam-size 2",
        "gsm_symbolic_crane_proportional_49x49_seed123.json",
        "--gsm-split-name train",
    )
    for fragment in required:
        assert fragment in text


def test_claude_resume_service_does_not_repeat_a_terminal_80_attempt_cycle():
    text = _SERVICE.read_text(encoding="utf-8")

    assert "ExecStart=/home/aadivyar/csd-generation/.recovery/claude-code-gsm14b/launch_resume_from55.sh" in text
    assert "Restart=on-abnormal" in text
    assert "Restart=on-failure" not in text
    assert "SuccessExitStatus=75" in text
    assert "KillMode=control-group" in text


def test_claude_resume_checkpoint_can_be_claimed_only_once(tmp_path):
    context = tmp_path / ".context" / "claude_code_resume_0715"
    context.mkdir(parents=True)
    (context / "gsm14b_attempt55.dfy").write_text("cost := 0;\n", encoding="utf-8")
    (context / "gsm14b_before55.json").write_text("[]\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_claude = tmp_path / "claude"
    fake_flock = tmp_path / "flock"
    for executable in (fake_python, fake_claude, fake_flock):
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    env = {
        **os.environ,
        "CLAIM_ONLY": "1",
        "REPO": str(tmp_path),
        "PY": str(fake_python),
        "CLAUDE_EXECUTABLE": str(fake_claude),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    first = subprocess.run(["bash", str(_LAUNCHER)], env=env, capture_output=True)
    second = subprocess.run(["bash", str(_LAUNCHER)], env=env, capture_output=True)

    assert first.returncode == 0
    assert second.returncode == 75
    assert (context / "attempt55.claim").is_dir()
    assert b"one_time_checkpoint_already_claimed" in second.stdout


def test_helper_refresh_launcher_uses_a_new_permanent_claim_and_same_boundary():
    launcher = _HELPER_LAUNCHER.read_text(encoding="utf-8")
    service = _HELPER_SERVICE.read_text(encoding="utf-8")

    assert "gsm14b_attempt55.dfy" in launcher
    assert "gsm14b_before55.json" in launcher
    assert "--initial-attempt-offset 54" in launcher
    assert "--max-iterations 26" in launcher
    assert "attempt55-helper-refresh.claim" in launcher
    assert "attempt55.claim" not in launcher.replace("attempt55-helper-refresh.claim", "")
    assert "--generation-backend claude" in launcher
    assert "bedrock" not in launcher.lower()
    assert "Restart=on-abnormal" in service
    assert "SuccessExitStatus=75" in service
    assert "launch_resume_from55_helpers.sh" in service


def test_temporary_provider_exit_releases_legacy_one_time_claim(tmp_path):
    context = tmp_path / ".context" / "claude_code_resume_0715"
    context.mkdir(parents=True)
    (context / "gsm14b_attempt55.dfy").write_text("cost := 0;\n", encoding="utf-8")
    (context / "gsm14b_before55.json").write_text("[]\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_claude = tmp_path / "claude"
    fake_flock = tmp_path / "flock"
    fake_python.write_text("#!/usr/bin/env bash\nexit 76\n", encoding="utf-8")
    fake_claude.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o700)
    fake_claude.chmod(0o700)
    fake_flock.chmod(0o700)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    env = {
        **os.environ,
        "REPO": str(tmp_path),
        "PY": str(fake_python),
        "CLAUDE_EXECUTABLE": str(fake_claude),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    for launcher, claim_name in (
        (_LAUNCHER, "attempt55.claim"),
        (_HELPER_LAUNCHER, "attempt55-helper-refresh.claim"),
    ):
        result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True)
        assert result.returncode == 76
        assert not (context / claim_name).exists()
