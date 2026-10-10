import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time

import pytest

from synthesis.generate.generator import ClaudeTransientError, StrategyGenerator


EXPECTED_ACCOUNT = "aadivya@fermi.ai"
MODEL = "claude-opus-5-5"


def _fake_claude(
    tmp_path: Path,
    *,
    account: str = EXPECTED_ACCOUNT,
    auth_overrides: dict | None = None,
    generation_output: str = "generated strategy",
    malformed_generation_json: bool = False,
    missing_result: bool = False,
    generation_exit_code: int = 0,
    generation_stderr: str = "",
    generation_is_error: bool = False,
    generation_sleep_seconds: float = 0,
    child_pid_path: Path | None = None,
) -> tuple[Path, Path]:
    capture_path = tmp_path / "capture.json"
    executable = tmp_path / "claude"
    auth_payload = {
        "loggedIn": True,
        "email": account,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
    auth_payload.update(auth_overrides or {})
    executable.write_text(
        f"""#!{sys.executable}
import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

if sys.argv[1:] == ["auth", "status", "--json"]:
    print(json.dumps({auth_payload!r}))
    raise SystemExit(0)

args = sys.argv[1:]
prompt_path = Path(args[args.index("--system-prompt-file") + 1])
capture = {{
    "args": args,
    "cwd": os.getcwd(),
    "cwd_entries": sorted(os.listdir(os.getcwd())),
    "environment_names": sorted(os.environ),
    "home": os.environ.get("HOME"),
    "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
    "system_prompt_path": str(prompt_path),
    "system_prompt_b64": base64.b64encode(prompt_path.read_bytes()).decode("ascii"),
    "system_prompt_mode": stat.S_IMODE(prompt_path.stat().st_mode),
    "user_prompt_b64": base64.b64encode(sys.stdin.buffer.read()).decode("ascii"),
}}
Path({str(capture_path)!r}).write_text(json.dumps(capture), encoding="utf-8")
child_pid_path = {str(child_pid_path) if child_pid_path else ''!r}
if child_pid_path:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")
time.sleep({generation_sleep_seconds!r})
if {generation_stderr!r}:
    print({generation_stderr!r}, file=sys.stderr)
if {generation_exit_code!r}:
    raise SystemExit({generation_exit_code!r})
if {malformed_generation_json!r}:
    print("not-json")
elif {generation_is_error!r}:
    print(json.dumps({{"is_error": True, "result": "provider rejected request"}}))
elif {missing_result!r}:
    print(json.dumps({{"type": "result"}}))
else:
    print(json.dumps({{"result": {generation_output!r}}}))
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable, capture_path


def _generator(tmp_path: Path, executable: Path, **kwargs) -> StrategyGenerator:
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir(mode=0o700, exist_ok=True)
    kwargs.setdefault("claude_max_retries", 0)
    kwargs.setdefault("claude_timeout_seconds", 5)
    return StrategyGenerator(
        backend="claude",
        model_name=MODEL,
        claude_executable=str(executable),
        claude_config_dir=str(config_dir),
        claude_expected_account=EXPECTED_ACCOUNT,
        **kwargs,
    )


def _streaming_fake_claude(tmp_path: Path, body: str) -> Path:
    """Create a fake CLI whose generation body writes stream-json lines."""
    executable = tmp_path / "claude-stream"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys
import time

if sys.argv[1:] == ["auth", "status", "--json"]:
    print(json.dumps({{
        "loggedIn": True,
        "email": {EXPECTED_ACCOUNT!r},
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }}), flush=True)
    raise SystemExit(0)

sys.stdin.buffer.read()
{body}
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def test_stream_telemetry_retains_full_thinking_text_and_usage(tmp_path, caplog):
    lines = [
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "PRIVATE REASONING"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "strategy body"}}},
        {"type": "result", "result": "strategy body", "usage": {"input_tokens": 111, "output_tokens": 222}},
    ]
    body = "\n".join(f"print(json.dumps({line!r}), flush=True)" for line in lines)
    executable = _streaming_fake_claude(tmp_path, body)
    telemetry = tmp_path / "private-telemetry"
    generator = _generator(
        tmp_path,
        executable,
        claude_telemetry_dir=str(telemetry),
        claude_idle_timeout_seconds=1,
        claude_max_retries=0,
    )
    caplog.set_level("WARNING")

    assert generator._generate_text("system", "user") == "strategy body"

    files = [path for path in telemetry.glob("*.jsonl") if ".timeline." not in path.name]
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert [json.loads(line) for line in files[0].read_text().splitlines()] == lines
    assert "PRIVATE REASONING" in caplog.text
    assert "strategy body" in caplog.text
    assert "input_tokens=111" in caplog.text
    assert "output_tokens=222" in caplog.text


def test_stream_telemetry_writes_timestamped_activity_timeline(tmp_path):
    payloads = [
        {"type": "stream_event", "event": {"delta": {"type": "thinking_delta", "thinking": "reasoning"}}},
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "answer"}}},
        {"type": "result", "result": "answer", "usage": {"input_tokens": 7, "output_tokens": 3}},
    ]
    body = "\n".join(f"print(json.dumps({payload!r}), flush=True)" for payload in payloads)
    executable = _streaming_fake_claude(tmp_path, body)
    telemetry = tmp_path / "private-telemetry"
    generator = _generator(
        tmp_path,
        executable,
        claude_telemetry_dir=str(telemetry),
        claude_idle_timeout_seconds=1,
    )

    assert generator._generate_text("system", "user") == "answer"

    timeline = next(telemetry.glob("*.timeline.jsonl"))
    events = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines()]
    assert [event["stream_kind"] for event in events] == ["thinking", "text", "result"]
    assert all(event["observed_at_utc"].endswith("Z") for event in events)
    assert [event["elapsed_seconds"] for event in events] == sorted(
        event["elapsed_seconds"] for event in events
    )
    assert events[-1]["input_tokens"] == 7
    assert events[-1]["output_tokens"] == 3
    assert stat.S_IMODE(timeline.stat().st_mode) == 0o600


def test_stream_activity_can_continue_past_old_absolute_timeout(tmp_path):
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": f"think-{index}"}}}
        for index in range(6)
    ]
    statements = []
    for event in events:
        statements.extend([f"print(json.dumps({event!r}), flush=True)", "time.sleep(0.04)"])
    statements.append("print(json.dumps({'type': 'result', 'result': 'done'}), flush=True)")
    executable = _streaming_fake_claude(tmp_path, "\n".join(statements))
    generator = _generator(
        tmp_path,
        executable,
        claude_timeout_seconds=0.08,
        claude_idle_timeout_seconds=0.08,
        claude_emergency_timeout_seconds=2,
        claude_max_retries=0,
        claude_telemetry_dir=str(tmp_path / "telemetry"),
    )

    assert generator._generate_text("system", "user") == "done"


def test_silent_stream_retries_the_exact_request_without_new_attempt(tmp_path):
    count = tmp_path / "calls.txt"
    body = f"""
from pathlib import Path
counter = Path({str(count)!r})
calls = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(calls))
if calls == 1:
    time.sleep(0.2)
else:
    print(json.dumps({{"type": "result", "result": "same-attempt-success"}}), flush=True)
"""
    executable = _streaming_fake_claude(tmp_path, body)
    generator = _generator(
        tmp_path,
        executable,
        claude_idle_timeout_seconds=0.05,
        claude_emergency_timeout_seconds=1,
        claude_max_retries=1,
        claude_retry_delay_seconds=0,
        claude_telemetry_dir=str(tmp_path / "telemetry"),
    )

    assert generator._generate_text("system", "user") == "same-attempt-success"
    assert count.read_text() == "2"
    raw_streams = [
        path
        for path in (tmp_path / "telemetry").glob("*.jsonl")
        if ".timeline." not in path.name
    ]
    assert len(raw_streams) == 2


def test_stdin_backpressure_is_covered_by_idle_timeout(tmp_path):
    executable = tmp_path / "claude-no-stdin"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys
import time
if sys.argv[1:] == ["auth", "status", "--json"]:
    print(json.dumps({{
        "loggedIn": True, "email": {EXPECTED_ACCOUNT!r}, "authMethod": "claude.ai",
        "apiProvider": "firstParty", "subscriptionType": "max"
    }}), flush=True)
    raise SystemExit(0)
time.sleep(60)
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    generator = _generator(
        tmp_path,
        executable,
        claude_idle_timeout_seconds=0.1,
        claude_emergency_timeout_seconds=1,
        claude_max_retries=0,
        claude_telemetry_dir=str(tmp_path / "telemetry"),
    )

    with pytest.raises(ClaudeTransientError, match="idle-timeout"):
        generator._generate_text("system", "x" * 5_000_000)


def test_stream_stderr_is_private_and_redacted_from_worker_log(tmp_path, caplog):
    executable = _streaming_fake_claude(
        tmp_path,
        "print('https://auth.example/callback?code=SECRET_TOKEN', file=sys.stderr, flush=True)\n"
        "print(json.dumps({'type': 'result', 'result': 'done'}), flush=True)",
    )
    telemetry = tmp_path / "telemetry"
    generator = _generator(
        tmp_path,
        executable,
        claude_telemetry_dir=str(telemetry),
        claude_idle_timeout_seconds=1,
    )
    caplog.set_level("WARNING")

    assert generator._generate_text("system", "user") == "done"
    stderr_file = next(telemetry.glob("*.stderr.log"))
    assert stat.S_IMODE(stderr_file.stat().st_mode) == 0o600
    assert "SECRET_TOKEN" in stderr_file.read_text()
    assert "SECRET_TOKEN" not in caplog.text
    assert "auth.example" not in caplog.text


def test_account_lock_allows_only_one_author_process_at_a_time(tmp_path):
    state = tmp_path / "active.json"
    body = f"""
from pathlib import Path
state = Path({str(state)!r})
record = json.loads(state.read_text()) if state.exists() else {{"active": 0, "maximum": 0}}
record["active"] += 1
record["maximum"] = max(record["maximum"], record["active"])
state.write_text(json.dumps(record))
time.sleep(0.12)
record = json.loads(state.read_text())
record["active"] -= 1
state.write_text(json.dumps(record))
print(json.dumps({{"type": "result", "result": "done"}}), flush=True)
"""
    executable = _streaming_fake_claude(tmp_path, body)
    lock = tmp_path / "author.lock"
    generators = [
        _generator(
            tmp_path,
            executable,
            claude_author_lock_file=str(lock),
            claude_telemetry_dir=str(tmp_path / f"telemetry-{index}"),
            claude_idle_timeout_seconds=1,
            claude_max_retries=0,
        )
        for index in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda gen: gen._generate_text("system", "user"), generators)) == [
            "done",
            "done",
        ]
    assert json.loads(state.read_text())["maximum"] == 1


def test_claude_transport_preserves_prompt_bytes_and_isolates_process(tmp_path):
    executable, capture_path = _fake_claude(tmp_path)
    generator = _generator(tmp_path, executable)
    system_prompt = "system prompt\nwith unicode: λ\n"
    user_prompt = "user prompt\nwithout an added newline"

    output = generator._generate_text(system_prompt, user_prompt)

    assert output == "generated strategy"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert base64.b64decode(capture["system_prompt_b64"]) == system_prompt.encode("utf-8")
    assert base64.b64decode(capture["user_prompt_b64"]) == user_prompt.encode("utf-8")
    assert capture["system_prompt_mode"] == 0o600
    assert capture["cwd_entries"] == []
    assert Path(capture["cwd"]) != Path.cwd()
    assert Path(capture["home"]) != Path.home()
    assert capture["config_dir"] == str(tmp_path / "claude-config")

    args = capture["args"]
    assert args[:4] == ["--print", "--model", MODEL, "--effort"]
    assert args[4] == "high"
    assert "--system-prompt-file" in args
    assert ["--output-format", "stream-json"] == args[
        args.index("--output-format") : args.index("--output-format") + 2
    ]
    assert "--include-partial-messages" in args
    assert "--verbose" in args
    assert ["--tools", ""] == args[args.index("--tools") : args.index("--tools") + 2]
    for flag in (
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--no-chrome",
    ):
        assert flag in args
    assert ["--setting-sources", ""] == args[
        args.index("--setting-sources") : args.index("--setting-sources") + 2
    ]

    forbidden_prefixes = (
        "AWS_",
        "ANTHROPIC_",
        "OPENAI_",
        "GOOGLE_",
        "VERTEX_",
        "FOUNDRY_",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    )
    assert not any(
        name.startswith(forbidden_prefixes) for name in capture["environment_names"]
    )
    assert not Path(capture["cwd"]).exists()
    assert not Path(capture["home"]).exists()
    assert not Path(capture["system_prompt_path"]).exists()


def test_claude_environment_raises_the_output_token_ceiling(tmp_path):
    # Authoring calls with large prompts exceeded the CLI's default 32000
    # output-token maximum (thinking counts against it); the env var is the
    # documented way to raise it.
    executable, _ = _fake_claude(tmp_path)
    generator = _generator(tmp_path, executable)
    env = generator._claude_environment(tmp_path / "isolated-home")
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


def test_claude_environment_caps_thinking_below_the_output_ceiling(tmp_path):
    # A GSM-2B authoring call spent all 128000 output tokens (two 64000-token
    # passes) on thinking with zero text; the thinking cap must leave room for
    # the strategy text inside the output budget.
    executable, _ = _fake_claude(tmp_path)
    generator = _generator(tmp_path, executable)
    env = generator._claude_environment(tmp_path / "isolated-home")
    thinking = int(env["MAX_THINKING_TOKENS"])
    output = int(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    assert thinking == 48000
    assert thinking < output
    # On Sonnet 4.6 the fixed thinking budget is ignored unless adaptive
    # reasoning is explicitly disabled (code.claude.com model-config docs).
    assert env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] == "1"


def test_claude_account_must_match_before_generation(tmp_path):
    executable, capture_path = _fake_claude(tmp_path, account="wrong@example.com")
    generator = _generator(tmp_path, executable)

    with pytest.raises(ValueError, match="aadivya@fermi.ai"):
        generator._generate_text("system", "user")

    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"loggedIn": False}, "logged in"),
        ({"authMethod": "apiKey"}, "claude.ai"),
        ({"apiProvider": "thirdParty"}, "firstParty"),
        ({"subscriptionType": "pro"}, "Max"),
    ],
)
def test_claude_account_requires_first_party_max(tmp_path, overrides, error):
    executable, capture_path = _fake_claude(tmp_path, auth_overrides=overrides)
    generator = _generator(tmp_path, executable)

    with pytest.raises(ValueError, match=error):
        generator._generate_text("system", "user")

    assert not capture_path.exists()


def test_claude_backend_rejects_any_other_model(tmp_path):
    executable, _ = _fake_claude(tmp_path)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()

    with pytest.raises(ValueError, match="claude-opus-5-5"):
        StrategyGenerator(
            backend="claude",
            model_name="claude-opus-4-7",
            claude_executable=str(executable),
            claude_config_dir=str(config_dir),
            claude_expected_account=EXPECTED_ACCOUNT,
        )


def test_claude_malformed_json_is_a_clear_error(tmp_path, caplog):
    executable, _ = _fake_claude(tmp_path, malformed_generation_json=True)
    generator = _generator(tmp_path, executable)

    with pytest.raises(RuntimeError, match="valid JSON"):
        generator._generate_text("system", "user")
    assert "exit_status=0 category=invalid-json" in caplog.text


def test_claude_missing_result_is_a_clear_error(tmp_path, caplog):
    executable, _ = _fake_claude(tmp_path, missing_result=True)
    generator = _generator(tmp_path, executable)

    with pytest.raises(RuntimeError, match="result"):
        generator._generate_text("system", "user")
    assert "exit_status=0 category=missing-result" in caplog.text


def test_claude_json_error_is_logged_with_category(tmp_path, caplog):
    executable, _ = _fake_claude(tmp_path, generation_is_error=True)
    generator = _generator(tmp_path, executable)

    with pytest.raises(RuntimeError, match="returned an error"):
        generator._generate_text("system", "user")

    assert "exit_status=0 category=cli-result" in caplog.text


def test_claude_nonzero_and_subscription_limit_errors_are_classified(tmp_path):
    executable, _ = _fake_claude(
        tmp_path,
        generation_exit_code=1,
        generation_stderr="weekly usage limit reached",
    )
    generator = _generator(tmp_path, executable)

    with pytest.raises(RuntimeError, match="subscription limit"):
        generator._generate_text("system", "user")


def test_claude_timeout_kills_the_process_group_and_cleans_up(tmp_path, caplog):
    child_pid_path = tmp_path / "child.pid"
    executable, capture_path = _fake_claude(
        tmp_path,
        generation_sleep_seconds=60,
        child_pid_path=child_pid_path,
    )
    generator = _generator(tmp_path, executable)
    generator.claude_idle_timeout_seconds = 0.2

    with pytest.raises(ClaudeTransientError, match="idle-timeout"):
        generator._generate_text("system", "user")
    assert "exit_status=timeout category=idle-timeout" in caplog.text

    deadline = time.time() + 3
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("Claude descendant process survived timeout cleanup")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert not Path(capture["cwd"]).exists()
    assert not Path(capture["system_prompt_path"]).exists()


def test_process_group_cleanup_stops_child_after_wrapper_already_exited(tmp_path):
    child_pid_path = tmp_path / "orphan.pid"
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-c",
                (
                    "import pathlib, subprocess, sys; "
                    "child=subprocess.Popen([sys.executable, '-c', "
                    "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "time.sleep(60)']); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
            ),
        ],
        start_new_session=True,
    )
    wrapper.wait(timeout=3)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    StrategyGenerator._stop_claude_process_group(wrapper)

    deadline = time.time() + 3
    while time.time() < deadline:
        status_path = Path(f"/proc/{child_pid}/stat")
        if not status_path.exists() or status_path.read_text().split()[2] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("Claude descendant survived after its wrapper exited")


def test_two_claude_calls_use_distinct_temporary_state(tmp_path):
    captures = []
    generators = []
    for index in range(2):
        call_dir = tmp_path / str(index)
        call_dir.mkdir()
        executable, capture_path = _fake_claude(call_dir)
        generators.append(_generator(call_dir, executable))
        captures.append(capture_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(
                lambda pair: pair[0]._generate_text("system", pair[1]),
                zip(generators, ("user one", "user two")),
            )
        )

    assert outputs == ["generated strategy", "generated strategy"]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in captures]
    assert records[0]["cwd"] != records[1]["cwd"]
    assert records[0]["system_prompt_path"] != records[1]["system_prompt_path"]


def test_nonzero_stdout_limit_error_gets_stable_access_marker(tmp_path, caplog):
    executable, _ = _fake_claude(
        tmp_path,
        generation_exit_code=1,
        generation_output="You've hit your limit for this billing period",
    )
    text = executable.read_text(encoding="utf-8")
    text = text.replace(
        f"if {1!r}:\n    raise SystemExit({1!r})",
        (
            "if True:\n"
            "    print(json.dumps({'is_error': True, 'result': "
            "\"You've hit your limit for this billing period\"}))\n"
            "    raise SystemExit(1)"
        ),
    )
    executable.write_text(text, encoding="utf-8")
    generator = _generator(tmp_path, executable)

    with pytest.raises(RuntimeError, match=r"\[claude-author-access\]"):
        generator._generate_text("system", "user")
    assert "[claude] failure" in caplog.text
    assert "exit_status=1" in caplog.text
    assert "category=access" in caplog.text
    assert "duration_seconds=" in caplog.text


def test_claude_error_text_redacts_urls_codes_and_tokens():
    safe = StrategyGenerator._safe_claude_error(
        "visit https://auth.example/callback?code=SECRET_CODE "
        "Bearer SECRET_BEARER api_key=SECRET_KEY sk-ant-SECRET_TOKEN"
    )

    for secret in (
        "https://auth.example",
        "SECRET_CODE",
        "SECRET_BEARER",
        "SECRET_KEY",
        "sk-ant-SECRET_TOKEN",
    ):
        assert secret not in safe


def test_explicit_claude_timeout_wins_over_ambient_environment(tmp_path, monkeypatch):
    executable, _ = _fake_claude(tmp_path)
    monkeypatch.setenv("CSD_CLAUDE_TIMEOUT_SECONDS", "999")

    generator = _generator(tmp_path, executable)

    assert generator.claude_timeout_seconds == 5


def test_claude_paths_never_call_any_api_transport(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    executable, _ = _fake_claude(tmp_path, generation_output="safe result")
    generator = _generator(tmp_path, executable)

    def forbidden(*args, **kwargs):
        pytest.fail("a paid API transport was called from the Claude Code backend")

    monkeypatch.setattr(generator, "_generate_bedrock", forbidden)
    monkeypatch.setattr(generator, "_generate_gemini", forbidden)
    monkeypatch.setattr(generator, "_generate_vertex", forbidden)
    monkeypatch.setattr(generator, "_summarize_rationale_claim_bedrock", forbidden)
    monkeypatch.setattr(generator, "_summarize_rationale_claim_anthropic", forbidden)
    monkeypatch.setattr(generator, "_summarize_rationale_claim_openai", forbidden)
    monkeypatch.setattr(generator, "_summarize_rationale_claim_gemini", forbidden)
    monkeypatch.setattr(generator, "_summarize_rationale_claim_vertex", forbidden)
    class ForbiddenClient:
        def __getattribute__(self, name):
            pytest.fail(f"provider client was accessed: {name}")

    generator._client = ForbiddenClient()
    generator._summary_client = ForbiddenClient()
    generator._summary_anthropic_client = ForbiddenClient()

    assert generator._generate_text("system", "user") == "safe result"
    assert generator.summarize_rationale_claim("rationale") == "safe result"
    assert "[claude] configuration" in caplog.text
    assert str(tmp_path / "claude-config") in caplog.text


def test_claude_rationale_summary_uses_same_transport(tmp_path, monkeypatch):
    executable, capture_path = _fake_claude(tmp_path, generation_output="one sentence")
    generator = _generator(tmp_path, executable)
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "bedrock")
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND", "anthropic")

    summary = generator.summarize_rationale_claim("Changed the span budget after failures.")

    assert summary == "one sentence"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert b"Summarize a CSD attempt rationale" in base64.b64decode(
        capture["system_prompt_b64"]
    )
    assert b"Changed the span budget" in base64.b64decode(capture["user_prompt_b64"])


def test_claude_rationale_failure_returns_full_rationale_without_api_fallback(
    tmp_path, monkeypatch
):
    executable, _ = _fake_claude(tmp_path, generation_exit_code=1)
    generator = _generator(tmp_path, executable)
    monkeypatch.setattr(
        generator,
        "_summarize_rationale_claim_bedrock",
        lambda *args, **kwargs: pytest.fail("Bedrock fallback was called"),
    )
    monkeypatch.setattr(
        generator,
        "_summarize_rationale_claim_anthropic",
        lambda *args, **kwargs: pytest.fail("Anthropic fallback was called"),
    )
    rationale = "Changed the span budget after failures."

    assert generator.summarize_rationale_claim(rationale) == rationale


def test_prompt_log_uses_canonical_backend_and_final_runtime_prompt(tmp_path, monkeypatch):
    executable, capture_path = _fake_claude(tmp_path)
    generator = _generator(tmp_path, executable)
    generator.set_synthesis_context("Qwen/Test", "gsm_symbolic", 900, 1)
    log_dir = tmp_path / "prompt-log"
    monkeypatch.setenv("CSD_PROMPT_LOG_DIR", str(log_dir))

    generator._generate_text("system", "user")

    record = json.loads((log_dir / "prompt_io.jsonl").read_text(encoding="utf-8"))
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert record["backend"] == "claude"
    assert record["model"] == MODEL
    assert record["system_prompt"].endswith(
        "- Evaluation model: Qwen/Test\n"
        "- Dataset: gsm_symbolic\n"
        "- maxSteps budget: 900\n"
        "- stepTokenBudget: 1\n"
    )
    assert base64.b64decode(capture["system_prompt_b64"]) == record[
        "system_prompt"
    ].encode("utf-8")


def test_provider_aliases_preserve_routes_and_warn(tmp_path):
    executable, _ = _fake_claude(tmp_path)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()

    with pytest.warns(FutureWarning, match="claude-code"):
        claude = StrategyGenerator(
            backend="claude-code",
            model_name=MODEL,
            claude_executable=str(executable),
            claude_config_dir=str(config_dir),
            claude_expected_account=EXPECTED_ACCOUNT,
        )
    with pytest.warns(FutureWarning, match="bedrock"):
        bedrock = StrategyGenerator(
            backend="bedrock",
            model_name="us.anthropic.claude-sonnet-4-6",
            api_key="test-token",
        )

    assert claude.backend == "claude"
    assert bedrock.backend == "claude-bedrock"
