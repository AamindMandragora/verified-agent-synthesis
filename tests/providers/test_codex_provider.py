import base64
import json
from pathlib import Path
import stat
import sys

import pytest

from synthesis.generate.generator import CodexTransientError, StrategyGenerator


MODEL = "gpt-5.6-sol"


def _fake_codex(
    tmp_path: Path,
    *,
    login_output: str = "Logged in using ChatGPT",
    login_stderr: str = "",
    generation_output: str = "generated strategy",
    generation_sleep_seconds: float = 0,
) -> tuple[Path, Path]:
    capture_path = tmp_path / "capture.json"
    executable = tmp_path / "codex"
    executable.write_text(
        f"""#!{sys.executable}
import base64
import json
import os
from pathlib import Path
import sys
import time

if sys.argv[1:] == [\"login\", \"status\"]:
    print({login_output!r})
    print({login_stderr!r}, file=sys.stderr)
    raise SystemExit(0)

args = sys.argv[1:]
output_path = Path(args[args.index(\"--output-last-message\") + 1])
capture = {{
    \"args\": args,
    \"cwd\": os.getcwd(),
    \"cwd_entries\": sorted(os.listdir(os.getcwd())),
    \"home\": os.environ.get(\"HOME\"),
    \"codex_home\": os.environ.get(\"CODEX_HOME\"),
    \"prompt_b64\": base64.b64encode(sys.stdin.buffer.read()).decode(\"ascii\"),
}}
Path({str(capture_path)!r}).write_text(json.dumps(capture), encoding=\"utf-8\")
time.sleep({generation_sleep_seconds!r})
output_path.write_text({generation_output!r}, encoding=\"utf-8\")
print(\"provider status, not the final answer\")
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable, capture_path


def _generator(tmp_path: Path, executable: Path, **kwargs) -> StrategyGenerator:
    kwargs.setdefault("codex_executable", str(executable))
    kwargs.setdefault("codex_max_retries", 0)
    kwargs.setdefault("codex_timeout_seconds", 5)
    return StrategyGenerator(backend="codex", model_name=MODEL, **kwargs)


def test_codex_backend_requires_the_fixed_model(tmp_path):
    executable, _ = _fake_codex(tmp_path)

    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        StrategyGenerator(
            backend="codex",
            model_name="gpt-5.5",
            codex_executable=str(executable),
        )


def test_codex_requires_chatgpt_login_before_generation(tmp_path):
    executable, capture_path = _fake_codex(
        tmp_path,
        login_output="Logged in using API key",
    )
    generator = _generator(tmp_path, executable)

    with pytest.raises(ValueError, match="ChatGPT"):
        generator._generate_text("system prompt", "user prompt")

    assert not capture_path.exists()


def test_codex_accepts_chatgpt_login_reported_on_stderr(tmp_path):
    executable, capture_path = _fake_codex(
        tmp_path,
        login_output="",
        login_stderr="Logged in using ChatGPT",
        generation_output="stderr-authenticated result",
    )
    generator = _generator(tmp_path, executable)

    assert generator._generate_text("system", "user") == "stderr-authenticated result"
    assert capture_path.exists()


def test_codex_uses_isolated_read_only_exec_and_exact_final_message(tmp_path, monkeypatch, caplog):
    executable, capture_path = _fake_codex(tmp_path, generation_output="FINAL CODEX ANSWER")
    generator = _generator(tmp_path, executable)
    monkeypatch.setenv("CSD_PROMPT_LOG_DIR", str(tmp_path / "prompt-log"))
    caplog.set_level("INFO")

    assert generator._generate_text("SYSTEM SECRET", "USER SECRET") == "FINAL CODEX ANSWER"

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    args = capture["args"]
    assert args[:2] == ["exec", "--model"]
    assert args[2] == MODEL
    assert "--sandbox" in args and args[args.index("--sandbox") + 1] == "read-only"
    for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
        assert flag in args
    assert capture["cwd_entries"] == []
    prompt = base64.b64decode(capture["prompt_b64"]).decode("utf-8")
    assert "SYSTEM SECRET" in prompt and "USER SECRET" in prompt
    assert "SYSTEM SECRET" not in caplog.text
    assert "USER SECRET" not in caplog.text
    assert "FINAL CODEX ANSWER" not in caplog.text
    assert "provider=codex" in caplog.text
    assert "model=gpt-5.6-sol" in caplog.text
    assert not (tmp_path / "prompt-log" / "prompt_io.jsonl").exists()


def test_codex_alias_normalizes_to_codex_backend(tmp_path):
    executable, _ = _fake_codex(tmp_path)
    with pytest.warns(FutureWarning, match="codex-cli"):
        generator = StrategyGenerator(
            backend="codex-cli",
            model_name=MODEL,
            codex_executable=str(executable),
        )
    assert generator.backend == "codex"


def test_codex_timeout_returns_transient_error(tmp_path):
    executable, _ = _fake_codex(tmp_path, generation_sleep_seconds=60)
    generator = _generator(
        tmp_path,
        executable,
        codex_timeout_seconds=0.1,
    )

    with pytest.raises(CodexTransientError, match="timed out"):
        generator._generate_text("system", "user")


def test_codex_retries_transient_cli_failure_without_changing_prompt(tmp_path):
    counter = tmp_path / "attempts"
    executable = tmp_path / "codex-retry"
    executable.write_text(
        f"""#!{sys.executable}
import json
from pathlib import Path
import sys

if sys.argv[1:] == [\"login\", \"status\"]:
    print(\"Logged in using ChatGPT\")
    raise SystemExit(0)
count = int(Path({str(counter)!r}).read_text()) + 1 if Path({str(counter)!r}).exists() else 1
Path({str(counter)!r}).write_text(str(count))
if count == 1:
    raise SystemExit(1)
output = Path(sys.argv[sys.argv.index(\"--output-last-message\") + 1])
sys.stdin.buffer.read()
output.write_text(\"retried result\")
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    generator = _generator(
        tmp_path,
        executable,
        codex_max_retries=1,
        codex_retry_delay_seconds=0,
    )

    assert generator._generate_text("system", "user") == "retried result"
    assert counter.read_text() == "2"
