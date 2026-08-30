import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from synthesis.generate.generator import CodexTransientError, StrategyGenerator
from synthesis.generate.pi_oauth import contract as pi_contract


MODEL = "gpt-5.6-sol"
ACCOUNT_SHA256 = "a" * 64


def _configured_real_pi_node() -> Path:
    configured = os.environ.get("CSD_PI_NODE_EXECUTABLE")
    if not configured:
        pytest.skip("real Pi bridge tests require CSD_PI_NODE_EXECUTABLE")
    return pi_contract.resolve_pi_node(configured)


def _fake_pi_node(
    tmp_path: Path,
    *,
    authenticated: bool = True,
    generation_output: str = "generated strategy",
    generation_sleep_seconds: float = 0,
    fail_first_generation: bool = False,
) -> tuple[Path, Path, Path, Path]:
    capture_path = tmp_path / "capture.jsonl"
    counter_path = tmp_path / "attempts"
    bridge_path = tmp_path / "bridge.mjs"
    bridge_path.write_text("// fake bridge entrypoint\n", encoding="utf-8")
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    executable = tmp_path / "node"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys
import time

if sys.argv[1:] == ["--version"]:
    print("v24.5.0")
    raise SystemExit(0)

request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
capture = {{
    "argv": sys.argv[1:],
    "request": request,
    "environment": {{
        name: os.environ.get(name)
        for name in (
            "HOME",
            "PATH",
            "CSD_PI_AUTH_PATH",
            "PI_OFFLINE",
            "PI_SKIP_VERSION_CHECK",
            "OPENAI_API_KEY",
            "CODEX_HOME",
        )
    }},
}}
with Path({str(capture_path)!r}).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(capture) + "\\n")

route = {{
    "auth_mode": "chatgpt_codex_oauth",
    "provider": "openai-codex",
    "model": {MODEL!r},
    "account_id_sha256": {ACCOUNT_SHA256!r},
}}
if request["operation"] == "check_auth":
    if {authenticated!r}:
        print(json.dumps({{"ok": True, "route": route}}))
    else:
        print(json.dumps({{"ok": False, "error_category": "authentication"}}))
        raise SystemExit(2)
    raise SystemExit(0)

count_path = Path({str(counter_path)!r})
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
if {fail_first_generation!r} and count == 1:
    print(json.dumps({{"ok": False, "error_category": "transport"}}))
    raise SystemExit(3)
time.sleep({generation_sleep_seconds!r})
print(json.dumps({{
    "ok": True,
    "text": {generation_output!r},
    "route": route,
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable, bridge_path, auth_path, capture_path


def _generator(
    tmp_path: Path,
    executable: Path,
    bridge_path: Path,
    auth_path: Path,
    **kwargs,
) -> StrategyGenerator:
    kwargs.setdefault("pi_node_executable", str(executable))
    kwargs.setdefault("pi_bridge_path", str(bridge_path))
    kwargs.setdefault("pi_auth_path", str(auth_path))
    kwargs.setdefault("codex_max_retries", 0)
    kwargs.setdefault("codex_timeout_seconds", 5)
    return StrategyGenerator(backend="codex", model_name=MODEL, **kwargs)


def _captures(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_codex_backend_requires_the_fixed_model(tmp_path):
    executable, bridge_path, auth_path, _ = _fake_pi_node(tmp_path)

    with pytest.raises(ValueError, match="gpt-5.6-sol"):
        StrategyGenerator(
            backend="codex",
            model_name="gpt-5.5",
            pi_node_executable=str(executable),
            pi_bridge_path=str(bridge_path),
            pi_auth_path=str(auth_path),
        )


def test_pi_oauth_must_be_valid_before_generation(tmp_path):
    executable, bridge_path, auth_path, capture_path = _fake_pi_node(
        tmp_path,
        authenticated=False,
    )
    generator = _generator(tmp_path, executable, bridge_path, auth_path)

    with pytest.raises(ValueError, match="ChatGPT/Codex OAuth"):
        generator._generate_text("system prompt", "user prompt")

    assert [item["request"]["operation"] for item in _captures(capture_path)] == [
        "check_auth"
    ]


def test_pi_provider_keeps_system_and_user_prompts_separate_and_has_no_tools(
    tmp_path,
    monkeypatch,
    caplog,
):
    executable, bridge_path, auth_path, capture_path = _fake_pi_node(
        tmp_path,
        generation_output="FINAL MODEL ANSWER",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_HOME", "/must/not/reach/child")
    monkeypatch.setenv("CSD_PROMPT_LOG_DIR", str(tmp_path / "prompt-log"))
    generator = _generator(tmp_path, executable, bridge_path, auth_path)
    caplog.set_level("INFO")

    assert generator._generate_text("SYSTEM SECRET", "USER SECRET") == "FINAL MODEL ANSWER"

    auth_capture, generation_capture = _captures(capture_path)
    assert auth_capture["request"] == {"operation": "check_auth"}
    assert generation_capture["argv"] == [str(bridge_path)]
    assert generation_capture["request"] == {
        "operation": "complete",
        "model": MODEL,
        "system_prompt": "SYSTEM SECRET",
        "user_prompt": "USER SECRET",
        "reasoning": "high",
        "tools": [],
        "tool_choice": "none",
        "previous_response_id": None,
        "conversation": None,
    }
    assert generation_capture["environment"]["CSD_PI_AUTH_PATH"] == str(auth_path)
    assert generation_capture["environment"]["PI_OFFLINE"] == "1"
    assert generation_capture["environment"]["PI_SKIP_VERSION_CHECK"] == "1"
    assert generation_capture["environment"]["OPENAI_API_KEY"] is None
    assert generation_capture["environment"]["CODEX_HOME"] is None
    assert "SYSTEM SECRET" not in caplog.text
    assert "USER SECRET" not in caplog.text
    assert "FINAL MODEL ANSWER" not in caplog.text
    assert "provider=openai-codex" in caplog.text
    assert "model=gpt-5.6-sol" in caplog.text
    assert not (tmp_path / "prompt-log" / "prompt_io.jsonl").exists()


def test_codex_alias_normalizes_to_codex_backend(tmp_path):
    executable, bridge_path, auth_path, _ = _fake_pi_node(tmp_path)
    with pytest.warns(FutureWarning, match="codex-cli"):
        generator = StrategyGenerator(
            backend="codex-cli",
            model_name=MODEL,
            pi_node_executable=str(executable),
            pi_bridge_path=str(bridge_path),
            pi_auth_path=str(auth_path),
        )
    assert generator.backend == "codex"


def test_pi_provider_timeout_returns_transient_error(tmp_path):
    executable, bridge_path, auth_path, _ = _fake_pi_node(
        tmp_path,
        generation_sleep_seconds=60,
    )
    generator = _generator(
        tmp_path,
        executable,
        bridge_path,
        auth_path,
        codex_timeout_seconds=0.1,
    )

    with pytest.raises(CodexTransientError, match="timed out"):
        generator._generate_text("system", "user")


def test_pi_provider_retries_transport_failure_with_identical_request(tmp_path):
    executable, bridge_path, auth_path, capture_path = _fake_pi_node(
        tmp_path,
        generation_output="retried result",
        fail_first_generation=True,
    )
    generator = _generator(
        tmp_path,
        executable,
        bridge_path,
        auth_path,
        codex_max_retries=1,
        codex_retry_delay_seconds=0,
    )

    assert generator._generate_text("system", "user") == "retried result"
    generations = [
        item["request"]
        for item in _captures(capture_path)
        if item["request"]["operation"] == "complete"
    ]
    assert len(generations) == 2
    assert generations[0] == generations[1]


def test_author_route_binds_pi_oauth_account_and_request_contract(tmp_path):
    executable, bridge_path, auth_path, _ = _fake_pi_node(tmp_path)
    generator = _generator(tmp_path, executable, bridge_path, auth_path)

    generator._generate_text("system", "user")

    route = generator.author_route_identity()
    assert route["auth_mode"] == "chatgpt_codex_oauth"
    assert route["provider"] == "openai-codex"
    assert route["model"] == MODEL
    assert route["account_id_sha256"] == ACCOUNT_SHA256
    assert route["account_verified"] is True
    assert route["harness"] == "pi-provider-only"
    assert route["pi_version"] == "0.84.4"
    assert route["request_contract"] == "system-instructions-single-user-no-tools-v1"
    assert route["node_executable"] == str(executable.resolve())
    assert route["bridge_path"] == str(bridge_path.resolve())
    assert len(route["node_sha256"]) == 64
    assert len(route["bridge_sha256"]) == 64
    assert route["pi_install_file_count"] > 0
    assert len(route["pi_install_sha256"]) == 64
    assert "access" not in json.dumps(route).lower()
    assert "refresh" not in json.dumps(route).lower()


def test_pi_runtime_binding_rejects_a_different_direct_package(tmp_path, monkeypatch):
    executable, bridge_path, _, _ = _fake_pi_node(tmp_path)
    provider_dir = tmp_path / "provider"
    installed = provider_dir / "node_modules" / "@earendil-works" / "pi-coding-agent"
    installed.mkdir(parents=True)
    (installed / "package.json").write_text(
        json.dumps({"name": "@earendil-works/pi-coding-agent", "version": "0.84.4"}),
        encoding="utf-8",
    )
    package_lock = provider_dir / "package-lock.json"
    package_lock.write_text(
        json.dumps(
            {
                "packages": {
                    "": {
                        "dependencies": {
                            "@earendil-works/pi-coding-agent": "0.84.4"
                        }
                    },
                    "node_modules/@earendil-works/pi-coding-agent": {
                        "version": "0.84.4",
                        "resolved": "https://registry.npmjs.org/not-the-reviewed-package.tgz",
                        "integrity": "sha512-wrong",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pi_contract, "DEFAULT_PI_PACKAGE_LOCK", package_lock)

    with pytest.raises(ValueError, match="reviewed Pi package"):
        pi_contract.pi_runtime_binding(
            node_executable=str(executable),
            bridge_path=str(bridge_path),
        )


def test_pi_runtime_binding_changes_when_installed_bytes_change(tmp_path, monkeypatch):
    executable, bridge_path, _, _ = _fake_pi_node(tmp_path)
    provider_dir = tmp_path / "provider"
    installed = provider_dir / "node_modules" / "@earendil-works" / "pi-coding-agent"
    installed.mkdir(parents=True)
    (installed / "package.json").write_text(
        json.dumps({"name": "@earendil-works/pi-coding-agent", "version": "0.84.4"}),
        encoding="utf-8",
    )
    payload = installed / "dist.js"
    payload.write_text("first bytes\n", encoding="utf-8")
    package_lock = provider_dir / "package-lock.json"
    package_lock.write_text(
        json.dumps(
            {
                "packages": {
                    "": {
                        "dependencies": {
                            "@earendil-works/pi-coding-agent": "0.84.4"
                        }
                    },
                    "node_modules/@earendil-works/pi-coding-agent": {
                        "version": "0.84.4",
                        "resolved": pi_contract.PI_PACKAGE_RESOLVED,
                        "integrity": pi_contract.PI_PACKAGE_INTEGRITY,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pi_contract, "DEFAULT_PI_PACKAGE_LOCK", package_lock)

    first = pi_contract.pi_runtime_binding(
        node_executable=str(executable),
        bridge_path=str(bridge_path),
    )
    payload.write_text("changed bytes\n", encoding="utf-8")
    second = pi_contract.pi_runtime_binding(
        node_executable=str(executable),
        bridge_path=str(bridge_path),
    )

    assert first["pi_install_file_count"] == 2
    assert second["pi_install_file_count"] == 2
    assert first["pi_install_sha256"] != second["pi_install_sha256"]


def test_pi_node_requires_an_explicit_runtime_binding(monkeypatch):
    monkeypatch.delenv("CSD_PI_NODE_EXECUTABLE", raising=False)

    with pytest.raises(ValueError, match="CSD_PI_NODE_EXECUTABLE"):
        pi_contract.resolve_pi_node()


def test_real_pi_bridge_never_echoes_oauth_tokens_on_auth_failure(tmp_path):
    node = _configured_real_pi_node()
    bridge = pi_contract.resolve_pi_bridge()
    auth = tmp_path / "auth.json"
    access_secret = "access-token-must-never-appear"
    refresh_secret = "refresh-token-must-never-appear"
    auth.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": access_secret,
                    "refresh": refresh_secret,
                    "expires": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(node), str(bridge)],
        input=json.dumps({"operation": "check_auth"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=pi_contract.pi_bridge_environment(auth, tmp_path / "isolated-home"),
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert access_secret not in combined
    assert refresh_secret not in combined
    assert json.loads(result.stdout) == {
        "ok": False,
        "error_category": "authentication",
    }


def test_real_pi_bridge_loads_an_unexpired_stored_oauth_credential(tmp_path):
    node = _configured_real_pi_node()
    bridge = pi_contract.resolve_pi_bridge()
    account_id = "account-for-local-auth-test"

    def encoded(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    access = ".".join(
        (
            encoded({"alg": "none"}),
            encoded(
                {
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": account_id
                    }
                }
            ),
            "unsigned",
        )
    )
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": access,
                    "refresh": "unused-unexpired-refresh-token",
                    "expires": 4_102_444_800_000,
                    "accountId": account_id,
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(node), str(bridge)],
        input=json.dumps({"operation": "check_auth"}),
        capture_output=True,
        text=True,
        timeout=30,
        env=pi_contract.pi_bridge_environment(auth, tmp_path / "isolated-home"),
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "ok": True,
        "route": {
            "auth_mode": "chatgpt_codex_oauth",
            "provider": "openai-codex",
            "model": MODEL,
            "account_id_sha256": hashlib.sha256(
                account_id.encode("utf-8")
            ).hexdigest(),
        },
    }
