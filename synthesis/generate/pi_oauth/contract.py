"""Small, sealed process boundary around Pi's ChatGPT/Codex OAuth provider."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


PI_PROVIDER_ID = "openai-codex"
PI_MODEL = "gpt-5.6-sol"
PI_VERSION = "0.84.4"
PI_SOURCE_COMMIT = "853a80d26c90a14c1886f0ebb8ffaae133ca2185"
PI_PACKAGE_INTEGRITY = (
    "sha512-jmOlrqUmvhh/siNWFRXjYLJzhKFIHNsAQaysRwzQPQFnPAaV/"
    "vhqHsLH/MBsIISA1Rjj7WTUFR3nJrpXoLx39w=="
)
PI_PACKAGE_RESOLVED = (
    "https://registry.npmjs.org/@earendil-works/pi-coding-agent/-/"
    "pi-coding-agent-0.84.4.tgz"
)
PI_REQUEST_CONTRACT = "system-instructions-single-user-no-tools-v1"
PI_PROVIDER_DIR = Path(__file__).resolve().parent / "provider"
DEFAULT_PI_BRIDGE_PATH = PI_PROVIDER_DIR / "bridge.mjs"
DEFAULT_PI_PACKAGE_LOCK = PI_PROVIDER_DIR / "package-lock.json"


class PiBridgeTimeout(RuntimeError):
    """The bounded Pi provider process did not finish in time."""


class PiBridgeFailure(RuntimeError):
    """The Pi provider process failed without exposing provider response text."""

    def __init__(self, category: str):
        self.category = category
        super().__init__(f"Pi provider failed: {category}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_file(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def resolve_pi_node(value: str | None = None) -> Path:
    """Resolve one explicit Node runtime without invoking a shell."""
    configured = value or os.environ.get("CSD_PI_NODE_EXECUTABLE")
    if not configured:
        raise ValueError("Pi OAuth requires CSD_PI_NODE_EXECUTABLE")
    resolved = shutil.which(configured)
    if resolved is None:
        raise ValueError(f"Pi Node executable not found: {configured!r}")
    return Path(resolved).resolve()


def resolve_pi_bridge(value: str | None = None) -> Path:
    configured = value or os.environ.get("CSD_PI_BRIDGE_PATH")
    return _resolve_file(configured or DEFAULT_PI_BRIDGE_PATH, "Pi bridge")


def resolve_pi_auth(value: str | None = None) -> Path:
    configured = value or os.environ.get("CSD_PI_AUTH_PATH")
    if not configured:
        raise ValueError("Pi OAuth requires CSD_PI_AUTH_PATH")
    return _resolve_file(configured, "Pi OAuth file")


def _node_version(node: Path) -> str:
    try:
        result = subprocess.run(
            [str(node), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Pi Node runtime cannot report its version") from exc
    version = result.stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if result.returncode != 0 or match is None:
        raise ValueError("Pi Node runtime returned an invalid version")
    numeric = tuple(int(part) for part in match.groups())
    if numeric < (22, 19, 0):
        raise ValueError("Pi requires Node 22.19.0 or newer")
    return version


def _validated_pi_install(package_lock: Path) -> Path:
    """Require the reviewed direct package and its installed version."""
    try:
        lock = json.loads(package_lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Pi package lock is unreadable or invalid") from exc
    packages = lock.get("packages") if isinstance(lock, dict) else None
    root = packages.get("") if isinstance(packages, dict) else None
    direct = (
        packages.get("node_modules/@earendil-works/pi-coding-agent")
        if isinstance(packages, dict)
        else None
    )
    dependency = (
        root.get("dependencies", {}).get("@earendil-works/pi-coding-agent")
        if isinstance(root, dict)
        else None
    )
    if (
        dependency != PI_VERSION
        or not isinstance(direct, dict)
        or direct.get("version") != PI_VERSION
        or direct.get("resolved") != PI_PACKAGE_RESOLVED
        or direct.get("integrity") != PI_PACKAGE_INTEGRITY
    ):
        raise ValueError("Pi package lock does not bind the reviewed Pi package")

    install_root = package_lock.parent / "node_modules"
    installed_package = (
        install_root
        / "@earendil-works"
        / "pi-coding-agent"
        / "package.json"
    )
    try:
        installed = json.loads(installed_package.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("reviewed Pi package is not installed") from exc
    if (
        not isinstance(installed, dict)
        or installed.get("name") != "@earendil-works/pi-coding-agent"
        or installed.get("version") != PI_VERSION
    ):
        raise ValueError("installed package is not the reviewed Pi package")
    return install_root


def _hash_install_tree(root: Path) -> tuple[int, str]:
    """Hash path, mode, link target, and bytes for the complete npm install."""
    resolved_root = root.resolve(strict=True)
    digest = hashlib.sha256()
    count = 0
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        mode = path.lstat().st_mode & 0o777
        if path.is_symlink():
            target = os.readlink(path)
            resolved_target = (path.parent / target).resolve(strict=True)
            if not resolved_target.is_relative_to(resolved_root):
                raise ValueError("Pi install contains a link outside node_modules")
            kind = b"link"
            content_hash = hashlib.sha256(target.encode("utf-8")).digest()
        elif path.is_file():
            kind = b"file"
            content_hash = bytes.fromhex(_hash_file(path))
        elif path.is_dir():
            continue
        else:
            raise ValueError("Pi install contains an unsupported filesystem entry")
        digest.update(kind + b"\0")
        digest.update(relative + b"\0")
        digest.update(f"{mode:o}".encode("ascii") + b"\0")
        digest.update(content_hash)
        count += 1
    if count == 0:
        raise ValueError("reviewed Pi package install is empty")
    return count, digest.hexdigest()


def pi_runtime_binding(
    *,
    node_executable: str | None = None,
    bridge_path: str | None = None,
) -> dict[str, Any]:
    """Bind the exact local runtime and provider bridge bytes."""
    node = resolve_pi_node(node_executable)
    bridge = resolve_pi_bridge(bridge_path)
    package_lock = _resolve_file(DEFAULT_PI_PACKAGE_LOCK, "Pi package lock")
    install_root = _validated_pi_install(package_lock)
    install_file_count, install_sha256 = _hash_install_tree(install_root)
    return {
        "harness": "pi-provider-only",
        "pi_version": PI_VERSION,
        "pi_source_commit": PI_SOURCE_COMMIT,
        "pi_package_integrity": PI_PACKAGE_INTEGRITY,
        "request_contract": PI_REQUEST_CONTRACT,
        "node_executable": str(node),
        "node_version": _node_version(node),
        "node_sha256": _hash_file(node),
        "bridge_path": str(bridge),
        "bridge_sha256": _hash_file(bridge),
        "package_lock_sha256": _hash_file(package_lock),
        "pi_install_file_count": install_file_count,
        "pi_install_sha256": install_sha256,
    }


def _stored_account_id(auth_path: Path) -> str:
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Pi OAuth file is unreadable or invalid") from exc
    credential = payload.get(PI_PROVIDER_ID) if isinstance(payload, dict) else None
    if not isinstance(credential, dict) or credential.get("type") != "oauth":
        raise ValueError("Pi OAuth file has no ChatGPT/Codex OAuth credential")
    account_id = credential.get("accountId")
    if isinstance(account_id, str) and account_id:
        return account_id
    access = credential.get("access")
    if not isinstance(access, str):
        raise ValueError("Pi OAuth credential has no account identity")
    try:
        encoded = access.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        account_id = claims["https://api.openai.com/auth"]["chatgpt_account_id"]
    except (IndexError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Pi OAuth credential has no account identity") from exc
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("Pi OAuth credential has no account identity")
    return account_id


def stored_pi_oauth_route(
    *,
    node_executable: str | None = None,
    bridge_path: str | None = None,
    auth_path: str | None = None,
) -> dict[str, Any]:
    """Bind stored account identity and local runtime without a network call."""
    auth = resolve_pi_auth(auth_path)
    account_id = _stored_account_id(auth)
    return {
        "auth_mode": "chatgpt_codex_oauth",
        "provider": PI_PROVIDER_ID,
        "model": PI_MODEL,
        "account_id_sha256": hashlib.sha256(account_id.encode("utf-8")).hexdigest(),
        "account_verified": True,
        **pi_runtime_binding(
            node_executable=node_executable,
            bridge_path=bridge_path,
        ),
    }


def pi_bridge_environment(auth_path: Path, home: Path) -> dict[str, str]:
    """Build an allowlisted child environment with no API-key fallback."""
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.update(
        {
            "HOME": str(home),
            "CSD_PI_AUTH_PATH": str(auth_path),
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
    )
    return environment


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_pi_bridge(
    request: dict[str, Any],
    *,
    node_executable: str | None = None,
    bridge_path: str | None = None,
    auth_path: str | None = None,
    timeout_seconds: float = 1800,
) -> tuple[dict[str, Any], float]:
    """Run one JSON request through the provider bridge and parse one JSON reply."""
    node = resolve_pi_node(node_executable)
    bridge = resolve_pi_bridge(bridge_path)
    auth = resolve_pi_auth(auth_path)
    input_bytes = json.dumps(
        request, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with (
        tempfile.TemporaryDirectory(prefix="csd-pi-cwd-") as cwd_name,
        tempfile.TemporaryDirectory(prefix="csd-pi-home-") as home_name,
    ):
        started = time.monotonic()
        process = subprocess.Popen(
            [str(node), str(bridge)],
            cwd=cwd_name,
            env=pi_bridge_environment(auth, Path(home_name)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate(
                input=input_bytes,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            _stop_process_group(process)
            raise PiBridgeTimeout("Pi provider request timed out") from exc
        except BaseException:
            _stop_process_group(process)
            raise
        duration = time.monotonic() - started
    if len(stdout) > 16 * 1024 * 1024:
        raise PiBridgeFailure("invalid_output")
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiBridgeFailure("invalid_output") from exc
    if not isinstance(payload, dict):
        raise PiBridgeFailure("invalid_output")
    if process.returncode != 0 or payload.get("ok") is not True:
        category = str(payload.get("error_category") or "provider")
        if category not in {
            "authentication",
            "invalid_request",
            "model_unavailable",
            "quota",
            "timeout",
            "transport",
            "provider",
        }:
            category = "provider"
        raise PiBridgeFailure(category)
    return payload, duration


def pi_oauth_probe(
    *,
    node_executable: str | None = None,
    bridge_path: str | None = None,
    auth_path: str | None = None,
    timeout_seconds: float = 90,
) -> dict[str, Any]:
    """Verify the stored OAuth route without sending a model prompt."""
    payload, duration = run_pi_bridge(
        {"operation": "check_auth"},
        node_executable=node_executable,
        bridge_path=bridge_path,
        auth_path=auth_path,
        timeout_seconds=timeout_seconds,
    )
    route = payload.get("route")
    if (
        not isinstance(route, dict)
        or route.get("auth_mode") != "chatgpt_codex_oauth"
        or route.get("provider") != PI_PROVIDER_ID
        or route.get("model") != PI_MODEL
        or re.fullmatch(r"[0-9a-f]{64}", str(route.get("account_id_sha256")))
        is None
    ):
        raise PiBridgeFailure("authentication")
    return {
        "status": "ready",
        "duration_seconds": duration,
        "route": {
            **route,
            "account_verified": True,
            **pi_runtime_binding(
                node_executable=node_executable,
                bridge_path=bridge_path,
            ),
        },
    }
