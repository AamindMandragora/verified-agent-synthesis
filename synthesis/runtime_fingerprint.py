"""Stable identity for the Python interpreter and installed packages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def current_runtime_fingerprint() -> dict[str, Any]:
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', '').lower()}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    package_bytes = json.dumps(
        packages, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256(package_bytes).hexdigest(),
    }


def main() -> None:
    print(json.dumps(current_runtime_fingerprint(), sort_keys=True))


if __name__ == "__main__":
    main()
