"""Campaign-safe summaries for prompt, response, and strategy text."""

from __future__ import annotations

import hashlib
import os
from typing import Any


SAFE_LOGGING_ENV = "CSD_REDACT_SENSITIVE_LOGS"


def safe_logging_enabled() -> bool:
    return os.environ.get(SAFE_LOGGING_ENV, "").strip() == "1"


def text_metadata(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "chars": len(text),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def display_text(label: str, text: str) -> str:
    if not safe_logging_enabled():
        return f"{label}: {text}"
    metadata = text_metadata(text)
    return (
        f"{label}: [redacted chars={metadata['chars']} "
        f"sha256={metadata['sha256']}]"
    )
