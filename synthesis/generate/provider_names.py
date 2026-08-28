"""Canonical names for hosted CSD author providers."""

from __future__ import annotations

import warnings


GENERATION_BACKENDS = (
    "huggingface",
    "vllm",
    "openai",
    "codex",
    "claude",
    "claude-bedrock",
    "anthropic",
    "gemini",
    "vertex",
    "claude-code",
    "bedrock",
)

_DEPRECATED_ALIASES = {
    "claude-code": "claude",
    "codex-cli": "codex",
    "bedrock": "claude-bedrock",
}


def normalize_generation_backend(name: str, *, warn: bool = True) -> str:
    """Return the canonical provider name and warn on old aliases."""
    canonical = _DEPRECATED_ALIASES.get(name, name)
    if warn and canonical != name:
        warnings.warn(
            f"generation backend {name!r} is deprecated; use {canonical!r}",
            FutureWarning,
            stacklevel=2,
        )
    return canonical
