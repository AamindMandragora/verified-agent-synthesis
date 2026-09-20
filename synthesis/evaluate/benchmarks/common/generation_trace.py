"""Say why one generation call ended, from what the call left behind.

The decoders return text only, so a log reader could not tell "the model emitted its
end token" from "the token budget ran out" from "a stop word matched". Kept free of
torch/transformers, like constraint_audit.py, so it is unit-testable anywhere.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def describe_generation_end(
    *,
    generated_token_ids: Sequence[int],
    stop_token_ids: Iterable[int],
    max_new_tokens: int | None,
    completion: str,
    stop_words: Sequence[str] | None,
) -> dict[str, Any]:
    """Return ``stop_reason`` plus the counts it was decided from.

    Checked in the order the generate loops themselves check: end token, then stop
    word, then budget. ``unknown`` means none of the three explains the stop, which
    is itself worth seeing in a log.
    """
    ids = [int(token_id) for token_id in generated_token_ids]
    last_token_id = ids[-1] if ids else None
    if last_token_id is not None and last_token_id in {int(i) for i in stop_token_ids}:
        stop_reason = "end_token"
    elif stop_words and any(word in completion for word in stop_words):
        stop_reason = "stop_word"
    elif max_new_tokens is not None and len(ids) >= int(max_new_tokens):
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown"
    return {
        "stop_reason": stop_reason,
        "generated_tokens": len(ids),
        "last_token_id": last_token_id,
    }
