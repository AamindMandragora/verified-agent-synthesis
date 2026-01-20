"""
Utilities for extracting tool-choice rationale from generated Dafny strategy bodies.

We embed rationale as a Dafny comment block at the top of the method body:

  // CSD_RATIONALE_BEGIN
  // ... explanation ...
  // CSD_RATIONALE_END
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RationaleExtraction:
    rationale: str | None
    body_without_rationale: str
    has_markers: bool


BEGIN_MARKER = "// CSD_RATIONALE_BEGIN"
END_MARKER = "// CSD_RATIONALE_END"


def extract_rationale(strategy_body: str) -> RationaleExtraction:
    """
    Extract an embedded rationale block from a Dafny method body.

    - Returns `rationale=None` if markers are missing or empty.
    - Leaves the remaining code in `body_without_rationale` (best-effort).
    """
    lines = strategy_body.splitlines()

    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == BEGIN_MARKER:
            begin_idx = i
            break
    if begin_idx is None:
        return RationaleExtraction(rationale=None, body_without_rationale=strategy_body, has_markers=False)

    for j in range(begin_idx + 1, len(lines)):
        if lines[j].strip() == END_MARKER:
            end_idx = j
            break
    if end_idx is None:
        # Marker start without end: treat as missing to avoid mis-parsing code.
        return RationaleExtraction(rationale=None, body_without_rationale=strategy_body, has_markers=False)

    rationale_lines: list[str] = []
    for raw in lines[begin_idx + 1 : end_idx]:
        s = raw.strip()
        if not s:
            continue
        # Require // rationale lines; if not present, still capture best-effort as text.
        if s.startswith("//"):
            s = s[2:].lstrip()
        rationale_lines.append(s)

    rationale = "\n".join([ln for ln in rationale_lines if ln]).strip() or None

    # Remove the rationale block from the body (including markers), preserving remaining lines.
    body_without = "\n".join(lines[:begin_idx] + lines[end_idx + 1 :]).lstrip("\n")

    return RationaleExtraction(rationale=rationale, body_without_rationale=body_without, has_markers=True)


