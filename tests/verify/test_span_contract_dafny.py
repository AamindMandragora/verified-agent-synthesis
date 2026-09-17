"""The Dafny library must verify on its own, and the span contract must do its job.

The synthesis pipeline only verifies GeneratedCSD.dfy, and Dafny does not re-check
included files, so nothing else notices a proof gap inside the library.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[2] / "synthesis" / "verify" / "library"
FIXTURES = Path(__file__).resolve().parent / "span_contract_bodies"
MARKER = "// QWEN_INSERT_STRATEGY_HERE"

pytestmark = pytest.mark.skipif(shutil.which("dafny") is None, reason="dafny not installed")


def _verify(path: Path) -> str:
    done = subprocess.run(
        ["dafny", "verify", "--verification-time-limit", "300", str(path)],
        capture_output=True, text=True, timeout=1500,
    )
    return done.stdout + done.stderr


def _verify_body(name: str, tmp_name: str) -> str:
    template = (LIB / "GeneratedCSD.dfy").read_text()
    assert MARKER in template
    target = LIB / tmp_name
    target.write_text(template.replace(MARKER, (FIXTURES / name).read_text()))
    try:
        return _verify(target)
    finally:
        target.unlink()


def test_library_verifies_on_its_own():
    out = _verify(LIB / "VerifiedAgentSynthesis.dfy")
    assert " 0 errors" in out, out[-2000:]


def test_honest_body_verifies():
    out = _verify_body("honest.dfy", "_SpanContractHonest.dfy")
    assert " 0 errors" in out, out[-2000:]


def test_body_that_drops_the_span_flag_and_appends_is_rejected():
    out = _verify_body("drops_flag_then_appends.dfy", "_SpanContractCheat.dfy")
    assert " 0 errors" not in out
    assert "Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)" in out
