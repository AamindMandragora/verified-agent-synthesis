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
REFERENCES = Path(__file__).resolve().parents[2] / "synthesis" / "verify" / "reference"
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


# Every reference decoder is a BODY that goes into the one template. There is no
# whole-file reference format any more, so the only thing that can verify is the
# body inside GeneratedCSD.dfy -- which is also exactly what
# run_reference_strategy.py compiles and evaluates.
REFERENCE_BODIES = [
    "cars.dfy",
    "crane.dfy",
    "crane_faithful.dfy",
    "gcd.dfy",
    "itergen.dfy",
    "unconstrained.dfy",
]


def test_references_are_bodies_not_whole_files():
    """A reference must carry no module or method header of its own."""
    for name in REFERENCE_BODIES:
        text = (REFERENCES / name).read_text()
        assert "module " not in text, name
        assert "method MyCSDStrategy" not in text, name
        assert "include " not in text, name


@pytest.mark.parametrize("name", REFERENCE_BODIES)
def test_reference_body_verifies_in_the_template(name):
    """Each reference must satisfy the current span contract, not a stale copy."""
    from synthesis.generate.generator import inject_strategy_into_template

    target = LIB / f"_Ref_{name}"
    target.write_text(inject_strategy_into_template((REFERENCES / name).read_text()))
    try:
        out = _verify(target)
    finally:
        target.unlink()
    assert " 0 errors" in out, out[-2000:]
