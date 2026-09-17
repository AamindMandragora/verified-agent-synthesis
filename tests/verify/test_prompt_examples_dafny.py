"""The examples we show the author must satisfy the contract we ask them to satisfy.

Two ways the prompt goes stale, one test each:
  * an example body that no longer verifies inside the current template;
  * a contract quoted in SYSTEM_PROMPT that no longer matches the template.

The second one is kept honest at the source: prompts.py reads the contract out of
GeneratedCSD.dfy at import time, so this test checks the wiring rather than a copy.
"""
import re

import pytest

from synthesis.generate import prompts

from tests.verify.test_span_contract_dafny import LIB, _verify, pytestmark  # noqa: F401

TEMPLATE = LIB / "GeneratedCSD.dfy"


def _example_bodies() -> list[tuple[str, str]]:
    """Every body in VERIFIED_EXAMPLES, named by the first line of its rationale.

    The examples live inside a format string, so literal Dafny braces are doubled
    there; undouble them the way str.format does before handing them to Dafny.
    """
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?=// CSD_RATIONALE_BEGIN)", prompts.VERIFIED_EXAMPLES)
        if "// CSD_RATIONALE_BEGIN" in chunk
    ]
    bodies = []
    for chunk in chunks:
        body = chunk.split("```")[0].rstrip().replace("{{", "{").replace("}}", "}")
        name = body.splitlines()[1].strip("/ ").split(".")[0]
        bodies.append((name, body))
    return bodies


EXAMPLES = _example_bodies()


def test_there_are_examples_to_check():
    assert len(EXAMPLES) >= 10


@pytest.mark.parametrize("name,body", EXAMPLES, ids=[name for name, _ in EXAMPLES])
def test_verified_example_verifies_in_the_template(name, body):
    from synthesis.generate.generator import inject_strategy_into_template

    safe = re.sub(r"[^A-Za-z0-9]+", "_", name)
    target = LIB / f"_Example_{safe}.dfy"
    target.write_text(inject_strategy_into_template(body))
    try:
        out = _verify(target)
    finally:
        target.unlink()
    assert " 0 errors" in out, out[-2000:]


def _contract_lines() -> list[str]:
    """The requires/ensures lines of AuthorBody, straight out of the template."""
    text = TEMPLATE.read_text()
    start = text.index("  method AuthorBody(")
    header = text[start:text.index("\n  {", start)]
    return [
        line.strip()
        for line in header.splitlines()
        if line.strip().startswith(("requires ", "ensures "))
    ]


def test_system_prompt_quotes_the_template_contract():
    lines = _contract_lines()
    assert len(lines) >= 10
    for line in lines:
        assert line in prompts.SYSTEM_PROMPT, line


def test_system_prompt_carries_no_stale_contract_lines():
    """Nothing in the prompt's contract block that the template does not have."""
    block = prompts.AUTHOR_BODY_CONTRACT
    quoted = [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith(("requires ", "ensures ", "decreases "))
    ]
    assert quoted == _contract_lines()


def test_prompt_has_no_hidden_span_wording():
    """There are no hidden spans any more; every span carries a visible `<<`."""
    for stale in ("hidden constrained", "hidden span", "NO visible", "no visible"):
        assert stale not in prompts.SYSTEM_PROMPT, stale
        assert stale not in prompts.INITIAL_GENERATION_PROMPT, stale
        assert stale not in prompts.VERIFIED_EXAMPLES, stale
