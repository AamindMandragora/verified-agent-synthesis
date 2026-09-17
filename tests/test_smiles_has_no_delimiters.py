"""No SMILES prompt may ask the model for `<< >>` delimiters.

What was wrong
--------------
All three SMILES prompt builders told the model to produce delimiters:

    format_prompt                   "Wrap your answer molecule in << >> delimiters"
    format_prompt_expression_only   "Return exactly one line containing `<<SMILES>>`"
    format_prompt_chain_of_thought  "wrap your final SMILES in << >> delimiters"

while the grammar (`smiles_*.lark`, `start: smiles`) has no delimiter in it and
the baselines SMILES is compared against never ask for one either. Asking for
something the grammar cannot produce made the comparison unfair and confused
the strategy-writing AI.

The decision (2026-07-28) was to take the delimiters out of the prompts. That
still holds, and it is what this file pins.

Note the span is a separate matter. Since the span contract was unified, the
SMILES OUTPUT does carry `<< >>`: the runtime opens the span itself by seeding
a literal `<<` into the output, and closes it when the molecule is complete.
None of that touches the PROMPT -- the model sees exactly what a baseline model
sees -- which is why this file's claim survived the change. `clean_smiles_output`
strips `<<`/`>>` rather than requiring them, so extraction is unaffected.
"""

from __future__ import annotations

import importlib

import pytest


PROMPT_BUILDERS = [
    "format_prompt",
    "format_prompt_expression_only",
    "format_prompt_chain_of_thought",
]

EXAMPLE = {"prompt": "Give a molecule from the acrylates class."}


def _smiles_eval_logic():
    return importlib.import_module("synthesis.evaluate.benchmarks.smiles.eval_logic")


@pytest.mark.parametrize("builder_name", PROMPT_BUILDERS)
def test_no_prompt_asks_the_model_for_delimiters(builder_name):
    builder = getattr(_smiles_eval_logic(), builder_name)
    prompt = builder(None, EXAMPLE)

    assert "<<" not in prompt and ">>" not in prompt, (
        f"{builder_name} tells the model to emit << >> delimiters. The SMILES "
        "grammar has no delimiter in it and the baselines do not ask for one; "
        "the span is opened by the RUNTIME, in the output, not by the prompt."
    )


def test_the_runtime_opens_the_span_instead(monkeypatch):
    """SMILES generation asks the runtime to open the span, not the model."""
    from synthesis.evaluate.benchmarks.smiles import generation

    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return ("", 0, 0.0, [], [])

    monkeypatch.setattr(generation, "run_crane_csd", _capture)
    _smiles_eval_logic().get_generation_runner()()

    assert seen.get("force_open_span") is True, (
        "SMILES generation must run with force_open_span=True: nothing in the "
        "prompt asks the model for a `<<`, so if the runtime does not open the "
        "span, a strategy waiting for one waits forever."
    )
