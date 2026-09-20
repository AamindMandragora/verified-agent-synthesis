"""Every in-process SMILES lane sends its task through the model's chat template.

Qwen3.5 only ends an answer at the end of an assistant turn. Handed bare text it
keeps writing: measured on Qwen3.5-2B after the complete acrylate
"CCCCCOC(=O)C=C" the end token sits at logit 14.94 behind "." at 15.69, and the
same task as a chat turn puts the end token on top at 23.12. GCD was moved onto
the shared helper first; CRANE, IterGen and unconstrained were still sending raw
text, and the 10-example trace of 2026-09-20 shows what that costs (CRANE 2B
acrylates: 9 of 10 answers ran to the 400-token cap; unconstrained: syntax
invalid on all ten).

CARS is deliberately absent: ConstrainedModel.format_prompt in the cars repo
applies the same template itself, so rendering here too would wrap it twice.
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

MODEL = "Qwen/Qwen3.5-2B"
TASK = "Molecule: CCCOC(=O)C=C\nMolecule: "

LANES = [
    "_crane_via_adaptive_syncode",
    "_run_itergen_legacy_adapter_inner",
    "run_unconstrained_smiles_adapter",
    "run_gcd_legacy_adapter",
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def test_the_shared_helper_opens_an_assistant_turn_with_thinking_off(tokenizer) -> None:
    rendered = lane_module._render_chat_prompt(tokenizer, TASK)

    assert "<|im_start|>assistant" in rendered
    assert "</think>" in rendered
    assert TASK.strip() in rendered


@pytest.mark.parametrize("lane", LANES)
def test_each_smiles_lane_renders_through_the_shared_helper(lane: str) -> None:
    source = inspect.getsource(getattr(lane_module, lane))

    assert "_render_chat_prompt(" in source


def test_the_unconstrained_lane_renders_the_rolling_prompt_it_actually_sends() -> None:
    source = inspect.getsource(lane_module.run_unconstrained_smiles_adapter)

    assert "_render_chat_prompt(tokenizer, running_prompt)" in source
    assert "llm.generate([generation_prompt]" in source
    assert "tokenizer(generation_prompt" in source


def test_the_crane_lane_rewrites_the_smiles_prompt_before_rendering_it() -> None:
    source = inspect.getsource(lane_module._crane_via_adaptive_syncode)
    rewrite = source.index("_crane_prompt_for_generation(")
    render = source.index("_render_chat_prompt(")

    assert rewrite < render
