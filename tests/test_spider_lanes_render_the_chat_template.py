"""Every in-process Spider lane sends the task through the model's chat template.

Qwen3.5 only ends an answer at the end of an assistant turn. Handed the bare
few-shot text the raw path used to send, it never wants its end token: measured
on Qwen3.5-2B after the complete query " SELECT COUNT( * ) FROM singer",
<|im_end|> sits at logit 11.75 against 18.25 for ";" and 16.62 for "_in". The
grammar blocks ";", so the GCD lane took "_in" and ran to its 600-token cap on
all ten examples; the free lanes invented another "question: ... SQL:" pair
instead of stopping. The same task through the chat template with thinking off
puts <|im_end|> on top at 19.12.

IterGen and our own method already render through
``SpiderPromptParts.render_for_model``. These tests hold GCD, CRANE and
unconstrained to the same call and keep the raw path from coming back.
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts
from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

MODEL = "Qwen/Qwen3.5-2B"
TASK = "db_id: concert_singer\n\nquestion: How many singers do we have?\n"

LANES = ["run_gcd_legacy_adapter", "_crane_via_adaptive_syncode", "run_unconstrained_spider_adapter"]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def test_the_shared_helper_renders_an_assistant_turn(tokenizer) -> None:
    parts = SpiderPromptParts(TASK)
    rendered = lane_module._spider_generation_prompt(parts, tokenizer, MODEL)

    assert "<|im_start|>assistant" in rendered
    assert "</think>" in rendered
    assert rendered != parts.raw_text
    assert "How many singers" in rendered


def test_the_shared_helper_refuses_already_flattened_text(tokenizer) -> None:
    with pytest.raises(TypeError):
        lane_module._spider_generation_prompt(SpiderPromptParts(TASK).raw_text, tokenizer, MODEL)


@pytest.mark.parametrize("lane", LANES)
def test_each_spider_lane_uses_the_shared_helper(lane: str) -> None:
    source = inspect.getsource(getattr(lane_module, lane))

    assert "_spider_generation_prompt(" in source
    assert "str(prompt)" not in source
