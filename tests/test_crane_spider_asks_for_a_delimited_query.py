"""CRANE on Spider has to be told about the << >> delimiters, or it never constrains.

CRANE (arXiv 2502.09061) decodes freely until the model writes the opening
delimiter and only then applies the grammar mask. The Spider lane sent the same
bare prompt every other Spider lane sends, which never mentions "<<", so the
switch could not fire: all ten examples of the 2026-09-20 trace logged
spans_opened=0 constrained_steps=0 and the score came from free text.

The GSM CRANE prompt names the delimiters outright and shows the response shape;
the SMILES one asks for reasoning first and shows the shape in its examples.
Spider carries no few-shot example to show, so it names them the way GSM does.
The schema, the db_id and the question stay byte-identical to the other lanes.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.benchmarks.sql_spider.prompts import (
    SpiderPromptParts,
    format_spider_itergen_aligned_prompt,
)
from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

EXAMPLE = {
    "db_id": "concert_singer",
    "db_info": "# singer ( singer_id , name , country )",
    "question": "How many singers do we have?",
}


@pytest.fixture
def shared_prompt() -> SpiderPromptParts:
    return format_spider_itergen_aligned_prompt(EXAMPLE)


@pytest.fixture
def crane_prompt(shared_prompt: SpiderPromptParts) -> SpiderPromptParts:
    return lane_module._crane_prompt_for_generation("spider", shared_prompt)


def test_the_crane_prompt_names_both_delimiters(crane_prompt) -> None:
    text = crane_prompt.raw_text

    assert "<<" in text
    assert ">>" in text


def test_the_crane_prompt_asks_for_reasoning_before_the_query(crane_prompt) -> None:
    text = crane_prompt.raw_text

    assert "Only output the SQL quey." not in text
    assert "reason" in text.lower()


def test_the_schema_and_question_are_unchanged(crane_prompt) -> None:
    text = crane_prompt.raw_text

    assert "db_id: concert_singer" in text
    assert "db_info: # singer ( singer_id , name , country )" in text
    assert "question: How many singers do we have?" in text


def test_the_result_still_renders_through_the_shared_spider_helper(crane_prompt) -> None:
    assert isinstance(crane_prompt, SpiderPromptParts)
    assert hasattr(crane_prompt, "render_for_model")


def test_a_prompt_without_the_expected_instruction_is_refused() -> None:
    with pytest.raises(ValueError):
        lane_module._crane_prompt_for_generation("spider", SpiderPromptParts("db_id: x\n"))


def test_gsm_still_passes_its_prompt_through_untouched() -> None:
    messages = [{"role": "user", "content": "a question"}]

    assert lane_module._crane_prompt_for_generation("gsm_symbolic", messages) is messages
