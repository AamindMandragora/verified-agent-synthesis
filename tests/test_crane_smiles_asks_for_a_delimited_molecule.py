"""CRANE on SMILES has to be told about the << >> delimiters, or it never constrains.

CRANE (arXiv 2502.09061) decodes freely and switches the grammar mask on only
after the model writes the opening delimiter. The SMILES lane asked for the
molecule "in exactly the same format as the examples" and never named "<<", so
the switch mostly did not fire: in the 2026-09-20 trace the 4B model opened no
span at all on 10 of 10 examples and the 2B model opened none on 7 of 10.

The Spider and GSM CRANE prompts both name the delimiters outright, so SMILES
now does too. The task sentences and the example molecules are untouched.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate import run_legacy_fixed_strategy as lane_module


@pytest.fixture
def shared_prompt() -> str:
    return get_smiles_task("acrylates")["prompt"]


@pytest.fixture
def crane_prompt(shared_prompt: str) -> str:
    return lane_module._crane_prompt_for_generation("smiles", shared_prompt)


def _instruction_text(prompt: str) -> str:
    return prompt.split("Molecule:", 1)[0]


def test_the_instruction_names_both_delimiters(crane_prompt) -> None:
    instruction = _instruction_text(crane_prompt)

    assert "<<" in instruction
    assert ">>" in instruction


def test_the_instruction_still_asks_for_reasoning_first(crane_prompt) -> None:
    instruction = _instruction_text(crane_prompt)

    assert "Your response must be a single SMILES molecule and nothing else." not in crane_prompt
    assert "reason" in instruction.lower()


def test_the_task_sentences_are_unchanged(crane_prompt, shared_prompt) -> None:
    first_line = shared_prompt.splitlines()[0]

    assert first_line in crane_prompt


def test_every_example_molecule_survives(crane_prompt, shared_prompt) -> None:
    examples = [
        line.split("Molecule:", 1)[1].strip()
        for line in shared_prompt.splitlines()
        if line.startswith("Molecule:") and line.split("Molecule:", 1)[1].strip()
    ]

    assert examples
    for molecule in examples:
        assert f"Molecule: <<{molecule}>>" in crane_prompt


def test_a_prompt_without_the_expected_instruction_is_refused() -> None:
    with pytest.raises(ValueError):
        lane_module._crane_prompt_for_generation("smiles", "Molecule:")
