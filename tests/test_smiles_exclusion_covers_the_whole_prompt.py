"""A baseline that copies a molecule straight out of its own prompt is repeating the
prompt, not answering. The exclusion list must therefore name every molecule the prompt
shows, not only the ones on labelled ``Molecule:`` lines.

Found 2026-09-20 in the trace run: the shared SMILES instruction line ends
"(example: CC(=O)OC=C)", that molecule was missing from the list, and CARS
chain-extenders scored it as a correct answer 14 times out of 34 rows.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.smiles.dataset import (
    SMILES_CLASSES,
    extract_prompt_exemplars,
    get_smiles_task,
)
from synthesis.evaluate.benchmarks.smiles import eval_logic
from synthesis.evaluate.benchmarks.smiles.eval_logic import format_prompt_expression_only
from synthesis.evaluate.run_legacy_fixed_strategy import _legacy_benchmark_prompt

_PROFILES = ("expression_only", "evaluator_default", "chain_of_thought")


def _example(class_name: str) -> dict:
    task = get_smiles_task(class_name)
    return {
        "class_name": class_name,
        "prompt": task["prompt"],
        "grammar_text": task["grammar_text"],
        "prompt_exemplars": list(task["prompt_exemplars"]),
    }


def test_the_instruction_lines_own_example_molecule_is_excluded():
    task = get_smiles_task("chain_extenders")
    prompt = format_prompt_expression_only(None, {"prompt": task["prompt"]})
    assert "CC(=O)OC=C" in prompt
    assert "CC(=O)OC=C" in extract_prompt_exemplars(prompt, task["grammar_text"])


def test_every_labelled_molecule_stays_excluded():
    for class_name in SMILES_CLASSES:
        task = get_smiles_task(class_name)
        prompt = format_prompt_expression_only(None, {"prompt": task["prompt"]})
        excluded = set(extract_prompt_exemplars(prompt, task["grammar_text"]))
        for line in task["prompt"].splitlines():
            if line.startswith("Molecule:") and line.split(":", 1)[1].strip():
                assert line.split(":", 1)[1].strip() in excluded


def test_ordinary_prose_is_not_mistaken_for_a_molecule():
    task = get_smiles_task("acrylates")
    prompt = format_prompt_expression_only(None, {"prompt": task["prompt"]})
    labelled = {
        line.split(":", 1)[1].strip()
        for line in task["prompt"].splitlines()
        if line.startswith("Molecule:") and line.split(":", 1)[1].strip()
    }
    extra = set(extract_prompt_exemplars(prompt, task["grammar_text"])) - labelled
    assert extra == {"CC(=O)OC=C"}


def test_each_lane_prompt_refreshes_the_exclusion_list():
    for profile in _PROFILES:
        example = _example("acrylates")
        example["prompt_exemplars"] = []
        prompt = _legacy_benchmark_prompt(eval_logic, None, example, profile)
        assert example["prompt_exemplars"], f"{profile} left the exclusion list empty"
        if "CC(=O)OC=C" in prompt:
            assert "CC(=O)OC=C" in example["prompt_exemplars"]


def test_a_rolling_suffix_molecule_is_excluded_on_the_next_draw():
    example = _example("acrylates")
    example["prompt"] = example["prompt"].rstrip() + " CCCCCOC(=O)C=C\nMolecule:"
    _legacy_benchmark_prompt(eval_logic, None, example, "expression_only")
    assert "CCCCCOC(=O)C=C" in example["prompt_exemplars"]
