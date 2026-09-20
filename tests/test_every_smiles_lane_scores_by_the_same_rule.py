"""One rule decides whether a SMILES row is correct, and every lane uses it.

Our own method scores a molecule through `benchmarks/smiles/eval_logic.is_correct`,
which asks for `unique_valid_candidate`: a non-empty molecule that parses under the
class grammar, that RDKit accepts, that carries the class motif, and that is not one
of the molecules the prompt already showed.

The unconstrained baseline lane had its own rule, `is_novel AND class_membership`,
which counted a repeat as wrong but let a prompt exemplar through. Two lanes measuring
two different things cannot be compared, so both now call one shared function.

CARS is deliberately outside this: its rows are the unique accepted molecules its own
paper scores, and that is not touched here.
"""

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.benchmarks.smiles import eval_logic
from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.benchmarks.smiles.metrics import (
    evaluate_smiles_output,
    molecule_is_correct,
)
from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

TASK = get_smiles_task("acrylates")
GRAMMAR = str(TASK["grammar_text"])
EXEMPLAR = "COC(=O)C=C"
NOVEL_ACRYLATE = "CCCCCCCCCCCCCCOC(=O)C=C"


def _row(molecule: str, exemplars=(EXEMPLAR,)):
    return evaluate_smiles_output("acrylates", molecule, GRAMMAR, list(exemplars))


def test_a_new_in_class_molecule_is_correct() -> None:
    assert molecule_is_correct(_row(NOVEL_ACRYLATE)) is True


def test_a_molecule_the_prompt_already_showed_is_not_correct() -> None:
    assert molecule_is_correct(_row(EXEMPLAR)) is False


def test_a_valid_molecule_outside_the_class_is_not_correct() -> None:
    assert molecule_is_correct(_row("C=O")) is False


def test_nonsense_is_not_correct() -> None:
    assert molecule_is_correct(_row("not a molecule")) is False


def test_a_missing_row_is_not_correct() -> None:
    assert molecule_is_correct(None) is False


def test_our_own_methods_rule_is_the_shared_one() -> None:
    for molecule in (NOVEL_ACRYLATE, EXEMPLAR, "C=O"):
        row = _row(molecule)
        verdict = eval_logic.is_correct(None, molecule, "", {}, row, molecule)

        assert verdict == molecule_is_correct(row)


def test_the_unconstrained_lane_keeps_no_rule_of_its_own() -> None:
    source = inspect.getsource(lane_module.run_unconstrained_smiles_adapter)

    assert "molecule_is_correct(smiles_eval)" in source
    assert 'smiles_eval.get("class_membership")' not in source
