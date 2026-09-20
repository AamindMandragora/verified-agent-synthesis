import pytest

from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.run_legacy_fixed_strategy import _crane_smiles_prompt, _crane_smiles_span


@pytest.mark.parametrize("cls", ["acrylates", "chain_extenders", "isocyanates"])
def test_prompt_shows_spans_and_ends_open_for_reasoning(cls):
    task = get_smiles_task(cls)
    out = _crane_smiles_prompt(task["prompt"])
    assert "nothing else" not in out
    assert "between << and >>" in out
    assert out.endswith("Reasoning:")
    for mol in task["prompt_exemplars"]:
        assert f"Molecule: <<{mol}>>" in out


def test_rolling_prompt_lines_are_wrapped_too():
    task = get_smiles_task("acrylates")
    rolled = task["prompt"].rstrip() + " CCOC(=O)C=C\nMolecule:"
    out = _crane_smiles_prompt(rolled)
    assert "Molecule: <<CCOC(=O)C=C>>" in out
    assert out.endswith("Reasoning:")


def test_scored_molecule_is_the_last_span_and_empty_without_one():
    assert _crane_smiles_span(" an ester of acrylic acid.\nMolecule: <<CCOC(=O)C=C>>") == "CCOC(=O)C=C"
    assert _crane_smiles_span("<<C>> no wait <<CC>>") == "CC"
    assert _crane_smiles_span("1,2,3,4,5") == ""
