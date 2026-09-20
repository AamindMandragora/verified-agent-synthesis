"""A chain extender is a small molecule with at least two reactive ends (alcohol
O-H and/or primary/secondary amine N-H groups) so it can join two polymer chains.
The old ``target_class_membership`` check for ``chain_extenders`` was a plain
substring match against ``("OC", "CO", "N")``, so methanol (``CO``), ammonia
(``N``), triethylamine (``CCN(CC)CC``), and dimethyl ether (``COC``) all counted
as chain extenders despite having zero or one reactive end.

This file pins the fixed rule: RDKit parses the molecule AND it has >= 2
reactive ends, counting hydroxyls that are not carboxylic acids and
primary/secondary amines that are not amides.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.benchmarks.smiles.metrics import (
    rdkit_available,
    target_class_membership,
)

import pytest

pytestmark = pytest.mark.skipif(
    not rdkit_available(), reason="chain_extenders membership requires RDKit"
)


# --- molecules with fewer than two reactive ends: must be rejected ---

REJECTED = [
    "CO",  # methanol: one hydroxyl
    "N",  # ammonia: not even a substituted amine
    "CCN(CC)CC",  # triethylamine: tertiary amine, no N-H
    "COC",  # dimethyl ether: no O-H, no N-H
    "CCO",  # ethanol: one hydroxyl
    "CC(=O)O",  # acetic acid: hydroxyl is on a carbonyl carbon
    "OC(=O)CCC(=O)O",  # succinic acid (diacid): both hydroxyls are carboxylic
    "CC(=O)NCCNC(C)=O",  # diamide: both nitrogens are amide N, not amine
]


@pytest.mark.parametrize("smiles", REJECTED)
def test_rejects_molecules_with_fewer_than_two_reactive_ends(smiles):
    assert target_class_membership("chain_extenders", smiles) is False


# --- molecules with two or more genuine reactive ends: must be accepted ---

ACCEPTED = [
    "OCCO",  # ethylene glycol: two hydroxyls
    "OCCCCO",  # 1,4-butanediol: two hydroxyls
    "NCCN",  # ethylenediamine: two amines
    "NCCO",  # ethanolamine: one amine + one hydroxyl
    "NCCCCCCN",  # hexamethylenediamine: two amines
    "Nc1ccc(Cc2ccc(N)cc2)cc1",  # methylenedianiline: two aromatic amines
    "OCCOc1ccc(OCCO)cc1",  # hydroquinone bis(2-hydroxyethyl) ether: two hydroxyls
]


@pytest.mark.parametrize("smiles", ACCEPTED)
def test_accepts_molecules_with_two_or_more_reactive_ends(smiles):
    assert target_class_membership("chain_extenders", smiles) is True


def test_every_reference_chain_extender_passes_the_new_check():
    """Every molecule in the chain_extenders prompt-exemplar dataset must be a
    chain extender under the new rule -- if the rule is right, it should not
    reject any of the molecules we ourselves show the model as examples of the
    class."""
    task = get_smiles_task("chain_extenders")
    molecules = task["prompt_exemplars"]
    assert molecules, "expected at least one reference chain_extenders molecule"
    failures = [
        smiles
        for smiles in molecules
        if target_class_membership("chain_extenders", smiles) is not True
    ]
    assert not failures, f"reference chain_extenders molecules rejected: {failures}"


# --- acrylates / isocyanates: behavior pinned unchanged (not part of this fix) ---

ACRYLATE_CASES = [
    ("C=CC(=O)OCC", True),  # ethyl acrylate
    ("CCO", False),  # ethanol: not an acrylate
]

ISOCYANATE_CASES = [
    ("O=C=Nc1ccccc1", True),  # phenyl isocyanate
    ("CCO", False),  # ethanol: not an isocyanate
]


@pytest.mark.parametrize("smiles,expected", ACRYLATE_CASES)
def test_acrylates_membership_unchanged(smiles, expected):
    assert target_class_membership("acrylates", smiles) is expected


@pytest.mark.parametrize("smiles,expected", ISOCYANATE_CASES)
def test_isocyanates_membership_unchanged(smiles, expected):
    assert target_class_membership("isocyanates", smiles) is expected
