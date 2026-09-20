"""Unconstrained SMILES and GCD SMILES used to build their prompts differently: the
unconstrained lane added a chain-of-thought prefix ("give brief step-by-step reasoning")
that is not in the CARS paper's own few-shot files, while GCD used the paper-faithful
"single molecule, nothing else" framing. This checks they now share one prompt."""

from __future__ import annotations

from synthesis.evaluate import run_legacy_fixed_strategy as runner
from synthesis.evaluate.benchmarks.smiles.eval_logic import format_prompt_expression_only


def test_unconstrained_smiles_seed_prompt_matches_gcd_expression_only_prompt():
    base_prompt = (
        "You are an expert in chemistry. ...\n\nMolecule: C=CC(=O)OC\nMolecule: "
    )
    seed_prompt = runner._unconstrained_smiles_seed_prompt(base_prompt)
    gcd_prompt = format_prompt_expression_only(None, {"prompt": base_prompt})
    assert seed_prompt == gcd_prompt


def test_unconstrained_smiles_seed_prompt_has_no_chain_of_thought_prefix():
    seed_prompt = runner._unconstrained_smiles_seed_prompt("Molecule: ")
    assert "step-by-step reasoning" not in seed_prompt
