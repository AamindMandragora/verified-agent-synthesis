"""The CARS paper (arXiv 2510.01902, Appendix C) collects N unique in-grammar molecules
and reports how many of them belong to the target class. Our SMILES adapter wrote a
scored row for every draw instead, repeats and rejects included, so accuracy was an
average over copies: the 2026-09-20 chain-extenders trace wrote 34 rows holding 10
distinct molecules and scored 1.0, with 14 of those rows being the molecule the prompt's
instruction line shows.

Rows are now the N unique in-grammar molecules. The entry ticket is being a molecule at
all -- non-empty, in-grammar, chemically valid, new, and not one the prompt already
showed. Class membership is deliberately NOT part of the ticket: it is the thing the
accuracy measures, so gating entry on it would make every run score 1.0. How many draws
it took and how many were rejected stay in the run's metrics, where they describe the run
without scoring it.
"""

from __future__ import annotations

from synthesis.evaluate.run_legacy_fixed_strategy import _cars_smiles_draw_verdict


def _replay(draws: list[tuple[str, bool, bool]]) -> tuple[list[str], dict[str, int]]:
    """Run the class loop's decision over a scripted sequence of
    ``(molecule, syntax_valid, is_prompt_exemplar)`` draws."""
    seen: set[str] = set()
    scored: list[str] = []
    stats = {"draws": 0, "duplicate_draws": 0, "rejected_draws": 0}
    for molecule, syntax_valid, is_exemplar in draws:
        stats["draws"] += 1
        verdict = _cars_smiles_draw_verdict(molecule, syntax_valid, is_exemplar, seen)
        if verdict == "accepted":
            seen.add(molecule)
            scored.append(molecule)
        else:
            stats[f"{verdict}_draws"] += 1
    return scored, stats


def test_a_repeated_molecule_is_scored_once():
    scored, stats = _replay(
        [("CCO", True, False), ("CCO", True, False), ("CCCO", True, False)]
    )
    assert scored == ["CCO", "CCCO"]
    assert stats == {"draws": 3, "duplicate_draws": 1, "rejected_draws": 0}


def test_a_draw_that_fails_the_grammar_is_not_scored():
    scored, stats = _replay([("not a molecule", False, False), ("CCO", True, False)])
    assert scored == ["CCO"]
    assert stats == {"draws": 2, "duplicate_draws": 0, "rejected_draws": 1}


def test_a_molecule_the_prompt_showed_is_not_scored():
    scored, stats = _replay([("CC(=O)OC=C", True, True), ("CCO", True, False)])
    assert scored == ["CCO"]
    assert stats["rejected_draws"] == 1


def test_a_molecule_outside_the_class_still_earns_a_row():
    # Class membership is what the accuracy counts, so an in-grammar molecule that
    # misses the class is one of the N and scores as a miss, not a discarded draw.
    scored, _stats = _replay([("CCO", True, False)])
    assert scored == ["CCO"]


def test_an_empty_draw_is_not_scored():
    scored, stats = _replay([("", True, False)])
    assert scored == []
    assert stats["rejected_draws"] == 1
