"""SMILES IterGen used to be a single forward() call with no validity check and no
retry -- unlike Spider's IterGen lane, which backs up and retries on a bad column or
table name. This gives SMILES the same forward/check/backward shape, using RDKit
validity and non-repeat/non-exemplar in place of Spider's schema check. The IterGen
paper has no molecule case study; this loop is our own adaptation of their SQL
protocol (2026-09-20 baseline repair, item 7)."""

from __future__ import annotations

from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.run_legacy_fixed_strategy import _itergen_generate_smiles

_TASK = get_smiles_task("acrylates")
_GRAMMAR_TEXT = _TASK["grammar_text"]
_EXEMPLARS = _TASK["prompt_exemplars"]
_EXAMPLE = {
    "class_name": "acrylates",
    "grammar_text": _GRAMMAR_TEXT,
    "prompt_exemplars": _EXEMPLARS,
}
_VALID_NEW_MOLECULE = "CCCCCOC(=O)C=C"  # grammar/RDKit-valid acrylate, not a prompt exemplar
_INVALID_TEXT = "not a molecule at all!!"


class FakeIterGen:
    def __init__(self, completions: list[str]):
        self._completions = completions
        self._index = -1
        self.backward_calls: list[str | None] = []
        self.started_with = None

    def start(self, prompt):
        self.started_with = prompt
        self._index = -1

    def forward(self, unit=None, units=None, num=1, **gen_args):
        self._index = min(self._index + 1, len(self._completions) - 1)
        return [self._completions[self._index]]

    def backward(self, unit=None, num=1):
        self.backward_calls.append(unit)

    def finished(self) -> bool:
        return False


def test_invalid_molecule_triggers_backward():
    iter_gen = FakeIterGen([_INVALID_TEXT] * 5)
    seen: set[str] = set(_EXEMPLARS)
    completion = _itergen_generate_smiles(
        iter_gen, "prompt", _EXAMPLE, seen=seen, max_iterations=5, backwards_limit=3
    )
    assert iter_gen.backward_calls, "an invalid molecule must trigger a backward retry"
    assert completion == _INVALID_TEXT
    assert _VALID_NEW_MOLECULE not in seen


def test_valid_unique_molecule_is_accepted_without_backing_up():
    iter_gen = FakeIterGen([_VALID_NEW_MOLECULE])
    seen: set[str] = set(_EXEMPLARS)
    completion = _itergen_generate_smiles(
        iter_gen, "prompt", _EXAMPLE, seen=seen, max_iterations=5, backwards_limit=3
    )
    assert completion == _VALID_NEW_MOLECULE
    assert not iter_gen.backward_calls
    assert _VALID_NEW_MOLECULE in seen


def test_a_prompt_exemplar_is_rejected_and_retried():
    exemplar = _EXEMPLARS[0]
    iter_gen = FakeIterGen([exemplar, _VALID_NEW_MOLECULE])
    seen: set[str] = set(_EXEMPLARS)
    completion = _itergen_generate_smiles(
        iter_gen, "prompt", _EXAMPLE, seen=seen, max_iterations=5, backwards_limit=3
    )
    assert iter_gen.backward_calls
    assert completion == _VALID_NEW_MOLECULE


def test_gives_up_after_the_backwards_limit():
    iter_gen = FakeIterGen([_INVALID_TEXT] * 10)
    seen: set[str] = set(_EXEMPLARS)
    _itergen_generate_smiles(
        iter_gen, "prompt", _EXAMPLE, seen=seen, max_iterations=10, backwards_limit=3
    )
    assert len(iter_gen.backward_calls) == 3
