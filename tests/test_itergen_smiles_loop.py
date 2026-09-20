"""SMILES IterGen forwards by the grammar's ``smiles`` symbol, and that symbol closes at
the end of every parenthesised branch as well as at the end of the molecule. So
``forward()`` hands back mid-molecule prefixes: the 2026-09-20 trace run accepted
``C=CC(=O)`` -- a branch close, not an acrylate -- as a finished answer on 2 draws of 10,
and every other accepted draw also ended on ``)``.

A molecule counts as written only once the model itself ends it. The validity,
not-a-repeat and not-a-prompt-exemplar checks run then, and a failed check still backs up
and retries. A molecule the token cap cut short is returned unchecked, the way every other
lane treats a capped generation, and the scorer judges it.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.run_legacy_fixed_strategy import _itergen_generate_smiles

_TASK = get_smiles_task("acrylates")
_EXEMPLARS = _TASK["prompt_exemplars"]
_EXAMPLE = {
    "class_name": "acrylates",
    "grammar_text": _TASK["grammar_text"],
    "prompt_exemplars": _EXEMPLARS,
}
_VALID_NEW_MOLECULE = "CCCCCOC(=O)C=C"  # grammar/RDKit-valid acrylate, not a prompt exemplar
_BRANCH_PREFIX = "C=CC(=O)"  # RDKit-valid on its own, but the model has not finished writing
_INVALID_TEXT = "not a molecule at all!!"


class FakeIterGen:
    """Each step is ``(text_so_far, the_model_ended_it)``, mirroring what ``forward()``
    returns and what ``finished()`` reports right after it."""

    def __init__(self, steps: list[tuple[str, bool]]):
        self._steps = steps
        self._index = -1
        self.forward_calls = 0
        self.backward_calls: list[str | None] = []

    def start(self, prompt):
        self._index = -1

    def forward(self, unit=None, units=None, num=1, **gen_args):
        self.forward_calls += 1
        self._index = min(self._index + 1, len(self._steps) - 1)
        return [self._steps[self._index][0]]

    def backward(self, unit=None, num=1):
        self.backward_calls.append(unit)

    def finished(self) -> bool:
        return self._index >= 0 and self._steps[self._index][1]


def _run(steps, *, max_iterations=5, backwards_limit=3):
    iter_gen = FakeIterGen(steps)
    seen: set[str] = set(_EXEMPLARS)
    completion = _itergen_generate_smiles(
        iter_gen,
        "prompt",
        _EXAMPLE,
        seen=seen,
        max_iterations=max_iterations,
        backwards_limit=backwards_limit,
    )
    return iter_gen, seen, completion


def test_a_mid_molecule_prefix_is_not_accepted():
    iter_gen, seen, completion = _run(
        [(_BRANCH_PREFIX, False), (_VALID_NEW_MOLECULE, True)]
    )
    assert completion == _VALID_NEW_MOLECULE
    assert _BRANCH_PREFIX not in seen
    assert not iter_gen.backward_calls, "keeping going is not a retry"


def test_the_model_ending_the_molecule_runs_the_checks():
    iter_gen, seen, completion = _run([(_VALID_NEW_MOLECULE, True)])
    assert completion == _VALID_NEW_MOLECULE
    assert _VALID_NEW_MOLECULE in seen
    assert not iter_gen.backward_calls


def test_an_invalid_finished_molecule_triggers_backward():
    iter_gen, seen, completion = _run([(_INVALID_TEXT, True)] * 5)
    assert iter_gen.backward_calls
    assert completion == _INVALID_TEXT
    assert _INVALID_TEXT not in seen


def test_a_prompt_exemplar_is_rejected_and_retried():
    iter_gen, _seen, completion = _run(
        [(_EXEMPLARS[0], True), (_VALID_NEW_MOLECULE, True)]
    )
    assert iter_gen.backward_calls
    assert completion == _VALID_NEW_MOLECULE


def test_gives_up_after_the_backwards_limit():
    iter_gen, _seen, _completion = _run(
        [(_INVALID_TEXT, True)] * 10, max_iterations=10, backwards_limit=3
    )
    assert len(iter_gen.backward_calls) == 3


def test_a_prefix_the_cap_cut_short_is_returned_unchecked():
    iter_gen = FakeIterGen([(_BRANCH_PREFIX, True)])
    iter_gen.generation_config = type("C", (), {"max_new_tokens": 3, "eos_token_id": None})()
    iter_gen.session_tokens = [[11, 22, 33]]
    iter_gen.start_from = 0
    seen: set[str] = set(_EXEMPLARS)
    completion = _itergen_generate_smiles(
        iter_gen, "prompt", _EXAMPLE, seen=seen, max_iterations=5, backwards_limit=3
    )
    assert completion == _BRANCH_PREFIX
    assert _BRANCH_PREFIX not in seen, "the cap stopping it is not the model accepting it"
