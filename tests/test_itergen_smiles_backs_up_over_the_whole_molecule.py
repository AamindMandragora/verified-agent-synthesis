"""A rejected molecule has to be backed out of, not nudged.

The SMILES loop checks a whole molecule and then called backward("smiles"),
which walks back to the start of the LAST smiles symbol. The SMILES grammars
close a smiles symbol at the end of every parenthesised branch, so the last one
starts at the end of the text and the backtrack removed nothing: the run of
2026-09-20 logged tokens_removed=0 on 70 of its 71 backtracks, and the loop
handed the identical string to the check ten times before giving up.

The unit the loop checks is the molecule, so that is the unit it backs up over:
every generated token, back to the prompt. The trace is kept, which is what lets
the recurrence penalty steer the redraw somewhere new.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.run_legacy_fixed_strategy import _itergen_generate_smiles


class FakeIterGen:
    """Answers with the same rejected molecule until something backs it out."""

    def __init__(self, molecule: str, token_count: int) -> None:
        self.molecule = molecule
        self.token_count = token_count
        self.start_from = 5
        self.session_tokens = list(range(5)) + list(range(100, 100 + token_count))
        self.generation_config = type("C", (), {"max_new_tokens": 400})()
        self.tokenizer = type("T", (), {"eos_token_id": 248046})()
        self.generation_stop_token_ids = {248046}
        self.backward_calls: list[tuple] = []

    def start(self, prompt) -> None:
        self.prompt = prompt

    def forward(self, units=None, num=1):
        return [self.molecule]

    def finished(self) -> bool:
        return True

    def backward(self, unit=None, num=1):
        self.backward_calls.append((unit, num))


EXAMPLE = {"class_name": "acrylates", "grammar_text": ""}


def run_loop(fake: FakeIterGen) -> None:
    _itergen_generate_smiles(
        fake, "Molecule: ", EXAMPLE, seen={"CC(=C)C(=O)OC1=CC=CC=C1"}, backwards_limit=3
    )


def test_it_backs_up_over_every_generated_token() -> None:
    fake = FakeIterGen("CC(=C)C(=O)OC1=CC=CC=C1", token_count=27)
    run_loop(fake)

    assert fake.backward_calls
    assert all(call == ("token", 27) for call in fake.backward_calls)


def test_it_no_longer_asks_for_one_nested_smiles_symbol() -> None:
    fake = FakeIterGen("CC(=C)C(=O)OC1=CC=CC=C1", token_count=27)
    run_loop(fake)

    assert all(unit != "smiles" for unit, _ in fake.backward_calls)


def test_nothing_generated_means_nothing_to_back_out_of() -> None:
    fake = FakeIterGen("", token_count=0)
    run_loop(fake)

    assert fake.backward_calls == []
