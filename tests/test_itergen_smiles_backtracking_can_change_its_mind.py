"""IterGen's backtracking needs the recurrence penalty, or it redraws the same string.

IterGen's answer to "this unit is wrong" is backward() then forward(). The only
thing that makes the redraw differ is the recurrence penalty: backward keeps the
trace, and forward multiplies the logits of the tokens already tried at that
position by the penalty (main.py, "if self.recurrence_penalty != 1.0"). With the
penalty at its 1.0 default the mechanism is switched off.

The SMILES lane ran with it off and the 2026-09-20 trace shows the result: 8 of
10 chain-extender examples burned all ten backtracks, several of them returning
the identical continuation every time. Every IterGen-mode experiment the authors
checked in uses 0.3 (case_studies/sql/eval_sql.py line 189, and the privacy run
in README.md); their unconstrained and syncode baselines use none. The SMILES
lane now uses the same 0.3 the Spider lane does.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.run_legacy_fixed_strategy import _itergen_generation_kwargs


def kwargs_for(dataset: str) -> dict:
    return _itergen_generation_kwargs(dataset=dataset, max_tokens=4496, max_new_tokens=400)


def test_smiles_backtracking_is_penalised_for_repeating_itself() -> None:
    assert kwargs_for("smiles")["recurrence_penalty"] == 0.3


def test_spider_keeps_the_penalty_it_already_had() -> None:
    assert kwargs_for("spider")["recurrence_penalty"] == 0.3


def test_smiles_still_samples() -> None:
    settings = kwargs_for("smiles")

    assert settings["do_sample"] is True
    assert settings["temperature"] == 1.0


def test_gsm_takes_no_penalty_because_its_lane_never_backtracks() -> None:
    assert "recurrence_penalty" not in kwargs_for("gsm_symbolic")
