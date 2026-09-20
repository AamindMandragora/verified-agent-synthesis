"""Every sampled SMILES lane seeds each draw from the same formula.

The CRANE, IterGen and unconstrained SMILES lanes each seeded a draw from the
draw's GLOBAL index, so a sharded lane does not replay another shard's draws. The
GCD lane seeded nothing at all, so its draws depended on whatever had already
consumed the process RNG and could not be reproduced.

All four now ask one function for the seed, so the formula lives in one place.
"""

import inspect
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

SAMPLED_SMILES_LANES = (
    lane_module.run_gcd_legacy_adapter,
    lane_module._crane_via_adaptive_syncode,
    lane_module._run_itergen_legacy_adapter_inner,
    lane_module.run_unconstrained_smiles_adapter,
)


def test_the_seed_is_decided_by_the_global_draw_index() -> None:
    seeds = [lane_module._smiles_draw_seed(i) for i in range(5)]

    assert len(set(seeds)) == 5
    assert seeds == [lane_module._smiles_draw_seed(i) for i in range(5)]


def test_a_shard_starting_later_gets_that_shards_own_seeds() -> None:
    assert lane_module._smiles_draw_seed(0) != lane_module._smiles_draw_seed(10)


def test_the_parity_seed_environment_variable_moves_every_draw(monkeypatch) -> None:
    monkeypatch.setenv("CSD_PARITY_SEED", "99")
    shifted = lane_module._smiles_draw_seed(0)
    monkeypatch.setenv("CSD_PARITY_SEED", "1234")

    assert shifted != lane_module._smiles_draw_seed(0)


def test_every_sampled_smiles_lane_asks_for_a_seed() -> None:
    for lane in SAMPLED_SMILES_LANES:
        assert "_smiles_draw_seed(" in inspect.getsource(lane), lane.__name__


def test_no_lane_carries_its_own_copy_of_the_formula() -> None:
    for lane in SAMPLED_SMILES_LANES:
        assert "1_000_003" not in inspect.getsource(lane), lane.__name__
