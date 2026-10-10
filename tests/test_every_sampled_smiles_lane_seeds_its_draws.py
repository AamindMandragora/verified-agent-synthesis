"""Every sampled SMILES lane seeds each draw from its split's seed range.

History: the CRANE, IterGen and unconstrained SMILES lanes seeded a draw from
CSD_PARITY_SEED + global draw index; GCD seeded nothing until 2026-09-20; CARS
only seeded the process once. Since 2026-09-26 SMILES has a train/test split made
of two disjoint seed ranges (synthesis/evaluate/benchmarks/smiles/seed_split.py),
so a draw's seed comes from (split, draw index) and nothing else: no environment
variable moves it, and every lane asks the one module for it.
"""

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate import run_legacy_fixed_strategy as lane_module

# Lanes that load their draws through the Evaluator and seed from the row.
ROW_SEEDED_LANES = (
    lane_module.run_gcd_legacy_adapter,
    lane_module._crane_via_adaptive_syncode,
    lane_module._run_itergen_legacy_adapter_inner,
)


def test_row_seeded_lanes_seed_each_draw_from_the_row():
    for lane in ROW_SEEDED_LANES:
        assert 'seed_smiles_draw(example["draw_seed"])' in inspect.getsource(lane), lane.__name__


def test_the_unconstrained_lane_asks_the_split_for_each_seed():
    source = inspect.getsource(lane_module.run_unconstrained_smiles_adapter)
    assert "smiles_draw_seed(args.smiles_split_name, args.smiles_trial, _i)" in source


def test_the_cars_lane_seeds_every_draw_it_spends():
    source = inspect.getsource(lane_module.run_cars_legacy_adapter)
    assert "smiles_draw_seed(" in source and "seed_smiles_draw(" in source


def test_the_old_environment_seeded_formula_is_gone():
    source = Path(lane_module.__file__).read_text()
    assert "_smiles_draw_seed" not in source
    for lane in (*ROW_SEEDED_LANES, lane_module.run_unconstrained_smiles_adapter):
        assert "1_000_003" not in inspect.getsource(lane), lane.__name__
