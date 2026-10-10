"""SMILES train/test split: two disjoint ranges of per-draw sampler seeds.

Why (user ruling 2026-09-26): SMILES has no list of examples to split. Every draw
of a class sends the same prompt; the only thing that makes draw 7 differ from
draw 8 is the random seed the sampler starts from. So a SMILES "example" is
(class, seed), and the fair split is two seed ranges that never overlap:

    train  seeds 100_000 .. 199_999   (draw k of train uses seed 100_000 + k)
    test   seeds 200_000 .. 299_999   (draw k of test  uses seed 200_000 + k)

Baselines on train seeds set the bar, synthesis scores train seeds, and the
winner is re-evaluated on test seeds against baselines on test seeds.

Trials (user ruling 2026-09-26, CARS paper Table 1: "3 trials, report mean and
std"). Rerunning a lane on the same seeds replays the same draws, so each trial
gets its own block of TRIAL_SEED_BLOCK = 10_000 seeds inside its split:

    trial t of a split: seeds  split_start + (t-1)*10_000  ..  split_start + t*10_000 - 1
    e.g. train trial 2, draw k  ->  seed 110_000 + k

Why 10_000: the biggest spender is CARS, which stops at 10 x its 100-molecule
target = 1_000 draws per class lane. 10_000 is 10x that, which leaves room for
a later draw offset (a confirmation slice or a shard) without ever reaching the
next trial. Three trials use 30_000 of the 100_000 seeds in each split. Train
and test use the same layout, so "trial 2" means the same offset on both sides.

Why these numbers: each range holds 100_000 draws, far more than any lane
spends (CARS stops at 10 x its 100-molecule target, i.e. 1_000 draws per
class). Both ranges also miss every seed the pre-split baselines used (1234,
2345, 3456 plus multiples of 1_000_003), so no old run replays a split draw.

The seed depends only on (split, draw index): not on the class, the process,
the shard, or an environment variable. The class is not folded in because each
class has its own prompt, so the same seed under two classes is two different
draws anyway.
"""

from __future__ import annotations

import logging
import random

LOGGER = logging.getLogger(__name__)

SMILES_SPLIT_NAMES: tuple[str, ...] = ("train", "test")
SMILES_TRIALS: tuple[int, ...] = (1, 2, 3)
TRIAL_SEED_BLOCK = 10_000

_SEED_RANGES: dict[str, range] = {
    "train": range(100_000, 200_000),
    "test": range(200_000, 300_000),
}


def seed_range(split_name: str) -> range:
    """The seeds that belong to ``split_name`` ('train' or 'test')."""
    if split_name not in _SEED_RANGES:
        raise ValueError(
            f"SMILES split name must be one of {list(SMILES_SPLIT_NAMES)}, got {split_name!r}."
        )
    return _SEED_RANGES[split_name]


def trial_seed_range(split_name: str, trial: int) -> range:
    """The TRIAL_SEED_BLOCK seeds that trial ``trial`` (1, 2 or 3) of ``split_name`` uses."""
    seeds = seed_range(split_name)
    if type(trial) is not int or trial not in SMILES_TRIALS:
        raise ValueError(f"SMILES trial must be one of {list(SMILES_TRIALS)}, got {trial!r}.")
    start = seeds.start + (trial - 1) * TRIAL_SEED_BLOCK
    return range(start, start + TRIAL_SEED_BLOCK)


def smiles_draw_seed(split_name: str, trial: int, draw_index: int) -> int:
    """The sampler seed for draw ``draw_index`` (0-based) of one trial of ``split_name``."""
    seeds = trial_seed_range(split_name, trial)
    index = int(draw_index)
    if not 0 <= index < len(seeds):
        raise ValueError(
            f"SMILES draw index {index} is outside {split_name} trial {trial} "
            f"(0 .. {len(seeds) - 1})."
        )
    return seeds[index]


def seed_range_record(split_name: str | None, trial: int | None) -> dict:
    """What a result JSON records about the SMILES split and trial (all None if no split)."""
    if split_name is None:
        return {"smiles_split_name": None, "smiles_trial": None, "smiles_seed_range": None}
    seeds = trial_seed_range(split_name, trial)
    return {
        "smiles_split_name": split_name,
        "smiles_trial": trial,
        "smiles_seed_range": [seeds.start, seeds.stop],
    }


def seed_smiles_draw(seed: int) -> None:
    """Seed every RNG a SMILES sampler may read, right before one draw.

    Python ``random``, NumPy, and torch (CPU and every CUDA device). A draw run
    after this is a function of its seed and its prompt; see the report of
    2026-09-26 for where a GPU kernel or vLLM batching can still break that.
    """
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    LOGGER.debug("[smiles-seed-split] seeded draw seed=%d", seed)


__all__ = [
    "SMILES_SPLIT_NAMES",
    "SMILES_TRIALS",
    "TRIAL_SEED_BLOCK",
    "seed_range",
    "seed_range_record",
    "seed_smiles_draw",
    "smiles_draw_seed",
    "trial_seed_range",
]
