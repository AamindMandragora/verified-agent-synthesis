"""Split-provenance guard and metadata for train/eval comparisons.

Why this module exists (incident 2026-07-17): synthesis runs default to the
TRAIN side of a split file, while baseline runs (run_legacy_fixed_strategy)
and re-evals default to the EVAL/TEST side. Three recorded cells (GSM
Qwen3.5-4B/9B, Spider Qwen3.5-4B) were launched with an acceptance bar
(--min-accuracy) that had been measured on the eval side, then scored on the
train side — an apples-to-oranges comparison that made metaDecode look far
under bars it was actually near. Nothing recorded which split a run used, so
the mismatch was only recoverable from wrapper logs.

This module fixes both halves at the root:

1. `check_bar_split_provenance(...)` — called at launch by run_synthesis.
   When a run uses a split file and a nonzero accuracy bar, the caller must
   declare which split the bar was measured on (--bar-from-train or
   --bar-from-test on run_synthesis), and it
   must match the split being evaluated. A mismatch always refuses to start —
   there is no override flag; cross-split comparison is exactly the bug this
   guard exists to prevent.

2. `split_provenance_metadata(...)` — the split file/name (and declared bar
   split) that every synthesis report and eval output JSON now embeds, so any
   future number is traceable to its split without log forensics.

Pure logic, no heavy imports — unit-testable off-GPU.
"""

from __future__ import annotations

from typing import Any


class BarSplitProvenanceError(ValueError):
    """Raised when an accuracy bar's split provenance is missing or mismatched."""


# One vocabulary for every dataset, zero aliases: train / test. These match
# the index keys the split files store (train_indices / test_indices). GSM
# manifests historically stored the held-out side as eval_indices; that key
# was renamed to test_indices on 2026-07-17 so one concept has one name
# everywhere. Spider's old "eval" alias was removed the same day — aliases
# are how the bar/eval split mixup stayed invisible.
SPLIT_NAMES_BY_DATASET = {
    "gsm_symbolic": ("train", "test"),
    "spider": ("train", "test"),
    # SMILES (2026-09-26): train/test are two disjoint per-draw seed ranges, not
    # an index file. See synthesis/evaluate/benchmarks/smiles/seed_split.py.
    "smiles": ("train", "test"),
}

# Datasets whose split is a seed range, so there is never a split file to point at.
SEED_RANGE_SPLIT_DATASETS = ("smiles",)


def validate_split_name(dataset: str, split_name: str | None, flag: str) -> None:
    """Reject split names outside the dataset's vocabulary (no aliases)."""
    valid = SPLIT_NAMES_BY_DATASET.get(dataset)
    if valid is None or split_name is None:
        return
    if split_name not in valid:
        hint = (
            " The held-out side is named 'test' (the 'eval' alias was removed 2026-07-17)."
            if split_name == "eval"
            else ""
        )
        raise BarSplitProvenanceError(
            f"{flag} '{split_name}' is not a valid split name for {dataset}; "
            f"expected one of {list(valid)}.{hint}"
        )


def check_bar_split_provenance(
    dataset: str,
    split_file: str | None,
    split_name: str | None,
    min_accuracy: float,
    bar_split_name: str | None,
) -> None:
    """Refuse to launch when the accuracy bar's split doesn't match the eval split.

    The guard only applies when a train/test split is actually in use (a split
    file for GSM/Spider; always, for SMILES, whose split is two seed ranges) and
    the run has a real accuracy bar.
    """
    if split_file is None:
        if dataset not in SEED_RANGE_SPLIT_DATASETS:
            return
        split_file = f"the {dataset} seed ranges (synthesis/evaluate/benchmarks/smiles/seed_split.py)"
    if min_accuracy is None or min_accuracy <= 0:
        return

    validate_split_name(dataset, split_name, "split name")
    validate_split_name(dataset, bar_split_name, "bar split")
    eval_side = split_name
    bar_side = bar_split_name

    if bar_side is None:
        valid = "|".join(SPLIT_NAMES_BY_DATASET.get(dataset, ("train", "test")))
        raise BarSplitProvenanceError(
            f"--min-accuracy {min_accuracy} is used as an acceptance bar on the "
            f"'{eval_side}' side of {split_file}, but no bar split was "
            "declared. Declare which split the bar was measured on "
            f"(--bar-from-train or --bar-from-test; valid splits: {valid}). "
            "Refusing to launch: comparing "
            "a bar from one split side against scores from the other produced "
            "wrong win/loss verdicts on 2026-07-17 (GSM Qwen3.5-4B/9B)."
        )
    if bar_side != eval_side:
        raise BarSplitProvenanceError(
            f"Bar split mismatch: --bar-from-{bar_side} but this run "
            f"evaluates the '{eval_side}' side of {split_file}. Either re-measure "
            f"the bar on '{eval_side}' or evaluate on '{bar_side}'. There is no "
            "override: cross-split bars produced wrong verdicts on 2026-07-17."
        )


def build_split_provenance(
    gsm_split_file: Any = None,
    gsm_split_name: str | None = None,
    spider_split_file: Any = None,
    spider_split_name: str | None = None,
    bar_split_name: str | None = None,
    smiles_split_name: str | None = None,
    smiles_trial: int | None = None,
) -> dict:
    """The ONE split-provenance dict shape embedded in every output JSON.

    Every writer goes through this builder: synthesis reports and re-evals
    call it via Evaluator.split_provenance(), the legacy runner calls it once
    from its parsed args. One shape, one place.
    """

    def _str_or_none(value: Any) -> str | None:
        return str(value) if value is not None else None

    return {
        "gsm_split_file": _str_or_none(gsm_split_file),
        "gsm_split_name": gsm_split_name,
        "spider_split_file": _str_or_none(spider_split_file),
        "spider_split_name": spider_split_name,
        "bar_split_name": bar_split_name,
        **_smiles_seed_range_record(smiles_split_name, smiles_trial),
    }


def _smiles_seed_range_record(smiles_split_name: str | None, smiles_trial: int | None) -> dict:
    """smiles_split_name, smiles_trial and that trial's smiles_seed_range ([start, stop)); all None off SMILES."""
    from synthesis.evaluate.benchmarks.smiles.seed_split import seed_range_record

    return seed_range_record(smiles_split_name, smiles_trial)
