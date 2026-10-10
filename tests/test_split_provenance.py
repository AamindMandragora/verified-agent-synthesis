"""Tests for synthesis/split_provenance.py after the 2026-07-17 simplification.

Covers: (1) the single train/test vocabulary for BOTH datasets — GSM's old
"eval" alias must now be rejected; (2) the launch guard's mismatch behavior is
unchanged; (3) build_split_provenance (which replaced split_provenance_metadata)
produces the one canonical dict shape.

Run: /usr/local/bin/python3 -m pytest test_split_provenance.py -q
"""

import pytest

from synthesis.split_provenance import (
    BarSplitProvenanceError,
    SPLIT_NAMES_BY_DATASET,
    build_split_provenance,
    check_bar_split_provenance,
    validate_split_name,
)


# --- vocabulary: one set of names, no aliases ---------------------------------

def test_vocabulary_is_train_test_for_both_datasets():
    assert SPLIT_NAMES_BY_DATASET["gsm_symbolic"] == ("train", "test")
    assert SPLIT_NAMES_BY_DATASET["spider"] == ("train", "test")


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider"])
@pytest.mark.parametrize("name", ["train", "test"])
def test_valid_names_pass(dataset, name):
    validate_split_name(dataset, name, "--bar-split-name")


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "spider"])
def test_eval_alias_rejected_with_hint(dataset):
    with pytest.raises(BarSplitProvenanceError, match="'test'"):
        validate_split_name(dataset, "eval", "--bar-split-name")


def test_unknown_name_rejected():
    with pytest.raises(BarSplitProvenanceError):
        validate_split_name("gsm_symbolic", "dev", "split name")


def test_none_and_unknown_dataset_are_noops():
    validate_split_name("gsm_symbolic", None, "x")
    validate_split_name("no_such_dataset", "anything", "x")
    # SMILES has named splits since 2026-09-26 (two seed ranges).
    with pytest.raises(BarSplitProvenanceError):
        validate_split_name("smiles", "anything", "x")


# --- launch guard (behavior unchanged) ----------------------------------------

def test_guard_mismatch_refuses():
    with pytest.raises(BarSplitProvenanceError, match="mismatch"):
        check_bar_split_provenance(
            "gsm_symbolic", "splits.json", "train", 0.32, "test"
        )


def test_guard_missing_bar_split_refuses():
    with pytest.raises(BarSplitProvenanceError, match="--bar-from-train or --bar-from-test"):
        check_bar_split_provenance("spider", "splits.json", "train", 0.52, None)


def test_guard_match_passes():
    check_bar_split_provenance("gsm_symbolic", "splits.json", "train", 0.32, "train")


def test_guard_exempt_cases():
    check_bar_split_provenance("smiles", None, "train", 0.0, None)  # no bar
    check_bar_split_provenance("spider", None, "train", 0.5, None)  # no split file
    check_bar_split_provenance("spider", "splits.json", "train", 0.0, None)  # no bar


# --- build_split_provenance: the one dict shape -------------------------------

def test_build_full():
    got = build_split_provenance(
        gsm_split_file="/a/gsm.json",
        gsm_split_name="train",
        spider_split_file=None,
        spider_split_name=None,
        bar_split_name="train",
    )
    assert got == {
        "gsm_split_file": "/a/gsm.json",
        "gsm_split_name": "train",
        "spider_split_file": None,
        "spider_split_name": None,
        "bar_split_name": "train",
        "smiles_split_name": None,
        "smiles_trial": None,
        "smiles_seed_range": None,
    }


def test_build_stringifies_path_objects():
    from pathlib import Path

    got = build_split_provenance(spider_split_file=Path("/b/spider.json"))
    assert got["spider_split_file"] == "/b/spider.json"
    assert got["gsm_split_file"] is None


def test_build_defaults_all_none():
    got = build_split_provenance()
    assert set(got) == {
        "gsm_split_file", "gsm_split_name",
        "spider_split_file", "spider_split_name", "bar_split_name",
        "smiles_split_name", "smiles_trial", "smiles_seed_range",
    }
    assert all(v is None for v in got.values())


def test_old_function_is_gone():
    from synthesis import split_provenance

    assert not hasattr(split_provenance, "split_provenance_metadata")
