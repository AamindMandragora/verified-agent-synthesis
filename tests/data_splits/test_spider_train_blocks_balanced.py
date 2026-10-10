"""Every 50-example block of the Spider train side must match the held-out difficulty mix.

Synthesis trains on train[0:50] and confirms on later blocks; held-out is the 300 test examples.
If a block is harder or easier than held-out, training scores stop predicting held-out scores.
"""
import collections
import json
import pathlib

SPLIT = pathlib.Path(__file__).resolve().parents[2] / "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
LEVELS = ("easy", "medium", "hard", "extra")
BLOCK = 50


def _mix(split, indices):
    counts = collections.Counter(split["difficulty_by_index"][str(i)] for i in indices)
    return {level: counts[level] for level in LEVELS}


def test_each_train_block_matches_heldout_mix_within_one_example():
    split = json.loads(SPLIT.read_text())
    train, test = split["train_indices"], split["test_indices"]
    heldout = _mix(split, test)
    for start in range(0, len(train), BLOCK):
        block = _mix(split, train[start:start + BLOCK])
        for level in LEVELS:
            expected = heldout[level] * BLOCK / len(test)
            assert abs(block[level] - expected) <= 1, (start, level, block[level], expected)


def test_train_and_test_sets_are_unchanged_and_disjoint():
    split = json.loads(SPLIT.read_text())
    assert len(set(split["train_indices"])) == len(split["train_indices"]) == 300
    assert len(set(split["test_indices"])) == 300
    assert not set(split["train_indices"]) & set(split["test_indices"])
    assert _mix(split, split["train_indices"]) == _mix(split, split["test_indices"])
