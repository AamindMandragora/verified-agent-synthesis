"""TDD test for the shared ring-balance helper in delimiter_hygiene.

A SMILES ring bond is opened and closed by the same label appearing twice: a
single digit (`C1CC1`), or `%` plus two digits for labels above 9 (`C%12CC%12`).
"Which labels are still open" is per-label odd/even parity: a label is open iff
it has appeared an odd number of times. Digits inside `[...]` are isotopes or
charges, not ring bonds, so they are ignored.

The grammar (a context-free approximation) cannot enforce this balance, so the
mask layer must: while any ring label is open, stopping (EOS / mark-complete)
is forbidden. This is applied to the shared grammar layer for every
grammar-constrained decoder, so it is a framework property, not a per-decoder
advantage.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delimiter_hygiene as D


def test_no_rings_balanced():
    assert D.ring_labels_open("CCO") == set()
    assert D.ring_balance_allows_stop("CCO") is True


def test_single_digit_open_then_closed():
    assert D.ring_labels_open("C1CC") == {"1"}
    assert D.ring_balance_allows_stop("C1CC") is False
    assert D.ring_labels_open("C1CC1") == set()
    assert D.ring_balance_allows_stop("C1CC1") is True


def test_two_digit_percent_label():
    assert D.ring_labels_open("C%12CC") == {"12"}
    assert D.ring_balance_allows_stop("C%12CC") is False
    assert D.ring_balance_allows_stop("C%12CC%12") is True


def test_digits_in_brackets_are_not_ring_bonds():
    # isotope 13, charge 2 -> not ring closures
    assert D.ring_labels_open("[13CH3]") == set()
    assert D.ring_balance_allows_stop("[NH4+]") is True
    # ring digit outside the bracket still counts
    assert D.ring_labels_open("[nH]1cccc") == {"1"}


def test_multiple_distinct_labels():
    # two open rings
    assert D.ring_labels_open("C1CC2") == {"1", "2"}
    # one closes, one stays open
    assert D.ring_labels_open("C1CC2CC1") == {"2"}
    # both close
    assert D.ring_balance_allows_stop("C1CC2CC1C2") is True


def test_reused_label_reopens():
    # naphthalene-style label reuse: 1 opens, closes, reopens, closes -> even parity
    assert D.ring_balance_allows_stop("C1CCCCC1CC1CCCCC1") is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
