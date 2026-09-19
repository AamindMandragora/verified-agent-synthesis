"""The "how many tokens can come next" count must use the exact grammar check.

Why: helpers use this count to decide "this is a narrow spot" and "stopping is allowed
because nothing can follow". The fast mask is loose, so a count taken from it says
thousands where the true number is far smaller. The exact check is too slow to run on
every token, so the count stops as soon as it reaches the cap the caller asks for.
"""
import pytest

from tests.test_prefix_check_does_not_depend_on_earlier_calls import _parser

PREFIX = list("O=C(O)C(=C)")


@pytest.fixture(scope="module")
def acrylates():
    return _parser("smiles_acrylates")


def test_the_count_matches_trying_every_token_text(acrylates):
    from synthesis.evaluate.benchmarks.common.parser_utils import dafny_seq_to_str

    text = "".join(PREFIX)
    mask = acrylates._get_accept_mask_for_text(text)
    offered = {dafny_seq_to_str(acrylates._token_list[i]) for i in mask.nonzero().flatten().tolist()}
    every_text = {dafny_seq_to_str(token) for token in acrylates._token_list} - {""}
    valid = {t for t in every_text if acrylates._is_valid_prefix(text + t)}

    # The mask is loose in one direction only, with one known gap: it never offers
    # runs of the wildcard atom such as "**". Those can never be picked, so they
    # do not count as possible next tokens.
    assert all("**" in t for t in valid - offered)
    assert 0 < len(valid & offered) < len(offered)
    assert acrylates.ValidNextTokenCountUpTo(PREFIX, 10 ** 6) == len(valid & offered)


@pytest.mark.parametrize("cap", [0, 1, 5])
def test_the_count_stops_at_the_cap(acrylates, cap):
    assert acrylates.ValidNextTokenCountUpTo(PREFIX, cap) == cap


def test_the_loose_count_is_gone(acrylates):
    assert not hasattr(acrylates, "ValidNextTokenCount")


def test_the_token_list_agrees_with_the_count(acrylates):
    """The Dafny contract says the count is the length of this list, so the list must be exact too."""
    listed = acrylates.ValidNextTokens(PREFIX)
    text = "".join(PREFIX)
    assert all(acrylates._is_valid_prefix(text + token) for token in listed)
    assert len(set(listed)) == acrylates.ValidNextTokenCountUpTo(PREFIX, 10 ** 6)
