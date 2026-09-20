"""GSM's "gcd" strategy must use our own greedy grammar-constrained path, not IterGen.

Why this file exists
---------------------
``crane_repo_runner._STRATEGY_TO_CRANE_MODE`` mapped ``"gcd": ("itergen", False)``: CRANE
main.py has no separate GCD mode, so "gcd" was implemented there as IterGen-with-reasoning-off.
That is not what GCD (greedy grammar-constrained decoding) means anywhere else in this
codebase -- Spider and SMILES GCD both go through ``run_gcd_legacy_adapter``, our own
SynCode-backed hard-mask decoder. GSM "gcd" must go through the same path, using
``grammars/gsm.lark`` and the shared GSM prompt, not through the CRANE-repo subprocess.
"""

from __future__ import annotations

from synthesis.evaluate.run_legacy_fixed_strategy import _routes_through_crane_repo


def test_gsm_gcd_does_not_route_through_crane_repo():
    assert _routes_through_crane_repo("gsm_symbolic", "gcd") is False


def test_gsm_unconstrained_still_routes_through_crane_repo():
    assert _routes_through_crane_repo("gsm_symbolic", "unconstrained") is True


def test_gsm_crane_still_routes_through_crane_repo():
    assert _routes_through_crane_repo("gsm_symbolic", "crane") is True


def test_gsm_itergen_still_routes_through_crane_repo():
    assert _routes_through_crane_repo("gsm_symbolic", "itergen") is True


def test_spider_gcd_does_not_route_through_crane_repo():
    # Spider was never routed through the CRANE repo at all (no Spider support there);
    # unaffected by this fix, kept here to pin that it stays that way.
    assert _routes_through_crane_repo("spider", "gcd") is False


def test_spider_unconstrained_does_not_route_through_crane_repo():
    assert _routes_through_crane_repo("spider", "unconstrained") is False
