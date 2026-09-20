"""A file copy silently deleted these once; make sure it cannot happen again.

Why this file exists
---------------------
Commit ``dc41fd63`` (Sep 18) copied an older version of
``run_legacy_fixed_strategy.py`` over the Aug 4 commit (``9d009711``) that had
made the baseline lanes faithful to their papers. That copy deleted 501 lines
and 19 functions -- among them the real IterGen back-up-and-retry loop for
Spider and the Transformers-5 compatibility shim IterGen needs to run at all.
The baseline campaign that week ran on code missing those repairs, and the
review that would normally catch a regression only looked at that day's
commits, so it missed a file-level overwrite.

This test does not check that any of these functions behaves correctly --
other tests do that. It only checks that each one still exists. A future
"restore an older copy of this file" mistake then fails a test immediately
instead of silently shipping a baseline that looks like it ran but did not.
"""

from __future__ import annotations

from synthesis.evaluate import run_legacy_fixed_strategy as legacy_runner

# Every name here was deleted by the Sep 18 file-copy regression and restored
# by commit 3a236e14. Listed explicitly, not matched by pattern: a pattern
# loose enough to survive renames would also be loose enough to miss a
# deletion.
REQUIRED_FUNCTION_NAMES = [
    "_itergen_generate_spider",
    "_itergen_spider_schema",
    "_itergen_generation_stop_token_ids",
    "_install_itergen_transformers_compat",
    "_itergen_generation_kwargs",
    "_gcd_generation_kwargs",
    "_crane_stop_words",
    "_crane_generation_kwargs",
    "_crane_prompt_for_generation",
    "_crane_completion_for_scoring",
    "_crane_adaptive_surface",
    "_legacy_fixed_max_new_tokens",
    "_completion_for_dataset",
    "_crane_smiles_prompt",
    "_crane_smiles_span",
]


def test_every_restored_baseline_function_is_still_defined():
    missing = [
        name for name in REQUIRED_FUNCTION_NAMES
        if not callable(getattr(legacy_runner, name, None))
    ]
    assert missing == [], (
        f"{len(missing)} function(s) are gone from run_legacy_fixed_strategy.py: "
        f"{missing}. This is the exact shape of the Sep 18 regression (a whole-file "
        "copy silently dropping functions the baselines depend on) -- restore them, "
        "do not just add stubs."
    )
