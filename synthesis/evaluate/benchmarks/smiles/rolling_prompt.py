"""Rolling few-shot prompt for the SMILES generative eval (Fix A, 2026-06-10).

Each successive eval example's prompt grows to include the molecules accepted from
earlier examples — the same diversity mechanism the CARS baseline adapter's
``smiles_prompt_suffix`` uses in run_legacy_fixed_strategy.py. Kept byte-identical to
that lane's mechanism; see tests/test_smiles_rolling_suffix.py.

As of 2026-09-20 the other baseline lanes (GCD/IterGen/CRANE/unconstrained) no longer
do this -- each of those now sends the same fixed prompt on every draw (see
tests/test_smiles_lanes_do_not_feed_molecules_back_into_the_prompt.py). This module's
parity claim covers CARS only; it is not evidence about the other lanes.
"""

MAX_SUFFIX_CHARS = 45000


def cap_suffix(suffix: str) -> str:
    """Drop the oldest appended molecules if suffix exceeds ``MAX_SUFFIX_CHARS``."""
    if len(suffix) <= MAX_SUFFIX_CHARS:
        return suffix
    lines = suffix.split("\n")
    while len("\n".join(lines)) > MAX_SUFFIX_CHARS and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


def apply_suffix(example: dict, suffix_by_class: dict) -> None:
    """Prepend the accumulated rolling suffix onto this example's prompt, in place."""
    cls = str(example.get("class_name", ""))
    example["prompt"] = example["prompt"].rstrip() + suffix_by_class.get(cls, "")


def update_suffix(example: dict, suffix_by_class: dict, actual: str, syntax_valid: bool) -> None:
    """After scoring, append this molecule to the class's rolling suffix if it is a
    syntactically-valid, non-empty molecule (the baseline's exact gate)."""
    if syntax_valid and actual:
        cls = str(example.get("class_name", ""))
        suffix_by_class[cls] = cap_suffix(
            suffix_by_class.get(cls, "") + f" {actual}\nMolecule:"
        )
