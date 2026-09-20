"""IterGen's length cap has to count new tokens, because that is what it is shown.

``forward()`` and ``finished()`` evaluate the stopping criteria on the generated
tokens alone (``session_tokens[:, start_from:]``), but ``start()`` used to set the
length the criteria compare against to ``max_new_tokens + len(prompt)``. The
prompt was therefore added to the budget twice over, and the effective ceiling
became the session ceiling. The SMILES IterGen lane, capped at 400 by
--eval-max-steps, logged examples at generated_tokens=1283 on 2026-09-20.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ITERGEN = REPO / "legacy" / "itergen"
for path in (str(REPO), str(ITERGEN)):
    if path not in sys.path:
        sys.path.insert(0, path)

from itergen.main import IterGen

budget = IterGen._generated_token_budget


def test_the_prompt_does_not_buy_extra_tokens() -> None:
    assert budget(4496, 400, 900) == 400


def test_a_long_prompt_eats_into_the_session_ceiling() -> None:
    assert budget(4496, 400, 4200) == 296


def test_no_cap_means_whatever_the_session_ceiling_leaves() -> None:
    assert budget(4496, None, 900) == 3596


def test_a_prompt_that_fills_the_ceiling_still_leaves_one_token() -> None:
    assert budget(4496, 400, 5000) == 1
