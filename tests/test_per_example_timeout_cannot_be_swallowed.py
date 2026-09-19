"""The per-example time limit must end the example even when code under it
catches ``Exception`` (the parser helpers and IterGen's decoder both do)."""
import time

import pytest

from synthesis.evaluate.evaluator import PerExampleTimeout, _PerExampleTimer


def test_timeout_escapes_a_blanket_except_exception():
    swallowed = []
    with pytest.raises(PerExampleTimeout):
        with _PerExampleTimer(0.2):
            deadline = time.time() + 3
            while time.time() < deadline:
                try:
                    time.sleep(0.01)
                except Exception as e:  # what parser_utils does
                    swallowed.append(e)
    assert swallowed == []
