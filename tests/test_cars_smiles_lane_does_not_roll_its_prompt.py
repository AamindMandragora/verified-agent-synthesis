"""The CARS authors encode the prompt once and reuse it for every draw
(~/cars/cars/cars.py, CARS.__init__ / get_sample), so our CARS SMILES lane must not
append accepted molecules to the prompt, and no switch may bring that back."""
from pathlib import Path

import synthesis.evaluate.run_legacy_fixed_strategy as harness

SOURCE = Path(harness.__file__).read_text()


def test_the_harness_keeps_no_rolling_prompt_state():
    assert "smiles_prompt_suffix" not in SOURCE
    assert "_cap_suffix" not in SOURCE


def test_no_env_switch_can_turn_a_rolling_prompt_on():
    assert "CSD_SMILES_ROLLING_PROMPT" not in SOURCE
