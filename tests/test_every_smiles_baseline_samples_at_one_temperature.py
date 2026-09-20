"""Every SMILES baseline lane samples at the same temperature, 1.0 (user ruling 2026-09-20)."""
import re
from pathlib import Path

from synthesis.evaluate import run_legacy_fixed_strategy as harness

SOURCE = Path(harness.__file__).read_text()


def test_the_shared_smiles_temperature_is_one():
    assert harness._SMILES_TEMPERATURE == 1.0


def test_gcd_and_crane_use_the_shared_temperature():
    assert harness._gcd_generation_kwargs("smiles")["temperature"] == harness._SMILES_TEMPERATURE
    assert harness._crane_generation_kwargs("smiles")["temperature"] == harness._SMILES_TEMPERATURE


def test_no_smiles_lane_writes_its_own_temperature_number():
    literals = [t for t in re.findall(r"temperature\s*=\s*(0\.\d+)", SOURCE) if float(t) > 0]  # 0.0 is greedy
    assert literals == [], f"hard-coded sampling temperatures remain: {literals}"
