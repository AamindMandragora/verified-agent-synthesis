"""The SMILES scorer writes no debug files.

evaluate_smiles_output carried a leftover block from an earlier session that
appended NDJSON to /tmp/debug-d7d0bd-*. It was capped at five writes and wrapped
in a bare except, so it was harmless, but a baseline run should not be writing
scratch files to /tmp.
"""

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis.evaluate.benchmarks.smiles import metrics


def test_the_scorer_opens_no_files() -> None:
    source = inspect.getsource(metrics.evaluate_smiles_output)

    assert "/tmp/" not in source
    assert "open(" not in source


def test_the_module_names_no_debug_path() -> None:
    source = Path(metrics.__file__).read_text()

    assert "debug-d7d0bd" not in source
    assert "region agent log" not in source
