"""The live progress view (runs/resynth-opus55-20261010/watch.py) reads one result line per eval example."""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "synthesis/evaluate/evaluator.py").read_text()


def test_result_line_follows_the_syntax_check():
    syntax_pass = SRC.index("example_syntax_pass = self._example_syntax_pass(")
    result = SRC.index("[EVAL] result example=")
    assert syntax_pass < result


def test_result_line_names_correct_syntax_answer_and_expected():
    line = SRC[SRC.index("[EVAL] result example="):][:400]
    for field in ("correct=", "syntax=", "answer=", "expected="):
        assert field in line
    assert re.search(r"flush=True", line)
