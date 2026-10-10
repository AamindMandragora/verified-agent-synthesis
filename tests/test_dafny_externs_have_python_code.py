"""Every {:extern} the Dafny library declares must have Python code behind it.

Dafny verifies a call to an extern without ever seeing its body, so a missing
Python implementation passes verification and then crashes every example at
runtime (AttributeError). That happened with SpanAppearsInPrompt,
SpanResemblanceToPromptExamples, ParseG and ExtractContentExtern.
"""

from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIBRARY = REPO_ROOT / "synthesis/verify/library/VerifiedAgentSynthesis.dfy"
COMMON = REPO_ROOT / "synthesis/evaluate/benchmarks/common"


def _class_bodies(src: str) -> dict[str, str]:
    bodies = {}
    for m in re.finditer(r"\n\s*class\s+(\w+)[^{\n]*\{", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        bodies[m.group(1)] = src[m.end():i]
    return bodies


def _externs(body: str) -> set[str]:
    return set(
        re.findall(
            r"(?:function|predicate|method)\s*\{:extern\}[^\n]*?\s(\w+)\s*[\(<]",
            body,
        )
    )


def _methods_of(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"class {class_name} not found in {path}")


def test_every_lm_extern_is_implemented_on_each_backend():
    externs = _externs(_class_bodies(LIBRARY.read_text())["LM"])
    base = _methods_of(COMMON / "model_utils.py", "_TensorizedLMBase")
    for backend in ("VllmLM", "HuggingFaceLM"):
        have = base | _methods_of(COMMON / "model_utils.py", backend)
        missing = externs - have
        assert not missing, f"{backend} lacks LM externs: {sorted(missing)}"


def test_every_parser_extern_is_implemented():
    missing = _externs(_class_bodies(LIBRARY.read_text())["Parser"]) - _methods_of(
        COMMON / "parser_utils.py", "SyncodeDafnyParser"
    )
    assert not missing, f"Parser externs with no Python code: {sorted(missing)}"


class _CodePoint(str):
    pass


class _Seq(list):
    pass


_FAKE_DAFNY = types.SimpleNamespace(
    CodePoint=_CodePoint, SeqWithoutIsStrInference=lambda xs: _Seq(xs)
)


def test_every_csdhelpers_extern_is_bound_when_modules_load():
    from synthesis.evaluate.benchmarks.gsm_symbolic import environment

    externs = _externs(_class_bodies(LIBRARY.read_text())["CSDHelpers"])
    agent = types.SimpleNamespace(CSDHelpers=type("CSDHelpers", (), {}))
    environment.bind_csdhelpers_externs(agent, _FAKE_DAFNY)
    missing = {n for n in externs if not hasattr(agent.CSDHelpers, n)}
    assert not missing, f"CSDHelpers externs never bound: {sorted(missing)}"

    as_dafny = lambda t: _Seq(map(_CodePoint, t))
    got = agent.CSDHelpers.ExtractContentExtern(
        as_dafny("Molecule: <<C=CC(=O)OC>> done"), as_dafny("<<"), as_dafny(">>")
    )
    assert isinstance(got, _Seq) and all(isinstance(c, _CodePoint) for c in got)
    assert "".join(got) == "C=CC(=O)OC"

    load_src = ast.get_source_segment(
        Path(environment.__file__).read_text(),
        next(
            n
            for n in ast.parse(Path(environment.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "load_compiled_modules"
        ),
    )
    assert "bind_csdhelpers_externs(VerifiedDecoderAgent, _dafny)" in load_src


def test_extract_content_extern_meets_its_dafny_contract():
    from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
        extract_content_between,
    )

    assert extract_content_between("a <<x+1>> b", "<<", ">>") == "x+1"
    assert extract_content_between("<<a>><<b>>", "<<", ">>") == "a"
    assert extract_content_between("no span here", "<<", ">>") == ""
    assert extract_content_between("<<unclosed", "<<", ">>") == ""
    assert extract_content_between(">> <<a>>", "<<", ">>") == "a"
    # Contract: non-empty content means input == pre + start + content + end + post.
    for text in ["q <<1*2>> r", "<<>>", "<<a<<b>>"]:
        c = extract_content_between(text, "<<", ">>")
        if c:
            assert "<<" + c + ">>" in text
