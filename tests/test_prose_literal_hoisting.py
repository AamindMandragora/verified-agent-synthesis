"""Long prose string literals are hidden from the prover, then restored for compiling.

Measured 2026-09-25 (runs/resynth-20260923/verify_timeout_probe/hoist_check): a
3,807-character guidance literal made AuthorBody take 1,031 s to verify. An opaque
function holding it still took 155 s; a body-less function took 14 s. So the proof
runs against `function {:extern} {:axiom} ProseLiteralN(): string` (contents
unknown), and the compiled program gets `function ProseLiteralN(): string { <text> }`.
Hiding a value from the prover can only make a proof fail, never pass wrongly.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from synthesis.generate.generator import (
    PROSE_LITERAL_MIN_CHARS,
    StrategyGenerator,
    compile_source,
    hoist_prose_literals,
    inject_strategy_into_template,
    load_template,
)

LONG = "Read the question carefully. " * 10


def _gen() -> StrategyGenerator:
    gen = StrategyGenerator.__new__(StrategyGenerator)
    gen._template = load_template()
    return gen


def _author_body(source: str) -> str:
    return source[source.index("method AuthorBody("):]


def test_verification_source_hides_long_literal_behind_bodyless_function():
    body = f'var guidance := "{LONG}";'
    source = _gen().inject_strategy(body)
    assert "function {:extern} {:axiom} ProseLiteral0(): string\n" in source
    assert "var guidance := ProseLiteral0();" in source
    assert LONG not in source


def test_compile_source_restores_exact_text():
    body = f'var guidance := "{LONG}";'
    source = _gen().compile_source(body)
    assert f'function ProseLiteral0(): string {{ "{LONG}" }}' in source
    assert "var guidance := ProseLiteral0();" in source
    assert "{:extern}" not in source.split("module GeneratedCSD")[1].split("method")[0]


def test_short_literals_stay_inline_and_sources_match():
    body = 'if tok == "<<" || tok == "COUNT(" { cost := 0; }'
    gen = _gen()
    assert gen.inject_strategy(body) == gen.compile_source(body)
    assert body in gen.inject_strategy(body)
    assert "ProseLiteral" not in gen.inject_strategy(body)


def test_quotes_in_comments_char_literals_and_escapes_are_not_split():
    long_with_escape = 'say \\"hi\\" ' + "x" * PROSE_LITERAL_MIN_CHARS
    body = (
        f'// a comment with "{LONG}" in it\n'
        "var q := '\"';\n"
        "var x' := 1;\n"
        f'var g := "{long_with_escape}";\n'
    )
    new_body, literals = hoist_prose_literals(body)
    assert f'// a comment with "{LONG}" in it' in new_body
    assert "var q := '\"';" in new_body
    assert "var x' := 1;" in new_body
    assert "var g := ProseLiteral0();" in new_body
    assert literals == [f'"{long_with_escape}"']


def test_every_long_literal_gets_its_own_function_in_order():
    a, b = "a" * PROSE_LITERAL_MIN_CHARS, "b" * PROSE_LITERAL_MIN_CHARS
    new_body, literals = hoist_prose_literals(f'var p := "{a}" + "{b}";')
    assert new_body == "var p := ProseLiteral0() + ProseLiteral1();"
    assert literals == [f'"{a}"', f'"{b}"']


def test_module_functions_match_generator_methods():
    body = f'var g := "{LONG}";'
    gen = _gen()
    assert inject_strategy_into_template(body) == gen.inject_strategy(body)
    assert compile_source(body) == gen.compile_source(body)


def test_synthesis_compiler_skips_reverification():
    src = Path(__file__).resolve().parents[1] / "synthesis/evaluate/feedback_loop.py"
    text = src.read_text()
    assert re.search(r'compiler = DafnyCompiler\([^)]*extra_args=\["--no-verify"\]', text)
    assert "compiler.compile(full_code," not in text
    assert "compiler.compile(winner_full_code," not in text


# --- slow: real Dafny on a strategy that timed out in a live run -------------

_SLOW_RUN = Path.home() / (
    "csd-generation/runs/resynth-20260923/spider_4b_20260924_010706_972d73/"
    "results/progress_report.json"
)
_LIBRARY = Path(__file__).resolve().parents[1] / "synthesis/verify/library"


@pytest.mark.skipif(not _SLOW_RUN.exists() or shutil.which("dafny") is None,
                    reason="needs the focal run record and dafny")
def test_attempt_that_timed_out_now_verifies_and_compiles_quickly(tmp_path):
    from synthesis.verify.compiler import DafnyCompiler
    from synthesis.verify.tooling import build_default_verifier

    attempts = json.loads(_SLOW_RUN.read_text())["attempts"]
    body = next(a for a in attempts if a["attempt_number"] == 24)["strategy_code"]
    gen = _gen()

    verified = build_default_verifier().verify(
        gen.inject_strategy(body)
    )
    assert verified.success, verified.get_error_summary()[:2000]

    compiled = DafnyCompiler(output_dir=tmp_path, extra_args=["--no-verify"]).compile(
        gen.compile_source(body), "GeneratedCSD"
    )
    assert compiled.success, compiled.get_error_summary()[:2000]
    py_text = "\n".join(p.read_text() for p in compiled.output_dir.rglob("*.py"))
    _, literals = hoist_prose_literals(body)
    assert literals
    for literal in literals:
        plain = next(chunk for chunk in re.split(r"\\.", literal[1:-1]) if len(chunk) >= 40)
        assert plain[:40] in py_text
