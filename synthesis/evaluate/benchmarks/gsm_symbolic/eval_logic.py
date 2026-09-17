"""GSM-Symbolic evaluation logic delegated from the global evaluator."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimited_output import extract_last_delimited_span
from synthesis.evaluate.benchmarks.common.ungradable import UngradableExample

force_open_span = defaults.force_open_span
example_syntax_pass = defaults.example_syntax_pass_from_segments
accuracy_applicable = defaults.accuracy_applicable_always
accuracy_upper_bound = defaults.accuracy_upper_bound_with_remaining
final_accuracy_denominator = defaults.final_accuracy_denominator_all_examples
invalid_outputs_excluded = defaults.invalid_outputs_excluded_none
accuracy_definition = defaults.accuracy_definition_standard

_MAX_GSM_SCORING_EXPRESSION_CHARS = 512
_MAX_GSM_SCORING_EXPRESSION_TOKENS = 160
_MAX_GSM_SCORING_EXPRESSION_OPERATORS = 80
_MAX_GSM_SCORING_DIGIT_RUN = 64


def _is_pathological_gsm_scoring_expression(expression: str) -> bool:
    """Return whether a generated GSM expression is too large for safe scoring.

    The guard is intentionally much looser than the active split's gold answers:
    a 2026-06-29 audit found max gold length 119 chars and max operator count
    22 across train+eval. Larger generated strings are treated as invalid
    before native parser/prover code can wedge the evaluator.
    """

    text = str(expression or "").strip()
    if len(text) > _MAX_GSM_SCORING_EXPRESSION_CHARS:
        return True
    if len(text.split()) > _MAX_GSM_SCORING_EXPRESSION_TOKENS:
        return True
    if len(re.findall(r"[+\-*/%()]", text)) > _MAX_GSM_SCORING_EXPRESSION_OPERATORS:
        return True
    if re.search(r"\d{" + str(_MAX_GSM_SCORING_DIGIT_RUN) + r",}", text):
        return True
    return False


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    """Static ``gsm.lark`` for CRANE-faithful syntax scoring only.

    Constrained decode uses dataset-owned ``crane_valid_vars_gsm_grammar`` per
    example (see ``build_dynamic_parser``); ``--grammars-dir`` does not change
    adaptive CRANE decode masks.
    """
    return grammars_dir / "gsm.lark"


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.gsm_symbolic.dataset import (
        load_gsm_from_crane_folder,
    )
    from synthesis.project_defaults import default_gsm_source_dir

    indices = evaluator._load_gsm_split_indices()

    crane_dir = evaluator.gsm_source_dir
    if crane_dir is None:
        crane_dir = default_gsm_source_dir()
    if crane_dir is None:
        raise RuntimeError(
            "GSM-Symbolic evaluation requires a CRANE JSON folder. "
            "HuggingFace loading has been removed — set --gsm-source-dir or "
            "ensure synthesis.project_defaults.default_gsm_source_dir() resolves."
        )

    ds = load_gsm_from_crane_folder(
        crane_dir=crane_dir,
        limit=evaluator.sample_size,
        indices=indices,
    )
    return list(ds)


def _gsm_question_text(example: dict[str, Any]) -> str:
    # Prefer symbolic `{placeholder}` text over instantiated HF `question` when both exist.
    return (
        example.get("question_parsed")
        or example.get("original_question")
        or example.get("question", "")
    )


def format_prompt(evaluator: Any, example: dict[str, Any]) -> list[dict]:
    # Multi-turn chat delivery (system + 8 user/assistant pairs + question),
    # matching CRANE. Lifted GSM-1.5B unconstrained 22.0% -> 30.0% over the
    # flattened single-user-message form. generation.py passes a list straight
    # through as chat_messages; the CSD path templates from the same messages.
    from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import reasoning_with_symbolic_expr_messages

    return reasoning_with_symbolic_expr_messages(_gsm_question_text(example))


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """GSM prompt without CoT instructions (e.g. legacy IterGen grammar-masked generation)."""
    from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import symbolic_expression_only_prompt

    return symbolic_expression_only_prompt(_gsm_question_text(example))


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> list[dict]:
    """Same instructions as ``format_prompt``: reasoning then ``<<expression>>``."""
    return format_prompt(evaluator, example)


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    symbolic = example.get("answer_parsed", "")
    if symbolic:
        return symbolic
    answer_str = example.get("answer", "")
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_str)
    if match:
        return match.group(1)
    return answer_str


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        crane_valid_vars_gsm_grammar,
        extract_variables_from_mapping,
    )

    variable_types = example.get("variable_types") or {}
    if not isinstance(variable_types, dict):
        return None
    allowed_variables = extract_variables_from_mapping(variable_types)
    if not allowed_variables:
        return None

    cache_key = tuple(sorted(allowed_variables))
    parser_factory = evaluator._dynamic_parser_factory_cache.get(cache_key)
    if parser_factory is None:
        grammar_text = crane_valid_vars_gsm_grammar(list(cache_key))
        parser_factory = create_lark_dafny_parser(
            grammar_text,
            env["VerifiedDecoderAgent"],
            env["_dafny"],
            start="start",
            tokenizer=env["tokenizer"],
            dfa_mode="grammar_strict",
            apply_forbidden_token_filter=False,
            constrained_span_opener="<<",
            complete_start="expr",
        )
        evaluator._dynamic_parser_factory_cache[cache_key] = parser_factory

    return parser_factory(env["lm"]._Tokens)


def extract_actual(evaluator: Any, scored_output: str, example: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    actual, found = extract_last_delimited_span(scored_output)
    source = "last_visible_span" if found else "none"
    return actual, source, None


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    vt = example.get("variable_types", {})
    # Nothing here is caught, deliberately. Turning an unreadable field into an
    # empty dict does not just lose the error -- {} is falsy, so the check below
    # skips the symbolic grader entirely and scores this example by numeric
    # comparison instead. That is a different measurement (a proof of algebraic
    # equivalence versus "do these two numbers match"), averaged in with the
    # rest of the split, with nothing recording that the method changed.
    #
    # Pinned by tests/test_unreadable_types_does_not_switch_graders.py.
    if isinstance(vt, str):
        vt = ast.literal_eval(vt)
    if not isinstance(vt, dict):
        raise UngradableExample(
            "GSM variable_types must be a mapping of variable name to type, "
            f"got {type(vt).__name__}: {vt!r}"
        )
    # There is exactly one way to grade GSM here, and it is CRANE's: prove the
    # model's formula equals the gold's for every legal value of every variable.
    #
    # A numeric comparison used to sit below this as a fallback. It is gone.
    # CRANE has no such thing -- its parse_answer (gsm_symbolic.py:28-56) either
    # runs validate_expression_equivalence or leaves correct = False -- so every
    # example that fallback graded was measured by a method CRANE does not have
    # and then averaged into a figure reported against CRANE's published number.
    # It was not a rare path either: answer_parsed defaults to '' (dataset.py:65
    # and :164), which is falsy, so any source row missing that field took it.
    #
    # A row that cannot be graded this way is a dataset problem, not evidence
    # about the model, so it stops the run.
    # Pinned by tests/test_unreadable_types_does_not_switch_graders.py.
    if not vt:
        raise UngradableExample(
            "GSM example has no variable_types, so its answer cannot be checked "
            "for symbolic equivalence the way CRANE does. Refusing to fall back "
            f"to a numeric comparison. Question: {str(example.get('question', ''))[:200]!r}"
        )
    if not example.get("answer_parsed"):
        raise UngradableExample(
            "GSM example has no parsed gold expression (answer_parsed), so there "
            "is nothing to prove the model's answer equivalent to. Refusing to "
            f"fall back to a numeric comparison. Question: {str(example.get('question', ''))[:200]!r}"
        )
    return evaluator._gsm_symbolic_equivalence(actual, expected, vt)


def get_generation_runner():
    from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd

    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    """Returns the per-example dynamic Lark parser (used only as a secondary
    diagnostic; the primary GSM syntax check is check_syntax / CRANE-faithful)."""
    from lark import Lark
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        build_dynamic_grammar,
        build_numeric_only_grammar,
        extract_variables_from_mapping,
    )

    if example is None:
        return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")

    variable_types = example.get("variable_types") or {}
    if not isinstance(variable_types, dict):
        grammar_text = build_numeric_only_grammar(evaluator._get_grammar_text())
        return Lark(grammar_text, start="start", parser="lalr")
    allowed_variables = extract_variables_from_mapping(variable_types)
    if not allowed_variables:
        grammar_text = build_numeric_only_grammar(evaluator._get_grammar_text())
        return Lark(grammar_text, start="start", parser="lalr")

    cache_key = tuple(sorted(allowed_variables))
    parser = evaluator._syntax_parser_cache.get(cache_key)
    if parser is None:
        grammar_text = build_dynamic_grammar(evaluator._get_grammar_text(), list(cache_key))
        parser = Lark(grammar_text, start="start", parser="lalr")
        evaluator._syntax_parser_cache[cache_key] = parser
    return parser


# ---------------------------------------------------------------------------
# CRANE-faithful GSM syntax check on the FINAL <<...>> block only.
#
# CRANE reports GSM syntax as `parses` from get_avgs.py -> check_gsm_parsed:
# it grades ONLY the last <<...>> block per example. The block is rejected if
# it is empty, contains a brace, or contains "round("; it must be wrapped in
# << >>; then the full block (delimiters included) is parsed against the GSM
# grammar. We mirror that here so our syntax_rate is the same kind of number
# CRANE reports (final block only, any identifier allowed, per-example).
#
# This replaces the OLD metric (all visible spans + per-example variable
# restriction) which was stricter than CRANE and caused phantom-variable
# expressions to count as syntax failures even though CRANE would pass them.
# The per-example CRANE-shaped grammar (see crane_valid_vars_gsm_grammar) is used at
# DECODE TIME in build_dynamic_parser to restrict what the LM can generate — this only
# changes the post-hoc scoring metric.
# ---------------------------------------------------------------------------

_FINAL_BLOCK_PARSERS: dict[str, Any] = {}


def _final_block_parser(evaluator: Any):
    grammar_text = evaluator._get_grammar_text()
    parser = _FINAL_BLOCK_PARSERS.get(grammar_text)
    if parser is None:
        from lark import Lark

        parser = Lark(grammar_text, start="syncode", parser="lalr")
        _FINAL_BLOCK_PARSERS[grammar_text] = parser
    return parser


def _extract_final_block(output: str) -> str:
    # CRANE parse_completion_regex: response[rfind('<<') : rfind('>>') + 2].
    # When a delimiter is missing, rfind returns -1 and the slice is empty.
    return output[output.rfind("<<") : output.rfind(">>") + 2]


def _check_gsm_parsed(block: str, parser) -> bool:
    # Faithful port of CRANE src/get_avgs.py check_gsm_parsed.
    if block == "" or "{" in block or "}" in block or "round(" in block:
        return False
    if _is_pathological_gsm_scoring_expression(block):
        return False
    if not block.startswith("<<") or not block.endswith(">>"):
        return False
    try:
        parser.parse(block)
        return True
    except Exception:
        return False


def check_syntax(
    evaluator: Any, output: str, example: dict[str, Any] | None
) -> tuple[bool, list[tuple[str, bool]]]:
    """CRANE-faithful GSM syntax check on the FINAL ``<<...>>`` block only.

    Returns ``(parses, [(block, parses)])`` when a final block exists, else
    ``(False, [])`` when no ``<<...>>`` block is present. Composed with the
    default ``example_syntax_pass_from_segments`` (``bool(segments) and
    all_valid``) this yields exactly CRANE's per-example pass/fail.

    Semantics:
    - Only the FINAL << >> block is checked (CRANE scores the last answer block)
    - Any identifier name is allowed (static grammar, VARIABLE = any CNAME)
    - Explicit rejections: block containing '{', '}', or 'round('
    - Must be a complete parse of the full delimited block including << >>
    """
    block = _extract_final_block(output)
    if block == "":
        return False, []
    if _is_pathological_gsm_scoring_expression(block):
        return False, [(block, False)]
    parses = _check_gsm_parsed(block, _final_block_parser(evaluator))
    return parses, [(block, parses)]


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}
