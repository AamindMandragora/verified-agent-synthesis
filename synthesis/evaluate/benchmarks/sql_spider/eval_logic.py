"""Spider evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
import re
import logging
from typing import Any

from synthesis.evaluate.benchmarks.common import benchmark_defaults as defaults
from synthesis.evaluate.benchmarks.common.delimiter_hygiene import walk_spans
from synthesis.evaluate.benchmarks.sql_spider.prompts import (
    SpiderPromptParts,
    format_spider_itergen_aligned_prompt,
    format_spider_messages,
)
from synthesis.evaluate.benchmarks.sql_spider.output_contract import validate_bare_sql


_CONTRACT_LOG = logging.getLogger(__name__)

def force_open_span() -> bool:
    # The SQL answer is parser-governed from the first token, so the runtime
    # opens the span itself: the output starts as "<<" and the strategy starts
    # inside it. The prompt is unchanged by this.
    return True


constrained_temperature = defaults.constrained_temperature


def example_syntax_pass(
    all_valid_syntax: bool,
    segments: list,
    aux: dict | None,
) -> bool:
    # Same verdict as before the visible span: the bare-SQL output contract
    # (validate_bare_sql, the IterGen-aligned rule), run in extract_actual on the
    # span content with the runtime's `<<`/`>>` stripped.
    return bool(aux and aux.get("syntax_valid"))


accuracy_applicable = defaults.accuracy_applicable_always
accuracy_upper_bound = defaults.accuracy_upper_bound_with_remaining
final_accuracy_denominator = defaults.final_accuracy_denominator_all_examples
invalid_outputs_excluded = defaults.invalid_outputs_excluded_none
accuracy_definition = defaults.accuracy_definition_standard


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    return grammars_dir / "sql.lark"


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

    split_indices = evaluator._load_spider_split_indices()
    ds = load_spider(
        source="auto",
        limit=evaluator.sample_size,
        random_sample=split_indices is None,
        seed=evaluator.sample_seed,
        indices=split_indices,
    )
    return list(ds)


def _structured_or_compat(evaluator: Any, prompt: SpiderPromptParts) -> str | SpiderPromptParts:
    """Bind model identity while retaining plain-string compatibility for callers."""
    bound = prompt.with_model_name(getattr(evaluator, "model_name", None))
    if evaluator is None or not hasattr(evaluator, "model_name"):
        return str(bound)
    return bound

def format_prompt(evaluator: Any, example: dict[str, Any]) -> str | SpiderPromptParts:
    # Flattened few-shot format for synthesis: concise output keeps step budget
    # well within limits so constrained <<SQL>> spans can complete.
    # The inline few-shot example is LOAD-BEARING: a zero-shot IterGen-aligned
    # prompt (no example) made both Qwen Instruct models stop emitting << >>
    # entirely -> syntax collapsed to 0.7%/3.3% and accuracy fell 57.3->43.7 (7B)
    # and 44->20.7 (1.5B), confirmed on seed334 held-out 300 on 2026-06-05.
    # (Multi-turn lifted unconstrained 38%->44% but exhausts max_steps in
    # constrained mode and produces 0%/0% — confirmed 2026-05-28.)
    #
    # The only Spider prompt is IterGen's EXACT bare prompt (no few-shot, no
    # << >> instruction), for a fair head-to-head. It stays byte-identical now
    # that the runtime opens the span: the `<<` is seeded into the OUTPUT, not
    # into the prompt, so the model sees exactly what IterGen's model sees.
    # SPIDER_PARITY_LEGACY_PROMPT=1: match run_itergen_legacy_adapter's
    # expression_only prompt (used to freeze spider_legacy_n5).
    import os

    if os.environ.get("SPIDER_PARITY_LEGACY_PROMPT") == "1":
        return format_prompt_expression_only(evaluator, example)

    return _structured_or_compat(evaluator, format_spider_itergen_aligned_prompt(example))


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str | SpiderPromptParts:
    """Hard-mask / constrained decoders: IterGen's bare prompt."""
    return _structured_or_compat(evaluator, format_spider_itergen_aligned_prompt(example))


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> list[dict]:
    """Legacy CRANE-style runs: require explicit reasoning before the delimited query."""
    return format_spider_messages(
        example,
        instruction=(
            "Write a SINGLE SQL query answering the question, using ONLY the tables "
            "and columns in the schema.\n\n"
            "Reason step by step (tables, joins, filters). "
            "Then output SQL: followed by your query wrapped in << >>. "
            "Stop after the closing >>."
        ),
        few_shot_answer_line=(
            "Let's think step by step. We only need the singer table. "
            "SQL: <<SELECT count(*) FROM singer>>"
        ),
    )


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    return (example.get("query") or "").strip()


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    return None


def _active_removed_terminal_token_count(evaluator: Any) -> int:
    evidence = getattr(evaluator, "_active_generation_token_evidence", None)
    if not isinstance(evidence, dict):
        return 0
    return len(evidence.get("removed_terminal_token_ids", ()))


def _span_content_for_scoring(scored_output: str) -> str:
    """The content of the last closed `<< >>` span, verbatim.

    An opened but never-closed span is not an answer (it scores as empty), the same rule
    GSM and SMILES apply. Only an output with no delimiter at all, i.e. a legacy
    unconstrained baseline, is scored whole.
    """
    # Qwen3.5 opens every answer with a (usually empty) <think>...</think> block. It is
    # reasoning, not the answer: drop it. A block that never closes means the model ran
    # out of budget while still reasoning, so everything from there on is dropped too.
    text = re.sub(r"<think>.*?</think>", "", scored_output or "", flags=re.DOTALL)
    text = text.split("<think>", 1)[0].strip()
    spans = walk_spans(text).spans
    if not spans:
        # A span-free baseline continues the few-shot format and opens with "SQL:".
        # The label is not part of the query, and the answer ends at the first blank line:
        # after it the model starts over ("SQL: ..." again) until the budget runs out.
        return re.sub(r"^sql\s*:\s*", "", text, flags=re.IGNORECASE).split("\n\n", 1)[0].strip()
    closed = [span for span in spans if span.closed]
    return closed[-1].content if closed else ""


def extract_actual(evaluator: Any, scored_output: str, example: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    # Score the SPAN CONTENT, verbatim: the runtime's `<<`/`>>` are stripped and
    # everything inside is handed to the same output contract as before, with no
    # reformatting, so a `<<SELECT ...>>` output scores exactly like the bare
    # query did. An output with no span at all is scored whole, as before.
    parser = evaluator._get_syntax_parser(example) if hasattr(evaluator, "_get_syntax_parser") else None
    result = validate_bare_sql(_span_content_for_scoring(scored_output), parser=parser)
    removed_terminal_token_count = _active_removed_terminal_token_count(evaluator)
    _CONTRACT_LOG.info(
        "[spider-output-contract] contract_valid=%s rejection_reason=%s "
        "raw_chars=%d candidate_chars=%d removed_terminal_token_count=%d",
        result.accepted,
        result.rejection_reason,
        len(result.raw_output),
        len(result.sql or ""),
        removed_terminal_token_count,
    )
    aux = {
        "syntax_valid": result.accepted,
        "removed_terminal_token_count": removed_terminal_token_count,
        "output_contract_valid": result.accepted,
        "output_rejection_reason": result.rejection_reason,
    }
    if result.accepted:
        return result.sql, "bare_sql", aux
    return None, "spider_output_contract_rejected", aux


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    if not actual or not expected:
        return False
    from synthesis.evaluate.benchmarks.sql_spider.executor import prediction_matches_gold

    return prediction_matches_gold(actual, example)


def get_generation_runner():
    from synthesis.evaluate.benchmarks.sql_spider import generation

    def _forced_span_runner(*args, **kwargs):
        kwargs.setdefault("force_open_span", True)
        kwargs.setdefault("constrained_temperature", constrained_temperature())
        return generation.run_crane_csd(*args, **kwargs)

    return _forced_span_runner


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}
