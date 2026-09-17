"""SMILES evaluation logic delegated from the global evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_classes(evaluator: Any) -> list[str]:
    from synthesis.evaluate.benchmarks.smiles.dataset import normalize_smiles_classes

    return normalize_smiles_classes(evaluator.smiles_classes)


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task

    classes = normalize_classes(evaluator)
    if len(classes) != 1:
        raise ValueError(
            "SMILES CSD evaluation uses class-specific grammars; "
            "pass exactly one class via --smiles-classes."
        )
    return Path(get_smiles_task(classes[0])["grammar_path"])


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles

    return load_smiles(classes=normalize_classes(evaluator), samples_per_class=evaluator.sample_size)


def format_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    """Match CARS ``expression_only`` prompt (legacy freeze / fair compare)."""
    base_prompt = example.get("prompt", "")
    return (
        base_prompt.rstrip()
        + "\n\nGive exactly one SMILES molecule on a single line.\n"
        "Molecule: "
    )


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Grammar-masked legacy adapters: single SMILES span, no delimiters."""
    base_prompt = example.get("prompt", "")
    return (
        base_prompt.rstrip()
        + "\n\nReturn exactly one line containing the SMILES molecule "
        "(example: CC(=O)OC=C).\n"
        "Molecule: "
    )


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> str:
    """CRANE-style adaptive SMILES: reasoning allowed before the final molecule."""
    base_prompt = example.get("prompt", "")
    return (
        base_prompt.rstrip()
        + "\n\nThink step by step about how to satisfy the structural constraints, "
        "then give your final SMILES molecule.\n"
        "Molecule: "
    )


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    return str(example.get("class_name", ""))


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    grammar_text = example.get("grammar_text", "")
    class_name = str(example.get("class_name", "smiles"))
    cache_key = ("smiles", class_name, grammar_text)
    parser_factory = evaluator._dynamic_parser_factory_cache.get(cache_key)
    if parser_factory is None:
        # CARS uses llguidance bitmasks; Syncode DFA rejects multi-atom BPE tokens
        # like ``CC`` at the empty prefix (legacy first token). Default ON for SMILES.
        parser_factory = create_lark_dafny_parser(
            grammar_text,
            env["VerifiedDecoderAgent"],
            env["_dafny"],
            start="start",
            tokenizer=env["tokenizer"],
            accept_mask_backend="llguidance",
        )
        evaluator._dynamic_parser_factory_cache[cache_key] = parser_factory
    return parser_factory(env["lm"]._Tokens)


def extract_actual(
    evaluator: Any,
    scored_output: str,
    example: dict[str, Any],
) -> tuple[str | None, str, dict[str, Any] | None]:
    import re
    from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output

    class_name = example.get("class_name", "smiles")
    grammar_text = example.get("grammar_text", "")
    prompt_exemplars = example.get("prompt_exemplars", [])

    from synthesis.evaluate.benchmarks.common.delimited_output import extract_last_delimited_span

    span, found = extract_last_delimited_span(scored_output)
    candidate = span if found else scored_output

    smiles_eval = evaluate_smiles_output(
        class_name,
        candidate,
        grammar_text,
        prompt_exemplars,
        require_rdkit=True,
    )
    return smiles_eval["smiles"] or None, "smiles_eval", smiles_eval


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    return bool(aux and aux.get("unique_valid_candidate"))


def force_open_span() -> bool:
    # The molecule is parser-governed from the first token, so the runtime
    # opens the span itself: the output starts as "<<" and the strategy starts
    # inside it. The prompt is unchanged by this.
    return True


def example_syntax_pass(
    all_valid_syntax: bool,
    segments: list[tuple[str, bool]],
    aux: dict[str, Any] | None,
) -> bool:
    return bool(aux and aux.get("syntax_valid"))


def accuracy_applicable(aux: dict[str, Any] | None) -> bool:
    return bool(aux and aux.get("accuracy_applicable"))


def get_generation_runner():
    from synthesis.evaluate.benchmarks.smiles import generation

    def _forced_span_runner(*args, **kwargs):
        kwargs.setdefault("force_open_span", True)
        return generation.run_crane_csd(*args, **kwargs)

    return _forced_span_runner


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    grammar_text = (
        example.get("grammar_text", "")
        if isinstance(example, dict)
        else ""
    )
    cache_key = ("smiles", grammar_text)
    parser = evaluator._syntax_parser_cache.get(cache_key)
    if parser is None:
        parser = Lark(grammar_text, start="start", parser="lalr")
        evaluator._syntax_parser_cache[cache_key] = parser
    return parser


def ensure_runtime_prereqs(evaluator: Any) -> None:
    from synthesis.evaluate.benchmarks.smiles.metrics import rdkit_available

    if not rdkit_available():
        raise RuntimeError(
            "SMILES evaluation requires RDKit but it is not available in this environment. "
            "Install RDKit before running smiles synthesis/evaluation."
        )


def should_stop_collected(
    sample_outputs: list[dict[str, Any]],
    target_unique_valid: int = 100,
) -> str | None:
    """Paper-aligned stop: CARS generates until 100 unique-valid molecules are
    collected (subject to a 1000-sample cap). Once we cross the target the
    headline `samples_to_target_unique_valid` metric is already determined, so
    further generation is wasted work. Returns a reason string when the target
    is reached, else None.
    """
    if not sample_outputs:
        return None
    seen: set[str] = set()
    for sample in sample_outputs:
        smiles_eval = sample.get("smiles_eval") or {}
        if not smiles_eval.get("unique_valid_candidate"):
            continue
        smiles = str(smiles_eval.get("smiles") or "").strip()
        if smiles and smiles not in seen:
            seen.add(smiles)
            if len(seen) >= target_unique_valid:
                return (
                    "paper-aligned early stop: collected "
                    f"{len(seen)} unique-valid molecules (target {target_unique_valid}) "
                    f"after {len(sample_outputs)} samples."
                )
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    from synthesis.evaluate.benchmarks.smiles.metrics import smiles_trial_metrics

    paper_metrics = smiles_trial_metrics(
        sample_outputs,
        target_unique_valid=100,
        sample_cap=1000,
    )

    helper_events = [
        event
        for sample in sample_outputs
        for event in (sample.get("helper_trace") or [])
        if isinstance(event, dict)
    ]
    helper_count = len(helper_events)
    open_close_helpers = {
        "OpenConstrainedSpan",
        "CloseConstrainedSpan",
        "EnterObservedConstrainedSpan",
    }
    churn_calls = sum(
        1 for event in helper_events if event.get("helper") in open_close_helpers
    )
    delimiter_churn_ratio = churn_calls / max(1, helper_count)

    tiny_spans = 0
    for sample in sample_outputs:
        smiles_eval = sample.get("smiles_eval") or {}
        smiles = str(smiles_eval.get("smiles") or "")
        if smiles and len(smiles) <= 3:
            tiny_spans += 1
    tiny_span_rate = tiny_spans / max(1, len(sample_outputs))

    max_steps_hits = sum(1 for sample in sample_outputs if sample.get("hit_max_steps"))
    max_steps_hit_rate = max_steps_hits / max(1, len(sample_outputs))

    penalty = min(
        0.60,
        0.35 * delimiter_churn_ratio
        + 0.35 * tiny_span_rate
        + 0.30 * max_steps_hit_rate,
    )
    membership = float(paper_metrics.get("membership", 0.0) or 0.0)
    adjusted_membership = max(0.0, membership - penalty)

    anti = {
        "delimiter_churn_ratio": delimiter_churn_ratio,
        "tiny_span_rate": tiny_span_rate,
        "max_steps_hit_rate": max_steps_hit_rate,
        "penalty": penalty,
        "adjusted_membership_score": adjusted_membership,
    }
    return {
        "smiles_paper_trial": paper_metrics,
        "anti_degeneracy": anti,
    }


def accuracy_upper_bound(
    num_correct: int,
    remaining: int,
    num_accuracy_examples: int,
    total_planned_examples: int,
) -> float:
    return (num_correct + remaining) / max(1, num_accuracy_examples + remaining)


def final_accuracy_denominator(num_examples: int, num_accuracy_examples: int) -> int:
    return num_accuracy_examples


def invalid_outputs_excluded(num_examples: int, num_accuracy_examples: int) -> int:
    return num_examples - num_accuracy_examples


def accuracy_definition() -> str:
    return "unique_valid_rate_rdkit (distinct rdkit-valid + in-class + non-exemplar molecules / N)"


def override_accuracy(aux_metrics: dict[str, Any] | None, num_examples: int) -> float | None:
    """SMILES headline metric = unique-valid RATE, the CARS-paper axis.

    accuracy = unique_valid_count / N, where unique_valid_count (from
    smiles_trial_metrics) is the number of DISTINCT molecules that are RDKit-valid
    AND in-class AND non-exemplar. The denominator is N (all samples), so a collapsed
    strategy that emits one molecule x N scores ~1/N instead of the old gameable
    membership-rate's 1.0. Diversity (Tanimoto) and validity are reported alongside
    in smiles_paper_trial as comparable axes.

    Returns None if the trial metrics are absent (caller then keeps the default
    accuracy), so this hook stays inert for any dataset that does not define it.
    """
    trial = (aux_metrics or {}).get("smiles_paper_trial") or {}
    if not trial:
        return None
    unique_valid = int(trial.get("unique_valid_count", 0) or 0)
    return unique_valid / max(1, num_examples)
