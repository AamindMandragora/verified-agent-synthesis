"""Minimal baseline JSON storage helpers.

Baselines persist aggregate accuracy/syntax, optional timing and token metrics,
and per-row answers (question + generated text).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .evaluator import EvaluationResult


_REEVAL_LOG = logging.getLogger("csd.reevaluation_evidence")


def build_metrics_from_eval_samples(
    samples: list[dict[str, Any]],
    *,
    evaluator_total_time_seconds: float | None = None,
    evaluator_max_sample_time_seconds: float | None = None,
    run_wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate optional ``time_seconds`` / ``token_count`` fields from evaluator samples."""
    metrics: dict[str, Any] = {"num_examples": len(samples)}
    times = [
        float(s["time_seconds"])
        for s in samples
        if s.get("time_seconds") is not None
    ]
    toks = [
        int(s["token_count"])
        for s in samples
        if s.get("token_count") is not None
    ]
    works = [
        int(s["constrained_work"])
        for s in samples
        if s.get("constrained_work") is not None
    ]
    if works:
        metrics["total_constrained_work"] = int(sum(works))
        metrics["mean_constrained_work"] = round(sum(works) / len(works), 4)
        metrics["examples_with_constrained_work"] = len(works)
    if times:
        metrics["total_generation_seconds"] = round(sum(times), 4)
        metrics["mean_generation_seconds_per_example"] = round(sum(times) / len(times), 6)
        metrics["examples_with_generation_timing"] = len(times)
    if toks:
        metrics["total_output_tokens"] = int(sum(toks))
        metrics["mean_output_tokens_per_example"] = round(sum(toks) / len(toks), 4)
        metrics["examples_with_token_counts"] = len(toks)
    if evaluator_total_time_seconds is not None:
        metrics["evaluator_total_time_seconds"] = round(float(evaluator_total_time_seconds), 4)
    if evaluator_max_sample_time_seconds is not None:
        metrics["evaluator_max_sample_time_seconds"] = round(
            float(evaluator_max_sample_time_seconds), 4
        )
    if run_wall_time_seconds is not None:
        metrics["run_wall_time_seconds"] = round(float(run_wall_time_seconds), 4)
    return metrics


_REEVALUATION_TRACE_KEYS = frozenset(
    {
        "helper",
        "event",
        "status",
        "result_status",
        "generated_len_before",
        "generated_len_after",
        "candidate_count",
        "prefix_count",
        "steps",
        "failure_location",
        "trace_tag",
    }
)
_PROMPT_CONTRACT_KEYS = (
    "renderer",
    "family",
    "mode",
    "template_used",
    "raw_prompt",
    "chat_message_count",
    "user_message_count",
    "add_generation_prompt",
    "enable_thinking",
    "render_succeeded",
    "prompt_chars",
)


def _safe_helper_trace(trace: Any) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for event in trace if isinstance(trace, (list, tuple)) else ():
        if not isinstance(event, dict):
            continue
        safe_event = {
            key: event[key]
            for key in _REEVALUATION_TRACE_KEYS
            if key in event and isinstance(event[key], (bool, int, float, str))
        }
        if safe_event:
            safe.append(safe_event)
    return safe


def _safe_prompt_contract(contract: Any) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    return {
        key: contract.get(key)
        for key in _PROMPT_CONTRACT_KEYS
        if key in contract
    }


def _safe_int_list(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    return [int(value) for value in values]


def _safe_generation_token_evidence(evidence: Any) -> dict[str, Any] | None:
    if not isinstance(evidence, dict):
        return None
    return {
        "raw_token_ids": _safe_int_list(evidence.get("raw_token_ids")),
        "raw_decoded_text": evidence.get("raw_decoded_text"),
        "removed_terminal_token_ids": _safe_int_list(
            evidence.get("removed_terminal_token_ids")
        ),
        "decoded_text": evidence.get("decoded_text"),
    }


def build_reevaluation_sample_evidence(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Export audit fields while keeping the historical answers rows intact."""
    evidence_rows: list[dict[str, Any]] = []
    for evaluated_index, sample in enumerate(samples):
        token_evidence = _safe_generation_token_evidence(
            sample.get("generation_token_evidence")
        )
        removed_terminal_ids = (
            token_evidence.get("removed_terminal_token_ids", [])
            if token_evidence is not None
            else []
        )
        source_index = sample.get("source_index")
        if source_index is None:
            source_index = sample.get("spider_source_index")
        if source_index is None:
            source_index = sample.get("crane_source_index")
        error_type = sample.get("error_type")
        if error_type is None and sample.get("error"):
            error_type = "evaluation_error"
        error_status = sample.get("error_status")
        if error_status is None and sample.get("error"):
            error_status = "failed"
        if token_evidence is not None:
            # The token evidence is authoritative; never export a conflicting
            # caller-provided count.
            removed_count = len(removed_terminal_ids)
        else:
            removed_count = sample.get("removed_terminal_token_count")
        evidence_rows.append(
            {
                "evaluated_index": evaluated_index,
                "example_index": sample.get("example_index"),
                "source_index": source_index,
                "is_correct": sample.get("is_correct"),
                "accuracy_applicable": sample.get("accuracy_applicable"),
                "is_syntax_valid": sample.get("is_syntax_valid"),
                "answer_source": sample.get("answer_source"),
                "has_extracted_answer": sample.get("has_extracted_answer"),
                "output_contract_valid": sample.get("output_contract_valid"),
                "output_rejection_reason": sample.get("output_rejection_reason"),
                "timed_out": sample.get("timed_out"),
                "error_type": error_type,
                "error_status": error_status,
                "removed_terminal_token_count": removed_count,
                "generation_token_evidence": token_evidence,
                "constrained_work": sample.get("constrained_work"),
                "strategy_output_relation": sample.get("strategy_output_relation"),
                "strategy_mutation": sample.get("strategy_mutation"),
                "strategy_removed_sampled_token_ids": _safe_int_list(
                    sample.get("strategy_removed_sampled_token_ids")
                ),
                "helper_trace": _safe_helper_trace(sample.get("helper_trace")),
                "provenance_tags": [
                    str(value) for value in (sample.get("provenance_tags") or ())
                ],
                "failure_location": sample.get("failure_location"),
                "prompt_contract": _safe_prompt_contract(sample.get("prompt_contract")),
            }
        )
    return evidence_rows


def build_minimal_baseline_record(
    result: EvaluationResult,
    eval_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimal baseline payload from an evaluation result.

    ``eval_split`` is the split-provenance block (which split file/side the
    numbers were measured on — see synthesis/split_provenance.py). Callers
    should pass it so saved JSONs are self-describing; comparisons across
    split sides caused wrong win/loss verdicts on 2026-07-17.
    """
    samples = result.sample_outputs or []
    answers: list[dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question", ""))
        generated_answer = sample.get("scored_output")
        if not generated_answer:
            generated_answer = sample.get("full_output", "")
        row: dict[str, Any] = {
            "question": question,
            "generated_answer": str(generated_answer),
        }
        for key in (
            "example_index",
            "source_index",
            "crane_source_index",
            "spider_source_index",
            "db_id",
            "id_orig",
            "id_shuffled",
        ):
            if sample.get(key) is not None:
                row[key] = sample.get(key)
        if sample.get("token_count") is not None:
            row["num_tokens"] = int(sample["token_count"])
        if sample.get("constrained_work") is not None:
            row["constrained_work"] = int(sample["constrained_work"])
        if sample.get("time_seconds") is not None:
            row["generation_seconds"] = round(float(sample["time_seconds"]), 6)
        # Per-example outcome flags so saved JSONs support offline diffing
        # against baseline-side annotations without regrading.
        for key in ("is_correct", "is_syntax_valid"):
            if sample.get(key) is not None:
                row[key] = bool(sample[key])
        answers.append(row)

    metrics = build_metrics_from_eval_samples(
        samples,
        evaluator_total_time_seconds=result.total_time_seconds,
        evaluator_max_sample_time_seconds=result.max_sample_time_seconds,
    )

    record = {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "metrics": metrics,
        "answers": answers,
        "reevaluation_sample_evidence": build_reevaluation_sample_evidence(samples),
    }
    if eval_split is not None:
        record["eval_split"] = eval_split
    # Preserve the CARS-paper SMILES metrics (unique_valid_count, diversity_tanimoto,
    # validity_rdkit, samples_to_target_unique_valid) so saved JSONs carry the real
    # comparison axes instead of just the headline accuracy. Inert for other datasets.
    trial = (result.aux_metrics or {}).get("smiles_paper_trial")
    if trial:
        record["smiles_paper_trial"] = trial
    return record


def baseline_payload_from_success_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build the same baseline JSON shape as legacy exports from ``success_report.json``."""
    evaluation = report.get("evaluation_result") or {}
    samples = report.get("sample_outputs") or []

    answers: list[dict[str, Any]] = []
    for sample in samples:
        question = str(sample.get("question", ""))
        generated_answer = sample.get("scored_output")
        if not generated_answer:
            generated_answer = sample.get("full_output", "")
        row: dict[str, Any] = {
            "question": question,
            "generated_answer": str(generated_answer),
        }
        if sample.get("token_count") is not None:
            row["num_tokens"] = int(sample["token_count"])
        if sample.get("constrained_work") is not None:
            row["constrained_work"] = int(sample["constrained_work"])
        if sample.get("time_seconds") is not None:
            row["generation_seconds"] = round(float(sample["time_seconds"]), 6)
        # Per-example outcome flags so saved JSONs support offline diffing
        # against baseline-side annotations without regrading.
        for key in ("is_correct", "is_syntax_valid"):
            if sample.get(key) is not None:
                row[key] = bool(sample[key])
        answers.append(row)

    metrics = build_metrics_from_eval_samples(
        samples,
        evaluator_total_time_seconds=evaluation.get("total_time_seconds"),
        evaluator_max_sample_time_seconds=evaluation.get("max_sample_time_seconds"),
    )

    return {
        "accuracy": float(evaluation.get("accuracy", 0.0)),
        "syntax_rate": float(evaluation.get("syntax_rate", 0.0)),
        "metrics": metrics,
        "answers": answers,
    }


def save_minimal_baseline_json(
    result: EvaluationResult,
    json_path: Path,
    eval_split: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal baseline JSON file and return its path."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_minimal_baseline_record(result, eval_split=eval_split)
    if metadata:
        payload.update(metadata)
    # The answers list remains the historical minimal compatibility surface;
    # this dedicated list carries the audit fields for every reevaluation row.
    payload["reevaluation_sample_evidence"] = build_reevaluation_sample_evidence(
        result.sample_outputs or []
    )
    _REEVAL_LOG.info(
        "[reevaluation-evidence] rows=%d fields=%d",
        len(payload["reevaluation_sample_evidence"]),
        len(payload["reevaluation_sample_evidence"][0])
        if payload["reevaluation_sample_evidence"]
        else 0,
    )
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(json_path)
    return json_path
