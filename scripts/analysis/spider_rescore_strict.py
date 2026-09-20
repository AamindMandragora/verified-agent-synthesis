#!/usr/bin/env python
"""Rescore saved Spider baseline lane answers with the strict scorer.

Strict scorer = synthesis.evaluate.benchmarks.sql_spider.eval_logic.extract_actual /
is_correct on branch span-contract-fix (this worktree). Scores generated_answer
directly: run_legacy_fixed_strategy.py already stores completion_for_scoring(...)
output (prompt stripped), so no extra stripping is needed here.

Lanes live at:
  ~/modal-baseline-rerun-20260914/runs/baseline-rerun-20260918-focal/lanes/spider-<method>-q35-<model>-s<NN>/output.json

Each lane is a 75-example shard of the 300-example held-out split
(environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json,
"test" side). Shard s corresponds to test_indices[s*75:(s+1)*75] of that split
(verified 2026-09-19 by comparing shard manifest eval_indices to the parent
split's test_indices slice — exact match for all 4 shards, both models, all
methods sharing shard json).

Read-only: does not touch ~/csd-generation or any lane output file.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("SPIDER_TOKEN0_CONSTRAINED", "0")

REPO = Path(__file__).resolve().parents[2]  # .../span-contract-fix-20260917
sys.path.insert(0, str(REPO))

SPLIT_FILE = REPO / "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
LANES_ROOT = Path.home() / "modal-baseline-rerun-20260914/runs/baseline-rerun-20260918-focal/lanes"

METHODS = ["unconstrained", "crane", "gcd", "cars", "itergen"]
MODELS = ["2b", "4b"]
N_SHARDS = 4
SHARD_SIZE = 75


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def load_examples():
    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic("spider")
    ev = Evaluator(
        dataset_name="spider",
        model_name="unused",
        backend="huggingface",
        device="cpu",
        sample_size=300,
        max_steps=600,
        step_token_budget=1,
        spider_split_file=str(SPLIT_FILE),
        spider_split_name="test",
    )
    examples = logic.load_dataset_sample(ev)
    assert len(examples) == 300, len(examples)
    return logic, ev, examples


def score_cell(logic, ev, examples, method, model):
    """Return None if any shard's output.json is missing; else a result dict."""
    lane_dirs = [LANES_ROOT / f"spider-{method}-q35-{model}-s{s:02d}" for s in range(N_SHARDS)]
    missing = [str(d) for d in lane_dirs if not (d / "output.json").exists()]
    if missing:
        return {"status": "missing_lanes", "missing": missing}

    all_rows = []
    stored_acc = []
    stored_syn = []
    for s, d in enumerate(lane_dirs):
        payload = json.load(open(d / "output.json"))
        answers = payload["answers"]
        assert len(answers) == SHARD_SIZE, (d, len(answers))
        stored_acc.append(payload["accuracy"])
        stored_syn.append(payload["syntax_rate"])

        shard_examples = examples[s * SHARD_SIZE : (s + 1) * SHARD_SIZE]
        mismatches = sum(
            1
            for a, ex in zip(answers, shard_examples)
            if norm(a["question"]) != norm(ex.get("question", ""))
        )
        if mismatches:
            return {
                "status": "alignment_failed",
                "shard": str(d),
                "mismatches": mismatches,
                "n": SHARD_SIZE,
            }

        for a, ex in zip(answers, shard_examples):
            scored_output = a["generated_answer"]
            expected = logic.expected_answer(ev, ex)
            actual, _source, aux = logic.extract_actual(ev, scored_output, ex)
            correct = bool(logic.is_correct(ev, actual, expected, ex, aux, scored_output))
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            reason = (aux or {}).get("output_rejection_reason")
            all_rows.append(
                {
                    "correct": correct,
                    "syntax_valid": syntax_valid,
                    "rejection_reason": reason,
                }
            )

    n = len(all_rows)
    strict_acc = sum(r["correct"] for r in all_rows) / n
    strict_syn = sum(r["syntax_valid"] for r in all_rows) / n
    reasons = Counter(r["rejection_reason"] for r in all_rows if r["rejection_reason"])
    top_reasons = reasons.most_common(3)

    return {
        "status": "ok",
        "n": n,
        "stored_acc_mean": sum(stored_acc) / len(stored_acc),
        "stored_syn_mean": sum(stored_syn) / len(stored_syn),
        "strict_acc": strict_acc,
        "strict_syn": strict_syn,
        "top_reasons": top_reasons,
    }


def main():
    logic, ev, examples = load_examples()
    results = {}
    for method in METHODS:
        for model in MODELS:
            key = f"{method}-{model}"
            print(f"Scoring {key}...", file=sys.stderr)
            results[key] = score_cell(logic, ev, examples, method, model)
            print(f"  {key}: {results[key].get('status')}", file=sys.stderr)

    out_path = Path(__file__).with_name("spider_rescore_strict_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
