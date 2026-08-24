"""Run fixed baseline strategies using legacy repository code paths."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
import subprocess
from pathlib import Path
from typing import Any

from synthesis.evaluate.completion_text import completion_for_scoring, strip_prompt_prefix
from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts


LOGGER = logging.getLogger(__name__)
_MAX_PROMPT_CHARS = 50000  # ~12.5K tokens; leaves room for generation within 16384-token context
_MAX_SUFFIX_CHARS = 45000


def _completion_for_dataset(dataset: str, prompt: Any, raw_output: Any) -> str:
    """Use generated-only text for Spider; preserve legacy stripping elsewhere."""
    if dataset == "spider":
        return str(raw_output or "")
    return completion_for_scoring(prompt, raw_output)


def _maybe_seed_parity_rng() -> None:
    """Optional deterministic RNG for fair stochastic baseline compares."""
    import os
    raw = os.environ.get("CSD_PARITY_SEED", "").strip()
    if not raw:
        return
    seed = int(raw)
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass




def _maybe_reseed_parity_attempt(example_index: int, attempt_index: int) -> None:
    """Re-seed RNG per CARS get_sample attempt (pairs with remake CSD_PARITY_SEED_PER_ATTEMPT)."""
    import os
    if os.environ.get("CSD_PARITY_SEED_PER_ATTEMPT", "0") != "1":
        return
    raw = os.environ.get("CSD_PARITY_SEED", "").strip()
    if not raw:
        return
    seed = int(raw) + int(example_index) * 1_000_003 + int(attempt_index) * 9176
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _maybe_reseed_parity_example(example_index: int) -> None:
    """Re-seed RNG per example so rolling-prompt parity is not poisoned by prior draw counts."""
    import os
    if os.environ.get("CSD_PARITY_SEED_PER_EXAMPLE", "0") != "1":
        return
    raw = os.environ.get("CSD_PARITY_SEED", "").strip()
    if not raw:
        return
    seed = int(raw) + int(example_index) * 1_000_003
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _truncate_prompt(prompt: str, base_prompt: str) -> str:
    """Keep *prompt* within ``_MAX_PROMPT_CHARS`` by dropping the oldest appended molecules."""
    if len(prompt) <= _MAX_PROMPT_CHARS:
        return prompt
    suffix = prompt[len(base_prompt):]
    lines = suffix.split("\n")
    while len(base_prompt) + len("\n".join(lines)) > _MAX_PROMPT_CHARS and len(lines) > 1:
        lines.pop(0)
    return base_prompt + "\n".join(lines)


def _cap_suffix(suffix: str) -> str:
    """Drop the oldest appended molecules if suffix exceeds ``_MAX_SUFFIX_CHARS``."""
    if len(suffix) <= _MAX_SUFFIX_CHARS:
        return suffix
    lines = suffix.split("\n")
    while len("\n".join(lines)) > _MAX_SUFFIX_CHARS and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


def _ensure_repo_cache_env() -> Path:
    """Point HF + SynCode pickles at a single repo-local ``cache/`` directory.

    Legacy CRANE/IterGen historically defaulted to ``legacy/CRANE/src/iter_cache/``
    (cwd-relative), duplicating multi‑GB model snapshots. Setting ``CSD_CACHE_ROOT``
    (or these defaults) keeps Hugging Face checkpoints and ``mask_stores/`` / ``parsers/``
    together under ``<repo>/cache/``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("CSD_CACHE_ROOT", str(repo_root / "cache"))).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    root_s = str(cache_root)

    os.environ.setdefault("CSD_CACHE_ROOT", root_s)
    os.environ.setdefault("HF_HOME", root_s)
    os.environ.setdefault("HF_CACHE", root_s)
    os.environ.setdefault("TRANSFORMERS_CACHE", root_s)

    syn_existing = os.environ.get("SYNCODE_CACHE") or os.environ.get("ITER_SYNCODE_CACHE")
    if syn_existing:
        syn = syn_existing if syn_existing.endswith(os.sep) else syn_existing + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)
    else:
        syn = root_s if root_s.endswith(os.sep) else root_s + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)

    return cache_root


def _normalize_dataset(dataset: str) -> str:
    return "gsm_symbolic" if dataset == "gsm" else dataset


def _baseline_row_question(dataset: str, example: dict[str, Any], fallback: str) -> str:
    """Question string stored in baseline JSON and used for GSM example lookup."""
    if dataset == "gsm_symbolic":
        text = (
            example.get("question_parsed")
            or example.get("original_question")
            or example.get("question")
            or example.get("prompt")
        )
        return str(text or fallback)
    return str(example.get("question") or example.get("prompt") or fallback)


def _legacy_benchmark_prompt(
    logic: Any,
    evaluator: Any,
    example: dict[str, Any],
    profile: str,
) -> str | SpiderPromptParts:
    """User-message text for legacy fixed strategies (not used by metadecode).

    profile:
      - ``expression_only``: one delimited answer; used by IterGen and GCD.
      - ``chain_of_thought``: explicit reasoning then answer; used by CRANE adaptive SMILES.
        NOTE: for Spider, ``format_prompt_chain_of_thought`` returns a list[dict] (multi-turn
        chat messages) — do NOT use this profile for Spider; use ``evaluator_default`` instead.
      - ``evaluator_default``: ``logic.format_prompt``; for token-0 Spider this
        returns shared structured prompt parts. Fixed IterGen renders those parts;
        other legacy adapters stringify them at their existing raw-text boundary.
    """
    if profile == "evaluator_default":
        return logic.format_prompt(evaluator, example)
    if profile == "expression_only":
        return logic.format_prompt_expression_only(evaluator, example)
    if profile == "chain_of_thought":
        cot = getattr(logic, "format_prompt_chain_of_thought", None)
        if callable(cot):
            return cot(evaluator, example)
        return logic.format_prompt(evaluator, example)
    raise ValueError(f"Unknown legacy prompt profile: {profile}")


def _cars_prompt_profile(dataset: str) -> str:
    if dataset == "spider":
        return "evaluator_default"
    return "expression_only"


_SPIDER_COMPLETION_MARKERS = (
    r"\bHuman\s*:",
    r"\bAssistant\s*:",
    r"\bUser\s*:",
    r"\bSystem\s*:",
    r"\bdb_id\s*:",
    r"\bdb_info\s*:",
    r"\bquestion\s*:",
    r"\bSQL\s*:",
)


def _spider_completion_has_clean_boundary(text: str) -> bool:
    """True once later text would be discarded by Spider SQL cleanup."""
    if not text:
        return False
    if "\n\n" in text:
        return True
    semicolon = text.find(";")
    if semicolon > 0:
        return True
    repeated_select = re.search(r"\s+SelEct\s+", text)
    if repeated_select and repeated_select.start() > 0:
        return True
    for marker in _SPIDER_COMPLETION_MARKERS:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match and match.start() > 0:
            return True
    return False


def _spider_hf_stopping_criteria(tokenizer: Any, prompt_token_count: int) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAtCleanBoundary(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            new_ids = input_ids[0][prompt_token_count:]
            if getattr(new_ids, "numel", lambda: 0)() == 0:
                return False
            suffix = tokenizer.decode(new_ids, skip_special_tokens=True)
            return _spider_completion_has_clean_boundary(suffix)

    return StoppingCriteriaList([_StopAtCleanBoundary()])


def _gcd_stop_words(dataset: str) -> list[str] | None:
    if dataset == "gsm_symbolic":
        return [">>"]
    if dataset == "spider":
        return [";"]
    return None


def _gcd_generation_kwargs(dataset: str) -> dict[str, Any]:
    if dataset == "smiles":
        return {"do_sample": True, "temperature": 0.7}
    return {"do_sample": False}


def _crane_stop_words(dataset: str) -> list[str] | None:
    return [">>"]


def _crane_generation_kwargs(dataset: str) -> dict[str, Any]:
    if dataset == "smiles":
        return {"do_sample": True, "temperature": 0.7}
    return {"do_sample": False}


_CRANE_SMILES_REASONING_GUIDANCE = (
    "Think through the requested molecular class, then put only the final SMILES "
    "between << and >>."
)


def _crane_prompt_for_generation(dataset: str, prompt: str) -> str:
    if dataset == "smiles":
        return f"{prompt.rstrip()}\n\n{_CRANE_SMILES_REASONING_GUIDANCE}"
    return prompt



def _legacy_gsm_symbolic_grammar_base(repo_root: Path, examples: list[dict[str, Any]]) -> str:
    """Tighten ``gsm.lark`` from a batch (allowed vars / numeric-only), matching GCD semantics.

    Returns grammar text still using ``syncode: \"<<\" start \">>\"`` (full delimited span).
    """
    from synthesis.evaluate.benchmarks.gsm_symbolic.grammar import (
        build_dynamic_grammar,
        build_numeric_only_grammar,
        extract_variables_from_mapping,
    )

    base_gsm_grammar = (repo_root / "synthesis" / "evaluate" / "grammars" / "gsm.lark").read_text()
    variable_names: set[str] = set()
    for ex in examples:
        vt = ex.get("variable_types") or {}
        if isinstance(vt, dict):
            variable_names.update(extract_variables_from_mapping(vt))
    gsm_allowed_variables = sorted(variable_names)
    if gsm_allowed_variables:
        return build_dynamic_grammar(base_gsm_grammar, gsm_allowed_variables)
    return build_numeric_only_grammar(base_gsm_grammar)


def _gsm_symbolic_completion_to_delimited(
    completion: str,
    example: dict[str, Any],
    eval_runtime: Any,
    logic: Any,
) -> str:
    """Normalize raw GSM completion (prompt already ends with ``<<``) to ``<<expr>>`` for scoring.

    Matches GCD's ``_gcd_output`` so IterGen and Syncode GSM baselines use the same delimited form
    and syntax checks as ``benchmarks/gsm_symbolic/eval_logic.py`` extraction.
    """
    expr = completion.strip().splitlines()[0].strip()
    if expr.startswith("<<"):
        wrapped = expr if ">>" in expr else f"{expr}>>"
        expr = re.findall(r"<<\s*(.*?)\s*>>", wrapped, flags=re.DOTALL)
        expr = expr[-1] if expr else ""
    if expr.endswith(">>"):
        expr = expr[:-2].strip()
    if not expr:
        return ""

    for end in range(len(expr), 0, -1):
        candidate = expr[:end].strip()
        if not candidate:
            continue
        wrapped = f"<<{candidate}>>"
        all_valid, segments = eval_runtime._check_syntax_validity(
            wrapped,
            example=example,
        )
        if logic.example_syntax_pass(all_valid, segments, False, None):
            return wrapped
    return f"<<{expr}>>"


def _resolve_device(device: str, backend: str) -> str:
    """Turn the "auto" default into a device the vLLM backend accepts."""
    if device == "auto" and backend == "vllm":
        return "cuda"
    return device


def _legacy_local_cuda_device(device_arg: str) -> str:
    """CUDA device string valid for the GPUs visible in this process."""
    if device_arg and device_arg not in {"auto", "cuda"}:
        if device_arg in {"mps", "cpu"}:
            return device_arg
        if device_arg.startswith("cuda"):
            return device_arg
        return f"cuda:{device_arg}"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cuda"


def _configure_fixed_eval_runtime(eval_runtime: Any, args: argparse.Namespace, dataset: str) -> None:
    if dataset == "gsm_symbolic":
        repo_root = Path(__file__).resolve().parents[2]
        env_gsm = os.environ.get("CRANE_GSM_SYMBOLIC_DIR")
        eval_runtime.gsm_source_dir = (
            Path(env_gsm).expanduser()
            if env_gsm
            else repo_root / "legacy" / "CRANE" / "src" / "gsm_symbolic"
        )
        if args.gsm_split_file:
            eval_runtime.gsm_split_file = Path(args.gsm_split_file)
        eval_runtime.gsm_split_name = args.gsm_split_name
    if dataset == "spider":
        if args.spider_split_file:
            eval_runtime.spider_split_file = Path(args.spider_split_file)
        eval_runtime.spider_split_name = args.spider_split_name


def _crane_grammar_name(dataset: str) -> str:
    if dataset == "gsm_symbolic":
        return "gsm"
    if dataset == "spider":
        return "sql"
    if dataset == "smiles":
        return "text"
    raise ValueError(f"CRANE adapter supports gsm_symbolic/spider/smiles, got {dataset}")


def _mode_for_strategy(strategy: str) -> tuple[str, bool]:
    if strategy == "unconstrained":
        # Match CRANE GSM/SQL protocol: always request chain-of-thought from the subprocess model.
        return "original", True
    if strategy == "crane":
        return "adaptive", True
    raise ValueError(f"Unsupported CRANE-backed strategy: {strategy}")


def _compose_baseline_answer_row(question: str, generated: str, row: dict[str, Any]) -> dict[str, Any]:
    prompt_used = row.get("prompt_used")
    if isinstance(prompt_used, str) and prompt_used:
        generated = strip_prompt_prefix(prompt_used, generated)
    entry: dict[str, Any] = {"question": question, "generated_answer": generated}
    if row.get("num_tokens") is not None:
        entry["num_tokens"] = int(row["num_tokens"])
    if row.get("generation_seconds") is not None:
        entry["generation_seconds"] = round(float(row["generation_seconds"]), 6)
    return entry


def _slice_eval_examples(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = int(getattr(args, "eval_start_index", 0) or 0)
    raw_end = getattr(args, "eval_end_index", None)
    total = len(examples)
    end = total if raw_end is None else int(raw_end)

    if start < 0:
        raise ValueError("--eval-start-index must be >= 0")
    if end < start:
        raise ValueError("--eval-end-index must be >= --eval-start-index")
    if end > total:
        raise ValueError(
            f"--eval-end-index {end} is past the loaded sample size {total}"
        )

    if start == 0 and end == total:
        return examples, {}

    return examples[start:end], {
        "eval_sample_total_examples": total,
        "eval_start_index": start,
        "eval_end_index": end,
    }


def _aggregate_run_metrics(
    rows: list[dict[str, Any]],
    *,
    run_wall_time_seconds: float | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_examples": len(rows)}
    times = [
        float(r["generation_seconds"])
        for r in rows
        if r.get("generation_seconds") is not None
    ]
    toks = [int(r["num_tokens"]) for r in rows if r.get("num_tokens") is not None]
    if times:
        metrics["total_generation_seconds"] = round(sum(times), 4)
        metrics["mean_generation_seconds_per_example"] = round(sum(times) / len(times), 6)
        metrics["examples_with_generation_timing"] = len(times)
    if toks:
        metrics["total_output_tokens"] = int(sum(toks))
        metrics["mean_output_tokens_per_example"] = round(sum(toks) / len(toks), 4)
        metrics["examples_with_token_counts"] = len(toks)
    if run_wall_time_seconds is not None:
        metrics["run_wall_time_seconds"] = round(float(run_wall_time_seconds), 4)
    return metrics


def _build_minimal_json(
    rows: list[dict[str, Any]],
    output_json: Path,
    *,
    run_wall_time_seconds: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    metrics = _aggregate_run_metrics(rows, run_wall_time_seconds=run_wall_time_seconds)
    if extra_metrics:
        metrics.update(extra_metrics)
    if not rows:
        payload = {
            "accuracy": 0.0,
            "syntax_rate": 0.0,
            "metrics": metrics,
            "answers": [],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        return

    correct_vals: list[float] = []
    syntax_vals: list[float] = []
    answers: list[dict[str, Any]] = []

    syntax_keys = [
        "is_syntax_valid",
        "syntax_valid",
        "grammar_valid",
        "out_parse_success",
    ]

    for row in rows:
        if isinstance(row.get("correct"), bool):
            correct_vals.append(1.0 if row["correct"] else 0.0)

        syntax_value = None
        for key in syntax_keys:
            if key in row and isinstance(row[key], bool):
                syntax_value = 1.0 if row[key] else 0.0
                break
        if syntax_value is None:
            syntax_value = 0.0
        syntax_vals.append(syntax_value)

        question = str(row.get("question") or row.get("prompt") or "")
        generated = row.get("llm_response")
        if generated is None:
            generated = row.get("response")
        if generated is None:
            generated = row.get("pred")
        if generated is None:
            generated = ""
        answers.append(_compose_baseline_answer_row(question, str(generated), row))

    payload = {
        "accuracy": sum(correct_vals) / max(1, len(correct_vals)),
        "syntax_rate": sum(syntax_vals) / max(1, len(syntax_vals)),
        "metrics": metrics,
        "answers": answers,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


def _cars_partial_json_path(output_json: Path) -> Path:
    if output_json.suffix:
        return output_json.with_name(f"{output_json.stem}.partial{output_json.suffix}")
    return output_json.with_name(f"{output_json.name}.partial.json")


def _write_cars_partial_json(
    rows: list[dict[str, Any]],
    output_json: Path,
    *,
    run_wall_time_seconds: float | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    partial_json = _cars_partial_json_path(output_json)
    _build_minimal_json(
        rows,
        partial_json,
        run_wall_time_seconds=run_wall_time_seconds,
        extra_metrics=extra_metrics,
    )
    return partial_json


def _load_latest_crane_results(crane_src_dir: Path, dataset: str) -> list[dict[str, Any]]:
    dataset_dir = crane_src_dir / "logging" / dataset
    if not dataset_dir.exists():
        return []

    candidates = sorted(dataset_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return []

    latest = candidates[0]
    rows: list[dict[str, Any]] = []
    for line in latest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _annotate_legacy_rows_with_syntax(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dataset: str,
) -> list[dict[str, Any]]:
    """Add benchmark syntax booleans to legacy rows that only report correctness."""
    if not rows:
        return rows
    if any(isinstance(row.get("syntax_valid"), bool) for row in rows):
        return rows

    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
        # Forward the per-class SMILES filter; without this the Evaluator loads the
        # default sample (all three classes x sample_size) and --smiles-classes is ignored.
        smiles_classes=(
            [s.strip() for s in args.smiles_classes.split(",") if s.strip()]
            if dataset == "smiles" and getattr(args, "smiles_classes", None)
            else None
        ),
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples, _slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )
    examples_by_question: dict[str, dict[str, Any]] = {}
    for example in examples:
        if dataset == "gsm_symbolic":
            keys = [
                example.get("question_parsed"),
                example.get("original_question"),
                example.get("question"),
                example.get("prompt"),
                example.get("question_instantiated"),
            ]
            for key in keys:
                if key:
                    examples_by_question[str(key)] = example
        else:
            q = str(example.get("question") or example.get("prompt") or "")
            if q:
                examples_by_question[q] = example

    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("prompt") or "")
        example = examples_by_question.get(question)
        if example is None and idx < len(examples):
            example = examples[idx]
        if dataset == "gsm_symbolic" and example is not None:
            example = dict(example)
            if not isinstance(example.get("variable_types"), dict):
                gold_answer = str(row.get("gold_answer") or "")
                numeric_vars = sorted(
                    set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", gold_answer))
                    - {"int", "ToInt", "z3_floor_div"}
                )
                if numeric_vars:
                    example["variable_types"] = {name: "int" for name in numeric_vars}
        output_text = str(
            row.get("parsed_completion")
            or row.get("llm_response")
            or row.get("response")
            or row.get("pred")
            or ""
        )
        prompt_used = str(row.get("prompt_used") or "")
        completion = _completion_for_dataset(dataset, prompt_used or None, output_text)
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        all_valid, segments = eval_runtime._check_syntax_validity(
            scored_output,
            example=example,
        )
        if dataset == "spider":
            actual, _answer_source, _aux = logic.extract_actual(
                eval_runtime,
                scored_output,
                example or {},
            )
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        else:
            syntax_valid = bool(
                logic.example_syntax_pass(all_valid, segments, False, None)
            )
        row["syntax_valid"] = syntax_valid
    return rows


def _cars_model_id(eval_model: str) -> str:
    """Return the HuggingFace model ID to pass to CARS's run_task.py."""
    return eval_model


def _load_latest_cars_results(cars_root_dir: Path, class_name: str) -> list[dict[str, Any]]:
    runs_root = cars_root_dir / "runs_log"
    prefix = f"smiles_{class_name}"
    candidates = sorted(
        runs_root.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if prefix not in candidate.as_posix():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            return steps
    return []


def _load_cars_results_from_log_dir(log_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            return steps
    return []


def _extract_cars_log_dir(stdout: str, cars_root_dir: Path) -> Path | None:
    for line in stdout.splitlines():
        match = re.search(r"Saving results in folder\s+(.+)$", line.strip())
        if match:
            return cars_root_dir / match.group(1)
    return None


def _cars_add_import_paths(cars_root_dir: Path) -> None:
    candidate_str = str(cars_root_dir.resolve())
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _cars_tokens_to_text(tokens: list[Any]) -> str:
    eos_markers = {"<|eot_id|>", "<|im_end|>", "<|endoftext|>"}
    text_tokens: list[str] = []
    for token in tokens:
        token_str = str(token)
        if token_str in eos_markers:
            continue
        text_tokens.append(token_str)
    return "".join(text_tokens).strip()


def _cars_sampler_steps(
    cars_model: Any,
    prompt: str,
    *,
    n_steps: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run upstream ``cars.CARS`` multi-step sampling; return logged step dicts."""
    from cars.cars import CARS

    with tempfile.TemporaryDirectory(prefix="vas_cars_") as log_dir:
        runner = CARS(cars_model, prompt, "cars", log_dir)
        runner.get_samples(
            n_samples=1,
            n_steps=max(1, n_steps),
            stop_after=1,
            max_new_tokens=max(1, max_new_tokens),
        )
        return _load_cars_results_from_log_dir(Path(log_dir))


def _cars_completion_from_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    tokens = steps[-1].get("tokens")
    if not isinstance(tokens, list):
        return ""
    return _cars_tokens_to_text(tokens)


def _cars_normalize_gsm_symbolic_output(raw: str) -> str:
    """Wrap bare GSM expressions so benchmark scoring can see them.

    GSM-Symbolic ``extract_actual`` only reads ``<< ... >>`` spans. Legacy CARS often
    emits a grammar-valid expression body with no delimiters, which yields
    ``actual is None`` and zero accuracy despite plausible-looking output.
    """
    text = (raw or "").strip()
    if not text:
        return raw
    if re.findall(r"<<\s*(.*?)\s*>>", text, flags=re.DOTALL):
        return raw
    expr = text.splitlines()[0].strip()
    return f"<<{expr}>>" if expr else raw


def _cars_set_cached_grammar(
    cars_model: Any,
    grammar_text: str,
    grammar_cache: dict[str, Any],
) -> None:
    cached = grammar_cache.get(grammar_text)
    if cached is None:
        cars_model._set_grammar_constraint(grammar_text)
        grammar_cache[grammar_text] = cars_model.grammar_constraint
    else:
        cars_model.grammar_constraint = cached


def run_cars_legacy_adapter(args: argparse.Namespace) -> int:
    _maybe_seed_parity_rng()
    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]

    model_id = _cars_model_id(args.eval_model)

    cars_root_override = os.environ.get("CARS_REPO_DIR")
    if cars_root_override:
        cars_root = Path(cars_root_override).expanduser().resolve()
    else:
        upstream_cars = Path(os.path.expanduser("~/cars")).resolve()
        if upstream_cars.exists():
            cars_root = upstream_cars
        else:
            cars_root = Path(__file__).resolve().parents[2] / "legacy" / "cars"
    if not cars_root.exists():
        raise RuntimeError(f"cars directory not found: {cars_root}")

    _cars_add_import_paths(cars_root)

    import torch
    from cars.lib import ConstrainedModel
    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
        # Forward the per-class SMILES filter; without this the Evaluator loads the
        # default sample (all three classes x sample_size) and --smiles-classes is ignored.
        smiles_classes=(
            [s.strip() for s in args.smiles_classes.split(",") if s.strip()]
            if dataset == "smiles" and getattr(args, "smiles_classes", None)
            else None
        ),
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)

    examples, slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )
    cars_model = ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)

    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}
    grammar_cache: dict[str, Any] = {}
    n_steps = max(1, int(getattr(args, "cars_search_steps", 200)))
    max_new_tokens = max(32, int(args.eval_max_steps))

    gsm_cars_grammar = ""
    if dataset == "gsm_symbolic":
        gsm_cars_grammar = _legacy_gsm_symbolic_grammar_base(repo_root, examples)
    spider_cars_grammar = ""
    if dataset == "spider":
        spider_cars_grammar = (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()

    for _ex_i, example in enumerate(examples):
        _maybe_reseed_parity_example(_ex_i)
        __import__("os").environ["CSD_PARITY_EXAMPLE_INDEX"] = str(_ex_i)
        if dataset == "gsm_symbolic":
            grammar_text = gsm_cars_grammar
        elif dataset == "spider":
            grammar_text = spider_cars_grammar
        elif dataset == "smiles":
            grammar_text = str(example.get("grammar_text", ""))
        else:
            raise ValueError(f"Unsupported dataset for CARS adapter: {dataset}")
        _cars_set_cached_grammar(cars_model, grammar_text, grammar_cache)

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(
            logic,
            eval_runtime,
            example,
            _cars_prompt_profile(dataset),
        )
        if dataset == "spider":
            prompt = str(prompt)
        gen_started = time.perf_counter()
        steps = _cars_sampler_steps(
            cars_model,
            prompt,
            n_steps=n_steps,
            max_new_tokens=max_new_tokens,
        )
        output_text = _cars_completion_from_steps(steps)
        gen_seconds = time.perf_counter() - gen_started
        if dataset == "gsm_symbolic":
            output_text = _cars_normalize_gsm_symbolic_output(output_text)
        completion = _completion_for_dataset(dataset, prompt, output_text)
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                    smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": completion,
                "prompt_used": prompt,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )
        _write_cars_partial_json(
            rows,
            args.output_json,
            run_wall_time_seconds=time.perf_counter() - run_started,
            extra_metrics=slice_metrics,
        )

    if not rows:
        raise RuntimeError("CARS produced no rows; refusing to write an empty baseline JSON")

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics=slice_metrics,
    )
    try:
        _cars_partial_json_path(args.output_json).unlink()
    except FileNotFoundError:
        pass
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_gcd_legacy_adapter(args: argparse.Namespace) -> int:
    """GCD baseline via vendored SynCode (grammar_strict mode = pure hard-mask constrained decoding)."""
    run_started = time.perf_counter()
    import sys

    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]

    syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    syncode_pkg = syncode_root / "syncode"
    for p in [str(syncode_root), str(syncode_pkg)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from syncode.infer import Syncode
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)

    from synthesis.evaluate.evaluator import Evaluator
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
        # Forward the per-class SMILES filter; without this the Evaluator loads the
        # default sample (all three classes x sample_size) and --smiles-classes is ignored.
        smiles_classes=(
            [s.strip() for s in args.smiles_classes.split(",") if s.strip()]
            if dataset == "smiles" and getattr(args, "smiles_classes", None)
            else None
        ),
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_local_cuda_device(args.device)
    base_gsm_grammar = ""
    if dataset == "gsm_symbolic":
        base_gsm_grammar = _legacy_gsm_symbolic_grammar_base(repo_root, examples)
        # The prompt already ends with "<<"; constrain the expression body plus closing marker.
        base_gsm_grammar = base_gsm_grammar.replace(
            'syncode: "<<" start ">>"',
            'syncode: start ">>"',
            1,
        )

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return base_gsm_grammar
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for GCD adapter: {dataset}")

    def _gcd_max_new_tokens() -> int:
        return _legacy_fixed_max_new_tokens(
            dataset,
            args.eval_max_steps,
            strategy="gcd",
        )

    def _gcd_prompt(prompt: str) -> str:
        if dataset == "gsm_symbolic":
            return prompt.rstrip() + "<<"
        return prompt

    def _gcd_output(completion: str, example: dict[str, Any]) -> str:
        if dataset != "gsm_symbolic":
            return completion
        return _gsm_symbolic_completion_to_delimited(completion, example, eval_runtime, logic)

    syncode_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in syncode_cache:
            syncode_cache[cache_key] = Syncode(
                model=args.eval_model,
                mode="grammar_strict",
                quantize=False,
                device=device,
                grammar=grammar_text,
                parse_output_only=True,
                log_level=0,
                max_new_tokens=_gcd_max_new_tokens(),
                num_return_sequences=1,
                opp=False,
                **_gcd_generation_kwargs(dataset),
            )
        sc = syncode_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "expression_only")
        if dataset == "spider":
            prompt = str(prompt)
        gen_started = time.perf_counter()
        gcd_prompt = _gcd_prompt(prompt)
        completions = sc.infer(gcd_prompt, stop_words=_gcd_stop_words(dataset))
        gen_seconds = time.perf_counter() - gen_started
        raw_output = _gcd_output(completions[0] if completions else "", example)
        completion = _completion_for_dataset(dataset, gcd_prompt, raw_output)
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                    smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": completion,
                "prompt_used": gcd_prompt,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0

def _itergen_add_import_paths(itergen_root: Path) -> None:
    candidates = [
        itergen_root,
        itergen_root / "itergen" / "syncode",
        itergen_root / "itergen" / "syncode" / "syncode",
    ]
    for candidate in candidates:
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _install_itergen_transformers_compat(itergen_cls: type[Any]) -> None:
    """Keep legacy IterGen generation working with Transformers 5.

    Transformers 5 removed ``_get_logits_warper`` and folds sampling warpers
    into ``_get_logits_processor``. Greedy generation keeps an identity list;
    sampling uses Transformers 5's own processor builder.
    """
    if getattr(itergen_cls, "_csd_transformers_compat_installed", False):
        return

    original_start = getattr(itergen_cls, "start", None)

    def _compatible_update_gen_args(self: Any, **gen_args: Any) -> None:
        self.generation_config.update(**gen_args)
        # Transformers 5 leaves these unset, while legacy IterGen expects the
        # Transformers 4 greedy defaults and compares both values as integers.
        if getattr(self.generation_config, "num_beams", None) is None:
            self.generation_config.num_beams = 1
        if getattr(self.generation_config, "num_beam_groups", None) is None:
            self.generation_config.num_beam_groups = 1
        get_logits_warper = getattr(self.model, "_get_logits_warper", None)
        if get_logits_warper is not None:
            self.logit_warper = get_logits_warper(
                self.generation_config,
                device=self.device,
            )
            return
        if self.generation_config.do_sample:
            get_logits_processor = getattr(self.model, "_get_logits_processor", None)
            if get_logits_processor is None:
                raise RuntimeError(
                    "Legacy IterGen sampling requires Transformers' "
                    "_get_logits_processor when _get_logits_warper is unavailable."
                )
            self.logit_warper = get_logits_processor(
                generation_config=self.generation_config,
                input_ids_seq_length=0,
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=None,
                device=self.device,
                model_kwargs={},
            )
            LOGGER.info(
                "[legacy-itergen] using Transformers 5 sampling processors "
                "device=%s temperature=%s",
                self.device,
                getattr(self.generation_config, "temperature", None),
            )
            return

        from transformers import LogitsProcessorList

        self.logit_warper = LogitsProcessorList()
        print(
            "[legacy-itergen] _get_logits_warper is unavailable; using the "
            "identity warper for greedy generation.",
            file=sys.stderr,
        )

    def _compatible_start(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_start(self, *args, **kwargs)
        model_config = self.model.config
        get_text_config = getattr(model_config, "get_text_config", None)
        text_config = (
            get_text_config(decoder=True)
            if get_text_config is not None
            else model_config
        )
        layer_types = getattr(text_config, "layer_types", None) or []
        if any(layer_type in {"linear_attention", "hybrid"} for layer_type in layer_types):
            from transformers.cache_utils import DynamicCache

            self.model_kwargs["past_key_values"] = DynamicCache(config=model_config)
            print(
                "[legacy-itergen] initialized a config-aware cache for the "
                "model's linear-attention layers.",
                file=sys.stderr,
            )
        return result

    itergen_cls.update_gen_args = _compatible_update_gen_args
    if original_start is not None:
        itergen_cls.start = _compatible_start
    itergen_cls._csd_transformers_compat_installed = True


def _itergen_generate(iter_gen: Any, prompt: Any) -> str:
    iter_gen.start(prompt)
    generated = iter_gen.forward()
    if isinstance(generated, list):
        if not generated:
            return ""
        return str(generated[0])
    return str(generated)


def _itergen_generation_stop_token_ids(iter_gen: Any, tokenizer: Any) -> set[int]:
    for owner in (iter_gen, tokenizer):
        value = getattr(owner, "generation_stop_token_ids", None)
        if value is None:
            continue
        if isinstance(value, int):
            return {int(value)}
        return {int(item) for item in value}
    for owner in (iter_gen, tokenizer):
        value = getattr(owner, "eos_token_id", None)
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, int):
            return {int(value)}
        return {int(item) for item in value}
    return set()


def _itergen_generation_token_evidence(iter_gen: Any) -> dict[str, Any] | None:
    """Capture only generated IterGen IDs, excluding prompt/session prefix IDs."""
    session_tokens = getattr(iter_gen, "session_tokens", None)
    start_from = getattr(iter_gen, "start_from", None)
    tokenizer = getattr(iter_gen, "tokenizer", None)
    if session_tokens is None or start_from is None or tokenizer is None:
        return None
    try:
        generated_tokens = session_tokens[..., int(start_from):]
    except (IndexError, TypeError, ValueError):
        try:
            generated_tokens = session_tokens[int(start_from):]
        except (IndexError, TypeError, ValueError):
            return None
    if hasattr(generated_tokens, "detach"):
        generated_tokens = generated_tokens.detach().cpu()
    if hasattr(generated_tokens, "tolist"):
        generated_tokens = generated_tokens.tolist()
    while (
        isinstance(generated_tokens, list)
        and len(generated_tokens) == 1
        and isinstance(generated_tokens[0], list)
    ):
        generated_tokens = generated_tokens[0]
    if not isinstance(generated_tokens, list):
        return None
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        generation_token_evidence,
    )

    evidence = generation_token_evidence(
        generated_tokens,
        tokenizer,
        terminal_stop_token_ids=_itergen_generation_stop_token_ids(iter_gen, tokenizer),
    )
    LOGGER.info(
        "[legacy-itergen-spider] token-boundary generated_ids=%d removed_terminal_ids=%d",
        len(evidence["raw_token_ids"]),
        len(evidence["removed_terminal_token_ids"]),
    )
    return evidence


def _legacy_fixed_max_new_tokens(
    dataset: str,
    eval_max_steps: int,
    *,
    strategy: str,
) -> int:
    """Honor the campaign budget except for the documented GSM safety cap."""
    max_steps = int(eval_max_steps)
    if dataset == "gsm_symbolic":
        return min(96, max(32, max_steps))
    if dataset == "smiles":
        return max(64, max_steps)
    if dataset == "spider" and strategy == "itergen":
        return min(512, max(64, max_steps))
    return max(32, max_steps)


def _itergen_generation_kwargs(
    *,
    dataset: str,
    max_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Return dataset-faithful IterGen constructor settings."""
    kwargs: dict[str, Any] = {
        "parse_output_only": True,
        "quantize": False,
        "max_tokens": max_tokens,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
    }
    if dataset == "smiles":
        kwargs.update(do_sample=True, temperature=0.7)
    else:
        kwargs["do_sample"] = False
    if dataset == "spider":
        # The checked-in upstream Spider experiment uses 0.3 for its greedy
        # iterative search. This discourages immediate token recurrence without
        # changing the deterministic baseline into sampling.
        kwargs["recurrence_penalty"] = 0.3
    return kwargs


def _itergen_spider_schema(example: dict[str, Any]) -> dict[str, set[str]]:
    """Parse the Spider ``db_info`` surface used by the upstream adapter."""
    schema: dict[str, set[str]] = {}
    for table in str(example.get("db_info") or "").split("#")[1:]:
        try:
            table_name, columns_text = table.split("(", 1)
        except ValueError:
            continue
        columns = columns_text.split(")", 1)[0].split(",")
        schema[table_name.strip().lower()] = {
            column.strip().lower() for column in columns if column.strip()
        }
    return schema


def _itergen_spider_column_exists(schema: dict[str, set[str]], value: str) -> bool:
    column = value.strip().lower()
    if column == "*":
        return True
    if "." in column:
        table, column_name = column.split(".", 1)
        return column_name in schema.get(table, set())
    return any(column in columns for columns in schema.values())


def _itergen_generate_spider(
    iter_gen: Any,
    prompt: Any,
    example: dict[str, Any],
    *,
    max_iterations: int = 20,
    backwards_limit: int = 10,
) -> str:
    """Run the checked-in upstream Spider IterGen protocol."""
    iter_gen.start(prompt)
    schema = _itergen_spider_schema(example)
    generated: Any = ""
    num_backwards = 0
    iterations = 0
    LOGGER.info(
        "[legacy-itergen-spider] start tables=%d max_iterations=%d backwards_limit=%d",
        len(schema),
        max_iterations,
        backwards_limit,
    )
    while not iter_gen.finished() and iterations < max_iterations:
        iterations += 1
        generated = iter_gen.forward(units=["column_name", "table_name"], num=1)

        column_names = iter_gen.view("column_name")[0]
        last_column = column_names[-1] if column_names else None
        if (
            last_column is not None
            and not _itergen_spider_column_exists(schema, str(last_column))
            and num_backwards < backwards_limit
        ):
            iter_gen.backward("column_name")
            num_backwards += 1
            LOGGER.info(
                "[legacy-itergen-spider] backtrack unit=column_name value=%r count=%d",
                last_column,
                num_backwards,
            )
            continue

        table_names = iter_gen.view("table_name")[0]
        last_table = table_names[-1] if table_names else None
        if (
            last_table is not None
            and str(last_table).strip().lower() not in schema
            and num_backwards < backwards_limit
        ):
            iter_gen.backward("table_name")
            num_backwards += 1
            LOGGER.info(
                "[legacy-itergen-spider] backtrack unit=table_name value=%r count=%d",
                last_table,
                num_backwards,
            )

    if isinstance(generated, list):
        completion = str(generated[0]) if generated else ""
    else:
        completion = str(generated)
    LOGGER.info(
        "[legacy-itergen-spider] finish iterations=%d backtracks=%d finished=%s output_chars=%d",
        iterations,
        num_backwards,
        bool(iter_gen.finished()),
        len(completion),
    )
    return completion


# Robustness guard, not an evaluation change: greedy (do_sample=False) IterGen can enter a
# non-terminating regeneration loop on a degenerate example (e.g. an unbounded ``.``-repeated
# SMILES) and never return from ``forward()``. A per-example wall-clock cap treats such a stuck
# example as a non-answer (empty completion -> scored incorrect + syntax-invalid), which is exactly
# how a fair harness handles a baseline that cannot produce an answer in bounded time. It does NOT
# touch the grammar, grader, or scorer, and applies symmetrically to every method/example.
# Override with CSD_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS. Default 300s: a legitimate bounded decode
# normally finishes well inside the cap, which is reserved for pathological non-terminating cases.
_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS = float(
    os.environ.get("CSD_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS", "300")
)


class _ItergenPerExampleTimeout(Exception):
    """Raised when a single IterGen ``forward()`` exceeds the per-example wall-clock cap."""


def _itergen_generate_with_timeout(
    iter_gen: Any,
    prompt: Any,
    cap_seconds: float,
    *,
    spider_example: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Run ``_itergen_generate`` under a SIGALRM wall-clock cap.

    Returns ``(completion, timed_out)``. On timeout returns ``("", True)`` so the caller scores the
    example as a non-answer. A cap of <= 0 disables the guard (unbounded, legacy behaviour).
    """
    def _generate() -> str:
        if spider_example is not None:
            return _itergen_generate_spider(iter_gen, prompt, spider_example)
        return _itergen_generate(iter_gen, prompt)

    if cap_seconds <= 0:
        return _generate(), False

    def _handler(signum: int, frame: Any) -> None:
        raise _ItergenPerExampleTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, cap_seconds)
    try:
        return _generate(), False
    except _ItergenPerExampleTimeout:
        return "", True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_itergen_legacy_adapter(args: argparse.Namespace) -> int:
    prev_recursion = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev_recursion, 100_000))
    try:
        return _run_itergen_legacy_adapter_inner(args)
    finally:
        sys.setrecursionlimit(prev_recursion)


def _run_itergen_legacy_adapter_inner(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    dataset = _normalize_dataset(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]
    itergen_root = repo_root / "legacy" / "itergen"
    if not itergen_root.exists():
        raise RuntimeError(f"Legacy itergen directory not found: {itergen_root}")

    _itergen_add_import_paths(itergen_root)
    from itergen.main import IterGen

    _install_itergen_transformers_compat(IterGen)

    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
        # Forward the per-class SMILES filter; without this the Evaluator loads the
        # default sample (all three classes x sample_size) and --smiles-classes is ignored.
        smiles_classes=(
            [s.strip() for s in args.smiles_classes.split(",") if s.strip()]
            if dataset == "smiles" and getattr(args, "smiles_classes", None)
            else None
        ),
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_local_cuda_device(args.device)

    base_gsm_grammar_text = ""
    if dataset == "gsm_symbolic":
        base_gsm_grammar_text = _legacy_gsm_symbolic_grammar_base(repo_root, examples)
        # GCD alignment: constrain body after the prompt's ``<<`` through closing ``>>``.
        base_gsm_grammar_text = base_gsm_grammar_text.replace(
            'syncode: "<<" start ">>"',
            'syncode: start ">>"',
            1,
        )

    _new_tok = _legacy_fixed_max_new_tokens(
        dataset,
        args.eval_max_steps,
        strategy="itergen",
    )

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return base_gsm_grammar_text
        if dataset == "spider":
            return (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for itergen adapter: {dataset}")

    itergen_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in itergen_cache:
            # Session ceiling must cover prompt + capped decode (``_new_tok``), not ``eval_max_steps`` alone.
            _session_ceiling = min(16384, max(2048, _new_tok + 4096))
            # Do not use ``stop_strings`` here: ``["\\n\\n"]`` fires HuggingFace stopping criteria and ends
            # generation before the LALR parser can reach a complete / EOF-ready state (e.g. GSM expr cut
            # mid-expression). Rely on grammar ``start`` completion + ``max_new_tokens`` instead.
            itergen_cache[cache_key] = IterGen(
                grammar=grammar_text,
                model_id=args.eval_model,
                device=device,
                **_itergen_generation_kwargs(
                    dataset=dataset,
                    max_tokens=_session_ceiling,
                    max_new_tokens=_new_tok,
                ),
            )
        iter_gen = itergen_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "expression_only")
        if dataset == "gsm_symbolic":
            prompt = prompt.rstrip() + "<<"

        if dataset == "spider":
            generation_prompt = prompt.render_for_model(
                iter_gen.tokenizer,
                model_name=args.eval_model,
            )
        else:
            generation_prompt = prompt
        gen_started = time.perf_counter()
        raw_completion, _timed_out = _itergen_generate_with_timeout(
            iter_gen,
            generation_prompt,
            _ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS,
            spider_example=example if dataset == "spider" else None,
        )
        if _timed_out:
            print(
                f"[itergen] per-example wall-clock timeout after "
                f"{_ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS:g}s on {dataset} example "
                f"{example.get('class_name', example.get('id', '?'))} -- scoring as non-answer"
            )
        generation_token_evidence = (
            _itergen_generation_token_evidence(iter_gen)
            if dataset == "spider"
            else None
        )
        if (
            dataset == "spider"
            and not _timed_out
            and generation_token_evidence is not None
        ):
            raw_completion = generation_token_evidence["decoded_text"]
        if dataset == "gsm_symbolic":
            raw_completion = _gsm_symbolic_completion_to_delimited(
                raw_completion, example, eval_runtime, logic
            )
        gen_seconds = time.perf_counter() - gen_started
        prompt_for_scoring = str(prompt) if dataset == "spider" else prompt
        completion = _completion_for_dataset(dataset, prompt_for_scoring, raw_completion)
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        expected = logic.expected_answer(eval_runtime, example)
        actual, answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        if dataset == "spider":
            removed_terminal_token_count = len(
                (generation_token_evidence or {}).get("removed_terminal_token_ids", [])
            )
            LOGGER.info(
                "[spider-output-contract] contract_valid=%s rejection_reason=%s "
                "raw_chars=%d candidate_chars=%d removed_terminal_token_count=%d",
                bool(aux and aux.get("output_contract_valid")),
                aux.get("output_rejection_reason") if aux else None,
                len(str(scored_output)),
                len(str(actual or "")),
                removed_terminal_token_count,
            )
            if aux is not None:
                aux["removed_terminal_token_count"] = removed_terminal_token_count
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider" and aux is not None:
            syntax_valid = bool(aux.get("syntax_valid", aux.get("output_contract_valid", False)))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                    smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")
        question = _baseline_row_question(dataset, example, expected)
        row_out = {
            "question": question,
            "llm_response": completion,
            "prompt_used": prompt_for_scoring,
            "correct": bool(is_correct),
            "syntax_valid": bool(syntax_valid),
            "generation_seconds": gen_seconds,
            "timed_out": bool(_timed_out),
        }
        if dataset == "spider":
            row_out.update(
                actual=actual,
                answer_source=answer_source,
                output_contract_valid=(aux.get("output_contract_valid") if aux is not None else None),
                output_rejection_reason=aux.get("output_rejection_reason") if aux else None,
                generation_token_evidence=generation_token_evidence,
                removed_terminal_token_count=removed_terminal_token_count,
            )
        rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_unconstrained_smiles_adapter(args: argparse.Namespace) -> int:
    """Unconstrained SMILES generation: generate without grammar masking, then score.

    Each sample appends all previously generated valid molecules to the prompt
    so the model is nudged to produce novel strings (matching the CARS paper
    protocol).  Temperature 0.8 gives diversity; greedy would repeat the same
    molecule every time.
    """
    from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES, get_smiles_task
    from synthesis.evaluate.benchmarks.smiles.metrics import (
        clean_smiles_output,
        evaluate_smiles_output,
    )

    run_started = time.perf_counter()
    device = _legacy_local_cuda_device(args.device)
    selected_classes = [
        part.strip()
        for part in (args.smiles_classes or ",".join(SMILES_CLASSES)).split(",")
        if part.strip()
    ]
    unknown = sorted(set(selected_classes) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(
            f"Unknown SMILES class(es): {unknown}. Expected one of {SMILES_CLASSES}."
        )

    n_per_class = max(1, args.smiles_samples_per_class or args.eval_sample_size)
    uc_smiles_cot_prefix = (
        "For each molecule requested below, give brief step-by-step reasoning about the "
        "constraints, then write the SMILES string.\n\n"
    )

    if args.eval_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )

    rows: list[dict[str, Any]] = []
    seen_per_class: dict[str, set[str]] = {}

    for class_name in selected_classes:
        task = get_smiles_task(class_name)
        base_prompt = task["prompt"]
        grammar_text = str(task["grammar_text"])
        prompt_exemplars = list(task.get("prompt_exemplars", []))
        seen_per_class[class_name] = set(prompt_exemplars)
        seed_prompt = uc_smiles_cot_prefix + base_prompt
        running_prompt = seed_prompt

        for _i in range(n_per_class):
            running_prompt = _truncate_prompt(running_prompt, seed_prompt)
            if args.eval_backend == "vllm":
                from vllm import SamplingParams as _SP

                gen_started = time.perf_counter()
                sp = _SP(max_tokens=args.eval_max_steps, temperature=0.8, stop=["\n\n"])
                outputs = llm.generate([running_prompt], sp)
                gen_seconds = time.perf_counter() - gen_started
                completion = outputs[0].outputs[0]
                gen_text = completion.text
                num_toks = len(completion.token_ids) if getattr(completion, "token_ids", None) else None
            else:
                inputs = tokenizer(running_prompt, return_tensors="pt").to(model.device)
                gen_started = time.perf_counter()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=args.eval_max_steps,
                        do_sample=True, temperature=0.8,
                    )
                gen_seconds = time.perf_counter() - gen_started
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                num_toks = int(new_ids.numel())
                gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

            smiles_eval = evaluate_smiles_output(
                class_name=class_name,
                output=gen_text,
                grammar_text=grammar_text,
                prompt_exemplars=prompt_exemplars,
                require_rdkit=True,
            )
            cleaned = clean_smiles_output(gen_text)
            is_novel = bool(
                smiles_eval.get("syntax_valid")
                and cleaned
                and cleaned not in seen_per_class[class_name]
            )
            if cleaned and smiles_eval.get("syntax_valid"):
                seen_per_class[class_name].add(cleaned)
                running_prompt = running_prompt.rstrip() + f" {cleaned}\nMolecule: "

            row_out: dict[str, Any] = {
                "question": class_name,
                "llm_response": gen_text,
                "correct": bool(
                    is_novel and smiles_eval.get("class_membership")
                ),
                "syntax_valid": bool(smiles_eval.get("syntax_valid", False)),
                "generation_seconds": gen_seconds,
            }
            if num_toks is not None:
                row_out["num_tokens"] = num_toks
            rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_unconstrained_spider_adapter(args: argparse.Namespace) -> int:
    """Unconstrained Spider baseline without grammar masking in legacy CRANE ``main.py``.

    Uses the same chain-of-thought Spider prompt as other legacy adapters and scores via
    execution match against the local Spider databases (``SPIDER_DB_DIR`` or repository layout).
    """
    from synthesis.evaluate.benchmarks.registry import get_logic
    from synthesis.evaluate.evaluator import Evaluator

    run_started = time.perf_counter()
    logic = get_logic("spider")
    eval_runtime = Evaluator(
        dataset_name="spider",
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        spider_split_file=args.spider_split_file,
        spider_split_name=args.spider_split_name,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, "spider")
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_local_cuda_device(args.device)

    if args.eval_backend == "vllm":
        from vllm import LLM

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )

    rows: list[dict[str, Any]] = []
    max_new = max(32, int(args.eval_max_steps))

    for example in examples:
        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "evaluator_default")
        if not isinstance(prompt, str):
            prompt = str(prompt)
        num_toks: int | None = None
        if args.eval_backend == "vllm":
            from vllm import SamplingParams as _SP

            gen_started = time.perf_counter()
            sp = _SP(max_tokens=max_new, temperature=0.0)
            outputs = llm.generate([prompt], sp)
            gen_seconds = time.perf_counter() - gen_started
            completion = outputs[0].outputs[0]
            suffix = completion.text
            num_toks = len(completion.token_ids) if getattr(completion, "token_ids", None) else None
        else:
            import torch

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_started = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    stopping_criteria=_spider_hf_stopping_criteria(
                        tokenizer, inputs["input_ids"].shape[1]
                    ),
                )
            gen_seconds = time.perf_counter() - gen_started
            new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            num_toks = int(new_ids.numel())
            suffix = tokenizer.decode(new_ids, skip_special_tokens=True)

        scored_output = suffix
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))

        question = _baseline_row_question("spider", example, expected)
        row_out: dict[str, Any] = {
            "question": question,
            "llm_response": suffix,
            "prompt_used": prompt,
            "correct": bool(is_correct),
            "syntax_valid": bool(syntax_valid),
            "generation_seconds": gen_seconds,
        }
        if num_toks is not None:
            row_out["num_tokens"] = num_toks
        rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _crane_delimited_start_grammar(grammar_text: str) -> str:
    """Make a grammar's default ``start`` rule include CRANE's delimiters.

    CRANE switches into constrained mode only after the model emits ``<<``.
    Once that happens its parser sees partial text prefixed with ``<<``, so
    grammars whose default start rule is just the payload body must be wrapped
    as ``start: "<<" body ">>"``.  The evaluator grammars keep delimiter
    handling in ``syncode``/``csd_start`` helpers, which are not the default
    Lark start rule used by the CRANE-style decoder.
    """
    if re.search(r'(?m)^\??start\s*:\s*"<<".*">>"', grammar_text):
        return grammar_text

    start_rule = re.compile(r"(?m)^(\??)start(\s*:)")
    if not start_rule.search(grammar_text):
        raise ValueError("Cannot wrap CRANE grammar: no default start rule found")

    wrapped_body = start_rule.sub(r"\1crane_body\2", grammar_text, count=1)
    return 'start: "<<" crane_body ">>"\n' + wrapped_body


def _crane_adaptive_surface(dataset: str, grammar_text: str) -> dict[str, Any]:
    """Return the adaptive-decoder surface declared by each benchmark."""
    return {
        "grammar": _crane_delimited_start_grammar(grammar_text),
        "start_symbol": "<<",
        "start_in_grammar": True,
        "end_symbol": ">>",
        "end_in_grammar": True,
        "start_inside_constrained": False,
    }


def _crane_completion_for_scoring(dataset: str, prompt: str, raw_output: str) -> str:
    if dataset == "spider":
        return str(raw_output or "")
    completion = completion_for_scoring(prompt, raw_output)
    if dataset != "smiles":
        return completion

    complete_spans = re.findall(r"<<(.*?)>>", completion, flags=re.DOTALL)
    if not complete_spans:
        LOGGER.warning(
            "[legacy-crane] SMILES output had no complete constrained span "
            "output_chars=%d",
            len(completion),
        )
        return ""
    return complete_spans[-1].strip()


def _crane_via_adaptive_syncode(args: argparse.Namespace, dataset: str) -> int:
    """Run CRANE-style adaptive constrained decoding via vendored AdaptiveSynCode.

    This path uses the shared synthesis evaluator to choose examples and score
    answers, so CRANE is compared on the same split as MetaDecode, GCD, IterGen,
    and CARS. AdaptiveSynCode implements CRANE's << >> switching logic.
    """
    run_started = time.perf_counter()
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[2]
    syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    syncode_pkg = syncode_root / "syncode"
    for p in [str(syncode_root), str(syncode_pkg)]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    from syncode.infer import AdaptiveSynCode
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset)

    from synthesis.evaluate.evaluator import Evaluator
    eval_runtime = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=_resolve_device(args.device, args.eval_backend),
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name,
        # Forward the per-class SMILES filter; without this the Evaluator loads the
        # default sample (all three classes x sample_size) and --smiles-classes is ignored.
        smiles_classes=(
            [s.strip() for s in args.smiles_classes.split(",") if s.strip()]
            if dataset == "smiles" and getattr(args, "smiles_classes", None)
            else None
        ),
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples = logic.load_dataset_sample(eval_runtime)

    device = _legacy_local_cuda_device(args.device)
    base_gsm_grammar_text = ""
    if dataset == "gsm_symbolic":
        base_gsm_grammar_text = (
            _legacy_gsm_symbolic_grammar_base(repo_root, examples)
        )
    spider_grammar_text = ""
    if dataset == "spider":
        spider_grammar_text = (
            (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()
        )

    def _grammar_for_example(example: dict[str, Any]) -> str:
        if dataset == "gsm_symbolic":
            return base_gsm_grammar_text
        if dataset == "spider":
            return spider_grammar_text
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for AdaptiveSynCode adapter: {dataset}")

    syncode_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_prompt_suffix: dict[str, str] = {}

    for example in examples:
        surface = _crane_adaptive_surface(dataset, _grammar_for_example(example))
        grammar_text = str(surface["grammar"])
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in syncode_cache:
            syncode_cache[cache_key] = AdaptiveSynCode(
                model=args.eval_model,
                mode="grammar_strict",
                quantize=False,
                device=device,
                grammar=grammar_text,
                parse_output_only=True,
                log_level=0,
                start_symbol=surface["start_symbol"],
                start_in_grammar=surface["start_in_grammar"],
                end_symbol=surface["end_symbol"],
                end_in_grammar=surface["end_in_grammar"],
                start_inside_constrained=surface["start_inside_constrained"],
                max_new_tokens=max(32, int(args.eval_max_steps)),
                num_return_sequences=1,
                **_crane_generation_kwargs(dataset),
            )
            generation_kwargs = _crane_generation_kwargs(dataset)
            LOGGER.info(
                "[legacy-crane] initialized decoder dataset=%s model=%s "
                "do_sample=%s temperature=%s start_symbol=%r end_symbol=%r",
                dataset,
                args.eval_model,
                generation_kwargs["do_sample"],
                generation_kwargs.get("temperature"),
                surface["start_symbol"],
                surface["end_symbol"],
            )
        sc = syncode_cache[cache_key]

        if dataset == "smiles":
            cls = str(example.get("class_name", ""))
            if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                example["prompt"] = example["prompt"].rstrip() + smiles_prompt_suffix.get(cls, "")

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "evaluator_default")
        if dataset == "spider":
            prompt = str(prompt)
        prompt = _crane_prompt_for_generation(dataset, prompt)
        gen_started = time.perf_counter()
        completions = sc.infer(prompt, stop_words=_crane_stop_words(dataset))
        gen_seconds = time.perf_counter() - gen_started
        raw_output = completions[0] if completions else ""
        completion = _crane_completion_for_scoring(dataset, prompt, raw_output)
        LOGGER.info(
            "[legacy-crane] completed example dataset=%s raw_chars=%d scored_chars=%d",
            dataset,
            len(raw_output),
            len(completion),
        )
        scored_output = (
            eval_runtime._truncate_gsm_output(completion)
            if dataset == "gsm_symbolic"
            else completion
        )
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(actual and re.search(r"\bselect\b", actual, flags=re.IGNORECASE))
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            if syntax_valid and actual:
                cls = str(example.get("class_name", ""))
                if __import__("os").environ.get("CSD_SMILES_ROLLING_PROMPT", "1") != "0":
                    smiles_prompt_suffix[cls] = _cap_suffix(smiles_prompt_suffix.get(cls, "") + f" {actual}\nMolecule:")

        question = _baseline_row_question(dataset, example, expected)
        rows.append(
            {
                "question": question,
                "llm_response": completion,
                "prompt_used": prompt,
                "correct": bool(is_correct),
                "syntax_valid": bool(syntax_valid),
                "generation_seconds": gen_seconds,
            }
        )

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics={
            "adapter": "crane_shared_evaluator",
            "dataset_source": "synthesis.evaluate",
            "split_file": str(
                args.gsm_split_file if dataset == "gsm_symbolic" else args.spider_split_file
            )
            if dataset in {"gsm_symbolic", "spider"}
            and (args.gsm_split_file if dataset == "gsm_symbolic" else args.spider_split_file)
            else None,
            "split_name": args.gsm_split_name if dataset == "gsm_symbolic" else (
                args.spider_split_name if dataset == "spider" else None
            ),
        },
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def run_crane_legacy_adapter(args: argparse.Namespace) -> int:
    dataset = _normalize_dataset(args.dataset)

    if dataset == "smiles" and args.strategy == "unconstrained":
        return run_unconstrained_smiles_adapter(args)

    if dataset == "spider" and args.strategy == "unconstrained":
        return run_unconstrained_spider_adapter(args)

    if args.strategy == "crane":
        return _crane_via_adaptive_syncode(args, dataset)

    mode, do_cot = _mode_for_strategy(args.strategy)
    grammar = _crane_grammar_name(dataset)

    repo_root = Path(__file__).resolve().parents[2]
    crane_src_dir = repo_root / "legacy" / "CRANE" / "src"
    if not crane_src_dir.exists():
        raise RuntimeError(f"Legacy CRANE src directory not found: {crane_src_dir}")

    cmd = [
        "python",
        "main.py",
        "--dataset",
        dataset,
        "--num_examples",
        str(args.eval_sample_size),
        "--num_shots",
        "1",
        "--overwrite_results",
        "True",
        "--write_file",
        "True",
        "--regex_parser",
        "True",
        "--modify_system_prompt",
        "True",
        "--cot_model",
        args.eval_model,
        "--cot_grammar_mode",
        mode,
        "--cot_grammar",
        grammar if mode != "original" else "text",
        "--out_grammar",
        grammar if mode != "original" else "text",
    ]
    if do_cot:
        cmd.extend(["--do_cot", "True"])

    legacy_device = _legacy_local_cuda_device(args.device)
    cmd.extend(["--cot_device", legacy_device, "--llm_parser_device", legacy_device])

    if dataset == "gsm_symbolic":
        cmd.extend(["--start_symbol", "<<", "--end_symbol", ">>"])
    elif dataset in ("spider", "smiles"):
        cmd.extend(["--start_symbol", "<<", "--end_symbol", ">>"])

    if dataset == "smiles":
        smiles_classes = getattr(args, "smiles_classes", None) or "acrylates,chain_extenders,isocyanates"
        spc = getattr(args, "smiles_samples_per_class", None)
        if spc is None:
            spc = args.eval_sample_size
        cmd.extend(["--smiles_classes", smiles_classes])
        cmd.extend(["--smiles_samples_per_class", str(spc)])

    repo_syncode_root = repo_root / "synthesis" / "evaluate" / "syncode"
    repo_syncode_pkg = repo_syncode_root / "syncode"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    prefix_paths = [str(repo_root), str(repo_syncode_root), str(repo_syncode_pkg)]
    env["PYTHONPATH"] = os.pathsep.join(
        prefix_paths + ([existing_pythonpath] if existing_pythonpath else [])
    )

    crane_run_started = time.perf_counter()
    subprocess.run(cmd, cwd=crane_src_dir, check=True, env=env)

    rows = _annotate_legacy_rows_with_syntax(
        _load_latest_crane_results(crane_src_dir, dataset),
        args,
        dataset,
    )
    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - crane_run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy fixed strategy code and export minimal baseline JSON"
    )
    parser.add_argument("--strategy", required=True, choices=["unconstrained", "gcd", "crane", "itergen", "cars"])
    parser.add_argument("--dataset", required=True, choices=["gsm", "gsm_symbolic", "spider", "smiles"])
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--eval-sample-size", type=int, default=10)
    parser.add_argument(
        "--eval-start-index",
        type=int,
        default=0,
        help="Start index within the loaded evaluation sample, inclusive",
    )
    parser.add_argument(
        "--eval-end-index",
        type=int,
        default=None,
        help="End index within the loaded evaluation sample, exclusive",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-max-steps", type=int, default=150)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--dafny-path", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (default: 1; capped by VAS_MAX_CUDA_DEVICES)",
    )
    parser.add_argument("--gsm-split-file", type=str, default=None,
                        help="Optional GSM train/eval split manifest JSON")
    parser.add_argument("--gsm-split-name", type=str, choices=["train", "test"], default="test",
                        help="Which split from --gsm-split-file to use (default: test). "
                             "GSM's held-out side is named 'test'; the 'eval' alias was removed 2026-07-17.")
    parser.add_argument("--spider-split-file", type=str, default=None,
                        help="Optional Spider train/test split manifest JSON")
    parser.add_argument("--spider-split-name", type=str, choices=["train", "test", "eval"], default="eval",
                        help="Which split from --spider-split-file to use (default: eval)")
    parser.add_argument(
        "--smiles-classes",
        type=str,
        default=None,
        help="Comma-separated SMILES classes for legacy CRANE main.py (default: all three)",
    )
    parser.add_argument(
        "--smiles-samples-per-class",
        type=int,
        default=None,
        help="Samples per class for legacy CRANE main.py (default: eval-sample-size)",
    )
    parser.add_argument(
        "--cars-search-steps",
        type=int,
        default=200,
        help=(
            "Max stochastic CARS decode attempts per example (the grammar must accept "
            "a sample). Default 200. Independent of --eval-max-steps."
        ),
    )
    args = parser.parse_args()
    from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

    args.vllm_tensor_parallel_size = resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size)

    # Use process spawning instead of forking when vLLM starts worker processes.
    # Forking after CUDA is initialized in the parent crashes with
    # "Cannot re-initialize CUDA in forked subprocess".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _ensure_repo_cache_env()

    dataset_normalized = _normalize_dataset(args.dataset)
    # CRANE main.py only supports gsm_symbolic and FOL — no Spider support
    # (no prompt_templates/spider.yaml, no generate_spider_with_itergen function).
    # For Spider, fall through to the legacy shared-evaluator adapters which DO
    # support Spider and were the source of the 05-18 baselines.
    crane_repo_datasets = {"gsm_symbolic"}
    crane_repo_strategies = {"unconstrained", "gcd", "crane", "itergen"}
    if (
        dataset_normalized in crane_repo_datasets
        and args.strategy in crane_repo_strategies
    ):
        from synthesis.evaluate.baselines.crane_repo_runner import (
            run_crane_repo_baseline,
        )
        raise SystemExit(run_crane_repo_baseline(args, dataset_normalized))

    if args.strategy == "gcd":
        raise SystemExit(run_gcd_legacy_adapter(args))
    if args.strategy == "itergen":
        raise SystemExit(run_itergen_legacy_adapter(args))
    if args.strategy == "cars":
        raise SystemExit(run_cars_legacy_adapter(args))
    raise SystemExit(run_crane_legacy_adapter(args))


if __name__ == "__main__":
    main()
