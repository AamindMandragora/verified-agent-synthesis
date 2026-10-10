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


# One logger for every baseline adapter. Each line opens with the adapter's tag
# ("[legacy-gcd]", "[legacy-crane]", ...) and names the dataset, so a reader can tell
# from the log alone whether the step that defines each baseline really ran.
LOGGER = logging.getLogger("csd.baseline_trace")

_TRACE_TEXT_LIMIT = 300


def _configure_trace_logging() -> None:
    """CSD_TRACE_LOG=1 prints every ``csd.*`` INFO line to stdout.

    Nothing in this harness ever attached a log handler, so without this every
    ``LOGGER.info`` call is dropped. Lane ``stdout.log`` files capture stdout, so no
    separate log file is needed. Same on/off convention as ``CSD_GROUNDING_LOG``
    (model_utils.py), but for the whole ``csd`` logger tree rather than one logger.
    """
    if not os.environ.get("CSD_TRACE_LOG"):
        return
    csd_logger = logging.getLogger("csd")
    if any(getattr(handler, "_csd_trace_log", False) for handler in csd_logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    handler._csd_trace_log = True
    csd_logger.addHandler(handler)
    csd_logger.setLevel(logging.INFO)


def _clip(text: Any, limit: int = _TRACE_TEXT_LIMIT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[+{len(text) - limit} chars]"


def _token_id_list(value: Any) -> list[int] | str:
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(int(item) for item in value)
    return [int(value)]


def _log_lane_header(
    tag: str,
    dataset: str,
    *,
    prompt: Any,
    end_token_ids: Any,
    max_new_tokens: Any,
) -> None:
    """Once per lane, for the first example only: the one place a full prompt is logged."""
    prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
    LOGGER.info(
        "[%s] lane-header dataset=%s end_token_ids=%s max_new_tokens=%s "
        "first_example_prompt_chars=%d first_example_prompt=\n%s",
        tag,
        dataset,
        _token_id_list(end_token_ids),
        max_new_tokens,
        len(prompt_text),
        prompt_text,
    )


def _syncode_stop_token_ids(sc: Any) -> Any:
    return getattr(getattr(sc, "model", None), "_stop_token_ids", None)


def _syncode_generation_trace(sc: Any) -> dict[str, Any]:
    """What the vendored decoder recorded about its last call (language_model.py)."""
    return dict(getattr(getattr(sc, "model", None), "last_generation_trace", None) or {})


def _syncode_audit_counts(sc: Any) -> dict[str, Any] | None:
    audit = getattr(getattr(sc, "grammar_decoder", None), "constraint_audit", None)
    if audit is None:
        return None
    return {
        "masked": int(audit.total_masked_steps),
        "unmasked": int(audit.total_unconstrained_steps),
        "by_reason": dict(audit.counts_by_reason),
    }


def _audit_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    """The audit counts across a whole run; one example's share is after minus before."""
    if before is None or after is None:
        return None
    by_reason = {
        reason: count - before["by_reason"].get(reason, 0)
        for reason, count in after["by_reason"].items()
        if count - before["by_reason"].get(reason, 0)
    }
    return {
        "masked": after["masked"] - before["masked"],
        "unmasked": after["unmasked"] - before["unmasked"],
        "by_reason": by_reason,
    }


from synthesis.evaluate.benchmarks.smiles.seed_split import seed_smiles_draw, smiles_draw_seed


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


def _refresh_smiles_prompt_exemplars(example: dict[str, Any], prompt: Any) -> None:
    """Rebuild the SMILES exclusion list from the prompt the model is about to see.

    The list built at dataset load only covers the few-shot file. The instruction line
    the harness appends shows one more molecule inline, and CARS's rolling suffix adds
    the ones this run already accepted; a baseline that copies any of them back is
    repeating the prompt. Non-SMILES examples carry no ``grammar_text`` and are left
    alone.
    """
    from synthesis.evaluate.benchmarks.smiles.dataset import extract_prompt_exemplars

    grammar_text = example.get("grammar_text")
    if not grammar_text or not example.get("class_name") or not isinstance(prompt, str):
        return
    example["prompt_exemplars"] = extract_prompt_exemplars(prompt, str(grammar_text))


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
        prompt = logic.format_prompt(evaluator, example)
    elif profile == "expression_only":
        prompt = logic.format_prompt_expression_only(evaluator, example)
    elif profile == "chain_of_thought":
        cot = getattr(logic, "format_prompt_chain_of_thought", None)
        prompt = cot(evaluator, example) if callable(cot) else logic.format_prompt(evaluator, example)
    else:
        raise ValueError(f"Unknown legacy prompt profile: {profile}")
    _refresh_smiles_prompt_exemplars(example, prompt)
    return prompt


def _cars_prompt_profile(dataset: str) -> str:
    if dataset == "spider":
        return "evaluator_default"
    return "expression_only"


def _gcd_prompt_profile(dataset: str) -> str:
    """GCD forces its answer's constrained span to start immediately (see ``_gcd_prompt``),
    so on GSM it cannot itself produce chain-of-thought text. It still uses the same task
    header and few-shot examples every other GSM lane uses (``format_prompt`` /
    ``evaluator_default``) -- the "<<" append is the only difference the method needs."""
    if dataset == "gsm_symbolic":
        return "evaluator_default"
    return "expression_only"


def _render_chat_prompt(tokenizer: Any, messages: Any) -> str:
    """Render a lane's task as one chat turn with thinking off, as generation.py does.

    Qwen3.5 only ends an answer at the end of an assistant turn. Handed bare few-shot
    text it keeps writing, and a grammar-masked lane then runs to its token cap:
    measured on Qwen3.5-2B after the complete acrylate "CCCCCOC(=O)C=C", the end token
    sits at logit 14.94 behind "." at 15.69, and mid-molecule at 12.94 behind "(" at
    19.00. The same task as a chat turn puts the end token on top at 23.12. A plain
    string is the single user message; a list is already messages.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _spider_completion_has_clean_boundary(text: str) -> bool:
    """True once a blank line has closed the model's own paragraph.

    This is the ONLY structural stop the unconstrained baseline is allowed at
    generation time: a blank line is a property of the text itself, not an
    assumption about what a SQL query looks like. The semicolon / repeated-SELECT /
    repeated-few-shot-label checks that used to live here were SQL-aware stops --
    exactly the kind of grammar-driven help an "unconstrained" baseline must not
    get. That cleanup still happens, just later: uniformly, for every span-free
    Spider answer, in the scorer (``_span_content_for_scoring`` in
    ``benchmarks/sql_spider/eval_logic.py``). The model's own end token is what
    actually stops generation in the normal case; this function only covers the
    "kept rambling past the answer" case.
    """
    if not text:
        return False
    return "\n\n" in text


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


# Every SMILES baseline samples at this one temperature, so no lane gains or loses from
# its sampling setting. 1.0 is what CARS and the CARS paper's GCD baseline
# (arXiv 2510.01902) use: the model's own distribution, unsharpened.
_SMILES_TEMPERATURE = 1.0


def _gcd_generation_kwargs(dataset: str) -> dict[str, Any]:
    """SMILES samples rather than decoding greedily -- a molecule grammar accepts many
    strings, so greedy would always draw the same one path."""
    if dataset == "smiles":
        return {"do_sample": True, "temperature": _SMILES_TEMPERATURE}
    return {"do_sample": False}


def _crane_stop_words(dataset: str) -> list[str] | None:
    """Halting generation the moment ``>>`` appears would stop CRANE after its
    FIRST span, so the model could never return to free text and open a
    second one -- exactly the "many spans per answer" behaviour Algorithm 1
    describes. Spider and SMILES answers can need more than one span (a
    second query attempt, a second candidate molecule after reasoning about
    why the first is wrong); GSM's answer is one expression, so stopping
    there is still faithful. Everywhere else, rely on the model's own end
    token or the token budget (see ``HuggingFaceModel._stop_token_ids`` /
    ``_generate`` in syncode's ``language_model.py``), same as an unmasked
    decode. The scorer already takes the LAST closed span
    (``_span_content_for_scoring`` for Spider, ``_crane_smiles_span`` for
    SMILES), so a second span is used once the model can produce one.
    """
    if dataset == "gsm_symbolic":
        return [">>"]
    return None


def _crane_generation_kwargs(dataset: str) -> dict[str, Any]:
    if dataset == "smiles":
        # SMILES is scored on unique molecules, so greedy would repeat one string.
        return {"do_sample": True, "temperature": _SMILES_TEMPERATURE, "top_p": 1.0}
    return {"do_sample": False}


_CRANE_SMILES_ANSWER_ONLY = "Your response must be a single SMILES molecule and nothing else."
# The delimiters are named outright, the way the Spider and GSM CRANE prompts name
# them. CRANE turns the grammar mask on only after the model writes "<<", and an
# instruction that merely points at the examples did not get the model there: the
# 2026-09-20 trace opened no span on 10 of 10 4B examples and 7 of 10 2B ones. A model
# that quotes "<<" mid-reasoning opens a span early, which is ordinary CRANE behaviour:
# the scorer takes the LAST closed span (``_crane_smiles_span``).
_CRANE_SMILES_REASON_FIRST = (
    "First reason briefly, step by step, about what defines this class of molecule and how "
    "yours differs from the examples. Then write the final molecule on a new line wrapped in "
    "<< >>, exactly like the examples. Stop after the closing >>."
)


def _crane_smiles_prompt(prompt: str) -> str:
    """CRANE only constrains text after the model writes the opening delimiter, so its SMILES
    prompt asks for reasoning first and shows every example molecule inside a span."""
    if _CRANE_SMILES_ANSWER_ONLY not in prompt:
        raise ValueError("SMILES prompt changed: CRANE prompt rewrite no longer applies")
    body = prompt.replace(_CRANE_SMILES_ANSWER_ONLY, _CRANE_SMILES_REASON_FIRST).rstrip()
    body = re.sub(r"^Molecule: (\S+)$", r"Molecule: <<\1>>", body, flags=re.MULTILINE)
    if not body.endswith("Molecule:"):
        raise ValueError("SMILES prompt must end with an open 'Molecule:' line")
    return body[: -len("Molecule:")] + "Reasoning:"


def _crane_smiles_span(output: str) -> str:
    """The scored molecule is the content of the last complete span."""
    spans = re.findall(r"<<(.*?)>>", output, flags=re.DOTALL)
    return spans[-1].strip() if spans else ""


_SPIDER_ANSWER_ONLY = "Only output the SQL quey. "
_SPIDER_CRANE_HEADER = (
    "Write a SINGLE SQL query answering the question, using ONLY the tables and columns "
    "in the schema.\n\n"
    "Reason step by step (tables, joins, filters). Then output SQL: followed by your "
    "query wrapped in << >>. Stop after the closing >>.\n\n"
    "You will always respond in the format described below: \n"
    "Let's think step by step. <reasoning> SQL: <<SQL query>>"
)


def _crane_spider_prompt(prompt: Any) -> Any:
    """CRANE only starts masking after the model writes "<<", so the Spider prompt has
    to ask for it. The shared Spider prompt says "Only output the SQL quey.", which both
    rules out reasoning and never names a delimiter, and the 2026-09-20 trace shows the
    cost: 10 of 10 answers with spans_opened=0, scored on free text.

    The header mirrors CRANE's own GSM prompt, which names both delimiters outright and
    shows the response shape. SMILES could avoid the literal delimiters because its
    few-shot molecules demonstrate the format; Spider carries no few-shot example, so
    there is nothing else to learn it from. A model that quotes "<<" mid-reasoning opens
    a span early, which is ordinary CRANE behaviour: the scorer takes the LAST closed
    span (``_span_content_for_scoring``).

    The db_id, the schema and the question are copied through byte for byte.
    """
    task_text = getattr(prompt, "task_text", None)
    if task_text is None or _SPIDER_ANSWER_ONLY not in task_text:
        raise ValueError("Spider prompt changed: CRANE prompt rewrite no longer applies")
    return type(prompt)(
        _SPIDER_CRANE_HEADER + "\n\n" + task_text.replace(_SPIDER_ANSWER_ONLY, ""),
        answer_cue="Let's think step by step.",
        guidance=prompt.guidance,
        model_name=prompt.model_name,
    )


def _crane_prompt_for_generation(dataset: str, prompt: Any) -> Any:
    if dataset == "smiles":
        return _crane_smiles_prompt(prompt)
    if dataset == "spider":
        return _crane_spider_prompt(prompt)
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


def _gcd_gsm_grammar(base_gsm_grammar: str) -> str:
    """Grammar for the text GCD writes after the prompt's "<<": the expression, then ">>".

    SynCode parses from the rule named ``start``, so the closing marker has to sit in
    that rule. Without it ">>" is never allowed and no answer can end.
    """
    unused_rule = 'syncode: "<<" start ">>"\n'
    body_rule = "\nstart: any_expr\n"
    if unused_rule not in base_gsm_grammar or body_rule not in base_gsm_grammar:
        raise ValueError("gsm.lark no longer has the rules GCD rewrites")
    return base_gsm_grammar.replace(unused_rule, "", 1).replace(body_rule, '\nstart: any_expr ">>"\n', 1)


def _spider_generation_prompt(prompt: Any, tokenizer: Any, model_name: str) -> str:
    """The text a Spider lane sends the model: the task through its chat template.

    Qwen3.5 only ends an answer at the end of an assistant turn. Handed the bare
    few-shot text, it never wants its end token: measured on Qwen3.5-2B after the
    complete query " SELECT COUNT( * ) FROM singer", <|im_end|> sits at logit 11.75
    against 18.25 for ";" and 16.62 for "_in". A grammar-masked lane therefore runs
    to its token cap (";" is not in the SQL grammar, so "_in" wins) and a free lane
    invents another "question: ... SQL:" pair. The same task through the chat
    template with thinking off puts <|im_end|> on top at 19.12.

    IterGen and our own method already render this way, so this is the one call the
    rest of the lanes share. CARS is not here on purpose: ``ConstrainedModel`` in the
    cars repo applies the same template with ``enable_thinking=False`` itself, and
    rendering here too would wrap the prompt twice.
    """
    render = getattr(prompt, "render_for_model", None)
    if render is None:
        raise TypeError(
            f"Spider lanes render SpiderPromptParts, got {type(prompt).__name__}"
        )
    return render(tokenizer, model_name=model_name)


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
        if logic.example_syntax_pass(all_valid, segments, None):
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
    metrics: dict[str, Any] = {
        "num_examples": len(rows),
        "timing_definitions": {
            "total_generation_seconds": "Generation time summed over retained/scored outputs; excludes discarded outer draws.",
            "mean_generation_seconds_per_example": "Mean generation time over retained/scored outputs, not collection cost per requested output.",
            "run_wall_time_seconds": "Total adapter collection wall time, including setup, scoring, and rejected work.",
        },
    }
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
    eval_split: dict[str, Any] | None = None,
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
        if eval_split is not None:
            payload["eval_split"] = eval_split
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
    if eval_split is not None:
        payload["eval_split"] = eval_split

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
    eval_split: dict[str, Any] | None = None,
) -> Path:
    partial_json = _cars_partial_json_path(output_json)
    _build_minimal_json(
        rows,
        partial_json,
        run_wall_time_seconds=run_wall_time_seconds,
        extra_metrics=extra_metrics,
        eval_split=eval_split,
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
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
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
            syntax_valid = bool(_aux and _aux.get("syntax_valid"))  # the Spider output contract's verdict
        else:
            syntax_valid = bool(
                logic.example_syntax_pass(all_valid, segments, None)
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


def _load_cars_results_from_log_dir(log_dir: Path) -> dict[str, list[Any]]:
    """``steps`` holds one entry per ACCEPTED sample; ``successes`` one bool per try."""
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            successes = payload.get("successes")
            return {
                "steps": steps,
                "successes": successes if isinstance(successes, list) else [True] * len(steps),
            }
    return {"steps": [], "successes": []}


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


def _cars_max_new_tokens(dataset: str, eval_max_steps: int) -> int:
    """CARS paper settings (arXiv 2510.01902, Appendix C): 256 new tokens for SMILES,
    512 for Spider -- fixed per benchmark, not derived from the shared --eval-max-steps
    campaign flag other methods share (we previously used 150 for both)."""
    if dataset == "smiles":
        return 256
    if dataset == "spider":
        return 512
    return max(32, int(eval_max_steps))


def _cars_smiles_draw_verdict(
    molecule: str, syntax_valid: bool, is_prompt_exemplar: bool, molecules_seen: set[str]
) -> str:
    """Whether a CARS SMILES draw earns a scored row: accepted, duplicate or rejected.

    The CARS paper (arXiv 2510.01902, Appendix C) collects N unique in-grammar molecules
    and reports how many of them belong to the target class. So a draw earns a row by
    being a molecule at all: non-empty, in-grammar, chemically valid, new, and not one
    the prompt already showed. Class membership is deliberately not part of that test,
    because it is what the accuracy over these rows measures; gating entry on it would
    make every run score 1.0. A repeat is a draw the run spent, not a second answer to
    average in.
    """
    if not molecule or not syntax_valid or is_prompt_exemplar:
        return "rejected"
    return "duplicate" if molecule in molecules_seen else "accepted"


def _cars_smiles_class_targets(examples: list[dict[str, Any]]) -> dict[str, int]:
    """One draw target per class: however many rows the dataset sample already
    allocated to that class (so sharded lanes keep drawing shard-sized targets)."""
    counts: dict[str, int] = {}
    for ex in examples:
        cls = str(ex.get("class_name", ""))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _cars_sampler_steps(
    cars_model: Any,
    prompt: str,
    *,
    n_steps: int,
    max_new_tokens: int,
) -> dict[str, list[Any]]:
    """Run upstream ``cars.CARS`` multi-step sampling; return its logged steps and tries.

    The CARS repo's own log directory is a tempdir deleted on return, so anything a
    reader needs from it has to be returned here and logged at our call site."""
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


def _cars_draw_stop_reason(*, tries: int, search_steps_cap: int, produced_text: bool) -> str:
    if produced_text:
        return "sample_accepted"
    if tries >= search_steps_cap:
        return f"try_cap_{search_steps_cap}"
    return "no_sample"


# Sampling fields that a model's generation_config.json can set. Transformers 5.5.4 fills each one
# that the caller leaves unset (None) from the model's file before the global defaults.
_MODEL_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "typical_p", "repetition_penalty")


def _use_global_sampling_defaults(hf_model: Any, *, tag: str) -> None:
    """Make CARS and GCD decode the same way on every model (user, 2026-10-09).

    Qwen2.5 ships temperature 0.7, top-p 0.8, top-k 20 and a repetition penalty (1.05 on 7B, 1.1 on
    1.5B). CARS leaves these unset and GCD passes keyword values only, so both took Qwen2.5's values:
    SMILES draws collapsed onto a few molecules, and even greedy GCD on GSM and Spider got the
    repetition penalty. Qwen3.5 ships no sampling values. Clearing the model's values gives every
    model the Transformers global defaults: temperature 1.0, top-p 1.0, top-k 50, repetition penalty
    1.0. (The CARS paper says no top-k. The user chose top-k 50 so that the Qwen2.5 lanes match the
    Qwen3.5 lanes, which already ran with it.) End tokens stay as they are. CRANE and IterGen build
    their config from config.json, and the unconstrained lanes pass explicit vLLM SamplingParams, so
    they never read the model's file.
    """
    config = getattr(hf_model, "generation_config", None)
    if config is None:
        LOGGER.info("[%s] no generation_config on the model; nothing to clear", tag)
        return
    cleared = {f: getattr(config, f, None) for f in _MODEL_SAMPLING_FIELDS if getattr(config, f, None) is not None}
    for field in cleared:
        setattr(config, field, None)
    LOGGER.info("[%s] cleared model sampling defaults %s; generate uses the global defaults", tag, cleared or "{}")


def _cars_log_first_generate_call(cars_model: Any, *, dataset: str) -> None:
    """Log the sampling settings the CARS library hands to ``generate``, once.

    The library builds its own GenerationConfig inside ``ConstrainedModel._generate``,
    so the only honest place to read the values is the call itself. The model's own
    defaults are logged beside them because Transformers can fall back to those for
    any field the passed config leaves at its default.
    """
    hf_model = getattr(cars_model, "model", None)
    original_generate = getattr(hf_model, "generate", None)
    if original_generate is None:
        LOGGER.info("[legacy-cars] sampling dataset=%s unavailable (no HF model on the CARS object)", dataset)
        return
    logged: list[bool] = []

    def _generate_and_log_once(*args: Any, **kwargs: Any) -> Any:
        if not logged:
            logged.append(True)
            passed = kwargs.get("generation_config")
            defaults = getattr(hf_model, "generation_config", None)
            LOGGER.info(
                "[legacy-cars] sampling dataset=%s passed_to_generate: temperature=%s top_p=%s top_k=%s "
                "do_sample=%s max_new_tokens=%s eos_token_id=%s | model_default_temperature=%s "
                "model_default_top_p=%s model_default_top_k=%s",
                dataset,
                getattr(passed, "temperature", "unset"),
                getattr(passed, "top_p", "unset"),
                getattr(passed, "top_k", "unset"),
                getattr(passed, "do_sample", "unset"),
                getattr(passed, "max_new_tokens", "unset"),
                getattr(passed, "eos_token_id", "unset"),
                getattr(defaults, "temperature", "unset"),
                getattr(defaults, "top_p", "unset"),
                getattr(defaults, "top_k", "unset"),
            )
        return original_generate(*args, **kwargs)

    hf_model.generate = _generate_and_log_once


def _cars_set_cached_grammar(
    cars_model: Any,
    grammar_text: str,
    grammar_cache: dict[str, Any],
) -> None:
    cached = grammar_cache.get(grammar_text)
    if cached is None:
        # LLGuidance reads Lark productions but cannot read its priority
        # annotations. Parse the grammar itself so dots in literals, regexes
        # and comments stay untouched. The final benchmark checker still
        # receives the original grammar, including its lexer priorities.
        import lark

        grammar_parser = lark.Lark.open(
            "grammars/lark.lark", rel_to=lark.__file__, parser="lalr",
            propagate_positions=True,
        )
        tree = grammar_parser.parse(grammar_text)
        priorities = sorted(
            ((node.meta.start_pos, node.meta.end_pos) for node in tree.find_data("priority")),
            reverse=True,
        )
        backend_grammar = grammar_text
        for start, end in priorities:
            backend_grammar = backend_grammar[:start] + backend_grammar[end:]
        if priorities:
            LOGGER.info(
                "[legacy-cars] grammar compatibility: removed %d Lark priority "
                "annotations for LLGuidance; productions unchanged; final checker unchanged",
                len(priorities),
            )
        cars_model._set_grammar_constraint(backend_grammar)
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
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)

    examples, slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )
    cars_model = ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)

    rows: list[dict[str, Any]] = []
    smiles_draw_metrics: dict[str, dict[str, int]] = {}
    grammar_cache: dict[str, Any] = {}
    n_steps = max(1, int(getattr(args, "cars_search_steps", 200)))
    max_new_tokens = _cars_max_new_tokens(dataset, args.eval_max_steps)
    LOGGER.info(
        "[legacy-cars] settings dataset=%s model=%s max_new_tokens=%d search_steps_cap=%d "
        "smiles_draw_cap=10x_target (sampling values: see the '[legacy-cars] sampling' line, "
        "logged on the first real generate call)",
        dataset,
        model_id,
        max_new_tokens,
        n_steps,
    )
    _use_global_sampling_defaults(getattr(cars_model, "model", cars_model), tag="legacy-cars")
    _cars_log_first_generate_call(cars_model, dataset=dataset)
    try_totals = {"tries": 0, "rejected_tries": 0, "draws_hitting_try_cap": 0}

    gsm_cars_grammar = ""
    if dataset == "gsm_symbolic":
        gsm_cars_grammar = _legacy_gsm_symbolic_grammar_base(repo_root, examples)
    spider_cars_grammar = ""
    if dataset == "spider":
        spider_cars_grammar = (repo_root / "synthesis" / "evaluate" / "grammars" / "sql.lark").read_text()

    lane_header_logged = False

    def _cars_draw(_ex_i: int, example: dict[str, Any]) -> tuple[str, bool, bool, dict[str, Any]]:
        """Run one CARS draw for ``example``; return (actual, syntax_valid, is_correct, row).

        The caller decides whether the row is scored: outside SMILES every draw is one
        example and every row counts, while SMILES scores unique accepted molecules only.
        """
        nonlocal lane_header_logged
        _maybe_reseed_parity_example(_ex_i)
        os.environ["CSD_PARITY_EXAMPLE_INDEX"] = str(_ex_i)
        if dataset == "smiles":
            seed_smiles_draw(example["draw_seed"])
        if dataset == "gsm_symbolic":
            grammar_text = gsm_cars_grammar
        elif dataset == "spider":
            grammar_text = spider_cars_grammar
        elif dataset == "smiles":
            grammar_text = str(example.get("grammar_text", ""))
        else:
            raise ValueError(f"Unsupported dataset for CARS adapter: {dataset}")
        _cars_set_cached_grammar(cars_model, grammar_text, grammar_cache)

        prompt = _legacy_benchmark_prompt(
            logic,
            eval_runtime,
            example,
            _cars_prompt_profile(dataset),
        )
        if dataset == "spider":
            prompt = str(prompt)
        if not lane_header_logged:
            lane_header_logged = True
            _log_lane_header(
                "legacy-cars",
                dataset,
                prompt=prompt,
                end_token_ids=getattr(getattr(cars_model, "tokenizer", None), "eos_token_id", None),
                max_new_tokens=max_new_tokens,
            )
        gen_started = time.perf_counter()
        sampler_run = _cars_sampler_steps(
            cars_model,
            prompt,
            n_steps=n_steps,
            max_new_tokens=max_new_tokens,
        )
        steps = sampler_run["steps"]
        raw_cars_text = _cars_completion_from_steps(steps)
        output_text = raw_cars_text
        gen_seconds = time.perf_counter() - gen_started
        tries = len(sampler_run["successes"])
        rejected_tries = sum(1 for ok in sampler_run["successes"] if not ok)
        draw_stop_reason = _cars_draw_stop_reason(
            tries=tries, search_steps_cap=n_steps, produced_text=bool(steps)
        )
        try_totals["tries"] += tries
        try_totals["rejected_tries"] += rejected_tries
        try_totals["draws_hitting_try_cap"] += int(draw_stop_reason.startswith("try_cap"))
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
            syntax_valid = bool(aux and aux.get("syntax_valid"))  # the Spider output contract's verdict
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))

        LOGGER.info(
            "[legacy-cars] draw=%d dataset=%s class=%s tries=%d rejected_tries=%d search_steps_cap=%d "
            "stop_reason=%s generated_tokens=%s syntax_valid=%s accepted=%s raw=%r scored=%r",
            _ex_i,
            dataset,
            example.get("class_name"),
            tries,
            rejected_tries,
            n_steps,
            draw_stop_reason,
            len(steps[-1].get("tokens") or []) if steps else 0,
            bool(syntax_valid),
            bool(syntax_valid and is_correct),
            _clip(raw_cars_text),
            _clip(scored_output),
        )
        row = {
            "question": _baseline_row_question(dataset, example, expected),
            "llm_response": completion,
            "prompt_used": prompt,
            "correct": bool(is_correct),
            "syntax_valid": bool(syntax_valid),
            "generation_seconds": gen_seconds,
        }
        return str(actual or ""), bool(syntax_valid), bool(aux and aux.get("is_prompt_exemplar")), row

    def _score_row(row: dict[str, Any]) -> None:
        rows.append(row)
        _write_cars_partial_json(
            rows,
            args.output_json,
            eval_split=_eval_split_record(args),
            run_wall_time_seconds=time.perf_counter() - run_started,
            extra_metrics=slice_metrics,
        )

    if dataset == "smiles":
        # CARS paper protocol (arXiv 2510.01902, Appendix C): draw until the target
        # number of unique in-grammar molecules is reached (prompt exemplars excluded,
        # and excluding molecules this run already accepted for the same class), capped
        # at 10x the target so a class that never validates cannot spin forever. Only
        # those unique accepted molecules are scored; a repeat or a rejected draw is a
        # draw the run spent, counted in the metrics below rather than averaged into
        # accuracy. The target per class is however many rows the sliced dataset sample
        # already allocated to it, so sharded lanes keep drawing shard-sized targets.
        class_templates: dict[str, dict[str, Any]] = {}
        for ex in examples:
            class_templates.setdefault(str(ex.get("class_name", "")), ex)
        target_counts = _cars_smiles_class_targets(examples)

        if int(getattr(args, "eval_start_index", 0) or 0) != 0:
            # CARS spends up to 10x its target in draws, so a shard starting at
            # draw k would spend seeds another shard also spends.
            raise ValueError(
                "CARS SMILES lanes cannot be sharded with --eval-start-index: each "
                "class draws seeds 0.. of its split until its target is met."
            )
        _ex_i = -1
        for cls, template in class_templates.items():
            target = max(1, target_counts.get(cls, 1))
            max_draws = target * 10
            accepted_molecules: set[str] = set()
            class_draws = 0
            class_rejected = 0
            class_duplicates = 0
            for _draw in range(max_draws):
                if len(accepted_molecules) >= target:
                    break
                _ex_i += 1
                class_draws += 1
                draw_example = dict(template)
                draw_example["draw_index"] = _draw
                draw_example["draw_seed"] = smiles_draw_seed(template["smiles_split_name"], template["smiles_trial"], _draw)
                actual, syntax_valid, is_exemplar, row = _cars_draw(_ex_i, draw_example)
                verdict = _cars_smiles_draw_verdict(
                    actual, syntax_valid, is_exemplar, accepted_molecules
                )
                if verdict == "accepted":
                    accepted_molecules.add(actual)
                    _score_row(row)
                elif verdict == "duplicate":
                    class_duplicates += 1
                else:
                    class_rejected += 1
            smiles_draw_metrics[cls] = {
                "target": target,
                "draw_cap": max_draws,
                "draws": class_draws,
                "rejected_draws": class_rejected,
                "duplicate_draws": class_duplicates,
                "scored_unique_molecules": len(accepted_molecules),
            }
            LOGGER.info(
                "[legacy-cars] class=%s dataset=smiles target=%d draw_cap=%d draws=%d rejected=%d "
                "duplicates=%d accepted_unique=%d stop_reason=%s",
                cls,
                target,
                max_draws,
                class_draws,
                class_rejected,
                class_duplicates,
                len(accepted_molecules),
                "target_reached" if len(accepted_molecules) >= target else "draw_cap_10x",
            )
    else:
        for _ex_i, example in enumerate(examples):
            _score_row(_cars_draw(_ex_i, example)[3])
    total_draws = sum(m["draws"] for m in smiles_draw_metrics.values()) or len(rows)
    LOGGER.info(
        "[legacy-cars] summary dataset=%s draws=%d scored_rows=%d tries=%d rejected_tries=%d "
        "draws_hitting_try_cap=%d search_steps_cap=%d",
        dataset,
        total_draws,
        len(rows),
        try_totals["tries"],
        try_totals["rejected_tries"],
        try_totals["draws_hitting_try_cap"],
        n_steps,
    )

    if not rows and not smiles_draw_metrics:
        raise RuntimeError("CARS produced no rows; refusing to write an empty baseline JSON")
    # A SMILES run that drew its whole cap without one molecule surviving the checks has
    # a real result -- zero of the N it was asked for -- so it writes a JSON scoring 0.0
    # with the draw counts attached, rather than looking like a crashed run.

    final_metrics = dict(slice_metrics or {})
    if smiles_draw_metrics:
        final_metrics["cars_smiles_draws_by_class"] = smiles_draw_metrics
    _build_minimal_json(
        rows,
        args.output_json,
        eval_split=_eval_split_record(args),
        run_wall_time_seconds=time.perf_counter() - run_started,
        extra_metrics=final_metrics,
    )
    try:
        _cars_partial_json_path(args.output_json).unlink()
    except FileNotFoundError:
        pass
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _unique_smiles_correct(
    class_name: str, molecule: str, correct: bool, syntax_valid: bool,
    seen_per_class: dict[str, set[str]],
) -> bool:
    """Count a cleaned SMILES string once per class; preserve syntax and row count."""
    seen = seen_per_class.setdefault(class_name, set())
    duplicate = molecule in seen
    if molecule and syntax_valid:
        seen.add(molecule)
    if duplicate:
        LOGGER.info("[baseline-smiles-score] class=%s duplicate=True counted_correct=False", class_name)
    return bool(correct and syntax_valid and molecule and not duplicate)


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
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples, _slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )

    device = _legacy_local_cuda_device(args.device)
    base_gsm_grammar = ""
    if dataset == "gsm_symbolic":
        base_gsm_grammar = _gcd_gsm_grammar(_legacy_gsm_symbolic_grammar_base(repo_root, examples))

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
    smiles_seen_per_class: dict[str, set[str]] = {}

    gsm_tokenizer: Any = None
    if dataset == "gsm_symbolic":
        from transformers import AutoTokenizer

        gsm_tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)

    for example in examples:
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in syncode_cache:
            gcd_generation_kwargs = _gcd_generation_kwargs(dataset)
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
                **gcd_generation_kwargs,
            )
            _use_global_sampling_defaults(
                getattr(getattr(syncode_cache[cache_key], "model", None), "model", None), tag="legacy-gcd"
            )
            LOGGER.info(
                "[legacy-gcd] settings dataset=%s model=%s mode=grammar_strict opp=False do_sample=%s "
                "temperature=%s top_p=%s top_k=%s max_new_tokens=%d stop_words=%s grammar_chars=%d",
                dataset,
                args.eval_model,
                gcd_generation_kwargs.get("do_sample"),
                gcd_generation_kwargs.get("temperature", "unset"),
                gcd_generation_kwargs.get("top_p", "unset"),
                gcd_generation_kwargs.get("top_k", "unset"),
                _gcd_max_new_tokens(),
                _gcd_stop_words(dataset),
                len(grammar_text),
            )
        sc = syncode_cache[cache_key]

        if dataset == "smiles":
            seed_smiles_draw(example["draw_seed"])

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, _gcd_prompt_profile(dataset))
        if dataset == "gsm_symbolic":
            prompt = _render_chat_prompt(gsm_tokenizer, prompt)
        if dataset == "spider":
            prompt = _spider_generation_prompt(prompt, sc.model.tokenizer, args.eval_model)
        if dataset == "smiles":
            prompt = _render_chat_prompt(sc.model.tokenizer, prompt)
        gen_started = time.perf_counter()
        gcd_prompt = _gcd_prompt(prompt)
        if not rows:
            _log_lane_header(
                "legacy-gcd",
                dataset,
                prompt=gcd_prompt,
                end_token_ids=_syncode_stop_token_ids(sc),
                max_new_tokens=_gcd_max_new_tokens(),
            )
        audit_before = _syncode_audit_counts(sc)
        completions = sc.infer(gcd_prompt, stop_words=_gcd_stop_words(dataset))
        gen_seconds = time.perf_counter() - gen_started
        gcd_trace = _syncode_generation_trace(sc)
        gcd_audit = _audit_delta(audit_before, _syncode_audit_counts(sc))
        generated_tokens = gcd_trace.get("generated_tokens")
        # GCD's defining claim: the grammar mask touched every generated token.
        LOGGER.info(
            "[legacy-gcd] example=%d dataset=%s generated_tokens=%s masked_steps=%s unmasked_steps=%s "
            "unmasked_by_reason=%s every_token_masked=%s stop_reason=%s last_token_id=%s raw=%r",
            len(rows),
            dataset,
            generated_tokens if generated_tokens is not None else "unavailable",
            gcd_audit["masked"] if gcd_audit else "unavailable",
            gcd_audit["unmasked"] if gcd_audit else "unavailable",
            gcd_audit["by_reason"] if gcd_audit else "unavailable",
            (
                gcd_audit["unmasked"] == 0 and gcd_audit["masked"] == generated_tokens
                if gcd_audit and generated_tokens is not None
                else "unavailable"
            ),
            gcd_trace.get("stop_reason", "unavailable"),
            gcd_trace.get("last_token_id", "unavailable"),
            _clip(completions[0] if completions else ""),
        )
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
            syntax_valid = bool(aux and aux.get("syntax_valid"))  # the Spider output contract's verdict
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            is_correct = _unique_smiles_correct(
                str(example.get("class_name", "")), str(actual or ""),
                is_correct, syntax_valid, smiles_seen_per_class,
            )

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
        eval_split=_eval_split_record(args),
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
    """One ``forward()`` to completion: no unit to check, so nothing to back out of."""
    tag = "legacy-itergen"
    iter_gen.start(prompt)
    _log_itergen_start(tag, iter_gen, detail="single_forward_no_check_no_backward")
    generated = iter_gen.forward()
    completion = (str(generated[0]) if generated else "") if isinstance(generated, list) else str(generated)
    _log_itergen_forward(tag, iter_gen, iteration=1, units=None, num=None, completion=completion)
    _log_itergen_finish(tag, iter_gen, iterations=1, max_iterations=1, backtracks=0, completion=completion)
    return completion


def _itergen_generation_stop_token_ids(iter_gen: Any, tokenizer: Any) -> set[int]:
    """Every end token the model can stop on, not just the one IterGen kept.

    IterGen's own ``start()`` overwrites ``generation_config.eos_token_id``
    with the tokenizer's single id (``<|im_end|>`` for Qwen3.5), which drops
    the model config's own end token (``<|endoftext|>``). Read both straight
    from the tokenizer and the underlying model's config instead of trusting
    whatever ``iter_gen`` narrowed it down to, so a completion that ends on
    either token is recognized.
    """
    for owner in (iter_gen, tokenizer):
        value = getattr(owner, "generation_stop_token_ids", None)
        if value is not None:
            if isinstance(value, int):
                return {int(value)}
            return {int(item) for item in value}

    ids: set[int] = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(getattr(iter_gen, "model", None), "config", None), "eos_token_id", None),
        getattr(getattr(iter_gen, "generation_config", None), "eos_token_id", None),
    ):
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (list, tuple, set)):
            ids.update(int(item) for item in value)
        else:
            ids.add(int(value))
    return ids


def _itergen_generated_token_ids(iter_gen: Any) -> list[Any] | None:
    """Generated IterGen IDs only, excluding prompt/session prefix IDs."""
    session_tokens = getattr(iter_gen, "session_tokens", None)
    start_from = getattr(iter_gen, "start_from", None)
    if session_tokens is None or start_from is None:
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
    return generated_tokens


def _itergen_end_reason(iter_gen: Any, *, iterations: int, max_iterations: int) -> str:
    """Which condition ended the loop: end token, token budget, IterGen's own
    ``finished()`` (grammar complete or a stop string), or our iteration limit."""
    ids = _itergen_generated_token_ids(iter_gen) or []
    stop_ids = _itergen_generation_stop_token_ids(iter_gen, getattr(iter_gen, "tokenizer", None))
    if ids and int(ids[-1]) in stop_ids:
        return f"end_token(id={int(ids[-1])})"
    budget = getattr(getattr(iter_gen, "generation_config", None), "max_new_tokens", None)
    if budget is not None and len(ids) >= int(budget):
        return f"token_budget(max_new_tokens={int(budget)})"
    if bool(iter_gen.finished()):
        return "itergen_finished(grammar_complete_or_stop_string)"
    if iterations >= max_iterations:
        return f"iteration_limit(max_iterations={max_iterations})"
    return "unknown"


def _log_itergen_settings(dataset: str, kwargs: dict[str, Any]) -> None:
    LOGGER.info(
        "[legacy-itergen] settings dataset=%s do_sample=%s temperature=%s recurrence_penalty=%s "
        "max_new_tokens=%s max_tokens=%s",
        dataset,
        kwargs.get("do_sample"),
        kwargs.get("temperature", "unset"),
        kwargs.get("recurrence_penalty", "unset (IterGen default 1.0 = off)"),
        kwargs.get("max_new_tokens"),
        kwargs.get("max_tokens"),
    )


def _log_itergen_start(tag: str, iter_gen: Any, *, detail: str) -> None:
    LOGGER.info(
        "[%s] start %s recurrence_penalty=%s itergen_eos_token_id=%s",
        tag,
        detail,
        getattr(iter_gen, "recurrence_penalty", "unavailable"),
        getattr(getattr(iter_gen, "generation_config", None), "eos_token_id", "unavailable"),
    )


def _log_itergen_forward(
    tag: str, iter_gen: Any, *, iteration: int, units: Any, num: Any, completion: str
) -> None:
    ids = _itergen_generated_token_ids(iter_gen)
    LOGGER.info(
        "[%s] forward iteration=%d units=%s num=%s generated_tokens_so_far=%s text=%r",
        tag,
        iteration,
        units,
        num,
        len(ids) if ids is not None else "unavailable",
        _clip(completion),
    )


def _itergen_backward_and_log(
    tag: str, iter_gen: Any, *, iteration: int, unit: str, value: Any, reason: str, count: int
) -> None:
    """Back up over the last unit of this name. Used by the Spider loop.

    Spider's units (column_name, table_name) do not nest, so the start of the last one
    sits before the text that failed the check and backing up over it removes that text.
    SMILES nests, which is why it has its own helper below.
    """
    before = _itergen_generated_token_ids(iter_gen)
    iter_gen.backward(unit)
    after = _itergen_generated_token_ids(iter_gen)
    LOGGER.info(
        "[%s] backward iteration=%d unit=%s units_removed=1 tokens_removed=%s value=%r reason=%s count=%d",
        tag,
        iteration,
        unit,
        len(before) - len(after) if before is not None and after is not None else "unavailable",
        _clip(value),
        reason,
        count,
    )


def _itergen_backward_over_the_molecule(
    tag: str, iter_gen: Any, *, iteration: int, value: Any, reason: str, count: int
) -> None:
    """Back out of the whole molecule, which is the unit the SMILES loop checked.

    backward("smiles") walks to the start of the LAST smiles symbol, and the SMILES
    grammars close one at the end of every parenthesised branch, so that start sits at
    the end of the text and nothing is removed: 70 of the 71 backtracks in the
    2026-09-20 run logged tokens_removed=0 and the loop re-offered the identical string
    until it ran out of tries. Backing up every generated token lands on char 0 of the
    generation (see IterGen.backward's "token" branch) and keeps the trace, so the
    recurrence penalty can push the redraw off the tokens just tried.
    """
    before = _itergen_generated_token_ids(iter_gen) or []
    if not before:
        return
    iter_gen.backward("token", num=len(before))
    after = _itergen_generated_token_ids(iter_gen)
    LOGGER.info(
        "[%s] backward iteration=%d unit=token num=%d tokens_removed=%s value=%r reason=%s count=%d",
        tag,
        iteration,
        len(before),
        len(before) - len(after) if after is not None else "unavailable",
        _clip(value),
        reason,
        count,
    )


def _log_itergen_finish(
    tag: str,
    iter_gen: Any,
    *,
    iterations: int,
    max_iterations: int,
    backtracks: int,
    completion: str,
    end_reason: str | None = None,
) -> None:
    ids = _itergen_generated_token_ids(iter_gen)
    LOGGER.info(
        "[%s] finish end_reason=%s iterations=%d backtracks=%d finished=%s generated_tokens=%s "
        "output_chars=%d text=%r",
        tag,
        end_reason or _itergen_end_reason(iter_gen, iterations=iterations, max_iterations=max_iterations),
        iterations,
        backtracks,
        bool(iter_gen.finished()),
        len(ids) if ids is not None else "unavailable",
        len(completion),
        _clip(completion),
    )


def _itergen_generation_token_evidence(iter_gen: Any) -> dict[str, Any] | None:
    """Capture only generated IterGen IDs, excluding prompt/session prefix IDs."""
    tokenizer = getattr(iter_gen, "tokenizer", None)
    generated_tokens = _itergen_generated_token_ids(iter_gen)
    if generated_tokens is None or tokenizer is None:
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
        kwargs.update(do_sample=True, temperature=_SMILES_TEMPERATURE)
    else:
        kwargs["do_sample"] = False
    if dataset in ("spider", "smiles"):
        # 0.3 is what every IterGen-mode experiment the authors checked in uses
        # (case_studies/sql/eval_sql.py line 189; the privacy run in README.md).
        # It is the only thing that makes a redraw after backward() differ from the
        # draw that was just rejected: backward keeps the trace, and forward scales
        # the logits of the tokens already tried at that position. Left at the 1.0
        # default, SMILES backtracking was inert -- 8 of 10 chain-extender examples
        # burned all ten backtracks on 2026-09-20, several returning the identical
        # continuation each time. GSM is not here: its lane takes one forward() and
        # never backtracks, so there is nothing for the penalty to act on.
        kwargs["recurrence_penalty"] = 0.3
    return kwargs


def _itergen_spider_authors_grammar_text() -> str:
    """IterGen's own SQL grammar (see ``baselines/itergen_authors/README.md``).

    Defines ``column_name`` / ``table_name``, the units ``_itergen_generate_spider``
    asks ``view()``/``backward()`` for. Our own ``grammars/sql.lark`` names the same
    concepts ``column_ref`` / ``table_ref``, so passing it here made ``view()``
    always empty and the back-up step unreachable: the "IterGen" lane was really
    just one unconstrained decode with no retry.
    """
    repo_root = Path(__file__).resolve().parents[2]
    grammar_path = (
        repo_root
        / "synthesis"
        / "evaluate"
        / "baselines"
        / "itergen_authors"
        / "sql_grammar.lark"
    )
    return grammar_path.read_text()


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
    tag = "legacy-itergen-spider"
    iter_gen.start(prompt)
    schema = _itergen_spider_schema(example)
    generated: Any = ""
    num_backwards = 0
    iterations = 0
    _log_itergen_start(
        tag,
        iter_gen,
        detail=f"tables={len(schema)} max_iterations={max_iterations} backwards_limit={backwards_limit}",
    )

    def _view_and_check(unit: str, in_schema: Any) -> tuple[Any, bool]:
        raw_view = iter_gen.view(unit)
        LOGGER.info("[%s] view iteration=%d unit=%s raw=%s", tag, iterations, unit, _clip(raw_view))
        names = raw_view[0]
        last = names[-1] if names else None
        passed = last is None or bool(in_schema(last))
        LOGGER.info(
            "[%s] check iteration=%d unit=%s value=%r passed=%s reason=%s",
            tag,
            iterations,
            unit,
            last,
            passed,
            "no_unit_yet" if last is None else ("in_schema" if passed else "not_in_schema"),
        )
        if not passed and num_backwards >= backwards_limit:
            LOGGER.info(
                "[%s] backward-skipped iteration=%d unit=%s reason=backwards_limit(%d)",
                tag,
                iterations,
                unit,
                backwards_limit,
            )
        return last, passed

    while not iter_gen.finished() and iterations < max_iterations:
        iterations += 1
        units = ["column_name", "table_name"]
        generated = iter_gen.forward(units=units, num=1)
        _log_itergen_forward(
            tag,
            iter_gen,
            iteration=iterations,
            units=units,
            num=1,
            completion=str(generated[0]) if isinstance(generated, list) and generated else str(generated),
        )

        last_column, column_ok = _view_and_check(
            "column_name", lambda value: _itergen_spider_column_exists(schema, str(value))
        )
        if not column_ok and num_backwards < backwards_limit:
            num_backwards += 1
            _itergen_backward_and_log(
                tag,
                iter_gen,
                iteration=iterations,
                unit="column_name",
                value=last_column,
                reason="not_in_schema",
                count=num_backwards,
            )
            continue

        last_table, table_ok = _view_and_check(
            "table_name", lambda value: str(value).strip().lower() in schema
        )
        if not table_ok and num_backwards < backwards_limit:
            num_backwards += 1
            _itergen_backward_and_log(
                tag,
                iter_gen,
                iteration=iterations,
                unit="table_name",
                value=last_table,
                reason="not_in_schema",
                count=num_backwards,
            )

    if isinstance(generated, list):
        completion = str(generated[0]) if generated else ""
    else:
        completion = str(generated)
    _log_itergen_finish(
        tag,
        iter_gen,
        iterations=iterations,
        max_iterations=max_iterations,
        backtracks=num_backwards,
        completion=completion,
    )
    return completion


def _itergen_model_ended_the_molecule(iter_gen: Any) -> bool:
    """True once the model itself ended the molecule.

    ``forward(units=["smiles"], num=1)`` returns as soon as the parser closes one
    ``smiles`` symbol, and the SMILES grammars close one at the end of every
    parenthesised branch as well as at the end of the molecule. So a single call normally
    lands mid-molecule: the 2026-09-20 trace accepted ``C=CC(=O)``, which is a branch
    close, not an acrylate. An ending is the model's own end token, or the parser closing
    the grammar's ``start`` symbol.

    Running out of token budget is neither, although ``finished()`` reports it. Treating
    a capped generation as an ending would accept exactly the kind of cut-off prefix this
    guards against, so the conservative reading wins: the text is handed back unchecked
    and the scorer judges it, the same way every other lane treats a capped generation.
    """
    if not iter_gen.finished():
        return False
    ids = _itergen_generated_token_ids(iter_gen)
    if not ids:
        return True
    budget = getattr(getattr(iter_gen, "generation_config", None), "max_new_tokens", None)
    if budget is None or len(ids) < int(budget):
        return True
    stop_ids = _itergen_generation_stop_token_ids(iter_gen, getattr(iter_gen, "tokenizer", None))
    return bool(stop_ids) and int(ids[-1]) in stop_ids


def _itergen_forward_to_molecule_end(
    iter_gen: Any, *, tag: str, iteration: int, forward_cap: int
) -> tuple[str, bool]:
    """Keep forwarding until the model ends the molecule. Returns the text and whether it
    did. ``forward_cap`` is the session's remaining token budget, which every call spends
    at least one of, so it cannot cut short a molecule the budget would have allowed."""
    completion = ""
    for _ in range(forward_cap):
        generated = iter_gen.forward(units=["smiles"], num=1)
        completion = str(generated[0]) if isinstance(generated, list) and generated else str(generated)
        _log_itergen_forward(
            tag, iter_gen, iteration=iteration, units=["smiles"], num=1, completion=completion
        )
        if _itergen_model_ended_the_molecule(iter_gen):
            return completion, True
        if iter_gen.finished():
            return completion, False
    return completion, False


def _itergen_generate_smiles(
    iter_gen: Any,
    prompt: Any,
    example: dict[str, Any],
    *,
    seen: set[str],
    max_iterations: int = 20,
    backwards_limit: int = 10,
) -> str:
    """Per-molecule IterGen loop: forward until the model ends a molecule, check RDKit
    validity and uniqueness (not a prompt exemplar, not already accepted this run),
    backward and retry -- bounded by ``backwards_limit`` -- on failure. Same
    forward/check/backward shape as ``_itergen_generate_spider``, adapted to SMILES since
    the IterGen paper has no molecule case study of its own (2026-09-20 baseline repair,
    item 7; our own adaptation, not something the IterGen authors defined).

    ``max_iterations`` counts molecule attempts, not ``forward()`` calls: one attempt
    takes as many calls as the grammar's nested ``smiles`` symbols need.

    ``seen`` is mutated in place with the accepted molecule, mirroring how the other
    SMILES adapters track already-accepted molecules across a rolling multi-draw run.
    """
    from synthesis.evaluate.benchmarks.smiles.metrics import (
        clean_smiles_output,
        evaluate_smiles_output,
    )

    tag = "legacy-itergen-smiles"
    iter_gen.start(prompt)
    class_name = str(example.get("class_name", ""))
    grammar_text = str(example.get("grammar_text", ""))
    completion = ""
    num_backwards = 0
    iterations = 0
    loop_exit: str | None = None
    budget = getattr(getattr(iter_gen, "generation_config", None), "max_new_tokens", None)
    forward_cap = max(1, int(budget)) if budget else max_iterations
    _log_itergen_start(
        tag,
        iter_gen,
        detail=f"class={class_name} max_iterations={max_iterations} backwards_limit={backwards_limit}",
    )
    while iterations < max_iterations:
        iterations += 1
        completion, model_ended = _itergen_forward_to_molecule_end(
            iter_gen, tag=tag, iteration=iterations, forward_cap=forward_cap
        )
        if not model_ended:
            loop_exit = "budget_ran_out_before_the_model_ended_the_molecule"
            break
        cleaned = clean_smiles_output(completion)
        smiles_eval = evaluate_smiles_output(
            class_name=class_name,
            output=completion,
            grammar_text=grammar_text,
            prompt_exemplars=list(seen),
            require_rdkit=True,
        )
        accepted = bool(smiles_eval.get("syntax_valid") and cleaned and cleaned not in seen)
        if accepted:
            check_reason = "valid_and_new"
        elif not smiles_eval.get("syntax_valid"):
            check_reason = "invalid_molecule"
        elif not cleaned:
            check_reason = "empty_after_cleanup"
        else:
            check_reason = "already_seen_or_prompt_exemplar"
        LOGGER.info(
            "[%s] check iteration=%d unit=smiles value=%r passed=%s reason=%s",
            tag,
            iterations,
            _clip(cleaned),
            accepted,
            check_reason,
        )
        if accepted:
            seen.add(cleaned)
            loop_exit = "molecule_accepted"
            break
        if num_backwards >= backwards_limit:
            loop_exit = f"backwards_limit({backwards_limit})"
            break
        num_backwards += 1
        _itergen_backward_over_the_molecule(
            tag,
            iter_gen,
            iteration=iterations,
            value=cleaned,
            reason=check_reason,
            count=num_backwards,
        )

    _log_itergen_finish(
        tag,
        iter_gen,
        iterations=iterations,
        max_iterations=max_iterations,
        backtracks=num_backwards,
        completion=completion,
        end_reason=loop_exit,
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


class _ItergenPerExampleTimeout(BaseException):  # BaseException: IterGen catches Exception and would swallow it
    """Raised when a single IterGen ``forward()`` exceeds the per-example wall-clock cap."""


def _itergen_generate_with_timeout(
    iter_gen: Any,
    prompt: Any,
    cap_seconds: float,
    *,
    spider_example: dict[str, Any] | None = None,
    smiles_example: dict[str, Any] | None = None,
    smiles_seen: set[str] | None = None,
) -> tuple[str, bool]:
    """Run ``_itergen_generate`` under a SIGALRM wall-clock cap.

    Returns ``(completion, timed_out)``. On timeout returns ``("", True)`` so the caller scores the
    example as a non-answer. A cap of <= 0 disables the guard (unbounded, legacy behaviour).
    """
    def _generate() -> str:
        if spider_example is not None:
            return _itergen_generate_spider(iter_gen, prompt, spider_example)
        if smiles_example is not None:
            assert smiles_seen is not None
            return _itergen_generate_smiles(iter_gen, prompt, smiles_example, seen=smiles_seen)
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
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples, _slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )

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
            # The IterGen loop backs up on the units its own grammar names
            # (column_name/table_name); our grammar names them differently.
            return _itergen_spider_authors_grammar_text()
        if dataset == "smiles":
            return str(example.get("grammar_text", ""))
        raise ValueError(f"Unsupported dataset for itergen adapter: {dataset}")

    itergen_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    smiles_seen_per_class: dict[str, set[str]] = {}

    for _ex_i, example in enumerate(examples):
        grammar_text = _grammar_for_example(example)
        cache_key = f"{dataset}:{hash(grammar_text)}"
        if cache_key not in itergen_cache:
            # Session ceiling must cover prompt + capped decode (``_new_tok``), not ``eval_max_steps`` alone.
            _session_ceiling = min(16384, max(2048, _new_tok + 4096))
            # Do not use ``stop_strings`` here: ``["\\n\\n"]`` fires HuggingFace stopping criteria and ends
            # generation before the LALR parser can reach a complete / EOF-ready state (e.g. GSM expr cut
            # mid-expression). Rely on grammar ``start`` completion + ``max_new_tokens`` instead.
            itergen_kwargs = _itergen_generation_kwargs(
                dataset=dataset,
                max_tokens=_session_ceiling,
                max_new_tokens=_new_tok,
            )
            itergen_cache[cache_key] = IterGen(
                grammar=grammar_text,
                model_id=args.eval_model,
                device=device,
                **itergen_kwargs,
            )
            _log_itergen_settings(dataset, itergen_kwargs)
        iter_gen = itergen_cache[cache_key]

        if dataset == "smiles":
            seed_smiles_draw(example["draw_seed"])

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "expression_only")
        if dataset == "gsm_symbolic":
            prompt = prompt.rstrip() + "<<"

        if dataset == "spider":
            generation_prompt = prompt.render_for_model(
                iter_gen.tokenizer,
                model_name=args.eval_model,
            )
        elif dataset == "smiles":
            generation_prompt = _render_chat_prompt(iter_gen.tokenizer, prompt)
        else:
            generation_prompt = prompt
        if not rows:
            _log_lane_header(
                "legacy-itergen",
                dataset,
                prompt=generation_prompt,
                end_token_ids=_itergen_generation_stop_token_ids(iter_gen, iter_gen.tokenizer),
                max_new_tokens=_new_tok,
            )
        gen_started = time.perf_counter()
        if dataset == "smiles":
            _smiles_cls = str(example.get("class_name", ""))
            smiles_seen_per_class.setdefault(
                _smiles_cls, set(example.get("prompt_exemplars", []) or [])
            )
        raw_completion, _timed_out = _itergen_generate_with_timeout(
            iter_gen,
            generation_prompt,
            _ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS,
            spider_example=example if dataset == "spider" else None,
            smiles_example=example if dataset == "smiles" else None,
            smiles_seen=smiles_seen_per_class.get(str(example.get("class_name", "")))
            if dataset == "smiles"
            else None,
        )
        if _timed_out:
            LOGGER.info(
                "[legacy-itergen] finish end_reason=wall_clock_timeout(%gs) dataset=%s example=%d",
                _ITERGEN_PER_EXAMPLE_TIMEOUT_SECONDS,
                dataset,
                _ex_i,
            )
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
        eval_split=_eval_split_record(args),
        run_wall_time_seconds=time.perf_counter() - run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _log_unconstrained_no_constraint(tag: str, dataset: str, backend: str) -> None:
    LOGGER.info(
        "[%s] constraint dataset=%s backend=%s this code path builds no logits processor and no "
        "grammar object; generate is called with sampling/stop settings only",
        tag,
        dataset,
        backend,
    )


def _vllm_stop_reason(completion: Any) -> str:
    """vLLM's own verdict: ``length`` is the budget; ``stop`` carries the stop string,
    or None when the model's end token ended it."""
    finish_reason = getattr(completion, "finish_reason", None)
    if finish_reason == "stop":
        return f"vllm:stop(stop_reason={getattr(completion, 'stop_reason', None)!r}; None means end token)"
    return f"vllm:{finish_reason}"


def _hf_stop_reason(new_ids: Any, *, end_token_ids: Any, max_new_tokens: int, other: str) -> str:
    from synthesis.evaluate.benchmarks.common.generation_trace import describe_generation_end

    end = describe_generation_end(
        generated_token_ids=new_ids.tolist() if hasattr(new_ids, "tolist") else list(new_ids),
        stop_token_ids=end_token_ids if isinstance(end_token_ids, list) else [],
        max_new_tokens=max_new_tokens,
        completion="",
        stop_words=None,
    )
    return f"hf:{other if end['stop_reason'] == 'unknown' else end['stop_reason']}"


# One draw is one line. The shared SMILES prompt asks for "exactly one line containing
# the SMILES molecule" and then leaves a bare "Molecule: " for the model to finish, and
# GCD, which sends this same prompt, gets one molecule per draw because its grammar ends
# the generation there. Stopping at a blank line instead let one unconstrained draw write
# a whole list, and the scorer keeps the text after the last "Molecule:" -- so that draw
# quietly got one try per line. The 2026-09-20 acrylates trace has a 352-token draw of
# about fifteen molecules scored on its last one.
_UNCONSTRAINED_SMILES_STOP = ["\n"]


def _unconstrained_smiles_seed_prompt(base_prompt: str) -> str:
    """Same expression-only framing GCD uses for SMILES (2026-09-20 baseline repair,
    item 5) -- the CARS paper's own few-shot files already say "a single SMILES
    molecule and nothing else", so a chain-of-thought prefix here would have asked
    for something the paper's own prompt already rules out. GCD and unconstrained
    now share this one prompt instead of building it two different ways."""
    from synthesis.evaluate.benchmarks.smiles.eval_logic import format_prompt_expression_only

    return format_prompt_expression_only(None, {"prompt": base_prompt})


def run_unconstrained_smiles_adapter(args: argparse.Namespace) -> int:
    """Unconstrained SMILES generation: generate without grammar masking, then score.

    Every draw sends the same fixed seed prompt (the CARS authors' own sampling loop,
    ``cars/cars.py::CARS.get_sample``/``get_samples``, encodes the prompt once and reuses
    it for every draw -- it does not append accepted molecules back in). Sampling gives
    diversity; greedy would repeat the same molecule every time. The scorer still
    excludes every molecule this run has already accepted for the class, tracked in
    ``seen_per_class`` independently of what is sent to the model.
    """
    from synthesis.evaluate.benchmarks.smiles.dataset import (
        SMILES_CLASSES,
        extract_prompt_exemplars,
        get_smiles_task,
    )
    from synthesis.evaluate.benchmarks.smiles.metrics import (
        clean_smiles_output,
        evaluate_smiles_output,
        molecule_is_correct,
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
    # Sharded lanes slice the draw stream with --eval-start-index/--eval-end-index,
    # the same way the other SMILES baselines do.
    draw_start = int(getattr(args, "eval_start_index", 0) or 0)
    raw_end = getattr(args, "eval_end_index", None)
    draw_end = n_per_class if raw_end is None else min(int(raw_end), n_per_class)
    if args.eval_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.eval_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )

    tag = "legacy-unconstrained-smiles"
    _log_unconstrained_no_constraint(tag, "smiles", args.eval_backend)
    hf_end_token_ids = (
        None
        if args.eval_backend == "vllm"
        else _token_id_list(getattr(model.generation_config, "eos_token_id", None))
    )
    rows: list[dict[str, Any]] = []
    seen_per_class: dict[str, set[str]] = {}

    for class_name in selected_classes:
        task = get_smiles_task(class_name)
        base_prompt = task["prompt"]
        grammar_text = str(task["grammar_text"])
        seed_prompt = _unconstrained_smiles_seed_prompt(base_prompt)
        seen_per_class[class_name] = set(
            extract_prompt_exemplars(seed_prompt, grammar_text)
        )
        # Every draw sends this same rendered prompt; it never grows across draws.
        generation_prompt = _render_chat_prompt(tokenizer, seed_prompt)

        for _i in range(draw_start, draw_end):
            draw_seed = smiles_draw_seed(args.smiles_split_name, args.smiles_trial, _i)
            # The dedup set accumulates every molecule this run has accepted so far
            # (seeded from the prompt's own exemplars), independently of the prompt
            # text, so scoring still excludes prior draws' molecules even though the
            # model is never shown them.
            prompt_exemplars = list(seen_per_class[class_name])
            if not rows:
                _log_lane_header(
                    tag,
                    "smiles",
                    prompt=generation_prompt,
                    end_token_ids=(
                        "vllm default end tokens + stop strings ['\\n']"
                        if args.eval_backend == "vllm"
                        else hf_end_token_ids
                    ),
                    max_new_tokens=args.eval_max_steps,
                )
            if args.eval_backend == "vllm":
                from vllm import SamplingParams as _SP

                gen_started = time.perf_counter()
                sp = _SP(
                    max_tokens=args.eval_max_steps,
                    temperature=_SMILES_TEMPERATURE,
                    stop=_UNCONSTRAINED_SMILES_STOP,
                    seed=draw_seed,
                )
                outputs = llm.generate([generation_prompt], sp)
                gen_seconds = time.perf_counter() - gen_started
                completion = outputs[0].outputs[0]
                gen_text = completion.text
                num_toks = len(completion.token_ids) if getattr(completion, "token_ids", None) else None
                stop_reason = _vllm_stop_reason(completion)
            else:
                torch.manual_seed(draw_seed)
                inputs = tokenizer(generation_prompt, return_tensors="pt").to(model.device)
                gen_started = time.perf_counter()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=args.eval_max_steps,
                        do_sample=True, temperature=_SMILES_TEMPERATURE,
                        stop_strings=_UNCONSTRAINED_SMILES_STOP, tokenizer=tokenizer,
                    )
                gen_seconds = time.perf_counter() - gen_started
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                num_toks = int(new_ids.numel())
                gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                stop_reason = _hf_stop_reason(
                    new_ids,
                    end_token_ids=hf_end_token_ids,
                    max_new_tokens=args.eval_max_steps,
                    other="unknown",
                )

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
            # raw = what the model wrote; scored = the molecule the scorer pulled out of it.
            LOGGER.info(
                "[%s] class=%s draw=%d dataset=smiles temperature=%s stop_reason=%s generated_tokens=%s "
                "syntax_valid=%s novel=%s raw=%r scored=%r",
                tag,
                class_name,
                _i,
                _SMILES_TEMPERATURE,
                stop_reason,
                num_toks,
                bool(smiles_eval.get("syntax_valid")),
                is_novel,
                _clip(gen_text),
                _clip(cleaned),
            )
            if cleaned and smiles_eval.get("syntax_valid"):
                seen_per_class[class_name].add(cleaned)

            row_out: dict[str, Any] = {
                "question": class_name,
                "llm_response": gen_text,
                "correct": molecule_is_correct(smiles_eval),
                "syntax_valid": bool(smiles_eval.get("syntax_valid", False)),
                "generation_seconds": gen_seconds,
            }
            if num_toks is not None:
                row_out["num_tokens"] = num_toks
            rows.append(row_out)

    _build_minimal_json(
        rows,
        args.output_json,
        eval_split=_eval_split_record(args),
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
    examples, _slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )

    device = _legacy_local_cuda_device(args.device)

    if args.eval_backend == "vllm":
        from vllm import LLM

        llm = LLM(
            model=args.eval_model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()
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
    tag = "legacy-unconstrained-spider"
    _log_unconstrained_no_constraint(tag, "spider", args.eval_backend)
    hf_end_token_ids = (
        None
        if args.eval_backend == "vllm"
        else _token_id_list(getattr(model.generation_config, "eos_token_id", None))
    )

    for example in examples:
        prompt = _spider_generation_prompt(
            _legacy_benchmark_prompt(logic, eval_runtime, example, "evaluator_default"),
            tokenizer,
            args.eval_model,
        )
        if not rows:
            _log_lane_header(
                tag,
                "spider",
                prompt=prompt,
                end_token_ids=(
                    "vllm default end tokens, no stop strings"
                    if args.eval_backend == "vllm"
                    else hf_end_token_ids
                ),
                max_new_tokens=max_new,
            )
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
            stop_reason = _vllm_stop_reason(completion)
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
            stop_reason = _hf_stop_reason(
                new_ids,
                end_token_ids=hf_end_token_ids,
                max_new_tokens=max_new,
                other="stopping_criteria(clean_boundary)",
            )

        scored_output = suffix
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        # raw = what the model wrote; scored = the query the scorer pulled out of it.
        LOGGER.info(
            "[%s] example=%d dataset=spider greedy=True stop_reason=%s generated_tokens=%s raw=%r scored=%r",
            tag,
            len(rows),
            stop_reason,
            num_toks,
            _clip(suffix),
            _clip(actual),
        )
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        syntax_valid = bool(aux and aux.get("syntax_valid"))  # the Spider output contract's verdict

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
        eval_split=_eval_split_record(args),
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
    """The saved answer. For SMILES this keeps the reasoning; only the scored output
    is narrowed to the last span (see ``_crane_smiles_span``)."""
    if dataset == "spider":
        return str(raw_output or "")
    return completion_for_scoring(prompt, raw_output)


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
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
    )
    _configure_fixed_eval_runtime(eval_runtime, args, dataset)
    examples, _slice_metrics = _slice_eval_examples(
        logic.load_dataset_sample(eval_runtime),
        args,
    )

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
    smiles_seen_per_class: dict[str, set[str]] = {}

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
            LOGGER.info(
                "[legacy-crane] settings dataset=%s opportunistic_masking=True top_p=%s "
                "max_new_tokens=%d stop_words=%s",
                dataset,
                generation_kwargs.get("top_p", "unset"),
                max(32, int(args.eval_max_steps)),
                _crane_stop_words(dataset),
            )
        sc = syncode_cache[cache_key]

        prompt = _legacy_benchmark_prompt(logic, eval_runtime, example, "evaluator_default")
        # The CRANE rewrite runs BEFORE the chat template so its instruction is part of
        # the user turn. Rendering first would leave the instruction after the assistant
        # turn has already opened, where the model reads it as its own output.
        prompt = _crane_prompt_for_generation(dataset, prompt)
        if dataset == "spider":
            prompt = _spider_generation_prompt(prompt, sc.model.tokenizer, args.eval_model)
        if dataset == "smiles":
            prompt = _render_chat_prompt(sc.model.tokenizer, prompt)
        if dataset == "smiles":
            seed_smiles_draw(example["draw_seed"])
        example_index = len(rows)
        if not rows:
            _log_lane_header(
                "legacy-crane",
                dataset,
                prompt=prompt,
                end_token_ids=_syncode_stop_token_ids(sc),
                max_new_tokens=max(32, int(args.eval_max_steps)),
            )
        audit_before = _syncode_audit_counts(sc)
        gen_started = time.perf_counter()
        completions = sc.infer(prompt, stop_words=_crane_stop_words(dataset))
        gen_seconds = time.perf_counter() - gen_started
        raw_output = completions[0] if completions else ""
        crane_trace = _syncode_generation_trace(sc)
        crane_audit = _audit_delta(audit_before, _syncode_audit_counts(sc))
        for switch in crane_trace.get("mode_switches", []):
            LOGGER.info(
                "[legacy-crane] switch example=%d dataset=%s to=%s token_index=%s",
                example_index,
                dataset,
                switch.get("to"),
                switch.get("token_index"),
            )
        # constrained_steps sits beside masked_steps on purpose. CRANE runs
        # AdaptiveSynCode's opportunistic masking, so on a constrained step the decoder
        # checks the model's own pick against the grammar and reaches for the mask only
        # when that pick is illegal. masked_steps=0 next to constrained_steps=13 therefore
        # says all thirteen tokens were already legal, not that nothing was constrained.
        # Read on its own, masked_steps=0 invites exactly the opposite reading.
        LOGGER.info(
            "[legacy-crane] constraint-audit example=%d dataset=%s constrained_steps=%s "
            "masked_steps=%s unmasked_steps=%s unmasked_by_reason=%s",
            example_index,
            dataset,
            crane_trace.get("constrained_steps", "unavailable"),
            crane_audit["masked"] if crane_audit else "unavailable",
            crane_audit["unmasked"] if crane_audit else "unavailable",
            crane_audit["by_reason"] if crane_audit else "unavailable",
        )
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
        if dataset == "smiles":
            # Score only the molecule in the last span; the saved answer keeps the reasoning.
            scored_output = _crane_smiles_span(completion)
            if not scored_output:
                LOGGER.warning(
                    "[legacy-crane] SMILES output had no complete constrained span "
                    "output_chars=%d",
                    len(completion),
                )
        expected = logic.expected_answer(eval_runtime, example)
        actual, _answer_source, aux = logic.extract_actual(eval_runtime, scored_output, example)
        is_correct = bool(logic.is_correct(eval_runtime, actual, expected, example, aux, scored_output))
        closed_spans = re.findall(r"<<(.*?)>>", str(raw_output), flags=re.DOTALL)
        # Every scorer takes the LAST closed span (see ``_crane_stop_words``).
        LOGGER.info(
            "[legacy-crane] example=%d dataset=%s stop_reason=%s generated_tokens=%s last_token_id=%s "
            "constrained_steps=%s free_steps=%s switches=%d spans_opened=%d spans_closed=%d "
            "picked_span=%s picked_text=%r scorer_answer=%r raw=%r",
            example_index,
            dataset,
            crane_trace.get("stop_reason", "unavailable"),
            crane_trace.get("generated_tokens", "unavailable"),
            crane_trace.get("last_token_id", "unavailable"),
            crane_trace.get("constrained_steps", "unavailable"),
            crane_trace.get("free_steps", "unavailable"),
            len(crane_trace.get("mode_switches", [])),
            str(raw_output).count("<<"),
            len(closed_spans),
            f"last_closed(index={len(closed_spans) - 1})" if closed_spans else "none",
            _clip(closed_spans[-1].strip()) if closed_spans else "",
            _clip(actual),
            _clip(raw_output),
        )

        syntax_valid, _segments = eval_runtime._check_syntax_validity(scored_output, example=example)
        if dataset == "spider":
            syntax_valid = bool(aux and aux.get("syntax_valid"))  # the Spider output contract's verdict
        if dataset == "smiles":
            syntax_valid = bool(aux and aux.get("syntax_valid"))
            is_correct = _unique_smiles_correct(
                str(example.get("class_name", "")), str(actual or ""),
                is_correct, syntax_valid, smiles_seen_per_class,
            )

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
        eval_split=_eval_split_record(args),
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
        eval_split=_eval_split_record(args),
        run_wall_time_seconds=time.perf_counter() - crane_run_started,
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0


def _routes_through_crane_repo(dataset: str, strategy: str) -> bool:
    """True when ``dataset``/``strategy`` should subprocess the upstream CRANE main.py.

    CRANE main.py only supports gsm_symbolic and FOL -- no Spider support (no
    prompt_templates/spider.yaml, no generate_spider_with_itergen function). For Spider,
    fall through to the legacy shared-evaluator adapters which DO support Spider and were
    the source of the 05-18 baselines.

    "gcd" is excluded even for gsm_symbolic: the CRANE repo has no separate GCD mode, so
    crane_repo_runner used to implement it as IterGen-with-reasoning-off, which is not
    what GCD (greedy grammar-constrained decoding) means anywhere else in this codebase.
    GSM's "gcd" instead falls through to run_gcd_legacy_adapter below, the same
    SynCode-backed hard-mask path Spider and SMILES GCD already use.
    """
    crane_repo_datasets = {"gsm_symbolic"}
    crane_repo_strategies = {"unconstrained", "crane", "itergen"}
    return dataset in crane_repo_datasets and strategy in crane_repo_strategies


def _eval_split_record(args: argparse.Namespace) -> dict[str, Any]:
    """The split-provenance dict every baseline JSON embeds (same shape as re-evals)."""
    from synthesis.split_provenance import build_split_provenance

    dataset = _normalize_dataset(args.dataset)
    return build_split_provenance(
        gsm_split_file=args.gsm_split_file if dataset == "gsm_symbolic" else None,
        gsm_split_name=args.gsm_split_name if dataset == "gsm_symbolic" and args.gsm_split_file else None,
        spider_split_file=args.spider_split_file if dataset == "spider" else None,
        spider_split_name=args.spider_split_name if dataset == "spider" and args.spider_split_file else None,
        smiles_split_name=args.smiles_split_name if dataset == "smiles" else None,
        smiles_trial=args.smiles_trial if dataset == "smiles" else None,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--smiles-split-name",
        choices=["train", "test"],
        default=None,
        help=(
            "Which SMILES seed range to draw from (required for --dataset smiles, no "
            "default). Train and test are disjoint per-draw seed ranges; see "
            "synthesis/evaluate/benchmarks/smiles/seed_split.py."
        ),
    )
    parser.add_argument(
        "--smiles-trial",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Which SMILES trial of that split (required for --dataset smiles, no default). "
            "Each trial is its own block of 10,000 seeds, so the 3 trials are independent draws."
        ),
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
    args = parser.parse_args(argv)
    if args.dataset == "smiles" and args.smiles_split_name is None:
        parser.error(
            "--smiles-split-name train|test is required for --dataset smiles "
            "(no default side; see synthesis/evaluate/benchmarks/smiles/seed_split.py)"
        )
    if args.dataset == "smiles" and args.smiles_trial is None:
        parser.error(
            "--smiles-trial 1|2|3 is required for --dataset smiles "
            "(no default; see synthesis/evaluate/benchmarks/smiles/seed_split.py)"
        )
    return args


def main() -> None:
    args = _parse_args()
    # CSD_TRACE_LOG=1 turns the baseline trace log on (see _configure_trace_logging).
    _configure_trace_logging()
    LOGGER.info(
        "[legacy-runner] lane strategy=%s dataset=%s model=%s backend=%s eval_max_steps=%s "
        "start_index=%s end_index=%s smiles_split=%s smiles_trial=%s",
        args.strategy,
        args.dataset,
        args.eval_model,
        args.eval_backend,
        args.eval_max_steps,
        args.eval_start_index,
        args.eval_end_index,
        args.smiles_split_name,
        args.smiles_trial,
    )
    from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

    args.vllm_tensor_parallel_size = resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size)

    # Use process spawning instead of forking when vLLM starts worker processes.
    # Forking after CUDA is initialized in the parent crashes with
    # "Cannot re-initialize CUDA in forked subprocess".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _ensure_repo_cache_env()

    dataset_normalized = _normalize_dataset(args.dataset)
    if _routes_through_crane_repo(dataset_normalized, args.strategy):
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
