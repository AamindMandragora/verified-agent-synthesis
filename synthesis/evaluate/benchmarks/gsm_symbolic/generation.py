"""
Generation methods for evaluation.

Provides CSD (Constrained Decoding Strategy) generation by delegating
entirely to the Dafny-verified CSD strategy.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union


from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str


from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
    SpiderEvidenceContractError,
)
_SPIDER_CONTRACT_LOG = logging.getLogger("csd.spider_output_contract")


def _strategy_sequence_texts(strategy_token_sequence):
    """Return the exact strategy token texts without re-tokenizing output."""
    if strategy_token_sequence is None:
        return None, None
    if isinstance(strategy_token_sequence, str):
        return [strategy_token_sequence], strategy_token_sequence
    try:
        items = list(strategy_token_sequence)
    except TypeError:
        items = [
            strategy_token_sequence[index]
            for index in range(len(strategy_token_sequence))
        ]
    token_texts = [
        item if isinstance(item, str) else dafny_seq_to_str(item)
        for item in items
    ]
    return token_texts, "".join(token_texts)


def _finalize_spider_generation_evidence(
    lm,
    spider_prompt_active: bool,
    scored_output: str | None = None,
    strategy_token_sequence=None,
) -> None:
    if not spider_prompt_active:
        return
    strategy_token_texts, strategy_output_text = _strategy_sequence_texts(
        strategy_token_sequence
    )
    explicit_strategy_output = strategy_token_sequence is not None
    if explicit_strategy_output and scored_output is not None:
        if strategy_output_text != str(scored_output):
            _SPIDER_CONTRACT_LOG.error(
                "[spider-output-contract] strategy_output_mismatch "
                "strategy_chars=%d scored_chars=%d",
                len(strategy_output_text),
                len(str(scored_output)),
            )
            raise SpiderEvidenceContractError(
                "Spider strategy output does not match scored output"
            )
    if not explicit_strategy_output and scored_output is not None:
        reconcile = getattr(lm, "_reconcile_generation_evidence", None)
        if callable(reconcile):
            reconciled = reconcile(str(scored_output))
            if reconciled is False:
                _SPIDER_CONTRACT_LOG.error(
                    "[spider-output-contract] evidence_reconcile_failed "
                    "reason=sampled_ids_do_not_match scored_chars=%d",
                    len(str(scored_output)),
                )
                raise SpiderEvidenceContractError(
                    "Spider committed token evidence does not match scored output"
                )
    def _attach_strategy_evidence(evidence):
        if not explicit_strategy_output or evidence is None:
            return
        raw_decoded_text = str(evidence.get("decoded_text", ""))
        removed_sampled_ids = [
            int(token_id)
            for token_id in getattr(
                lm, "_generation_alignment_removed_token_ids", []
            )
        ]
        strategy_mutation = bool(raw_decoded_text != strategy_output_text or removed_sampled_ids)
        origin_proven = bool(
            getattr(lm, "_strategy_output_origin_proven", False)
        )
        if removed_sampled_ids or raw_decoded_text != strategy_output_text or not origin_proven:
            relation = "mixed" if raw_decoded_text else "strategy_authored"
        else:
            relation = "sampled_output"
        evidence.update(
            {
                "strategy_output_text": strategy_output_text,
                "strategy_token_texts": list(strategy_token_texts),
                "strategy_output_relation": relation,
                "strategy_mutation": strategy_mutation,
                "strategy_removed_sampled_token_ids": removed_sampled_ids,
            }
        )
        _SPIDER_CONTRACT_LOG.info(
            "[spider-output-contract] strategy_output relation=%s "
            "strategy_chars=%d sampled_chars=%d removed_count=%d",
            evidence["strategy_output_relation"],
            len(strategy_output_text),
            len(raw_decoded_text),
            len(removed_sampled_ids),
        )
    finalizer = getattr(lm, "_finalize_generation_evidence", None)
    if callable(finalizer) and finalizer() is not None:
        evidence = getattr(lm, "_last_generation_evidence", None)
        if not explicit_strategy_output and scored_output is not None and evidence is not None:
            decoded_text = str(evidence.get("decoded_text", ""))
            if decoded_text != str(scored_output):
                _SPIDER_CONTRACT_LOG.error(
                    "[spider-output-contract] evidence_mismatch committed_chars=%d scored_chars=%d",
                    len(decoded_text),
                    len(str(scored_output)),
                )
                raise SpiderEvidenceContractError(
                    "Spider committed token evidence does not match scored output"
                )
        _attach_strategy_evidence(evidence)
        return
    token_ids = getattr(lm, "_generation_token_ids", None)
    tokenizer = getattr(lm, "tokenizer", None)
    if token_ids is None or tokenizer is None:
        return
    stop_ids = getattr(lm, "_generation_stop_token_ids", None)
    if stop_ids is None:
        stop_ids = getattr(lm, "generation_stop_token_ids", None)
    if stop_ids is None:
        stop_ids = getattr(tokenizer, "generation_stop_token_ids", None)
    if stop_ids is None:
        stop_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(stop_ids, int):
        stop_ids = {stop_ids}
    from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
        generation_token_evidence,
    )

    lm._last_generation_evidence = generation_token_evidence(
        token_ids,
        tokenizer,
        terminal_stop_token_ids=stop_ids or (),
    )
    if not explicit_strategy_output and scored_output is not None:
        decoded_text = str(lm._last_generation_evidence.get("decoded_text", ""))
        if decoded_text != str(scored_output):
            _SPIDER_CONTRACT_LOG.error(
                "[spider-output-contract] evidence_mismatch committed_chars=%d scored_chars=%d",
                len(decoded_text),
                len(str(scored_output)),
            )
            raise SpiderEvidenceContractError(
                "Spider committed token evidence does not match scored output"
            )
    _attach_strategy_evidence(lm._last_generation_evidence)




def _call_my_csd_strategy(
    GeneratedCSD,
    _dafny,
    lm,
    parser,
    generated_prefix,
    start_inside_constrained,
    current_constrained,
    max_steps,
    step_token_budget,
    valid_token_groups_dafny,
    valid_tokens_dafny,
    eos_token_dafny,
    param_names,
    n_params,
):
    if "validTokenGroups" in param_names:
        return GeneratedCSD.default__.MyCSDStrategy(
            lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
            start_inside_constrained, current_constrained,
            max_steps, step_token_budget, valid_token_groups_dafny, eos_token_dafny,
        )
    if "validTokens" in param_names or n_params >= 10:
        return GeneratedCSD.default__.MyCSDStrategy(
            lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
            start_inside_constrained, current_constrained,
            max_steps, step_token_budget, valid_tokens_dafny, eos_token_dafny,
        )
    if n_params >= 9:
        return GeneratedCSD.default__.MyCSDStrategy(
            lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
            start_inside_constrained, current_constrained,
            max_steps, step_token_budget, eos_token_dafny,
        )
    return GeneratedCSD.default__.MyCSDStrategy(
        lm, parser, _dafny.SeqWithoutIsStrInference([]), generated_prefix,
        start_inside_constrained, current_constrained,
        max_steps, eos_token_dafny,
    )


def _enforce_max_steps(result_tokens: List[str], max_steps: int) -> None:
    if len(result_tokens) > max_steps:
        raise RuntimeError(
            f"CSD exceeded max_steps: generated {len(result_tokens)} tokens > {max_steps}"
        )


def run_crane_csd(
    env: dict,
    prompt_text: Union[str, List[dict]],
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
    dynamic_parser=None,
    start_inside_constrained: bool = False,
    step_token_budget: int = 1,
    valid_tokens: Optional[List[str]] = None,
    valid_token_groups: Optional[List[List[str]]] = None,
    max_seconds: Optional[float] = None,
    completion_mode: bool = False,
    early_stop_on_answer: bool = False,
) -> Tuple[str, int, float, List[Tuple[str, bool]], List[dict]]:
    """
    Run generation using the Dafny-verified CSD strategy.

    Delegates entirely to the compiled Dafny strategy — no dataset-specific
    orchestration is performed here.

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text (set as lm.instruction_text)
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file for post-hoc segment validation
        debug_delimiters: Whether to print debug output
        dynamic_parser: Optional per-question parser
        start_inside_constrained: Begin with an internal constrained chunk
            already active. This is useful for tasks like Spider where the
            answer is parser-governed from the first token but chunk boundaries
            should not be serialized as visible delimiters.
        completion_mode: When True, set lm.instruction_text to the raw
            prompt_text string with no chat template applied. Required for base
            (non-instruction-tuned) completion models, which must see the prompt
            as a raw continuation rather than ChatML-wrapped.
        early_stop_on_answer: When True, stop generation as soon as the output
            contains a finished final-answer span ('final answer' followed by a
            complete <<...>> span), mirroring CRANE's answer stopping. The
            output-so-far is returned and scored through the normal path.

    Returns:
        Tuple of (output_text, token_count, time_seconds, constrained_segments, helper_trace, constrained_work)
    """
    _dafny = env["_dafny"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]
    spider_prompt_active = hasattr(prompt_text, "render_for_model")
    if hasattr(lm, "_last_generation_evidence"):
        lm._last_generation_evidence = None
    lm._last_prompt_contract = None
    if hasattr(lm, "_generation_token_ids"):
        lm._generation_token_ids = []

    if hasattr(prompt_text, "render_for_model"):
        if hasattr(lm, "ResetTaskGuidance"):
            lm.ResetTaskGuidance()
        model_name = env.get("model_name") or getattr(lm, "model_name", None)
        if not model_name:
            model_config = getattr(getattr(lm, "model", None), "config", None)
            model_name = getattr(model_config, "model_type", None)
        if hasattr(lm, "set_structured_prompt"):
            lm.set_structured_prompt(prompt_text, model_name=model_name)
        render_with_contract = getattr(
            prompt_text, "render_for_model_with_contract", None
        )
        if callable(render_with_contract):
            rendered_prompt, prompt_contract = render_with_contract(
                lm.tokenizer, model_name=model_name
            )
        else:
            rendered_prompt = prompt_text.render_for_model(
                lm.tokenizer, model_name=model_name
            )
            prompt_contract = {
                "renderer": "structured",
                "family": "unknown",
                "mode": "structured",
                "template_used": None,
                "raw_prompt": None,
                "chat_message_count": None,
                "user_message_count": None,
                "add_generation_prompt": None,
                "enable_thinking": None,
                "render_succeeded": True,
                "prompt_chars": len(rendered_prompt),
            }
        lm.instruction_text = rendered_prompt
        lm._last_prompt_contract = prompt_contract
    elif completion_mode:
        if not isinstance(prompt_text, str):
            raise ValueError("completion_mode requires prompt_text to be a string")
        if hasattr(lm, "ResetTaskGuidance"):
            lm.ResetTaskGuidance()
        lm.instruction_text = prompt_text
        lm._last_prompt_contract = {
            "renderer": "legacy",
            "family": "unknown",
            "mode": "raw_completion",
            "template_used": False,
            "raw_prompt": True,
            "chat_message_count": 0,
            "user_message_count": 0,
            "add_generation_prompt": False,
            "enable_thinking": None,
            "render_succeeded": True,
            "prompt_chars": len(prompt_text),
        }
    else:
        if hasattr(lm, "ResetTaskGuidance"):
            lm.ResetTaskGuidance()
        chat_messages = prompt_text if isinstance(prompt_text, list) else [{"role": "user", "content": prompt_text}]
        template_fallback = False
        try:
            lm.instruction_text = lm.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            template_fallback = True
            lm.instruction_text = lm.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        lm._last_prompt_contract = {
            "renderer": "legacy",
            "family": "unknown",
            "mode": "chat",
            "template_used": True,
            "raw_prompt": False,
            "chat_message_count": len(chat_messages),
            "user_message_count": sum(
                1 for message in chat_messages if message.get("role") == "user"
            ),
            "add_generation_prompt": True,
            "enable_thinking": None if template_fallback else False,
            "template_fallback": template_fallback,
            "render_succeeded": True,
            "prompt_chars": len(lm.instruction_text),
        }
        if hasattr(lm, "set_chat_messages"):
            lm.set_chat_messages(chat_messages)
    start_time = time.time()
    runtime_deadline = None
    if max_seconds is not None:
        runtime_deadline = time.monotonic() + max_seconds
    if hasattr(lm, "SetRuntimeDeadline"):
        lm.SetRuntimeDeadline(runtime_deadline)
    if early_stop_on_answer and hasattr(lm, "SetAnswerEarlyStop"):
        lm.SetAnswerEarlyStop(True)

    # IterGen length parity: HF MaxLengthCriteria is applied to GENERATED tokens
    # only (session[:, start_from:]), while max_length = prompt_len + max_new_tokens.
    # Effective generated ceiling is therefore prompt_len + max_new_tokens
    # (spider adapter: max_new_tokens = min(512, eval_max_steps)).
    _mnt = os.environ.get("CSD_ITERGEN_MAX_NEW_TOKENS", "").strip()
    if _mnt:
        _prompt_len = len(
            lm.tokenizer.encode(getattr(lm, "instruction_text", "") or "", add_special_tokens=False)
        )
        _old = int(max_steps)
        max_steps = int(_mnt) + int(_prompt_len)
        print(
            f"  [ITERGEN-LEN] max_steps {_old} -> {max_steps} "
            f"(max_new_tokens={_mnt} + prompt_len={_prompt_len})",
            flush=True,
        )

    eos_token_str = lm.tokenizer.eos_token or "<|endoftext|>"
    eos_token_dafny = _dafny.Seq(eos_token_str)
    generated_prefix = _dafny.SeqWithoutIsStrInference([])
    current_constrained = _dafny.SeqWithoutIsStrInference([])

    trace_state = env.get("csd_trace")
    if isinstance(trace_state, dict):
        trace_state["events"] = []
        trace_state.pop("_pending_spider_rollback_prefix", None)
        trace_state.pop("_spider_helper_wrapper_depth", None)

    if valid_token_groups is not None:
        token_groups = valid_token_groups
        flat_tokens = [t for group in token_groups for t in group]
    else:
        flat_tokens = valid_tokens or []
        token_groups = [flat_tokens] if flat_tokens else []

    valid_tokens_dafny = _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(t) for t in flat_tokens]
    )
    valid_token_groups_dafny = _dafny.SeqWithoutIsStrInference(
        [
            _dafny.SeqWithoutIsStrInference([_dafny.Seq(t) for t in group])
            for group in token_groups
        ]
    )

    import inspect
    _sig = inspect.signature(GeneratedCSD.default__.MyCSDStrategy)
    _param_names = list(_sig.parameters.keys())
    _n_params = len(_param_names)
    # CARS get_sample: up to N full attempts, keep first stop_after successes.
    # CSD_CARS_SEARCH_STEPS mirrors --cars-search-steps (default 1 = single pass).
    cars_steps = max(1, int(os.environ.get(
        "CSD_CARS_SEARCH_STEPS",
        os.environ.get("CSD_CARS_SAMPLES_PER_PROMPT", "1"),
    )))
    cars_stop_after = max(1, int(os.environ.get("CSD_CARS_STOP_AFTER", "1")))
    from synthesis.evaluate.benchmarks.common.model_utils import AnswerCompleteStop

    answer_early_stopped = False
    strategy_token_sequence = None
    try:
        result = None
        _successes = 0
        for _attempt in range(cars_steps):
            if spider_prompt_active and hasattr(lm, "_generation_token_ids"):
                lm._generation_token_ids = []
                if hasattr(lm, "_reset_generation_transactions"):
                    lm._reset_generation_transactions()
            if os.environ.get("CSD_PARITY_SEED_PER_ATTEMPT", "0") == "1":
                _raw = os.environ.get("CSD_PARITY_SEED", "").strip()
                if _raw:
                    import random as _random
                    _base = int(_raw)
                    _ex = int(os.environ.get("CSD_PARITY_EXAMPLE_INDEX", "0"))
                    _seed = _base + _ex * 1_000_003 + int(_attempt) * 9176
                    _random.seed(_seed)
                    try:
                        import numpy as _np
                        _np.random.seed(_seed % (2**32 - 1))
                    except Exception:
                        pass
                    try:
                        import torch as _torch
                        _torch.manual_seed(_seed)
                        if _torch.cuda.is_available():
                            _torch.cuda.manual_seed_all(_seed)
                    except Exception:
                        pass
            result = _call_my_csd_strategy(
                GeneratedCSD,
                _dafny,
                lm,
                parser,
                generated_prefix,
                start_inside_constrained,
                current_constrained,
                max_steps,
                step_token_budget,
                valid_token_groups_dafny,
                valid_tokens_dafny,
                eos_token_dafny,
                _param_names,
                _n_params,
            )
            # Success ≈ CARS generation_ended accepting: non-empty complete prefix.
            _out = result[0] if isinstance(result, tuple) else result
            _toks = [dafny_seq_to_str(t) for t in _out] if _out is not None else []
            _eos = dafny_seq_to_str(eos_token_dafny) if eos_token_dafny is not None else ""
            _body = "".join(t for t in _toks if t != _eos)
            _ok = bool(_body) and bool(getattr(parser, "is_complete", lambda s: False)(_body))
            if _ok:
                _successes += 1
                if _successes >= cars_stop_after:
                    break
            # else: failed attempt; oracle trie already updated via RejectLastInTrie
    except AnswerCompleteStop:
        # Generation-complete signal, not a failure: the final answer span is
        # finished. The per-step hook stashed the tokens generated so far;
        # return them through the normal scoring path below.
        answer_early_stopped = True
    finally:
        if hasattr(lm, "ClearRuntimeDeadline"):
            lm.ClearRuntimeDeadline()

    final_inside_constrained = False
    final_current_constrained = _dafny.SeqWithoutIsStrInference([])

    if answer_early_stopped:
        result_tokens = list(getattr(lm, "_early_stop_tokens", None) or [])
        total_cost = len(result_tokens)
        strategy_token_sequence = result_tokens
    elif isinstance(result, tuple) and len(result) == 4:
        csd_output, final_inside_constrained, final_current_constrained, total_cost = result
        result_tokens = [dafny_seq_to_str(t) for t in csd_output]
        strategy_token_sequence = csd_output
    elif isinstance(result, tuple):
        csd_output, total_cost = result
        result_tokens = [dafny_seq_to_str(t) for t in csd_output]
        strategy_token_sequence = csd_output
    else:
        csd_output = result
        total_cost = 0
        result_tokens = [dafny_seq_to_str(t) for t in csd_output]
        strategy_token_sequence = csd_output

    if early_stop_on_answer and hasattr(lm, "SetAnswerEarlyStop"):
        # Clear AFTER harvesting the stash so the flag cannot leak into the
        # next example's generation.
        lm.SetAnswerEarlyStop(False)
    _enforce_max_steps(result_tokens, max_steps)
    output_text = "".join(result_tokens)
    if isinstance(trace_state, dict):
        pending_prefix = trace_state.pop("_pending_spider_rollback_prefix", None)
        if pending_prefix is not None:
            align_prefix = getattr(lm, "_align_generation_history_to_prefix", None)
            if callable(align_prefix):
                align_prefix(pending_prefix)
                _SPIDER_CONTRACT_LOG.info(
                    "[spider-output-contract] final_rollback_alignment applied=1"
                )
        trace_state.pop("_spider_helper_wrapper_depth", None)
    _finalize_spider_generation_evidence(
        lm,
        spider_prompt_active,
        scored_output=output_text,
        strategy_token_sequence=strategy_token_sequence,
    )
    execution_time = time.time() - start_time

    constrained_segments: List[Tuple[str, bool]] = []
    helper_trace = list(trace_state.get("events", [])) if isinstance(trace_state, dict) else []
    task_guidance = getattr(lm, "task_guidance", None)
    if task_guidance:
        helper_trace.append({"helper": "AppendTaskGuidance", "detail": task_guidance})

    final_chunk_tokens = [
        dafny_seq_to_str(final_current_constrained[i])
        for i in range(len(final_current_constrained))
    ]
    final_chunk = "".join(final_chunk_tokens)
    hidden_chunk_used = start_inside_constrained and (
        bool(final_chunk)
        or any(
            event.get("helper") in {"AppendConstrainedToken", "ConstrainedStep", "CloseConstrainedSpan"}
            for event in helper_trace
        )
    )
    if hidden_chunk_used:
        # Hidden constrained chunks are real parser-governed chunks even though
        # no << / >> boundary tokens are rendered into user-visible output.
        constrained_segments.append((final_chunk or output_text, True))

    if debug_delimiters:
        print(f"  [DEBUG] Generation finished in {execution_time:.2f}s. Cost: {total_cost}. Tokens: {len(result_tokens)}")

    # Per-example timing dump: what fraction of wall time went to Python callbacks
    # (LM + parser) vs Dafny-internal work. If callbacks sum << execution_time,
    # the bottleneck is inside the generated Dafny strategy (not our Python code).
    try:
        from synthesis.evaluate.benchmarks.common.model_utils import _TIMINGS, _print_timings_breakdown
        from synthesis.evaluate.benchmarks.common.parser_utils import _PARSER_TIMINGS, print_parser_timings

        lm_total = sum(t for t, _ in _TIMINGS.values())
        parser_total = sum(t for t, _ in _PARSER_TIMINGS.values())
        callbacks_total = lm_total + parser_total
        dafny_internal = max(execution_time - callbacks_total, 0.0)
        print(
            f"[STEP_BREAKDOWN] wall={execution_time:.2f}s  lm_callbacks={lm_total:.2f}s  "
            f"parser_callbacks={parser_total:.2f}s  dafny_internal={dafny_internal:.2f}s  "
            f"({100*dafny_internal/max(execution_time,1e-6):.1f}% in dafny/uninstrumented)",
            flush=True,
        )
        _print_timings_breakdown(header="end_of_example (cumulative)")
        print_parser_timings(header="end_of_example (cumulative)")
    except Exception as _dbg_err:
        print(f"[STEP_BREAKDOWN] error printing timings: {_dbg_err}", flush=True)

    return output_text, len(result_tokens), execution_time, constrained_segments, helper_trace, total_cost
