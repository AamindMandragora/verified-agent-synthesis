"""
Environment setup utilities for GSM-Symbolic evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for constrained generation.
"""

from __future__ import annotations

import sys
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List


def _dafny_token_to_str(value: Any) -> str:
    """Best-effort conversion of a Dafny token/seq to Python text."""
    if isinstance(value, str):
        return value
    try:
        return "".join(value)
    except TypeError:
        try:
            return "".join(value[i] for i in range(len(value)))
        except (TypeError, AttributeError, IndexError):
            return str(value)


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _truncate(text: str, limit: int = 80) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summarize_helper_event(name: str, args: tuple[Any, ...], result: Any, cost_before: Any, cost_after: Any) -> Dict[str, Any]:
    # Helper for token redaction: keep universal delimiters/EOS verbatim
    # (they're not synthesis-sample-specific and the strategy code already
    # references them), redact everything else (schema/identifier content).
    UNIVERSAL_TOKENS = {"<<", ">>", "EOS", "<EOS>", "<eos>"}
    def _safe_token(raw):
        try:
            t = _truncate(_dafny_token_to_str(raw))
        except Exception:
            return "<redacted>"
        return t if t in UNIVERSAL_TOKENS else "<redacted>"

    event: Dict[str, Any] = {
        "helper": name,
        "cost_before": cost_before,
        "cost_after": cost_after,
    }

    if name in {
        "UnconstrainedStep",
        "ConstrainedStep",
        "PenalizedConstrainedStep",
        "BoostedConstrainedStep",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "GroupBoostedConstrainedStep",
        "AdaptiveConstrainedStep",
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
    }:
        # Keep delimiter tokens verbatim; redact other tokens (synthesis-sample-leaky).
        token = _safe_token(result)
        event["token"] = token
        event["detail"] = f"token={token}"
        return event

    if name == "UnconstrainedChunk":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and len(result) >= 1 else None
        stopped_on_open = bool(result[1]) if isinstance(result, tuple) and len(result) >= 2 else False
        stopped_on_eos = bool(result[2]) if isinstance(result, tuple) and len(result) >= 3 else False
        steps_used = int(result[3]) if isinstance(result, tuple) and len(result) >= 4 else None
        event["generated_len"] = generated_len
        event["stopped_on_open"] = stopped_on_open
        event["stopped_on_eos"] = stopped_on_eos
        event["steps_used"] = steps_used
        event["detail"] = (
            f"chunk steps={steps_used}, hit_open={stopped_on_open}, hit_eos={stopped_on_eos}, generated_len={generated_len}"
        )
        return event

    if name == "UnconstrainedGeneration":
        event["generated_len"] = _safe_len(result)
        event["detail"] = f"unconstrained_generation, generated_len={event['generated_len']}"
        return event

    if name == "ConstrainedGeneration":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and result else None
        terminated_by_eos = bool(result[1]) if isinstance(result, tuple) and len(result) >= 2 else False
        event["generated_len"] = generated_len
        event["terminated_by_eos"] = terminated_by_eos
        event["detail"] = (
            f"constrained_generation, generated_len={generated_len}, terminated_by_eos={terminated_by_eos}"
        )
        return event

    if name == "CraneGeneration":
        event["generated_len"] = _safe_len(result)
        event["detail"] = f"crane_generation, generated_len={event['generated_len']}"
        return event

    if name in {"SoftConstrainedStep", "SafeSoftConstrainedStep", "ConfidenceGatedStep"}:
        flag = bool(result[1]) if isinstance(result, tuple) and len(result) == 2 else False
        token = _safe_token(result[0]) if isinstance(result, tuple) and len(result) == 2 else _safe_token(result)
        event["token"] = token
        event["fallback_or_constrained_path"] = flag
        event["detail"] = f"token={token}, fallback_or_constrained_path={flag}"
        return event

    if name in {"OpenConstrainedSpan", "EnterObservedConstrainedSpan"}:
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and len(result) >= 1 else None
        event["generated_len"] = generated_len
        action = "entered observed span" if name == "EnterObservedConstrainedSpan" else "opened span"
        event["detail"] = f"{action}, generated_len={generated_len}"
        return event

    if name == "AppendConstrainedToken":
        current_len = _safe_len(result[2]) if isinstance(result, tuple) and len(result) >= 3 else None
        token = _safe_token(args[-1]) if args else "<redacted>"
        event["token"] = token
        event["current_len"] = current_len
        event["detail"] = f"append {token}, current_len={current_len}"
        return event

    if name == "CloseConstrainedSpan":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and len(result) >= 1 else None
        event["generated_len"] = generated_len
        event["detail"] = f"closed span, generated_len={generated_len}"
        return event

    if name == "CloseSpanIfComplete":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and result else None
        current_len = _safe_len(result[2]) if isinstance(result, tuple) and len(result) >= 3 else None
        closed = bool(result[3]) if isinstance(result, tuple) and len(result) >= 4 else None
        event["generated_len"] = generated_len
        event["current_len"] = current_len
        event["closed"] = closed
        event["detail"] = f"close_if_complete, generated_len={generated_len}, current_len={current_len}, closed={closed}"
        return event

    if name == "CarsTrieStep":
        next_len = _safe_len(result[0]) if isinstance(result, tuple) and result else _safe_len(result)
        success = bool(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        event["next_len"] = next_len
        event["success"] = success
        event["detail"] = f"cars_step, next_len={next_len}, success={success}"
        return event

    if name == "ForwardUntilSymbol":
        result_len = _safe_len(result)
        event["result_len"] = result_len
        event["detail"] = f"forward_until_symbol, result_len={result_len}"
        return event

    if name == "BackwardToSymbol":
        generated_len = _safe_len(args[1]) if len(args) > 1 else None
        result_len = _safe_len(result)
        event["generated_len_before"] = generated_len
        event["generated_len_after"] = result_len
        event["detail"] = f"rollback_to_symbol, generated_len={generated_len}->{result_len}"
        return event

    if name in {
        "RollbackConstrainedSpan",
        "RollbackConstrainedSuffix",
        "RollbackConstrainedToComplete",
        "RollbackToCompletePrefix",
        "RollbackToValidPrefix",
        "RollbackAndRegenerate",
        "RollbackAndContinue",
    }:
        generated_arg_index, current_arg_index = {
            "BackwardToSymbol": (1, None),
            "RollbackConstrainedSpan": (2, 3),
            "RollbackConstrainedSuffix": (1, 2),
            "RollbackConstrainedToComplete": (1, 2),
            "RollbackToCompletePrefix": (1, None),
            "RollbackToValidPrefix": (1, None),
            "RollbackAndRegenerate": (3, None),
            "RollbackAndContinue": (3, 4),
        }[name]
        generated_before = (
            _safe_len(args[generated_arg_index])
            if len(args) > generated_arg_index
            else None
        )
        generated_after = _safe_len(result[0]) if isinstance(result, tuple) and result else _safe_len(result)
        current_before = (
            _safe_len(args[current_arg_index])
            if current_arg_index is not None and len(args) > current_arg_index
            else None
        )
        current_after = _safe_len(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        event["generated_len_before"] = generated_before
        event["generated_len_after"] = generated_after
        if current_arg_index is not None:
            event["current_len_before"] = current_before
            event["current_len_after"] = current_after
        if len(args) <= generated_arg_index:
            event["generated_len"] = generated_after
            if isinstance(result, tuple) and len(result) >= 2:
                event["current_len"] = current_after
        event["detail"] = (
            f"rollback, generated_len={generated_before}->{generated_after}, "
            + (
                f"current_len={current_before}->{current_after}"
                if current_arg_index is not None
                else ""
            )
        )
        return event

    if name == "DeadEndAvoidingStep":
        next_len = _safe_len(result[0]) if isinstance(result, tuple) and result else _safe_len(result)
        success = bool(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        event["next_len"] = next_len
        event["success"] = success
        event["detail"] = f"dead_end_retry, next_len={next_len}, success={success}"
        return event

    if name in {"RegenerateUnitOnCheckFailure", "RegenerateUnitOnGroundingFailure"}:
        result_len = _safe_len(result)
        # args: lm, parser, prompt, currentConstrained, eosToken, maxStepsPerUnit,
        #       maxRetries, maxRollbackBudget, allowedUnits
        max_steps = args[5] if len(args) > 5 else None
        max_retries = args[6] if len(args) > 6 else None
        allowed_count = _safe_len(args[8]) if len(args) > 8 else None
        event["result_len"] = result_len
        event["max_steps_per_unit"] = str(max_steps)
        event["max_retries"] = str(max_retries)
        event["allowed_units_count"] = allowed_count
        event["detail"] = (
            f"unit_rewind result_len={result_len}, max_steps={max_steps}, "
            f"max_retries={max_retries}, allowed_units={allowed_count}"
        )
        return event

    if name == "SaveLogitsSnapshot":
        snapshot_size = _safe_len(result)
        event["snapshot_size"] = snapshot_size
        event["detail"] = f"saved logits snapshot, size={snapshot_size}"
        return event

    if name == "RestoreLogitsSnapshot":
        snapshot_size = _safe_len(args[1]) if len(args) > 1 else None
        event["snapshot_size"] = snapshot_size
        event["detail"] = f"restored logits snapshot, size={snapshot_size}"
        return event

    if name == "SpeculativeConstrainedRollout":
        candidate_count = _safe_len(result[0]) if isinstance(result, tuple) and result else None
        candidate_prefix_len = _safe_len(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        hit_complete = bool(result[2]) if isinstance(result, tuple) and len(result) >= 3 else False
        hit_eos = bool(result[3]) if isinstance(result, tuple) and len(result) >= 4 else False
        steps_used = int(result[4]) if isinstance(result, tuple) and len(result) >= 5 else None
        event["candidate_token_count"] = candidate_count
        event["candidate_prefix_len"] = candidate_prefix_len
        event["hit_complete"] = hit_complete
        event["hit_eos"] = hit_eos
        event["steps_used"] = steps_used
        event["detail"] = (
            f"speculative_rollout, candidates={candidate_count}, prefix_len={candidate_prefix_len}, "
            f"hit_complete={hit_complete}, hit_eos={hit_eos}, steps_used={steps_used}"
        )
        return event

    if name == "RolloutConstrainedWithPenalties":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and result else None
        steps_used = int(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        terminated_by_eos = bool(result[2]) if isinstance(result, tuple) and len(result) >= 3 else False
        event["generated_len"] = generated_len
        event["steps_used"] = steps_used
        event["terminated_by_eos"] = terminated_by_eos
        event["detail"] = (
            f"penalized_rollout, generated_len={generated_len}, steps_used={steps_used}, "
            f"terminated_by_eos={terminated_by_eos}"
        )
        return event

    if name in {"ConstrainedSymbol", "ConstrainedSymbolInGenerated"}:
        hit_eos = False
        steps_used = None
        current_len = None
        generated_len = None
        if isinstance(result, tuple):
            if name == "ConstrainedSymbol" and len(result) >= 3:
                current_len = _safe_len(result[0])
                hit_eos = bool(result[1])
                steps_used = int(result[2])
            elif name == "ConstrainedSymbolInGenerated" and len(result) >= 4:
                generated_len = _safe_len(result[0])
                current_len = _safe_len(result[1])
                hit_eos = bool(result[2])
                steps_used = int(result[3])
        event["current_len"] = current_len
        event["generated_len"] = generated_len
        event["hit_eos"] = hit_eos
        event["steps_used"] = steps_used
        event["detail"] = f"symbol steps={steps_used}, hit_eos={hit_eos}, current_len={current_len}, generated_len={generated_len}"
        return event

    if name == "DeadEndDetection":
        event["detail"] = f"narrow={bool(result)}"
        return event

    if name == "ValidTokenCount":
        event["count"] = int(result) if isinstance(result, int) else str(result)
        event["detail"] = f"count={event['count']}"
        return event

    if name in {"BoostTokenLogits", "PenalizeTokenLogits", "SafeBoostTokenLogits", "SafePenalizeTokenLogits"}:
        # Strategy chose these tokens; conservatively keep universals,
        # redact others to prevent strategy from passing through sampled
        # tokens.
        tokens = [_safe_token(t) for t in args[1]] if len(args) >= 2 else []
        amount = args[2] if len(args) >= 3 else None
        event["tokens"] = tokens
        event["amount"] = str(amount)
        event["detail"] = f"tokens={tokens[:4]}, amount={amount}"
        return event

    if name == "TopValidCandidates":
        k = args[-2] if len(args) >= 2 else None
        candidate_count = _safe_len(result)
        # Preview redacted: candidate tokens depend on parser state which
        # depends on synthesis-sample content.
        event["k_requested"] = str(k)
        event["candidate_count"] = candidate_count
        event["detail"] = f"k={k}, got={candidate_count}"
        return event

    if name == "IsTokenValidNext":
        # Strategy chose this token; gpt-5.4 already sees it in the strategy
        # code. Still apply _safe_token to be conservative (in case the
        # strategy passes a sampled token here, which would leak).
        token = _safe_token(args[-1]) if args else "<redacted>"
        event["token"] = token
        event["is_valid"] = bool(result)
        event["detail"] = f"token={token}, valid={bool(result)}"
        return event

    if name == "GenerateLogits":
        prefix_len = _safe_len(args[0]) if args else None
        event["prefix_len"] = prefix_len
        event["detail"] = f"forward_pass, prefix_len={prefix_len}"
        return event

    if name == "ChooseNextTokenUnconstrained":
        # Keep delimiter tokens verbatim; redact schema/identifier tokens.
        token = _safe_token(result)
        event["token"] = token
        event["detail"] = f"sampled token={token}"
        return event

    if name == "MaskValidNextAndEos":
        prefix_len = _safe_len(args[1]) if len(args) >= 2 else None
        event["prefix_len"] = prefix_len
        event["detail"] = f"mask_to_valid, prefix_len={prefix_len}"
        return event

    if name == "BoostValidNextAndEos":
        prefix_len = _safe_len(args[1]) if len(args) >= 2 else None
        amount = args[2] if len(args) >= 3 else None
        event["prefix_len"] = prefix_len
        event["amount"] = str(amount)
        event["detail"] = f"boost_valid, prefix_len={prefix_len}, amount={amount}"
        return event

    event["detail"] = "result redacted"
    return event


def _attach_helper_fastpath(VerifiedDecoderAgent) -> None:
    """Replace the compiled Python vocab-scan argmax with a tensor-argmax.

    The Dafny compiler emits GetHighestLogitToken as a pure Python ``while``
    loop over ``lm.Tokens``, calling ``lm.Logits[i]`` twice per iteration.
    Each such access goes through ``_LogitsProxy.__getitem__`` which does
    ``tensor[idx].item()`` (a GPU->CPU sync) + ``_dafny.BigRational(...)``
    construction. For Qwen-14B (vocab ~152k) and the typical MyCSDStrategy
    (which calls GetHighestLogitToken on both the unconstrained and
    constrained branches), that is ~300k GPU syncs per generation step and
    was measured at ~6.0 s/step — dominating total wall time.

    The Dafny spec only uses GetHighestLogitToken's postcondition (returns
    the token at argmax(Logits) over lm.Tokens). ``lm.ChooseNextToken()``
    is the vectorized implementation of exactly that spec:
    ``self._Tokens[self._logits_tensor.argmax().item()]``. Swapping
    implementations preserves the postcondition the verifier used to prove
    callers, so no Dafny proof can be invalidated by this shim.

    Safe to call multiple times; idempotent via ``_fastpath_patched``.
    """
    helpers_cls = getattr(VerifiedDecoderAgent, "CSDHelpers", None)
    if helpers_cls is None or getattr(helpers_cls, "_fastpath_patched", False):
        return

    def _fast_get_highest_logit_token(self, lm):
        # Delegate to the tensor-backed argmax. lm.ChooseNextToken does a
        # single argmax over the constrained logits tensor and returns the
        # corresponding Dafny token sequence -- exactly what the Dafny spec
        # for GetHighestLogitToken requires, but O(1) GPU syncs instead of
        # O(vocab_size).
        return lm.ChooseNextToken()

    def _fast_top_valid_candidates(self, lm, parser, prompt, prefix, maxCandidates, eosToken):
        """Vectorized top-K over parser-valid + EOS tokens.

        The Dafny-compiled version (a) calls GenerateLogits, (b) walks
        parser.ValidNextTokens(prefix) in Python while doing two
        lm.Logits[i] proxy reads per iteration, then (c) re-walks the pool
        K times to find each successive max. With per-element GPU syncs
        and the vocab-sized valid pool seen in SQL, that's hundreds of
        GPU round-trips per call and dominates wall time.

        This shim does the same thing in one masked_fill + one topk:
          1. Refresh logits (matches the original side effect).
          2. Build the parser-valid + EOS boolean mask (subset over the
             constrained vocab) using the same _parser_full_mask path
             MaskValidNextAndEos uses.
          3. masked_fill the cached _logits_tensor with -inf for invalid.
          4. torch.topk(masked_logits, k) -> indices.
          5. Map indices back to Dafny tokens via lm._Tokens.
          6. Bump self.cost by 1, matching the original Dafny postcondition.
        """
        import torch as _torch
        import _dafny

        # 1) Refresh logits (matches original Dafny semantics)
        lm.GenerateLogits(prompt + prefix)

        # 2) Build valid-token boolean mask (subset over constrained vocab)
        full_mask = lm._parser_full_mask(parser, prefix)
        if full_mask.numel() == len(lm._token_ids):
            subset_mask = full_mask.to(dtype=_torch.bool, device=lm._logits_tensor.device)
        else:
            full_mask = lm._expand_full_mask(full_mask)
            subset_mask = lm._subset_mask_from_full_mask(full_mask)

        # 3) Add EOS to mask
        eos_indices = lm._token_indices_for_token(eosToken)
        if eos_indices:
            subset_mask[eos_indices] = True

        # 4) Cost bump (matches original)
        self.cost = self.cost + 1

        valid_count = int(subset_mask.sum().item())
        if valid_count == 0:
            # Pathological: no valid tokens at all. Match original fallback
            # behaviour which returns a singleton with eosToken.
            return _dafny.SeqWithoutIsStrInference([eosToken])

        # 5) topk over masked logits in one GPU op
        masked = lm._logits_tensor.masked_fill(~subset_mask, -float('inf'))
        k = int(maxCandidates) if int(maxCandidates) < valid_count else valid_count
        _, topk_idx = _torch.topk(masked, k)

        # 6) Map indices back to Dafny tokens
        idx_list = topk_idx.cpu().tolist()
        chosen = [lm._Tokens[int(i)] for i in idx_list]
        return _dafny.SeqWithoutIsStrInference(chosen)

    def _fast_boost_token_logits(self, lm, tokens, amount):
        """Vectorized boost. The Dafny-compiled version loops in Python
        with two GPU syncs per token (read+write through _LogitsProxy).
        For a typical strategy that boosts 1-5 tokens per step, this
        adds tens of unnecessary GPU round-trips per emitted token.
        """
        import torch as _torch
        if tokens is None or len(tokens) == 0:
            return
        indices = []
        for i in range(len(tokens)):
            idx_list = lm._token_indices_for_token(tokens[i])
            if idx_list:
                indices.extend(idx_list)
        if not indices:
            return
        idx_t = _torch.tensor(indices, device=lm._logits_tensor.device, dtype=_torch.long)
        amount_f = float(amount)
        lm._logits_tensor[idx_t] = _torch.clamp(
            lm._logits_tensor[idx_t] + amount_f, min=-1e9, max=1e9
        )
        if lm._full_logits is not None:
            full_ids = lm._token_ids_tensor[idx_t].to(lm._full_logits.device)
            lm._full_logits[full_ids] = _torch.clamp(
                lm._full_logits[full_ids] + amount_f, min=-1e9, max=1e9
            )
        lm.Logits.update_tensors(lm._logits_tensor, lm._full_logits)
        lm._logits_dirty = True

    def _fast_penalize_token_logits(self, lm, tokens, amount):
        """Vectorized penalize. Same shape as boost, opposite sign."""
        import torch as _torch
        if tokens is None or len(tokens) == 0:
            return
        indices = []
        for i in range(len(tokens)):
            idx_list = lm._token_indices_for_token(tokens[i])
            if idx_list:
                indices.extend(idx_list)
        if not indices:
            return
        idx_t = _torch.tensor(indices, device=lm._logits_tensor.device, dtype=_torch.long)
        amount_f = float(amount)
        lm._logits_tensor[idx_t] = _torch.clamp(
            lm._logits_tensor[idx_t] - amount_f, min=-1e9, max=1e9
        )
        if lm._full_logits is not None:
            full_ids = lm._token_ids_tensor[idx_t].to(lm._full_logits.device)
            lm._full_logits[full_ids] = _torch.clamp(
                lm._full_logits[full_ids] - amount_f, min=-1e9, max=1e9
            )
        lm.Logits.update_tensors(lm._logits_tensor, lm._full_logits)
        lm._logits_dirty = True

    def _fast_scale_all_logits(self, lm, scalar):
        """Vectorized scale. Replaces a Python while-loop over the entire
        constrained vocab with a single tensor multiply."""
        import torch as _torch
        scalar_f = float(scalar)
        lm._logits_tensor.mul_(scalar_f)
        lm._logits_tensor.clamp_(min=-1e9, max=1e9)
        if lm._full_logits is not None:
            lm._full_logits.mul_(scalar_f)
            lm._full_logits.clamp_(min=-1e9, max=1e9)
        lm.Logits.update_tensors(lm._logits_tensor, lm._full_logits)
        lm._logits_dirty = True

    # Preserve the originals so the fastpath can fall back when the LM
    # isn't a _TensorizedLMBase subclass (legacy smoke-test LM stubs).
    _orig_get_highest = helpers_cls.GetHighestLogitToken
    _orig_top_valid = helpers_cls.TopValidCandidates
    _orig_boost = helpers_cls.BoostTokenLogits
    _orig_penalize = helpers_cls.PenalizeTokenLogits
    _orig_scale = helpers_cls.ScaleAllLogits

    def _ghl_with_fallback(self, lm):
        if not hasattr(lm, '_logits_tensor'):
            return _orig_get_highest(self, lm)
        return _fast_get_highest_logit_token(self, lm)

    def _tvc_with_fallback(self, lm, parser, prompt, prefix, maxCandidates, eosToken):
        if not hasattr(lm, '_logits_tensor'):
            return _orig_top_valid(self, lm, parser, prompt, prefix, maxCandidates, eosToken)
        return _fast_top_valid_candidates(self, lm, parser, prompt, prefix, maxCandidates, eosToken)

    def _boost_with_fallback(self, lm, tokens, amount):
        if not hasattr(lm, '_logits_tensor'):
            return _orig_boost(self, lm, tokens, amount)
        return _fast_boost_token_logits(self, lm, tokens, amount)

    def _penalize_with_fallback(self, lm, tokens, amount):
        if not hasattr(lm, '_logits_tensor'):
            return _orig_penalize(self, lm, tokens, amount)
        return _fast_penalize_token_logits(self, lm, tokens, amount)

    def _scale_with_fallback(self, lm, scalar):
        if not hasattr(lm, '_logits_tensor'):
            return _orig_scale(self, lm, scalar)
        return _fast_scale_all_logits(self, lm, scalar)

    helpers_cls.GetHighestLogitToken = _ghl_with_fallback
    helpers_cls.TopValidCandidates = _tvc_with_fallback
    helpers_cls.BoostTokenLogits = _boost_with_fallback
    helpers_cls.PenalizeTokenLogits = _penalize_with_fallback
    helpers_cls.ScaleAllLogits = _scale_with_fallback

    # ---- Pure-Dafny helpers: O(N*M) Python-list scans -> O(N+M) Python-set ops.
    # The Dafny-compiled bodies use `t in seq` which is a linear scan over the
    # right-hand seq each call. Strategies that pass `lm.Tokens` (~150K) as the
    # right-hand side melt down at O(|left| * 150K) ops per generation step.
    import _dafny

    def _seq_to_str(tok):
        try:
            return "".join(tok[i] for i in range(len(tok)))
        except Exception:
            return str(tok)

    def _fast_intersect_token_sets(a, b):
        b_set = set()
        for i in range(len(b)):
            b_set.add(_seq_to_str(b[i]))
        out = []
        seen_a = set()
        for i in range(len(a)):
            t = a[i]
            s = _seq_to_str(t)
            if s in b_set and s not in seen_a:
                seen_a.add(s)
                out.append(t)
        return _dafny.SeqWithoutIsStrInference(out)

    def _fast_subtract_token_sets(a, b):
        b_set = set()
        for i in range(len(b)):
            b_set.add(_seq_to_str(b[i]))
        out = []
        for i in range(len(a)):
            t = a[i]
            if _seq_to_str(t) not in b_set:
                out.append(t)
        return _dafny.SeqWithoutIsStrInference(out)

    def _fast_flatten_token_groups(groups):
        flat = []
        for i in range(len(groups)):
            g = groups[i]
            for j in range(len(g)):
                flat.append(g[j])
        return _dafny.SeqWithoutIsStrInference(flat)

    def _fast_group_containing(groups, tok):
        target = _seq_to_str(tok)
        for i in range(len(groups)):
            g = groups[i]
            for j in range(len(g)):
                if _seq_to_str(g[j]) == target:
                    return i
        return -1

    def _fast_rollback_to_boundary(parser, currentConstrained, boundaryToken):
        # Walk back in Python to find the last occurrence of boundaryToken in
        # currentConstrained, then return the prefix up to (and including) it.
        # This avoids the Dafny-compiled token-equality loop overhead.
        bt_str = _seq_to_str(boundaryToken)
        cc_len = len(currentConstrained)
        idx = cc_len
        while idx > 0 and _seq_to_str(currentConstrained[idx - 1]) != bt_str:
            idx -= 1
        sliced = _dafny.SeqWithoutIsStrInference(
            [currentConstrained[i] for i in range(idx)]
        )
        # Common case: every prefix of a parser-valid prefix is itself parser-
        # valid in the LR-style grammars we use, so one IsValidPrefix check
        # short-circuits RollbackToValidPrefix's per-token trim loop.
        if parser.IsValidPrefix(sliced) and not parser.IsDeadPrefix(sliced):
            return sliced
        # Fallback: full trim loop entered from `sliced` (already a strict
        # suffix-trimmed prefix of currentConstrained).
        repaired = sliced
        while not parser.IsValidPrefix(repaired) or parser.IsDeadPrefix(repaired):
            repaired = _dafny.SeqWithoutIsStrInference(
                [repaired[i] for i in range(len(repaired) - 1)]
            )
        return repaired

    def _fast_rollback_to_valid_prefix(parser, generated):
        # Short-circuit: if `generated` is already valid, no trimming needed.
        if parser.IsValidPrefix(generated) and not parser.IsDeadPrefix(generated):
            return generated
        # Fall back to per-token trim, but build the slice via Python list
        # ops rather than per-iteration Dafny seq slicing.
        repaired_list = [generated[i] for i in range(len(generated))]
        repaired = generated
        while not parser.IsValidPrefix(repaired) or parser.IsDeadPrefix(repaired):
            if not repaired_list:
                break
            repaired_list.pop()
            repaired = _dafny.SeqWithoutIsStrInference(list(repaired_list))
        return repaired

    helpers_cls.IntersectTokenSets = staticmethod(_fast_intersect_token_sets)
    helpers_cls.SubtractTokenSets = staticmethod(_fast_subtract_token_sets)
    helpers_cls.FlattenTokenGroups = staticmethod(_fast_flatten_token_groups)
    helpers_cls.GroupContaining = staticmethod(_fast_group_containing)
    helpers_cls.RollbackToBoundary = staticmethod(_fast_rollback_to_boundary)
    helpers_cls.RollbackToValidPrefix = staticmethod(_fast_rollback_to_valid_prefix)

    helpers_cls._fastpath_patched = True


def _attach_helper_trace(VerifiedDecoderAgent, trace_state: Dict[str, Any]) -> None:
    helpers_cls = getattr(VerifiedDecoderAgent, "CSDHelpers", None)
    if helpers_cls is None or getattr(helpers_cls, "_trace_wrapped", False):
        return

    helper_names = [
        "UnconstrainedStep",
        "UnconstrainedChunk",
        "OpenConstrainedSpan",
        "EnterObservedConstrainedSpan",
        "AppendConstrainedToken",
        "CloseConstrainedSpan",
        "CloseSpanIfComplete",
        "CarsTrieStep",
        "ForwardUntilSymbol",
        "ConstrainedStep",
        "UnconstrainedGeneration",
        "ConstrainedGeneration",
        "CraneGeneration",
        "ConfidenceGatedStep",
        "PenalizedConstrainedStep",
        "BoostedConstrainedStep",
        "SoftConstrainedStep",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "SafeBoostedConstrainedStep",
        "SafePenalizedConstrainedStep",
        "SafeRepetitionPenaltyStep",
        "SafeTemperatureConstrainedStep",
        "SafeSoftConstrainedStep",
        "GroupBoostedConstrainedStep",
        "GroupHasValidMember",
        "BoostValidGroups",
        "AdaptiveConstrainedStep",
        "AdaptiveConstrainedStepWithPenalties",
        "ConstrainedSymbol",
        "ConstrainedSymbolInGenerated",
        "BackwardToSymbol",
        "RollbackConstrainedSpan",
        "RollbackConstrainedSuffix",
        "RollbackConstrainedToComplete",
        "RollbackToCompletePrefix",
        "RollbackToValidPrefix",
        "RollbackAndRegenerate",
        "RollbackAndContinue",
        "RegenerateUnitOnCheckFailure",
        "RegenerateUnitOnGroundingFailure",
        "DeadEndAvoidingStep",
        "DeadEndDetection",
        "ValidTokenCount",
        "BoostTokenLogits",
        "PenalizeTokenLogits",
        "SafeBoostTokenLogits",
        "SafePenalizeTokenLogits",
        "MaskTokensInPrefix",
        "GetHighestLogitToken",
        "GetLogitGap",
        "GetTopKTokens",
        "GetTokenLogit",
        "ScaleAllLogits",
        "SaveLogitsSnapshot",
        "RestoreLogitsSnapshot",
        "SpeculativeConstrainedRollout",
        "RolloutConstrainedWithPenalties",
        "TopValidCandidates",
        "IsTokenValidNext",
        "LastTokenBefore",
        "ExtractAfterKeyword",
        "FlattenTokenGroups",
        "GroupContaining",
        "IntersectTokenSets",
        "SubtractTokenSets",
    ]

    rollback_method_names = {
        "BackwardToSymbol",
        "RollbackConstrainedSpan",
        "RollbackConstrainedSuffix",
        "RollbackConstrainedToComplete",
        "RollbackToCompletePrefix",
        "RollbackToValidPrefix",
        "RollbackAndRegenerate",
        "RollbackAndContinue",
    }
    full_prefix_method_names = rollback_method_names - {"BackwardToSymbol"}
    full_prefix_method_names.add("CraneGeneration")

    def _align_pending_rollback(args):
        # A rollback can call another traced rollback before its outer result
        # is available. Only the next top-level LM-bearing helper may consume
        # the private returned prefix; inner wrappers must leave it untouched.
        if int(trace_state.get("_spider_helper_wrapper_depth", 0)) != 0:
            return
        pending = trace_state.get("_pending_spider_rollback_prefix")
        if pending is None:
            return
        lm = next(
            (
                arg
                for arg in args
                if callable(getattr(arg, "_align_generation_history_to_prefix", None))
            ),
            None,
        )
        if lm is None:
            return
        lm._align_generation_history_to_prefix(pending)
        trace_state.pop("_pending_spider_rollback_prefix", None)

    def _enter_helper_wrapper() -> None:
        depth = int(trace_state.get("_spider_helper_wrapper_depth", 0))
        trace_state["_spider_helper_wrapper_depth"] = depth + 1

    def _leave_helper_wrapper() -> None:
        depth = int(trace_state.get("_spider_helper_wrapper_depth", 0)) - 1
        if depth > 0:
            trace_state["_spider_helper_wrapper_depth"] = depth
        else:
            trace_state.pop("_spider_helper_wrapper_depth", None)

    for name in helper_names:
        descriptor = inspect.getattr_static(helpers_cls, name, None)
        if descriptor is None:
            continue
        if isinstance(descriptor, staticmethod):
            method = descriptor.__func__
            descriptor_kind = "static"
        elif isinstance(descriptor, classmethod):
            method = descriptor.__func__
            descriptor_kind = "class"
        else:
            method = descriptor
            descriptor_kind = "instance"

        def _make_wrapper(method_name, method, descriptor_kind):
            def _record(args, result, cost_before, cost_after):
                if (
                    method_name in full_prefix_method_names
                    and int(trace_state.get("_spider_helper_wrapper_depth", 0)) == 1
                ):
                    prefix = result[0] if isinstance(result, tuple) and result else result
                    if hasattr(prefix, "__len__"):
                        # Keep raw prefix objects private; only lengths enter events.
                        trace_state["_pending_spider_rollback_prefix"] = prefix
                trace_state.setdefault("events", []).append(
                    _summarize_helper_event(method_name, args, result, cost_before, cost_after)
                )

            def _wrapped_instance(self, *args, **kwargs):
                _align_pending_rollback(args)
                _enter_helper_wrapper()
                try:
                    cost_before = getattr(self, "cost", None)
                    result = method(self, *args, **kwargs)
                    cost_after = getattr(self, "cost", None)
                    _record(args, result, cost_before, cost_after)
                    return result
                finally:
                    _leave_helper_wrapper()

            def _wrapped_static(*args, **kwargs):
                _align_pending_rollback(args)
                _enter_helper_wrapper()
                try:
                    result = method(*args, **kwargs)
                    _record(args, result, None, None)
                    return result
                finally:
                    _leave_helper_wrapper()

            def _wrapped_class(cls, *args, **kwargs):
                _align_pending_rollback(args)
                _enter_helper_wrapper()
                try:
                    cost_before = getattr(cls, "cost", None)
                    result = method(cls, *args, **kwargs)
                    cost_after = getattr(cls, "cost", None)
                    _record(args, result, cost_before, cost_after)
                    return result
                finally:
                    _leave_helper_wrapper()

            if descriptor_kind == "static":
                return staticmethod(_wrapped_static)
            if descriptor_kind == "class":
                return classmethod(_wrapped_class)
            return _wrapped_instance

        setattr(helpers_cls, name, _make_wrapper(name, method, descriptor_kind))

    helpers_cls._trace_wrapped = True


def _attach_lm_trace(lm: Any, trace_state: Dict[str, Any]) -> None:
    """Wrap LM-side primitives that drive real forward passes / masking so the
    behavioral trace captures them alongside CSDHelpers events."""
    if lm is None or getattr(lm, "_trace_wrapped", False):
        return

    lm_method_names = [
        "GenerateLogits",
        "ChooseNextTokenUnconstrained",
        "MaskValidNextAndEos",
        "BoostValidNextAndEos",
    ]

    for name in lm_method_names:
        original = getattr(lm, name, None)
        if original is None:
            continue

        def _make_wrapper(method_name, method):
            def _wrapped(*args, **kwargs):
                result = method(*args, **kwargs)
                trace_state.setdefault("events", []).append(
                    _summarize_helper_event(method_name, args, result, None, None)
                )
                return result
            return _wrapped

        setattr(lm, name, _make_wrapper(name, original))

    lm._trace_wrapped = True


def resolve_run_dir(run_dir: Path) -> Path:
    """
    Resolve a run directory path, handling 'latest' shortcut.

    If run_dir ends with 'latest' and doesn't exist as a directory,
    reads the actual path from 'latest_run.txt' in the parent directory.

    Args:
        run_dir: Path to the synthesis run directory (may be 'latest' shortcut)

    Returns:
        Resolved actual path to the run directory
    """
    if run_dir.name == "latest" and not run_dir.exists():
        latest_txt = run_dir.parent / "latest_run.txt"
        if latest_txt.exists():
            actual_path = Path(latest_txt.read_text().strip())
            if actual_path.exists():
                return actual_path
    return run_dir


def load_compiled_modules(run_dir: Path):
    """
    Load compiled CSD modules from a synthesis run directory.

    Args:
        run_dir: Path to the synthesis run directory

    Returns:
        Tuple of (_dafny, VerifiedDecoderAgent, GeneratedCSD) modules

    Raises:
        FileNotFoundError: If compiled modules are not found
    """
    # Resolve 'latest' shortcut if needed
    run_dir = resolve_run_dir(run_dir)

    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        # Check if GeneratedCSD.py is directly in run_dir
        if (run_dir / "GeneratedCSD.py").exists():
            module_dir = run_dir
        else:
            # Fallback to gsm_crane_csd or try to find the directory
            module_dir = run_dir / "gsm_crane_csd"
            if not module_dir.exists():
                # Try to find any directory that contains GeneratedCSD.py
                found = list(run_dir.glob("*/GeneratedCSD.py"))
                if found:
                    module_dir = found[0].parent
                else:
                    raise FileNotFoundError(f"Compiled module directory not found in {run_dir}")

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    # Force a fresh import from the selected compiled directory for each attempt.
    # The synthesis loop evaluates multiple compiled candidates in one Python
    # process; without clearing sys.modules, later attempts can accidentally reuse
    # GeneratedCSD / VerifiedDecoderAgent from an earlier attempt.
    for module_name in ["GeneratedCSD", "VerifiedDecoderAgent", "module_", "System_", "_dafny"]:
        sys.modules.pop(module_name, None)

    _dafny = importlib.import_module("_dafny")
    VerifiedDecoderAgent = importlib.import_module("VerifiedDecoderAgent")
    GeneratedCSD = importlib.import_module("GeneratedCSD")

    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    backend: str,
    device: str,
    grammar_file: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vllm_tensor_parallel_size: int | None = None,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int = 16384,
    vllm_enforce_eager: bool = True,
) -> Dict[str, Any]:
    """
    Load model and setup Dafny environment once.
    Returns reusable objects for generation.

    Args:
        run_dir: Path to the synthesis run directory
        model_name: Model identifier
        backend: Runtime backend ("huggingface" or "vllm")
        device: Device to run on ("cuda" or "cpu")
        grammar_file: Path to grammar file
        load_in_4bit: Whether to load in 4-bit quantization
        load_in_8bit: Whether to load in 8-bit quantization
        vllm_tensor_parallel_size: Explicit tensor parallel size for vLLM
        vllm_pipeline_parallel_size: Explicit pipeline parallel size for vLLM
        vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
        vllm_max_model_len: Max context length passed to vLLM
        vllm_enforce_eager: Disable cudagraph/compile in vLLM for stability

    Returns:
        Environment dict with:
        - "_dafny": Dafny runtime module
        - "VerifiedDecoderAgent": Dafny decoder agent module
        - "GeneratedCSD": Generated CSD module
        - "lm": Language model wrapper
        - "parser": Grammar parser
        - "tokenizer": Backend tokenizer
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)
    # Patch the compiled vocab-scan argmax to a tensor-argmax BEFORE the
    # trace wrapper wraps CSDHelpers methods — the trace wrapper still sees
    # the call, but the underlying work is now O(1) GPU syncs.
    _attach_helper_fastpath(VerifiedDecoderAgent)
    trace_state: Dict[str, Any] = {"events": []}
    _attach_helper_trace(VerifiedDecoderAgent, trace_state)

    from synthesis.evaluate.benchmarks.common.model_utils import create_runtime_lm, load_runtime_tokenizer
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    tok = load_runtime_tokenizer(model_name, backend=backend)

    lm = create_runtime_lm(
        model_name=model_name,
        backend=backend,
        device=device,
        VerifiedDecoderAgent=VerifiedDecoderAgent,
        _dafny=_dafny,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_enforce_eager=vllm_enforce_eager,
    )
    _attach_lm_trace(lm, trace_state)

    # Create the parser over the active constrained expression only. Delimiters are
    # tracked in generated output state, not inside currentConstrained.
    grammar_text = grammar_file.read_text()
    LarkDafnyParser = create_lark_dafny_parser(
        grammar_text,
        VerifiedDecoderAgent,
        _dafny,
        start="csd_start",
        tokenizer=tok,
    )
    parser = LarkDafnyParser(lm._Tokens)

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": tok,
        "csd_trace": trace_state,
        "model_name": model_name,
    }


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}
