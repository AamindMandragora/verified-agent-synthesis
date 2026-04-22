"""
Environment setup utilities for GSM-Symbolic evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for constrained generation.
"""

from __future__ import annotations

import sys
import importlib
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
    event: Dict[str, Any] = {
        "helper": name,
        "cost_before": cost_before,
        "cost_after": cost_after,
    }

    if name in {"UnconstrainedStep", "ConstrainedStep", "PenalizedConstrainedStep", "BoostedConstrainedStep", "RepetitionPenaltyStep", "TemperatureConstrainedStep"}:
        token = _truncate(_dafny_token_to_str(result))
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

    if name == "SoftConstrainedStep":
        token = _truncate(_dafny_token_to_str(result[0])) if isinstance(result, tuple) and len(result) == 2 else _truncate(str(result))
        is_valid = bool(result[1]) if isinstance(result, tuple) and len(result) == 2 else False
        event["token"] = token
        event["is_valid"] = is_valid
        event["detail"] = f"token={token}, is_valid={is_valid}"
        return event

    if name == "OpenConstrainedSpan":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and len(result) >= 1 else None
        event["generated_len"] = generated_len
        event["detail"] = f"opened span, generated_len={generated_len}"
        return event

    if name == "AppendConstrainedToken":
        token = _truncate(_dafny_token_to_str(args[-1])) if args else ""
        current_len = _safe_len(result[2]) if isinstance(result, tuple) and len(result) >= 3 else None
        event["token"] = token
        event["current_len"] = current_len
        event["detail"] = f"append {token}, current_len={current_len}"
        return event

    if name == "CloseConstrainedSpan":
        generated_len = _safe_len(result[0]) if isinstance(result, tuple) and len(result) >= 1 else None
        event["generated_len"] = generated_len
        event["detail"] = f"closed span, generated_len={generated_len}"
        return event

    if name == "RollbackConstrainedSpan":
        current_len = _safe_len(result[1]) if isinstance(result, tuple) and len(result) >= 2 else None
        event["current_len"] = current_len
        event["detail"] = f"rollback, current_len={current_len}"
        return event

    if name == "DeadEndDetection":
        event["detail"] = f"narrow={bool(result)}"
        return event

    if name in {"BoostTokenLogits", "PenalizeTokenLogits"}:
        tokens = [_truncate(_dafny_token_to_str(t)) for t in args[1]] if len(args) >= 2 else []
        amount = args[2] if len(args) >= 3 else None
        event["tokens"] = tokens
        event["amount"] = str(amount)
        event["detail"] = f"tokens={tokens[:4]}, amount={amount}"
        return event

    if name == "TopValidCandidates":
        k = args[-2] if len(args) >= 2 else None
        candidate_count = _safe_len(result)
        preview: list[str] = []
        try:
            for i in range(min(3, len(result))):
                preview.append(_truncate(_dafny_token_to_str(result[i])))
        except Exception:
            pass
        event["k_requested"] = str(k)
        event["candidate_count"] = candidate_count
        event["preview"] = preview
        event["detail"] = f"k={k}, got={candidate_count}, preview={preview}"
        return event

    if name == "IsTokenValidNext":
        token = _truncate(_dafny_token_to_str(args[-1])) if args else ""
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
        token = _truncate(_dafny_token_to_str(result)) if result is not None else ""
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

    event["detail"] = _truncate(str(result))
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

    helpers_cls.GetHighestLogitToken = _fast_get_highest_logit_token
    helpers_cls._fastpath_patched = True


def _attach_helper_trace(VerifiedDecoderAgent, trace_state: Dict[str, Any]) -> None:
    helpers_cls = getattr(VerifiedDecoderAgent, "CSDHelpers", None)
    if helpers_cls is None or getattr(helpers_cls, "_trace_wrapped", False):
        return

    helper_names = [
        "UnconstrainedStep",
        "UnconstrainedChunk",
        "OpenConstrainedSpan",
        "AppendConstrainedToken",
        "CloseConstrainedSpan",
        "ConstrainedStep",
        "PenalizedConstrainedStep",
        "BoostedConstrainedStep",
        "SoftConstrainedStep",
        "RepetitionPenaltyStep",
        "TemperatureConstrainedStep",
        "RollbackConstrainedSpan",
        "DeadEndDetection",
        "BoostTokenLogits",
        "PenalizeTokenLogits",
        "TopValidCandidates",
        "IsTokenValidNext",
    ]

    for name in helper_names:
        original = getattr(helpers_cls, name, None)
        if original is None:
            continue

        def _make_wrapper(method_name, method):
            def _wrapped(self, *args, **kwargs):
                cost_before = getattr(self, "cost", None)
                result = method(self, *args, **kwargs)
                cost_after = getattr(self, "cost", None)
                trace_state.setdefault("events", []).append(
                    _summarize_helper_event(method_name, args, result, cost_before, cost_after)
                )
                return result
            return _wrapped

        setattr(helpers_cls, name, _make_wrapper(name, original))

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
    vllm_max_model_len: int = 4096,
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

    from evaluations.common.model_utils import create_runtime_lm, load_runtime_tokenizer
    from evaluations.common.parser_utils import create_lark_dafny_parser

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
    }


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}
