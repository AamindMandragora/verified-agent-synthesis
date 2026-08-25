"""
Model loading and management utilities for CSD evaluation.

Supports both HuggingFace and vLLM runtimes. The hot path remains tensorized:
next-token logits are captured as tensors, masking uses tensor ops, constrained
token selection is argmax over masked tensors, and unconstrained token selection
samples from the model distribution.
"""

from __future__ import annotations

import os
import logging
import math
import multiprocessing as mp
import re
import time
from collections import defaultdict
from contextlib import contextmanager

from typing import Any

import torch


# Diagnostic logging for the prompt-grounding extern (SpanGrounded) and the
# tried-token recurrence penalty. Tagged "[grounding]" / "[recurrence]" so a run's
# decisions can be grepped out of the log. The synthesis entrypoint never configures
# root logging (defaults to WARNING), so these INFO lines are invisible by default.
# Set CSD_GROUNDING_LOG=1 to attach a stderr handler at INFO and make them show — an
# OPT-IN diagnostic only; with the env var unset this block is a no-op and behaviour
# (masks, scoring, decode) is byte-identical to before.
_GROUNDING_LOG = logging.getLogger("csd.grounding")
_SPIDER_CONTRACT_LOG = logging.getLogger("csd.spider_output_contract")


def _coerce_token_id_set(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset({int(value)})
    return frozenset(int(item) for item in value)


def _default_generation_stop_token_ids(tokenizer: Any) -> frozenset[int]:
    configured = getattr(tokenizer, "generation_stop_token_ids", None)
    if configured is None:
        configured = getattr(tokenizer, "eos_token_id", None)
    return _coerce_token_id_set(configured)


if os.environ.get("CSD_GROUNDING_LOG"):
    _grounding_handler = logging.StreamHandler()
    _grounding_handler.setFormatter(logging.Formatter("%(message)s"))
    _GROUNDING_LOG.addHandler(_grounding_handler)
    _GROUNDING_LOG.setLevel(logging.INFO)
    _GROUNDING_LOG.propagate = False

# Keywords/functions that are never schema identifiers; excluded from the
# grounding check so they are not mistaken for table/column names.
_GROUNDING_STOPWORDS = frozenset({
    "select", "from", "where", "group", "by", "order", "having", "limit", "offset",
    "and", "or", "not", "in", "as", "on", "join", "inner", "left", "right", "outer",
    "full", "cross", "natural", "union", "intersect", "except", "distinct", "count",
    "sum", "avg", "min", "max", "like", "between", "is", "null", "asc", "desc", "all",
    "any", "exists", "case", "when", "then", "else", "end", "values", "insert",
    "update", "delete", "set", "into", "using", "true", "false", "with", "over",
    "partition", "cast", "coalesce", "substr", "upper", "lower", "abs", "round",
})

_GROUNDING_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Short alias-like tokens: a single letter, or a letter (optionally repeated)
# followed by digits (e.g. t1, t2, a1) — common query aliases, not schema names.
_GROUNDING_ALIAS_RE = re.compile(r"^(?:[A-Za-z]|[A-Za-z]+\d+)$")
_GROUNDING_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def _parse_schema_support(prompt_text: str) -> set:
    """Schema identifier names (lowercased) for the CURRENT example's prompt.

    Mirrors sql_spider/schema_grammar.parse_schema_names, but applied only to the
    text after the LAST `db_info:` marker (the real example's schema) and before
    `question:`, so the few-shot example's schema does not leak into the support
    set. Returns an empty set when no `db_info:` block is present (e.g. non-SQL
    prompts) — in which case grounding is a no-op.
    """
    if not prompt_text or "db_info:" not in prompt_text:
        return set()
    block = prompt_text.rsplit("db_info:", 1)[1]
    block = block.split("question:", 1)[0]
    names: set = set()
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        line = line[1:].strip()
        m = re.match(r"(\w+)\s*\((.+)\)", line)
        if not m:
            continue
        names.add(m.group(1).lower())
        for col in m.group(2).split(","):
            col = col.strip()
            if "." in col:
                col = col.split(".")[-1].strip()
            if col and re.match(r"^\w+$", col):
                names.add(col.lower())
    return names


def _candidate_identifiers(text: str) -> list:
    """Identifier-like tokens in `text` that should be checked for grounding.

    Strips quoted string-literal contents (those are values, not identifiers),
    drops keywords/functions, and drops short alias-like tokens. Lowercased.
    """
    stripped = _GROUNDING_QUOTED_RE.sub(" ", text or "")
    out: list = []
    for tok in _GROUNDING_IDENT_RE.findall(stripped):
        low = tok.lower()
        if low in _GROUNDING_STOPWORDS:
            continue
        if _GROUNDING_ALIAS_RE.match(tok):
            continue
        out.append(low)
    return out


def _candidate_identifiers_with_pos(text: str) -> list:
    """Same identifiers as `_candidate_identifiers`, paired with each one's
    CHARACTER OFFSET in `text`. Returns `[(name_lower, char_offset), ...]`.

    Signal-identical to `_candidate_identifiers` (same stopword / alias / quoted
    filtering, same order) — the only addition is the offset. To keep offsets
    truthful, quoted regions are blanked with EQUAL-LENGTH spaces (not collapsed
    to one space), so every later identifier keeps its real position in `text`.
    """
    masked = _GROUNDING_QUOTED_RE.sub(lambda m: " " * len(m.group(0)), text or "")
    out: list = []
    for m in _GROUNDING_IDENT_RE.finditer(masked):
        tok = m.group(0)
        low = tok.lower()
        if low in _GROUNDING_STOPWORDS:
            continue
        if _GROUNDING_ALIAS_RE.match(tok):
            continue
        out.append((low, m.start()))
    return out


def _first_ungrounded_token_idx(token_strs: list, support: set) -> tuple:
    """Index of the token that CONTAINS the first out-of-schema identifier.

    Inputs:
      - token_strs: the unit's token strings in order (rendered by concatenation,
        matching RenderPrefix — no separators between tokens).
      - support: the schema identifier support set for the current example.
    Output: `(found, idx)`. `found` is True iff some candidate identifier in the
    rendered text is not in `support`; `idx` is the index of the token holding
    that identifier's first character. `(False, 0)` when fully grounded or when
    `support` is empty (no recognizable schema → grounding is a no-op).

    Pure: needs no model/tokenizer, so it is unit-testable on its own. The
    membership signal is identical to `SpanGrounded`; only the position is new.
    """
    if not support:
        return (False, 0)
    text = "".join(token_strs)
    bad_off = None
    for name, off in _candidate_identifiers_with_pos(text):
        if name not in support:
            bad_off = off
            break
    if bad_off is None:
        return (False, 0)
    cum = 0
    for i, s in enumerate(token_strs):
        nxt = cum + len(s)
        if bad_off < nxt:
            return (True, i)
        cum = nxt
    # Offset past the end of every token (should not happen given the text was
    # built from these tokens) — clamp to the last token to keep idx in range.
    return (True, max(len(token_strs) - 1, 0))


# Per-component timing. Keyed by a short label. Values are (total_seconds, call_count).
# Printed periodically from GenerateLogits to break down where per-step time goes.
# Set CSD_DISABLE_TIMING=1 to turn off.
_TIMINGS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_TIMINGS_ENABLED = os.environ.get("CSD_DISABLE_TIMING", "") == ""
_TIMINGS_PRINT_EVERY = int(os.environ.get("CSD_TIMINGS_PRINT_EVERY", "10"))


@contextmanager
def _timed(label: str):
    """Accumulate wall-clock time under `label` into `_TIMINGS`."""
    if not _TIMINGS_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        slot = _TIMINGS[label]
        slot[0] += elapsed
        slot[1] += 1


def _print_timings_breakdown(header: str = "") -> None:
    if not _TIMINGS_ENABLED or not _TIMINGS:
        return
    total = sum(t for t, _ in _TIMINGS.values())
    if total <= 0:
        return
    lines = [f"[TIMING] {header} total={total:.2f}s"]
    for label in sorted(_TIMINGS.keys(), key=lambda k: -_TIMINGS[k][0]):
        secs, calls = _TIMINGS[label]
        pct = 100.0 * secs / total
        avg_ms = 1000.0 * secs / max(calls, 1)
        lines.append(f"  {label:<30} {secs:7.2f}s  ({pct:5.1f}%)  calls={calls:<5} avg={avg_ms:7.2f}ms")
    print("\n".join(lines), flush=True)

    # Also print parser-side timings if available. Lazy import avoids circular deps
    # and keeps this file independent of parser_utils.
    try:
        from synthesis.evaluate.benchmarks.common.parser_utils import print_parser_timings
        print_parser_timings(header=header)
    except Exception:
        pass

_RUNTIME_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
_VLLM_ENGINE_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}

# vLLM's SamplingParams.logprobs controls how much of the next-token distribution
# is returned. `-1` means "full vocabulary" — for a 152k Qwen vocab this costs
# ~5-8s per step in Python-object construction + IPC alone. We only need the
# top of the distribution for argmax / masking / boost semantics: any token
# outside the top-K is effectively tail noise and gets masked to -1e9 (same
# value used by `MaskToken`), which is indistinguishable from a masked token.
# Raise this if strategies begin reporting all-masked argmaxes.
VLLM_TOPK_LOGPROBS = 1000

from transformers import AutoModelForCausalLM, AutoTokenizer


class _LogitsProxy:
    """Proxy for lm.Logits that writes through to tensor-backed storage."""

    def __init__(self, size, token_ids):
        self._size = size
        self._token_ids = token_ids
        self._constrained_tensor: torch.Tensor | None = None
        self._full_tensor: torch.Tensor | None = None

    def update_tensors(
        self,
        constrained_tensor: torch.Tensor,
        full_tensor: torch.Tensor,
    ) -> None:
        self._constrained_tensor = constrained_tensor
        self._full_tensor = full_tensor

    def __getitem__(self, idx: int):
        import _dafny

        with _timed("LogitsProxy.__getitem__"):
            if self._constrained_tensor is not None:
                return _dafny.BigRational(self._constrained_tensor[idx].item())
            return _dafny.BigRational(0)

    def __setitem__(self, idx: int, value) -> None:
        with _timed("LogitsProxy.__setitem__"):
            float_val = float(value)
            if self._constrained_tensor is not None:
                self._constrained_tensor[idx] = float_val
            if self._full_tensor is not None:
                full_id = self._token_ids[idx]
                self._full_tensor[full_id] = float_val

    def __len__(self) -> int:
        return self._size


torch.set_float32_matmul_precision("high")


def get_model_input_device(model) -> torch.device:
    """Find the device where a HuggingFace model expects inputs."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for key, device in model.hf_device_map.items():
            if "embed" in key.lower():
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    return next(model.parameters()).device


def get_max_input_length(model, tokenizer) -> int:
    """Choose a safe max input length for the runtime."""
    max_len = None
    if hasattr(model, "config") and getattr(model.config, "max_position_embeddings", None):
        max_len = int(model.config.max_position_embeddings)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 1_000_000:
        max_len = min(max_len, int(tok_max)) if max_len else int(tok_max)
    return max_len or 4096




def _configure_vllm_multiprocessing() -> None:
    """Prefer spawn workers for vLLM to avoid CUDA re-init failures under fork."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        # Another library may have already locked the start method.
        pass

def load_runtime_tokenizer(model_name: str, backend: str = "huggingface"):
    """Load the tokenizer matching the requested runtime backend."""
    cache_key = (backend, model_name)
    cached = _RUNTIME_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if backend == "vllm":
        _configure_vllm_multiprocessing()
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _RUNTIME_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _get_visible_devices_key() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")


def _get_cached_vllm_engine(
    model_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    vllm_kwargs: dict[str, Any],
):
    _configure_vllm_multiprocessing()
    from vllm import LLM

    _patch_vllm_mpclient_shutdown_order()

    cache_key = (
        model_name,
        _get_visible_devices_key(),
        tensor_parallel_size,
        pipeline_parallel_size,
        float(gpu_memory_utilization),
        int(max_model_len),
        bool(enforce_eager),
        repr(sorted(vllm_kwargs.items(), key=lambda item: item[0])),
    )
    cached = _VLLM_ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    print(f"Loading model: {model_name} on cuda with vLLM...")
    tokenizer = load_runtime_tokenizer(model_name, backend="vllm")
    # EngineCore init can fail transiently when GPU memory is fragmented
    # (typically after a prior failed attempt in the same subprocess, or
    # when another vLLM instance is competing for the same device). Retry
    # once after a GPU-state cleanup; this recovered ~20 of the 25 vLLM
    # init failures observed in the May 17 runs.
    import gc as _gc
    import time as _time

    llm = None
    for _vllm_init_attempt in range(2):
        try:
            llm = LLM(
                model=model_name,
                tokenizer=model_name,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                enable_prefix_caching=True,
                max_logprobs=-1,
                disable_log_stats=True,
                **vllm_kwargs,
            )
            break
        except Exception as exc:
            if _vllm_init_attempt == 1:
                raise
            print(
                f"[vllm] Engine init failed: {type(exc).__name__}: {str(exc)[:200]}",
                flush=True,
            )
            print("[vllm] Cleaning GPU state and retrying in 10s...", flush=True)
            try:
                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _torch.cuda.synchronize()
            except Exception:
                pass
            _time.sleep(10)
    _VLLM_ENGINE_CACHE[cache_key] = (llm, tokenizer)
    return llm, tokenizer


def max_cuda_devices_from_env(default: int = 1) -> int:
    """Max CUDA devices for local runs (override with VAS_MAX_CUDA_DEVICES or CSD_MAX_CUDA_DEVICES)."""
    raw = os.environ.get(
        "VAS_MAX_CUDA_DEVICES",
        os.environ.get("CSD_MAX_CUDA_DEVICES", str(default)),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def resolve_vllm_tensor_parallel_size(requested: int | None = None) -> int:
    """Resolve vLLM tensor parallel size, capped by max_cuda_devices_from_env()."""
    cap = max_cuda_devices_from_env()
    tensor_parallel_size = requested or 1
    return max(1, min(tensor_parallel_size, cap))


def limit_cuda_visible_devices(value: str | None, max_devices: int | None = None) -> str | None:
    """Keep at most ``max_devices`` entries from a comma-separated CUDA_VISIBLE_DEVICES value."""
    if not value:
        return value
    cap = max_devices if max_devices is not None else max_cuda_devices_from_env()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= cap:
        return value
    return ",".join(parts[:cap])


def visible_cuda_device_ids() -> list[str]:
    """Return CUDA device ids from CUDA_VISIBLE_DEVICES, or 0..N-1 when unset."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        try:
            import torch

            return [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            return ["0"]
    return [part.strip() for part in visible.split(",") if part.strip()]


def pick_cuda_device_index_with_most_free_memory() -> int:
    """Pick the visible CUDA index with the largest free memory pool."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return 0
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
            if free_bytes > best_free:
                best_free = free_bytes
                best_idx = idx
        return best_idx
    except Exception:
        return 0


def narrow_cuda_visible_devices_to_index(device_index: int) -> str:
    """Restrict CUDA_VISIBLE_DEVICES to one physical id from the current visible set."""
    visible_ids = visible_cuda_device_ids()
    if not visible_ids:
        chosen = str(device_index)
    elif 0 <= device_index < len(visible_ids):
        chosen = visible_ids[device_index]
    else:
        chosen = visible_ids[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


def _patch_vllm_mpclient_shutdown_order() -> None:
    """Set engine_dead before engine_manager.shutdown in vLLM MPClient.shutdown.

    Stock MPClient.shutdown stops the engine manager first, then calls resources()
    which sets engine_dead. The monitor thread can observe the process exit in
    between and log ERROR 'Engine core proc ... died unexpectedly'.
    """
    try:
        from vllm.v1.engine.core_client import MPClient
    except Exception:
        return
    current = getattr(MPClient, "shutdown", None)
    if current is None or getattr(current, "_csd_engine_dead_patched", False):
        return

    def _shutdown(self, timeout=None):  # type: ignore[no-untyped-def]
        resources = getattr(self, "resources", None)
        if resources is not None and hasattr(resources, "engine_dead"):
            try:
                resources.engine_dead = True
            except Exception:
                pass
        return current(self, timeout=timeout)

    _shutdown._csd_engine_dead_patched = True  # type: ignore[attr-defined]
    MPClient.shutdown = _shutdown  # type: ignore[method-assign]


def _mark_vllm_engine_expected_shutdown(llm: Any) -> bool:
    """Set MPClient.resources.engine_dead before EngineCore exits.

    vLLM's monitor thread logs ERROR 'Engine core proc ... died unexpectedly'
    if the process sentinel fires while engine_dead is still False. MPClient.shutdown
    currently stops engine_manager before setting that flag, so mark it first.
    Returns True if a resources.engine_dead flag was found and set.
    """
    marked = False
    seen: set[int] = set()
    stack: list[Any] = [llm]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        try:
            obj_id = id(obj)
        except Exception:
            continue
        if obj_id in seen:
            continue
        seen.add(obj_id)
        resources = getattr(obj, "resources", None)
        if resources is not None and hasattr(resources, "engine_dead"):
            try:
                resources.engine_dead = True
                marked = True
            except Exception:
                pass
        for attr_name in ("llm_engine", "engine_core", "client", "engine"):
            try:
                child = getattr(obj, attr_name, None)
            except Exception:
                child = None
            if child is not None:
                stack.append(child)
    return marked


def clear_vllm_engine_cache() -> None:
    """Release cached vLLM engines before switching back to a generator model."""
    _patch_vllm_mpclient_shutdown_order()
    cached_engines = list(_VLLM_ENGINE_CACHE.values())
    _VLLM_ENGINE_CACHE.clear()

    # #region agent log
    def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
        try:
            import json as _agent_json
            import time as _agent_time
            from pathlib import Path as _agent_Path

            payload = _agent_json.dumps(
                {
                    "sessionId": "fffd8e",
                    "runId": "post-fix",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(_agent_time.time() * 1000),
                }
            )
            for _agent_path in (
                "/Users/aadivyar/Documents/Research/dynamic csd gen clean/.cursor/debug-fffd8e.log",
                "/home/aadivyar/csd-generation/logs/debug-fffd8e.log",
                str(_agent_Path.cwd() / "logs" / "debug-fffd8e.log"),
            ):
                try:
                    _p = _agent_Path(_agent_path)
                    _p.parent.mkdir(parents=True, exist_ok=True)
                    with open(_p, "a", encoding="utf-8") as _agent_f:
                        _agent_f.write(payload + "\n")
                except Exception:
                    continue
        except Exception:
            pass

    _agent_dbg(
        "A",
        "model_utils.py:clear_vllm_engine_cache:entry",
        "clear_vllm_engine_cache enter",
        {"cached_engine_count": len(cached_engines)},
    )
    # #endregion

    for llm, _tokenizer in cached_engines:
        marked = _mark_vllm_engine_expected_shutdown(llm)
        # #region agent log
        _agent_dbg(
            "A",
            "model_utils.py:clear_vllm_engine_cache:marked",
            "marked engine_dead before shutdown",
            {"marked": marked},
        )
        # #endregion
        for attr_name in ("shutdown", "close"):
            maybe_shutdown = getattr(llm, attr_name, None)
            if callable(maybe_shutdown):
                try:
                    maybe_shutdown()
                except Exception:
                    pass

        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            _mark_vllm_engine_expected_shutdown(engine)
            for attr_name in ("shutdown", "close"):
                maybe_shutdown = getattr(engine, attr_name, None)
                if callable(maybe_shutdown):
                    try:
                        maybe_shutdown()
                    except Exception:
                        pass

    try:
        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    # vLLM can leave EngineCore child processes alive even after the Python
    # shutdown hooks return. Kill only direct EngineCore children of this
    # synthesis process so other GPU jobs on the host are left alone.
    try:
        import os
        import signal
        import subprocess
        import time

        current_pid = str(os.getpid())
        proc = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            check=False,
            capture_output=True,
            text=True,
        )
        child_pids: list[int] = []
        for line in proc.stdout.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) != 3:
                continue
            pid, ppid, args = parts
            if ppid == current_pid and "VLLM::EngineCore" in args:
                try:
                    child_pids.append(int(pid))
                except ValueError:
                    continue

        # #region agent log
        _agent_dbg(
            "E",
            "model_utils.py:clear_vllm_engine_cache:sigterm",
            "EngineCore children before SIGTERM",
            {"child_pids": child_pids},
        )
        # #endregion

        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if child_pids:
            time.sleep(1)
        for pid in child_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_vllm_quantization_kwargs(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict[str, Any]:
    """Translate project quantization flags to the installed vLLM config surface."""
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of load_in_4bit or load_in_8bit.")

    if not (load_in_4bit or load_in_8bit):
        return {}

    quant_config: dict[str, Any] = {
        "quant_method": "bitsandbytes",
    }
    if load_in_4bit:
        quant_config.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        )
    else:
        quant_config.update(
            {
                "load_in_8bit": True,
            }
        )

    return {
        "quantization": "bitsandbytes",
        "hf_overrides": {
            "quantization_config": quant_config,
        },
    }


class _TaskGuidanceState:
    """First-call-wins prompt guidance appended by generated CSDs."""

    MAX_GUIDANCE_CHARS = 1200
    HEADER = "Additional task guidance from CSD:"

    def __init__(self) -> None:
        self.accepted_guidance: str | None = None

    def reset(self) -> None:
        self.accepted_guidance = None

    def _coerce_guidance(self, guidance: object) -> str:
        text = str(guidance).strip()
        if not text:
            return ""
        return text[: self.MAX_GUIDANCE_CHARS]


def _eos_is_legal(parser, prefix, has_other_valid_token: bool) -> bool:
    """Decide whether END-OF-TURN may be selected right now.

    Stopping is legal when the prefix is already a complete query, or when
    there is genuinely nothing else the model could write (a dead end). If
    the parser can't answer "is this prefix complete", fall back to the old
    permissive behaviour (always allow) rather than guessing.
    """
    if not hasattr(parser, "IsCompletePrefix"):
        return True
    try:
        is_complete = bool(parser.IsCompletePrefix(prefix))
    except Exception:
        return True
    return is_complete or not has_other_valid_token


class AnswerCompleteStop(Exception):
    """Generation-complete signal, NOT a failure: the final answer span is
    finished, so generation can stop early (CRANE-style answer stopping).
    Raised by the per-step hooks when answer early-stop is enabled and caught
    in run_crane_csd, which returns the output-so-far through the normal
    scoring path. Must never reach the evaluator's per-example except-block
    (that path scores the example as a total failure with empty output)."""


def _answer_complete(text: str) -> bool:
    """True when the output contains the FINISHED final answer: 'final answer'
    (case-insensitive) followed by a complete <<...>> span whose closing '>>'
    is followed by a non-continuation character. The lookahead matters:
    strategies often close tiny spans mid-expression ("<<n1>> + <<mult>>"),
    and stopping at the first '>>' would freeze a fragment as the last span
    the grader extracts. Spans that closed BEFORE the phrase never count."""
    idx = text.lower().rfind("final answer")
    if idx == -1:
        return False
    tail = text[idx:]
    open_pos = tail.find("<<")
    if open_pos == -1:
        return False
    last_close = tail.rfind(">>")
    if last_close < open_pos + 2:
        return False
    after = tail[last_close + 2:].lstrip(" \t")
    if not after:
        return False  # not decidable yet — wait for the next token
    return after[0] not in "+-*/%(<"


class _TensorizedLMBase:
    """Shared tensorized behavior for Dafny LM wrappers."""

    def __init__(self, _dafny, tokenizer, tokens, tids, logits_device: torch.device | str = "cpu"):
        self._dafny = _dafny
        self.tokenizer = tokenizer
        self._Tokens = tokens
        self._token_ids = tids
        self._structured_prompt = None
        self.model_name: str | None = None
        self._generation_stop_token_ids = _default_generation_stop_token_ids(tokenizer)
        self._generation_token_ids: list[int] = []
        self._generation_transaction_checkpoints: dict[str, list[int]] = {}
        self._generation_alignment_removed_token_ids: list[int] = []
        self._active_generation_checkpoint_key: str | None = None
        self._active_generation_checkpoint_prefix: str | None = None
        self._active_generation_checkpoint_snapshot: list[int] | None = None
        self._generation_transaction_rollback_restored = False
        self.instruction_text = ""
        self._task_guidance = _TaskGuidanceState()
        # Chat-template scaffolding so AppendTaskGuidance can inject the
        # guidance INTO the last user message (re-templating) instead of
        # appending it after the trailing assistant-generation marker.
        self._chat_messages: list[dict] | None = None
        self._logits_device = torch.device(logits_device)

        n = len(tids)
        self.Logits = _LogitsProxy(n, list(tids))
        self._logits_tensor = torch.zeros(n, dtype=torch.float32, device=self._logits_device)
        self._token_ids_tensor = torch.tensor(tids, dtype=torch.long, device=self._logits_device)
        self._full_logits: torch.Tensor | None = None
        # Self-consistency: when > 0, the constrained-span selection
        # (ChooseNextToken) samples from softmax(logits / T) instead of argmax,
        # so running the SAME strategy k times yields k DIFFERENT decodes to vote
        # over. Default 0.0 => exact argmax behavior, byte-for-byte unchanged for
        # every benchmark that does not opt in via this env var.
        self._constrained_temperature = float(
            os.environ.get("CSD_CONSTRAINED_TEMPERATURE", "0.0")
        )
        self._generate_count = 0
        self._token_id_to_str: dict[int, str] = {}
        self._runtime_deadline: float | None = None
        # CRANE-style answer early stop (flag-gated, default OFF): when
        # enabled, per-step hooks raise AnswerCompleteStop once the output
        # contains a finished final-answer span; the tokens generated so far
        # are stashed here for run_crane_csd to return as the output.
        self._answer_early_stop_enabled: bool = False
        self._early_stop_tokens: list[str] | None = None

        self._last_generation_evidence: dict[str, Any] | None = None
        self._last_prompt_contract: dict[str, Any] | None = None
        # Prefix-cache short-circuit state.
        self._last_full_prompt: str | None = None
        self._logits_dirty: bool = False
        self._cache_hits: int = 0

        # Persistent tried-token penalty (faithful IterGen recurrence_penalty
        # analog). Maps full_prompt -> {constrained-subset-index: times_tried}.
        # A grounding rollback registers the first token of the failed unit here;
        # GenerateLogits then re-applies the down-weight EVERY time it regenerates
        # at that prefix, so a greedy rollback diverges to a different token
        # instead of looping on the same out-of-schema name. The map is empty for
        # any run that never rolls back, so decoding is byte-identical when
        # grounding never fires. Our logits are vLLM LOG-probs (<=0), so the
        # IterGen "score *= 0.3" is applied as "logprob += count * ln(0.3)"
        # (same intent: reduce the tried token's probability; cumulative so
        # repeated tries are guaranteed to eventually demote the token within the
        # retry budget). Factor 1.0 disables. Keyed by full_prompt (which embeds
        # the per-example instruction_text), so cross-example contamination is
        # impossible; cleared on instruction_text change to bound memory.
        self._tried_token_penalties: dict[str, dict[int, int]] = {}
        self._penalty_instruction_key: str | None = None
        self._recurrence_penalty = float(
            os.environ.get("CSD_RECURRENCE_PENALTY", "0.3")
        )
        # IterGen-faithful flat mode: when ON, each distinct previously-tried token
        # is down-weighted by ln(factor) EXACTLY ONCE regardless of how many times
        # it was re-tried (IterGen multiplies the fresh logits by 0.3 once per pass,
        # no compounding). Default OFF = our cumulative ln(factor)*count behavior,
        # which is strictly stronger on a stubborn high-gap token. Gated so the
        # default decode for every other cell is byte-identical.
        self._recurrence_flat = os.environ.get(
            "CSD_RECURRENCE_FLAT", ""
        ).strip().lower() not in ("", "0", "false", "no")
        if _GROUNDING_LOG.isEnabledFor(logging.INFO):
            _GROUNDING_LOG.info(
                "[recurrence] penalty mode=%s factor=%.3f",
                "flat(itergen)" if self._recurrence_flat else "cumulative",
                self._recurrence_penalty,
            )

        self._token_str_to_indices = {}
        for i in range(n):
            token_str = self._to_str(tokens[i])
            self._token_str_to_indices.setdefault(token_str, []).append(i)

        self._oracle_trie_root = _OracleTrieNode()
        self._oracle_node = self._oracle_trie_root
        self._oracle_depth = 0
        self._oracle_context_ids: list[int] = []
        self._oracle_recompute_needed = False
        self._oracle_instruction_key: str | None = None
        self._oracle_pending_reject_id: int | None = None
        self._decode_trace_token_ids: set[int] = set()

    def _to_str(self, obj):
        if isinstance(obj, str):
            return obj
        try:
            return "".join(obj[i] for i in range(len(obj)))
        except Exception:
            return str(obj)

    def SetRuntimeDeadline(self, deadline: float | None):
        self._runtime_deadline = deadline

    def ClearRuntimeDeadline(self):
        self._runtime_deadline = None

    def _check_runtime_deadline(self):
        if self._runtime_deadline is not None and time.monotonic() >= self._runtime_deadline:
            raise TimeoutError("CSD example exceeded its runtime budget")

    def SetAnswerEarlyStop(self, enabled: bool):
        self._answer_early_stop_enabled = bool(enabled)
        self._early_stop_tokens = None

    def _check_answer_early_stop(self, input_prefix):
        if not self._answer_early_stop_enabled:
            return
        tokens = [self._to_str(input_prefix[i]) for i in range(len(input_prefix))]
        if _answer_complete("".join(tokens)):
            self._early_stop_tokens = tokens
            raise AnswerCompleteStop("final answer span complete — stopping generation")

    def _prefix_text(self, prefix) -> str:
        return "".join(self._to_str(prefix[i]) for i in range(len(prefix)))

    def _full_input_ids(self, input_prefix) -> list[int]:
        """Instruction ids + each emitted token's vocab id (IterGen-style).

        Do not ``encode(instruction + "".join(prefix))``: that re-merges the
        prompt tail with whitespace pieces (Spider ex0: trailing space + tabs
        became one ``' \t\t'`` token and tab-trapped decode).
        """
        pieces = [self._to_str(input_prefix[i]) for i in range(len(input_prefix))]
        # Prefer the exact vocab id we sampled when the piece is in our table.
        ids = list(self.tokenizer.encode(self.instruction_text, add_special_tokens=False))
        for piece in pieces:
            indices = self._token_str_to_indices.get(piece) if hasattr(self, "_token_str_to_indices") else None
            if indices:
                ids.append(int(self._token_ids_tensor[indices[0]].item()))
            else:
                ids.extend(self.tokenizer.encode(piece, add_special_tokens=False))
        return ids

    def ResetTaskGuidance(self):
        """Clear accepted guidance and every per-example prompt/rebuild cache."""
        self._task_guidance.reset()
        self._structured_prompt = None
        self._chat_messages = None
        self.model_name = None
        self.instruction_text = ""
        self._last_generation_evidence = None
        self._generation_alignment_removed_token_ids = []
        self._generation_token_ids = []
        self._generation_transaction_checkpoints = {}
        self._active_generation_checkpoint_key = None
        self._active_generation_checkpoint_prefix = None
        self._active_generation_checkpoint_snapshot = None
        self._generation_transaction_rollback_restored = False
        self._last_full_prompt = None
        self._tried_token_penalties.clear()
        self._penalty_instruction_key = None
        self._grounding_cache_key = None
        self._grounding_cache_val = set()
        self._logits_dirty = True

    def _maybe_reset_penalties(self) -> None:
        """Drop the tried-token penalty map when the example (instruction_text)
        changes, so penalties never leak across examples and memory stays bounded."""
        it = self.instruction_text or ""
        if self._penalty_instruction_key != it:
            self._tried_token_penalties.clear()
            self._penalty_instruction_key = it
        self._maybe_reset_oracle_trie()

    def _maybe_reset_oracle_trie(self) -> None:
        it = self.instruction_text or ""
        if self._oracle_instruction_key != it:
            self.ResetOracleTrie()
            self._oracle_instruction_key = it

    def ResetOracleTrie(self) -> None:
        self._oracle_trie_root = _OracleTrieNode()
        self._oracle_node = self._oracle_trie_root
        self._oracle_depth = 0
        self._oracle_context_ids = []
        self._oracle_recompute_needed = False
        self._oracle_pending_reject_id = None

    def _primary_vocab_id_for_token_str(self, token_str: str) -> int:
        indices = self._token_str_to_indices.get(token_str) or []
        if indices:
            return int(self._token_ids_tensor[indices[0]].item())
        return int(self.tokenizer.encode(token_str, add_special_tokens=False)[-1])

    def _dafny_prefix_slice(self, prefix, end: int):
        n = len(prefix)
        if end <= 0:
            return self._dafny.SeqWithoutIsStrInference([])
        return self._dafny.SeqWithoutIsStrInference([prefix[i] for i in range(min(end, n))])

    def _oracle_sync_prefix(self, parser, prefix):
        node = self._oracle_trie_root
        ids: list[int] = []
        n = len(prefix)
        for i in range(n):
            tok = prefix[i]
            tok_str = self._to_str(tok)
            sub = self._dafny_prefix_slice(prefix, i + 1)
            if not parser.IsValidPrefix(sub):
                bad_id = self._primary_vocab_id_for_token_str(tok_str)
                return node, ids, bad_id, False
            tid = self._primary_vocab_id_for_token_str(tok_str)
            if tid not in node.children:
                node.children[tid] = _OracleTrieNode(parent=node)
            node = node.children[tid]
            ids.append(tid)
        return node, ids, None, True

    def _oracle_apply_grammar_mask_to_log_theta(self, log_theta, parser, prefix, eos_token):
        full_mask = self._parser_full_mask(parser, prefix)
        if full_mask.numel() < log_theta.shape[1]:
            pad = torch.zeros(log_theta.shape[1] - full_mask.numel(), dtype=torch.bool, device=full_mask.device)
            full_mask = torch.cat((full_mask, pad))
        elif full_mask.numel() > log_theta.shape[1]:
            full_mask = full_mask[: log_theta.shape[1]]
        eos_indices = self._token_indices_for_token(eos_token)
        if eos_indices:
            for idx in eos_indices:
                if 0 <= idx < full_mask.numel():
                    full_mask[idx] = True
        log_theta[0, ~full_mask.to(log_theta.device)] = float("-inf")

    def _oracle_recompute_in_trie(self) -> None:
        node = self._oracle_node
        depth = self._oracle_depth
        context = self._oracle_context_ids
        while depth > 0:
            new_log_theta = torch.log(torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum())
            depth -= 1
            node = node.parent
            node.log_theta[0, context[depth]] = new_log_theta
        self._oracle_recompute_needed = False

    def CarsAdvanceTrieAndAdjustScores(self, parser, prefix, constrainFirst) -> bool:
        self._maybe_reset_oracle_trie()
        if self._full_logits is None:
            raise RuntimeError("Must call GenerateLogits before CarsAdvanceTrieAndAdjustScores")
        node, ids, bad_id, ok = self._oracle_sync_prefix(parser, prefix)
        self._oracle_node = node
        self._oracle_depth = len(ids)
        self._oracle_context_ids = ids
        if not ok:
            self._oracle_pending_reject_id = bad_id
            return False
        self._oracle_pending_reject_id = None
        is_root = len(prefix) == 0
        constrain_first = bool(constrainFirst)
        if node.raw_logprob is None:
            raw = torch.log_softmax(self._full_logits.detach().float(), dim=0).cpu()
            node.raw_logprob = raw.unsqueeze(0)
            node.log_theta = torch.zeros(1, raw.shape[0])
            adjust_scores = is_root and constrain_first
            eos = self.tokenizer.eos_token or "<|endoftext|>"
            self._oracle_apply_grammar_mask_to_log_theta(node.log_theta, parser, prefix, eos)
            self._oracle_recompute_needed = True
        else:
            adjust_scores = True
        if adjust_scores:
            delta = node.log_theta.to(self._full_logits.device, non_blocking=True)[0]
            if delta.numel() < self._full_logits.numel():
                pad = torch.zeros(self._full_logits.numel() - delta.numel(), device=self._full_logits.device, dtype=self._full_logits.dtype)
                delta = torch.cat((delta, pad))
            elif delta.numel() > self._full_logits.numel():
                delta = delta[: self._full_logits.numel()]
            self._full_logits = self._full_logits + delta
            self._logits_tensor = self._full_logits[self._token_ids_tensor]
            self.Logits.update_tensors(self._logits_tensor, self._full_logits)
            self._logits_dirty = True
        return True

    def RejectLastInTrie(self) -> None:
        reject_id = self._oracle_pending_reject_id
        discard_recorded_token = False
        # CARS generation_failed eliminates generated_tokens[-1] (the bad token).
        # After CarsTrieStep samples an invalid next, pending is unset and
        # context_ids still ends at the prior valid prefix — prefer the just-
        # sampled unconstrained id so the trie matches CARS.
        if reject_id is None:
            reject_id = getattr(self, "_last_unconstrained_token_id", None)
            discard_recorded_token = reject_id is not None
            if discard_recorded_token:
                self._last_unconstrained_token_id = None
        if reject_id is None and self._oracle_context_ids:
            reject_id = self._oracle_context_ids[-1]
            discard_recorded_token = True
        if reject_id is None:
            return
        if discard_recorded_token:
            self._discard_last_generated_token_id(reject_id)
        node = self._oracle_node
        if node.log_theta is None:
            width = node.raw_logprob.shape[1] if node.raw_logprob is not None else 1
            node.log_theta = torch.zeros(1, width)
        node.log_theta[0, reject_id] = float("-inf")
        self._decode_trace_token_ids.add(int(reject_id))
        self._oracle_recompute_in_trie()
        self._oracle_pending_reject_id = None

    def ApplyTraceRecurrence(self, factor) -> None:
        factor_f = float(factor)
        if factor_f >= 1.0 or not self._decode_trace_token_ids or self._full_logits is None:
            return
        log_factor = math.log(factor_f)
        for tid in self._decode_trace_token_ids:
            if 0 <= tid < self._full_logits.numel():
                self._full_logits[tid] += log_factor
        self._logits_tensor = self._full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)
        self._logits_dirty = True

    def PenalizeTriedTokenAt(self, prefix, token):
        """Dafny extern: persistently down-weight `token` as a next-token at
        position `prefix`, so a later regeneration at this position (after a
        rollback) picks a DIFFERENT token instead of looping. Records the
        constrained-subset index of the token; the actual down-weight is
        (re)applied by GenerateLogits every time it regenerates at this prefix.
        Has NO effect on the current logits (only future regenerations).

        Faithful analog of IterGen's recurrence_penalty (which down-weights the
        previously-tried next-token at a rolled-back trace position). Fair: uses
        only previously-tried tokens — no gold labels, no execution feedback.
        """
        self._restore_generation_transaction()
        indices = self._token_indices_for_token(token)
        if not indices:
            # Token not in the constrained subset vocab — nothing to penalize.
            # (Avoids MaskToken's vocab-id/subset-index ambiguity entirely.)
            return
        self._maybe_reset_penalties()
        full_prompt = self.instruction_text + self._prefix_text(prefix)
        bucket = self._tried_token_penalties.setdefault(full_prompt, {})
        for idx in indices:
            bucket[idx] = bucket.get(idx, 0) + 1
        # Invalidate the prefix cache so the very next GenerateLogits at this
        # prefix recomputes fresh and re-applies the (now updated) penalty.
        self._logits_dirty = True
        _GROUNDING_LOG.info(
            "[recurrence] penalize subset_idx=%s at prefix_len=%d; counts now=%s",
            indices, len(prefix), {i: bucket[i] for i in indices},
        )

    def _apply_recurrence_penalty(self, full_prompt: str) -> None:
        """Re-apply persistent tried-token down-weight at a fresh prefix."""
        factor = self._recurrence_penalty
        if factor >= 1.0:
            return
        bucket = self._tried_token_penalties.get(full_prompt)
        if not bucket:
            return
        log_factor = math.log(factor)
        n = self._logits_tensor.numel()
        for idx, count in bucket.items():
            if 0 <= idx < n:
                weight = 1 if self._recurrence_flat else count
                self._logits_tensor[idx] += log_factor * weight

    def set_chat_messages(self, chat_messages: list[dict]) -> None:
        """Register chat messages for safe last-user-turn guidance rebuilding."""
        self._chat_messages = [dict(message) for message in chat_messages]
        self._structured_prompt = None

    def set_structured_prompt(self, prompt, *, model_name: str | None = None) -> None:
        """Register immutable benchmark prompt parts for safe guidance rebuilding."""
        self._structured_prompt = prompt
        self.model_name = model_name or getattr(prompt, "model_name", None)
        self._chat_messages = None

    def AppendTaskGuidance(self, guidance):
        """Rebuild the active prompt with first-call guidance before decoding."""
        from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptRenderError

        if self._task_guidance.accepted_guidance is not None:
            return
        text = self._task_guidance._coerce_guidance(self._to_str(guidance))
        if not text:
            return
        if self._structured_prompt is not None:
            candidate = self._structured_prompt.with_guidance(text)
            try:
                render_with_contract = getattr(
                    candidate, "render_for_model_with_contract", None
                )
                if callable(render_with_contract):
                    rendered, prompt_contract = render_with_contract(
                        self.tokenizer,
                        model_name=self.model_name,
                    )
                else:
                    rendered = candidate.render_for_model(
                        self.tokenizer,
                        model_name=self.model_name,
                    )
                    prompt_contract = dict(self._last_prompt_contract or {})
                    prompt_contract.setdefault("renderer", "structured")
                    prompt_contract["render_succeeded"] = True
                    prompt_contract["prompt_chars"] = len(rendered)
            except SpiderPromptRenderError:
                raise
            except Exception as exc:
                _GROUNDING_LOG.error(
                    "[guidance] structured prompt rebuild failed type=%s",
                    type(exc).__name__,
                )
                raise SpiderPromptRenderError(
                    "Task guidance could not rebuild the registered structured prompt"
                ) from exc
            self._structured_prompt = candidate
            self.instruction_text = rendered
            self._last_prompt_contract = prompt_contract
            self._task_guidance.accepted_guidance = text
            _GROUNDING_LOG.info(
                "[spider-prompt] guidance_rebuild mode=structured model_family=%s "
                "guidance_chars=%d rendered_chars=%d",
                self.model_name or "unknown",
                len(text),
                len(rendered),
            )
            return
        if self._chat_messages is not None:
            messages = [dict(message) for message in self._chat_messages]
            last_user_idx = next(
                (
                    index
                    for index in range(len(messages) - 1, -1, -1)
                    if messages[index].get("role") == "user"
                ),
                None,
            )
            if last_user_idx is None:
                _GROUNDING_LOG.error(
                    "[guidance] registered chat prompt has no user message"
                )
                raise SpiderPromptRenderError(
                    "Task guidance requires a user message in the registered chat prompt"
                )
            existing = messages[last_user_idx].get("content", "") or ""
            messages[last_user_idx] = dict(messages[last_user_idx])
            messages[last_user_idx]["content"] = (
                f"{existing}\n\n{self._task_guidance.HEADER}\n{text}"
            )
            identity = (
                self.model_name or ""
            ).lower().replace("-", "_").replace(".", "_")
            template_fallback = False
            try:
                if "qwen3_5" in identity or "qwen35" in identity:
                    rendered = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    try:
                        rendered = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        template_fallback = True
                        rendered = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
            except SpiderPromptRenderError:
                raise
            except Exception as exc:
                _GROUNDING_LOG.error(
                    "[guidance] chat prompt rebuild failed type=%s",
                    type(exc).__name__,
                )
                raise SpiderPromptRenderError(
                    "Task guidance could not rebuild the registered chat prompt"
                ) from exc
            prompt_contract = dict(self._last_prompt_contract or {})
            prompt_contract.update(
                {
                    "renderer": prompt_contract.get("renderer", "legacy"),
                    "mode": "chat",
                    "template_used": True,
                    "raw_prompt": False,
                    "chat_message_count": len(messages),
                    "user_message_count": sum(
                        1 for message in messages if message.get("role") == "user"
                    ),
                    "add_generation_prompt": True,
                    "enable_thinking": None if template_fallback else False,
                    "template_fallback": template_fallback,
                    "render_succeeded": True,
                    "prompt_chars": len(rendered),
                }
            )
            self._chat_messages = messages
            self.instruction_text = rendered
            self._last_prompt_contract = prompt_contract
            self._task_guidance.accepted_guidance = text
            _GROUNDING_LOG.info(
                "[spider-prompt] guidance_rebuild mode=chat model_family=%s "
                "guidance_chars=%d rendered_chars=%d",
                self.model_name or "unknown",
                len(text),
                len(rendered),
            )
            return
        _GROUNDING_LOG.error(
            "[guidance] no registered prompt state; refusing to apply task guidance"
        )
        raise SpiderPromptRenderError(
            "Task guidance requires a registered structured or chat prompt"
        )

    @property
    def task_guidance(self) -> str | None:
        return self._task_guidance.accepted_guidance

    def SpanGrounded(self, text):
        """Dafny extern: is every identifier-like token in `text` present in the
        support set derived from the prompt? True when no support set is found.

        Fair: the support set comes only from the prompt context (the same
        information visible in the prompt to any baseline), never from execution
        feedback or gold labels.
        """
        if not isinstance(text, str):
            text = self._to_str(text)
        support = self._grounding_support_set()
        if not support:
            return True
        cands = _candidate_identifiers(text)
        bad = [c for c in cands if c not in support]
        grounded = len(bad) == 0
        _GROUNDING_LOG.info(
            "[grounding] span=%r support_n=%d cand_n=%d bad=%s grounded=%s",
            (text or "")[:120], len(support), len(cands), bad[:8], grounded,
        )
        return grounded

    def FirstUngroundedIdentifierTokenIdx(self, unitTokens):
        """Dafny extern: index of the token holding the FIRST out-of-schema
        identifier in `unitTokens`. Returns `(found, idx)`.

        Renders `unitTokens` by concatenation (matching RenderPrefix), then reuses
        the EXACT membership signal of `SpanGrounded` (same support set, same
        `_candidate_identifiers` filtering) and additionally reports WHERE the
        first bad identifier sits, so a rollback can penalize that token rather
        than the unit's first token. `found=False, idx=0` when fully grounded or
        when no support set was parsed.

        Fair: support set comes only from the prompt context, never from gold
        labels or execution feedback — identical provenance to SpanGrounded.
        """
        n = len(unitTokens)
        token_strs = [self._to_str(unitTokens[i]) for i in range(n)]
        support = self._grounding_support_set()
        found, idx = _first_ungrounded_token_idx(token_strs, support)
        if support:  # only log for SQL-like prompts that have a parsed schema
            if found:
                _GROUNDING_LOG.info(
                    "[grounding] first-ungrounded token_idx=%d of %d; text=%r",
                    idx, n, ("".join(token_strs))[:120],
                )
            else:
                _GROUNDING_LOG.info(
                    "[grounding] unit fully grounded (n=%d tokens); text=%r",
                    n, ("".join(token_strs))[:120],
                )
        return (found, idx)

    def _grounding_support_set(self) -> set:
        """Return schema identifiers cached for the current instruction text."""
        instruction = self.instruction_text or ""
        if getattr(self, "_grounding_cache_key", None) == instruction:
            return self._grounding_cache_val
        support = _parse_schema_support(instruction)
        self._grounding_cache_key = instruction
        self._grounding_cache_val = support
        _GROUNDING_LOG.info(
            "[grounding] parsed %d support identifiers for current example",
            len(support),
        )
        return support

    def _token_str_from_id(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._token_id_to_str.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode([token_id])
            self._token_id_to_str[token_id] = cached
        return cached

    def _dafny_prefix_from_token_strs(self, token_strs: list[str]):
        return self._dafny.SeqWithoutIsStrInference(
            [self._dafny.Seq(token) for token in token_strs]
        )

    def _token_strs_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self._token_str_from_id(token_id) for token_id in token_ids]

    def _reset_generation_transactions(self) -> None:
        """Clear per-example prefix checkpoints for sampled-ID provenance."""
        self._generation_alignment_removed_token_ids = []
        self._generation_transaction_checkpoints = {}
        self._active_generation_checkpoint_key = None
        self._active_generation_checkpoint_prefix = None
        self._active_generation_checkpoint_snapshot = None
        self._generation_transaction_rollback_restored = False

    def _align_generation_history_to_prefix(self, input_prefix) -> None:
        """Keep sampled-ID occurrences that still form the current CSD prefix."""
        if getattr(self, "_structured_prompt", None) is None:
            return
        history = [
            int(token_id) for token_id in getattr(self, "_generation_token_ids", [])
        ]
        stop_ids = self._generation_stop_ids()
        content = list(history)
        terminal: list[int] = []
        while content and content[-1] in stop_ids:
            terminal.insert(0, content.pop())
        expected_pieces = [
            self._to_str(input_prefix[index]) for index in range(len(input_prefix))
        ]
        expected_text = "".join(expected_pieces)
        retained: list[int] = []
        matched_indices: set[int] = set()
        cursor = 0
        for index, candidate in enumerate(content):
            candidate_text = self._token_str_from_id(candidate)
            if not candidate_text:
                continue
            match_at = expected_text.find(candidate_text, cursor)
            if match_at == -1:
                continue
            retained.append(candidate)
            matched_indices.add(index)
            cursor = match_at + len(candidate_text)
        removed_ids = [
            token_id
            for index, token_id in enumerate(content)
            if index not in matched_indices
        ]
        if not removed_ids:
            return
        self._generation_token_ids = retained + terminal
        self._generation_alignment_removed_token_ids.extend(removed_ids)
        _SPIDER_CONTRACT_LOG.info(
            "[spider-output-contract] prefix_alignment "
            "prefix_tokens=%d before_ids=%d after_ids=%d removed_ids=%d",
            len(expected_pieces),
            len(history),
            len(self._generation_token_ids),
            len(removed_ids),
        )

    def _begin_generation_transaction(self, input_prefix) -> None:
        """Checkpoint the current accepted IDs before generating at a prefix."""
        if getattr(self, "_structured_prompt", None) is None:
            return
        self._align_generation_history_to_prefix(input_prefix)
        checkpoints = getattr(self, "_generation_transaction_checkpoints", None)
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self._generation_transaction_checkpoints = checkpoints
        prefix_text = self._prefix_text(input_prefix)
        key = f"{self.instruction_text}\x00{prefix_text}"
        current = [int(token_id) for token_id in getattr(self, "_generation_token_ids", [])]
        previous_prefix = getattr(self, "_active_generation_checkpoint_prefix", None)
        previous_snapshot = checkpoints.get(key)
        rollback_already_restored = bool(
            getattr(self, "_generation_transaction_rollback_restored", False)
        )
        rollback_revisit = (
            previous_snapshot is not None
            and not rollback_already_restored
            and previous_prefix is not None
            and current != previous_snapshot
            and (
                prefix_text == previous_prefix
                or (
                    len(prefix_text) < len(previous_prefix)
                    and previous_prefix.startswith(prefix_text)
                )
            )
        )
        checkpoints[key] = current
        if rollback_revisit:
            _SPIDER_CONTRACT_LOG.info(
                "[spider-output-contract] transaction_checkpoint_replaced "
                "prefix_chars=%d previous_ids=%d current_ids=%d",
                len(prefix_text),
                len(previous_snapshot),
                len(current),
            )
        active_snapshot = current
        self._active_generation_checkpoint_key = key
        self._active_generation_checkpoint_prefix = prefix_text
        self._active_generation_checkpoint_snapshot = active_snapshot
        self._generation_transaction_rollback_restored = False

    def _restore_generation_transaction(self) -> None:
        """Restore the active pre-choice snapshot for a real rollback callback."""
        if getattr(self, "_structured_prompt", None) is None:
            return
        key = getattr(self, "_active_generation_checkpoint_key", None)
        checkpoints = getattr(self, "_generation_transaction_checkpoints", {})
        snapshot = getattr(self, "_active_generation_checkpoint_snapshot", None)
        if snapshot is None and key is not None:
            snapshot = checkpoints.get(key)
        if snapshot is None:
            return
        committed = [int(token_id) for token_id in snapshot]
        current = [int(token_id) for token_id in getattr(self, "_generation_token_ids", [])]
        self._generation_transaction_rollback_restored = True
        if current == committed:
            return
        self._generation_token_ids = committed
        _SPIDER_CONTRACT_LOG.info(
            "[spider-output-contract] callback_rollback from_ids=%d to_ids=%d",
            len(current),
            len(committed),
        )

    def _record_generated_token_ids(self, token_ids) -> None:
        if getattr(self, "_structured_prompt", None) is None:
            return
        if not hasattr(self, "_generation_token_ids"):
            self._generation_token_ids = []
        self._generation_token_ids.extend(int(token_id) for token_id in token_ids)

    def _discard_last_generated_token_id(self, token_id: int) -> None:
        """Undo one speculative token when CARS rejects or rolls it back."""
        generated = getattr(self, "_generation_token_ids", None)
        if not generated or int(generated[-1]) != int(token_id):
            return
        generated.pop()
        _SPIDER_CONTRACT_LOG.info(
            "[spider-output-contract] discarded_speculative_token committed_count=%d",
            len(generated),
        )

    def _reconcile_generation_evidence(self, scored_output: str) -> bool:
        """Validate the current committed history or fail closed."""
        if getattr(self, "_structured_prompt", None) is None:
            return True

        expected = str(scored_output)
        history = [int(token_id) for token_id in getattr(self, "_generation_token_ids", [])]
        stop_ids = self._generation_stop_ids()

        def _split_terminal(token_ids: list[int]) -> tuple[list[int], list[int]]:
            content = list(token_ids)
            terminal: list[int] = []
            while content and content[-1] in stop_ids:
                terminal.insert(0, content.pop())
            return content, terminal

        def _decode(token_ids: list[int]) -> str:
            try:
                return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))
            except TypeError:
                return str(self.tokenizer.decode(token_ids))

        content_history, terminal_ids = _split_terminal(history)
        if _decode(content_history) != expected:
            _SPIDER_CONTRACT_LOG.error(
                "[spider-output-contract] evidence_reconcile_failed history_ids=%d "
                "scored_chars=%d",
                len(content_history),
                len(expected),
            )
            return False

        self._generation_token_ids = content_history + terminal_ids
        _SPIDER_CONTRACT_LOG.info(
            "[spider-output-contract] evidence_reconciled committed_ids=%d "
            "removed_speculative_ids=%d terminal_ids=%d",
            len(content_history),
            0,
            len(terminal_ids),
        )
        return True

    def _generation_stop_ids(self) -> frozenset[int]:
        return _coerce_token_id_set(
            getattr(
                self,
                "_generation_stop_token_ids",
                _default_generation_stop_token_ids(self.tokenizer),
            )
        )

    def _prepare_generated_token_ids(self, token_ids) -> list[int]:
        """Retain raw IDs for stopping while preserving exact boundary evidence."""
        from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
            generation_token_evidence,
        )

        raw_ids = [int(token_id) for token_id in token_ids]
        evidence = generation_token_evidence(
            raw_ids,
            self.tokenizer,
            terminal_stop_token_ids=self._generation_stop_ids(),
        )
        self._last_generation_evidence = evidence
        removed_count = len(evidence["removed_terminal_token_ids"])
        return raw_ids[:-removed_count] if removed_count else raw_ids

    def _finalize_generation_evidence(self) -> dict[str, Any] | None:
        if getattr(self, "_structured_prompt", None) is None:
            self._last_generation_evidence = None
            return None
        from synthesis.evaluate.benchmarks.sql_spider.output_contract import (
            generation_token_evidence,
        )

        evidence = generation_token_evidence(
            getattr(self, "_generation_token_ids", []),
            self.tokenizer,
            terminal_stop_token_ids=self._generation_stop_ids(),
        )
        self._last_generation_evidence = evidence
        return evidence

    def _build_unconstrained_chunk_result(self, token_ids, open_span_token, eos_token, max_new_tokens: int):
        from synthesis.evaluate.benchmarks.sql_spider.prompts import SpiderPromptParts

        spider_contract_active = isinstance(
            getattr(self, "_structured_prompt", None), SpiderPromptParts
        )
        token_ids = [int(token_id) for token_id in token_ids]
        if not spider_contract_active:
            # GSM, SMILES, and other legacy unconstrained surfaces must see the
            # exact generated IDs, including EOS, so their stop flags stay intact.
            self._last_generation_evidence = None
        if max_new_tokens <= 0:
            if spider_contract_active:
                self._finalize_generation_evidence()
            return self._dafny_prefix_from_token_strs([]), False, False, 0

        open_span_str = self._to_str(open_span_token)
        eos_str = self._to_str(eos_token)
        chunk_tokens: list[str] = []
        chunk_text = ""
        steps_used = 0
        stopped_on_open = False
        stopped_on_eos = False
        stop_ids = self._generation_stop_ids() if spider_contract_active else frozenset()

        for raw_token_id in token_ids:
            if steps_used >= max_new_tokens:
                break
            if spider_contract_active and raw_token_id in stop_ids:
                # A declared generation stop is committed, then removed only
                # from the final scored decode by generation_token_evidence.
                self._record_generated_token_ids([raw_token_id])
                stopped_on_eos = True
                break

            token_str = self._token_str_from_id(raw_token_id)
            steps_used += 1
            if not spider_contract_active and token_str == eos_str:
                stopped_on_eos = True
                break

            candidate_text = chunk_text + token_str
            open_idx = candidate_text.find(open_span_str)
            if open_idx != -1:
                if spider_contract_active:
                    self._record_generated_token_ids([raw_token_id])
                prefix_text = candidate_text[:open_idx]
                chunk_tokens = self._token_strs_from_text(prefix_text)
                chunk_tokens.append(open_span_str)
                stopped_on_open = True
                break

            if spider_contract_active:
                self._record_generated_token_ids([raw_token_id])
            chunk_tokens.append(token_str)
            chunk_text = candidate_text

        if spider_contract_active:
            evidence = self._finalize_generation_evidence()
            _SPIDER_CONTRACT_LOG.info(
                "[spider-output-contract] token-boundary committed_ids=%d "
                "removed_terminal_token_count=%d",
                len(evidence["raw_token_ids"]) if evidence else 0,
                len(evidence["removed_terminal_token_ids"]) if evidence else 0,
            )

        return self._dafny_prefix_from_token_strs(chunk_tokens), stopped_on_open, stopped_on_eos, steps_used

    def IdToLogit(self, id_):
        with _timed("IdToLogit"):
            return self._dafny.BigRational(self._logits_tensor[id_].item())

    def MaskToken(self, token):
        with _timed("MaskToken"):
            self._restore_generation_transaction()
            # All-index: a runtime "token" is tokenizer.decode([id]), so two
            # vocab ids can decode to the SAME string. Masking only TokenToId's
            # first match leaves duplicate copies samplable, which defeats
            # DeadEndAvoidingStep's resample loop. Mask every id for the string.
            # On ASCII grammars each token has exactly one id, so this is a no-op.
            indices = self._token_indices_for_token(token)
            if not indices:
                indices = [self.TokenToId(token)]
            for token_id in indices:
                self._logits_tensor[token_id] = -1e9
            self._logits_dirty = True

    def IsMasked(self, token):
        with _timed("IsMasked"):
            # All-index: the string is masked (un-samplable) only when EVERY id
            # that decodes to it is masked. Single-id ASCII tokens are unchanged.
            indices = self._token_indices_for_token(token)
            if not indices:
                indices = [self.TokenToId(token)]
            return all(self._logits_tensor[i].item() == -1e9 for i in indices)

    def _finalize_full_logits(self, full_logits: torch.Tensor) -> None:
        full_logits = full_logits.float().to(self._logits_device)
        self._full_logits = full_logits
        self._logits_tensor = full_logits[self._token_ids_tensor]
        self.Logits.update_tensors(self._logits_tensor, self._full_logits)

    def _sample_full_token_id(self) -> int:
        if self._full_logits is None:
            raise RuntimeError("Must call GenerateLogits before sampling unconstrained tokens")

        # Default greedy (argmax) to match CRANE / IterGen do_sample=False.
        # Multinomial(softmax) made unconstrained CoT diverge from frozen
        # baselines at the first token (before any grammar mask applies).
        temperature = float(os.environ.get("CSD_UNCONSTRAINED_TEMPERATURE", "0.0"))
        if temperature <= 0.0:
            # Exact logit ties (common in fp16): torch.argmax picks the lowest
            # index. On GSM ex0 that picks token " <<" over " $\\" at equal
            # 22.875 and opens constrained early. Prefer a tied token that does
            # not contain the CRANE start marker "<<", else the highest tied id.
            logits = self._full_logits
            # IterGen opportunistic SoftConstrained: never accept EOS on the first
            # unconstrained peek (HF stopping is separate; grammar mask fallback
            # also clears EOS). Otherwise empty-prefix Spider picks <|im_end|>.
            if os.environ.get("CSD_ITERGEN_OPPORTUNISTIC", "0") == "1":
                logits = logits.clone()
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None and 0 <= int(eos_id) < logits.numel():
                    logits[int(eos_id)] = -1e9
                for tid in (151643, 151644, 151645, 151646):
                    if tid < logits.numel():
                        logits[tid] = -1e9
            max_val = logits.max()
            tied = (logits == max_val).nonzero(as_tuple=False).flatten()
            if tied.numel() == 1:
                return int(tied[0].item())
            # H9: Only intervene when a CRANE start-marker token is among the ties;
            # otherwise keep torch.argmax semantics (lowest tied id). Best parity
            # so far: 1/5 substantial with 4/5 byte-identical (ex3 remaining).
            pieces = {int(tid): self.tokenizer.decode([int(tid)]) for tid in tied.tolist()}
            if not any("<<" in s for s in pieces.values()):
                return int(tied.min().item())
            best = None
            for tid, piece in pieces.items():
                if "<<" in piece:
                    continue
                if best is None or tid > best:
                    best = tid
            if best is not None:
                return best
            return int(tied.max().item())

        probs = torch.softmax(self._full_logits / temperature, dim=0)
        if torch.isnan(probs).any() or torch.sum(probs).item() <= 0.0:
            return int(self._full_logits.argmax().item())
        chosen = int(torch.multinomial(probs, num_samples=1).item())
        return chosen

    def _finalize_from_logprob_dict(self, logprob_dict: dict[int, Any]) -> None:
        # vLLM returns next-token logprobs as a Python dict. We previously fetched
        # the *full* vocab (logprobs=-1) and looped once per vocab entry — 152k
        # Python objects + a 152k-step fill loop per step, dominating runtime.
        #
        # Now we ask for top-K only (see VLLM_TOPK_LOGPROBS) and vectorize the
        # dict -> tensor conversion. Missing token IDs are filled with -1e9,
        # matching the Dafny invariant `Logits[i] >= -1e9` (used elsewhere for
        # masking). For argmax / additive-transform semantics this is equivalent
        # to "masked out" — tokens that never made it into the top-K are treated
        # as un-selectable, which is the correct semantic for constrained
        # decoding (any tail-distribution token is effectively noise anyway).
        if self._token_ids_tensor.numel() > 0:
            token_ids_max_self = int(self._token_ids_tensor.max().item())
        else:
            token_ids_max_self = 0

        if logprob_dict:
            ids_list = [int(tid) for tid in logprob_dict.keys()]
            logprobs_list = [float(info.logprob) for info in logprob_dict.values()]
            max_token_id = max(max(ids_list), token_ids_max_self)
        else:
            ids_list = []
            logprobs_list = []
            max_token_id = token_ids_max_self

        full_scores = torch.full(
            (max_token_id + 1,),
            -1e9,
            dtype=torch.float32,
            device=self._logits_device,
        )
        if ids_list:
            ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=self._logits_device)
            scores_tensor = torch.tensor(logprobs_list, dtype=torch.float32, device=self._logits_device)
            full_scores.scatter_(0, ids_tensor, scores_tensor)

        self._finalize_full_logits(full_scores)

    def ChooseNextToken(self):
        with _timed("ChooseNextToken"):
            best_idx = self._select_constrained_index()
            if getattr(self, "_structured_prompt", None) is not None:
                self._record_generated_token_ids(
                    [int(self._token_ids_tensor[best_idx].item())]
                )
            return self._Tokens[best_idx]

    def _select_constrained_index(self) -> int:
        """Pick an index into the masked constrained-subset logits.

        T <= 0  -> argmax (today's exact behavior; grammar mask already applied,
                   invalid tokens sit at -1e9).
        T  > 0  -> sample from softmax(logits / T). Masked (-1e9) tokens get ~0
                   probability, so the grammar is still respected; only the choice
                   AMONG valid tokens becomes stochastic. Falls back to argmax on
                   a degenerate distribution (nan or non-positive mass).
        """
        temperature = self._constrained_temperature
        if temperature <= 0.0:
            return int(self._logits_tensor.argmax().item())
        probs = torch.softmax(self._logits_tensor / temperature, dim=0)
        if torch.isnan(probs).any() or torch.sum(probs).item() <= 0.0:
            return int(self._logits_tensor.argmax().item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def ChooseNextTokenUnconstrained(self):
        with _timed("ChooseNextTokenUnconstrained"):
            if self._full_logits is None:
                raise RuntimeError("Must call GenerateLogits before ChooseNextTokenUnconstrained")
            sampled_idx = self._sample_full_token_id()
            # Used by RejectLastInTrie when CARS aborts on an invalid sampled next
            # (pending_reject_id is only set when CarsAdvanceTrie sync fails).
            self._last_unconstrained_token_id = int(sampled_idx)
            # #region agent log
            try:
                import json as _dbg_json, time as _dbg_time
                from pathlib import Path as _DbgPath
                _tok = self.tokenizer.decode([int(sampled_idx)])
                _n = getattr(self, "_dbg_unconst_n", 0)
                self._dbg_unconst_n = _n + 1
                # Log first 8 unconstrained picks and every bang/id0 (cap 40).
                _bang_n = getattr(self, "_dbg_bang_n", 0)
                _is_bang = (int(sampled_idx) == 0) or (_tok.strip() == "!")
                if _is_bang:
                    self._dbg_bang_n = _bang_n + 1
                if _n < 8 or (_is_bang and _bang_n < 40):
                    _fl = self._full_logits
                    _top = []
                    if _fl is not None:
                        import torch as _torch
                        _v, _i = _torch.topk(_fl, k=min(5, int(_fl.numel())))
                        for _vv, _ii in zip(_v.tolist(), _i.tolist()):
                            _top.append({"id": int(_ii), "logit": float(_vv), "tok": self.tokenizer.decode([int(_ii)])})
                    _payload = {
                        "sessionId": "d0e277",
                        "runId": "post-instrument",
                        "hypothesisId": "P",
                        "location": "model_utils.py:ChooseNextTokenUnconstrained",
                        "message": "unconst_pick",
                        "data": {
                            "n": _n,
                            "sampled_idx": int(sampled_idx),
                            "tok": _tok,
                            "is_bang_or_id0": _is_bang,
                            "full_logits_finite": bool(_fl is not None and bool(_fl.isfinite().all().item())),
                            "full_logits_max": float(_fl.max().item()) if _fl is not None else None,
                            "full_logits_min": float(_fl.min().item()) if _fl is not None else None,
                            "top5": _top,
                        },
                        "timestamp": int(_dbg_time.time() * 1000),
                    }
                    for _logp in (
                        _DbgPath("/home/aadivyar/csd-generation/logs/debug-d0e277.log"),
                        _DbgPath("/Users/aadivyar/Documents/Research/dynamic csd gen clean/.cursor/debug-d0e277.log"),
                    ):
                        try:
                            _logp.parent.mkdir(parents=True, exist_ok=True)
                            with _logp.open("a") as _lf:
                                _lf.write(_dbg_json.dumps(_payload) + "\n")
                        except Exception:
                            pass
            except Exception:
                pass
            # #endregion
            if getattr(self, "_structured_prompt", None) is not None:
                self._record_generated_token_ids([sampled_idx])
            return self._dafny.Seq(self.tokenizer.decode([sampled_idx]))

    def _token_indices_for_token(self, token) -> list[int]:
        token_str = self._to_str(token)
        return list(self._token_str_to_indices.get(token_str, []))

    def _expand_full_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        if self._full_logits is None:
            raise RuntimeError("Must call GenerateLogits before applying parser masks")
        full_mask = full_mask.to(dtype=torch.bool, device=self._full_logits.device)
        if full_mask.numel() < self._full_logits.numel():
            padding = torch.zeros(
                self._full_logits.numel() - full_mask.numel(),
                dtype=torch.bool,
                device=self._full_logits.device,
            )
            full_mask = torch.cat((full_mask, padding))
        elif full_mask.numel() > self._full_logits.numel():
            full_mask = full_mask[: self._full_logits.numel()]
        return full_mask

    def _subset_mask_from_full_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        subset = full_mask[self._token_ids_tensor.to(full_mask.device)]
        return subset.to(dtype=torch.bool, device=self._logits_tensor.device)

    def _parser_full_mask(self, parser, prefix):
        if hasattr(parser, "_get_accept_mask_for_prefix"):
            return parser._get_accept_mask_for_prefix(prefix)
        valid_tokens = parser.ValidNextTokens(prefix)
        fallback_mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
        for i in range(len(valid_tokens)):
            token_str = self._to_str(valid_tokens[i])
            indices = self._token_str_to_indices.get(token_str)
            if indices is not None:
                fallback_mask[indices] = True
        return fallback_mask

    def MaskValidNextAndEos(self, parser, prefix, eosToken):
        with _timed("MaskValidNextAndEos"):
            full_mask = self._parser_full_mask(parser, prefix)
            if full_mask.numel() == len(self._token_ids):
                subset_mask = full_mask.to(dtype=torch.bool, device=self._logits_tensor.device)
                full_mask = None
            else:
                full_mask = self._expand_full_mask(full_mask)
                subset_mask = self._subset_mask_from_full_mask(full_mask)

            eos_indices = self._token_indices_for_token(eosToken)
            _clear_eos = (
                os.environ.get("CSD_ITERGEN_HARD_MASK", "0") == "1"
                or os.environ.get("CSD_ITERGEN_OPPORTUNISTIC", "0") == "1"
            )
            if _clear_eos:
                # IterGen: grammar DFA mask only. Even grammar_strict can mark
                # <|im_end|> accepted; clear it so argmax continues (LIMIT…).
                if eos_indices:
                    subset_mask[eos_indices] = False
                    if full_mask is not None:
                        eos_full_ids = self._token_ids_tensor[eos_indices].to(full_mask.device)
                        full_mask[eos_full_ids] = False
            elif eos_indices:
                # Outside IterGen, stopping is only allowed once the prefix is a
                # complete query (or nothing else can follow). Marking end-of-turn
                # legal unconditionally let the model stop mid-query and the
                # fragment was then scored as its answer.
                has_other_valid_token = bool(torch.sum(subset_mask).item() > 0)
                if _eos_is_legal(parser, prefix, has_other_valid_token):
                    subset_mask[eos_indices] = True
                    if full_mask is not None:
                        eos_full_ids = self._token_ids_tensor[eos_indices].to(full_mask.device)
                        full_mask[eos_full_ids] = True

            if torch.sum(subset_mask).item() == 0:
                # IterGen `_apply_mask`: if accept mask is empty, warn and mask
                # nothing (do not raise). Do NOT force-EOS here — that stopped
                # Spider ex3 at `-- 1` instead of continuing the zero run.
                if (
                    os.environ.get("CSD_ITERGEN_OPPORTUNISTIC", "0") == "1"
                    or os.environ.get("CSD_ITERGEN_HARD_MASK", "0") == "1"
                ):
                    return
                raise RuntimeError("MaskValidNextAndEos found no valid next tokens including EOS")

            self._logits_tensor.masked_fill_(~subset_mask, -1e9)
            if self._full_logits is not None:
                if full_mask is not None:
                    self._full_logits.masked_fill_(~full_mask, -1e9)
                else:
                    # Parser mask was already vocab-aligned to the constrained
                    # subset. SoftConstrained samples via ChooseNextTokenUnconstrained
                    # on _full_logits — must project the subset mask onto full vocab
                    # or EOS stays unmasked and wins after complete SQL.
                    projected = torch.zeros(
                        self._full_logits.numel(),
                        dtype=torch.bool,
                        device=self._full_logits.device,
                    )
                    ids = self._token_ids_tensor.to(self._full_logits.device)
                    sm = subset_mask.to(self._full_logits.device)
                    # Only mark true positions; leave rest False then fill.
                    projected[ids[sm]] = True
                    self._full_logits.masked_fill_(~projected, -1e9)
            self._logits_dirty = True
            self.Logits.update_tensors(self._logits_tensor, self._full_logits)

    def BoostValidNextAndEos(self, parser, prefix, amount, eosToken):
        with _timed("BoostValidNextAndEos"):
            # IterGen opportunistic: unconstrained greedy first; mask only on SoftConstrained
            # fallback when the pick is invalid. Do not alter logits here.
            if os.environ.get("CSD_ITERGEN_OPPORTUNISTIC", "0") == "1":
                return
            # IterGen parity: hard-mask to grammar accept set and do NOT force-include
            # eos (IterGen argmaxes DFA mask only; eos is HF stopping criteria).
            if os.environ.get("CSD_ITERGEN_HARD_MASK", "0") == "1":
                self.MaskValidNextAndEos(parser, prefix, eosToken)
                # MaskValidNextAndEos honors CSD_ITERGEN_HARD_MASK to skip eos force.
                return
            amount_f = float(amount)
            full_mask = self._parser_full_mask(parser, prefix)
            if full_mask.numel() == len(self._token_ids):
                subset_mask = full_mask.to(dtype=torch.bool, device=self._logits_tensor.device)
                full_mask = None
            else:
                full_mask = self._expand_full_mask(full_mask)
                subset_mask = self._subset_mask_from_full_mask(full_mask)

            eos_indices = self._token_indices_for_token(eosToken)
            if eos_indices:
                has_other_valid_token = bool(torch.sum(subset_mask).item() > 0)
                if _eos_is_legal(parser, prefix, has_other_valid_token):
                    subset_mask[eos_indices] = True
                    if full_mask is not None:
                        eos_full_ids = self._token_ids_tensor[eos_indices].to(full_mask.device)
                        full_mask[eos_full_ids] = True

            self._logits_tensor[subset_mask] = torch.clamp(
                self._logits_tensor[subset_mask] + amount_f, min=-1e9, max=1e9
            )
            if self._full_logits is not None and full_mask is not None:
                self._full_logits[full_mask] = torch.clamp(
                    self._full_logits[full_mask] + amount_f, min=-1e9, max=1e9
                )
            self.Logits.update_tensors(self._logits_tensor, self._full_logits)
            self._logits_dirty = True

    def MaskTokensExcept(self, valid_tokens, debug=False):
        with _timed("MaskTokensExcept"):
            accept_mask = torch.zeros(len(self._token_ids), dtype=torch.bool)
            for i in range(len(valid_tokens)):
                token_str = self._to_str(valid_tokens[i])
                indices = self._token_str_to_indices.get(token_str)
                if indices is not None:
                    accept_mask[indices] = True

            if torch.sum(accept_mask) == 0:
                valid_preview = [self._to_str(valid_tokens[i]) for i in range(min(len(valid_tokens), 10))]
                raise RuntimeError(
                    "MaskTokensExcept found no LM tokens matching the provided valid token set. "
                    f"Sample valid tokens: {valid_preview}"
                )

            if len(self._logits_tensor) > len(accept_mask):
                padding = torch.zeros(len(self._logits_tensor) - len(accept_mask), dtype=torch.bool)
                accept_mask = torch.cat((accept_mask, padding))

            self._logits_tensor.masked_fill_(~accept_mask.to(self._logits_tensor.device), -1e9)
            self._logits_dirty = True



class _OracleTrieNode:
    __slots__ = ("parent", "children", "raw_logprob", "log_theta")

    def __init__(self, parent: "_OracleTrieNode | None" = None):
        self.parent = parent
        self.children: dict[int, _OracleTrieNode] = {}
        self.raw_logprob: torch.Tensor | None = None
        self.log_theta: torch.Tensor | None = None



def _build_tokens_dafny(_dafny, tokenizer, token_ids):
    return _dafny.SeqWithoutIsStrInference(
        [_dafny.Seq(tokenizer.decode([tid])) for tid in token_ids]
    )


def create_huggingface_lm(
    model_name: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """Create a HuggingFace LM wrapped with a Dafny-compatible interface."""
    prec_str = "FP16"
    if load_in_4bit:
        prec_str = "4-bit"
    elif load_in_8bit:
        prec_str = "8-bit"

    print(f"Loading model: {model_name} on {device}... ({prec_str})")
    tokenizer = load_runtime_tokenizer(model_name, backend="huggingface")

    if device.startswith("cuda"):
        kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
        else:
            kwargs["torch_dtype"] = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}.get(os.environ.get("CSD_HF_DTYPE", "float16").lower(), torch.float16)

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        input_device = get_model_input_device(model)
        print(f"Model loaded across {torch.cuda.device_count()} GPU(s), inputs go to {input_device}")
    elif device == "mps" and torch.backends.mps.is_available():
        # Apple Silicon path: FP16 on Metal. bitsandbytes is unsupported on MPS,
        # so load_in_4bit/load_in_8bit flags are ignored if requested here.
        if load_in_4bit or load_in_8bit:
            print("⚠️  4/8-bit quantization is not supported on MPS — loading FP16 instead.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype={"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}.get(os.environ.get("CSD_HF_DTYPE", "float16").lower(), torch.float16),
        )
        model = model.to("mps")
        input_device = torch.device("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        input_device = torch.device("cpu")

    model.eval()

    if token_ids is None:
        token_ids = list(range(len(tokenizer)))

    tokens_dafny = _build_tokens_dafny(_dafny, tokenizer, token_ids)

    class HuggingFaceLM(_TensorizedLMBase, VerifiedDecoderAgent.LM):
        def __init__(self, hf_model, hf_tokenizer, tokens, tids, dev):
            VerifiedDecoderAgent.LM.__init__(self)
            _TensorizedLMBase.__init__(self, _dafny, hf_tokenizer, tokens, tids, logits_device=dev)
            self.model = hf_model
            self._input_device = dev
            self._max_input_len = get_max_input_length(hf_model, hf_tokenizer)

        def GenerateLogits(self, input_prefix):
            self._check_runtime_deadline()
            self._check_answer_early_stop(input_prefix)
            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text
            self._begin_generation_transaction(input_prefix)

            # Prefix-cache short-circuit
            if full_prompt == self._last_full_prompt and not self._logits_dirty:
                self._cache_hits += 1
                return

            self._generate_count += 1
            if self._generate_count % 10 == 0:
                print(f"    [PROGRESS] GenerateLogits call #{self._generate_count}, prefix length: {len(input_prefix)}, cache_hits={self._cache_hits}")

            # Append per-token vocab ids (not join-then-retokenize).
            id_list = self._full_input_ids(input_prefix)
            if len(id_list) > self._max_input_len:
                id_list = id_list[-self._max_input_len :]
            input_ids = torch.tensor([id_list], dtype=torch.long, device=self._input_device)
            attn = torch.ones_like(input_ids)
            inputs = {"input_ids": input_ids, "attention_mask": attn}

            use_kv = os.environ.get("CSD_HF_KV_CACHE", "0") == "1"
            with torch.no_grad():
                if use_kv:
                    input_ids = inputs["input_ids"]
                    past = getattr(self, "_past_key_values", None)
                    prev_ids = getattr(self, "_session_input_ids", None)
                    # Incremental only when new ids are a strict extension of the
                    # previous session (CRANE/IterGen-style decode session).
                    if (
                        past is not None
                        and prev_ids is not None
                        and input_ids.shape[-1] > prev_ids.shape[-1]
                        and torch.equal(input_ids[:, : prev_ids.shape[-1]], prev_ids)
                    ):
                        new_ids = input_ids[:, prev_ids.shape[-1] :]
                        attn = torch.ones(
                            (1, input_ids.shape[-1]),
                            dtype=inputs.get("attention_mask", input_ids).dtype
                            if "attention_mask" in inputs
                            else torch.long,
                            device=self._input_device,
                        )
                        output = self.model(
                            input_ids=new_ids,
                            attention_mask=attn,
                            past_key_values=past,
                            use_cache=True,
                        )
                    else:
                        output = self.model(**inputs, use_cache=True)
                    self._past_key_values = output.past_key_values
                    self._session_input_ids = input_ids
                    logits = output.logits[0, -1, :]
                else:
                    output = self.model(**inputs)
                    logits = output.logits[0, -1, :]
                    self._past_key_values = None
                    self._session_input_ids = None

            self._finalize_full_logits(logits)
            self._apply_recurrence_penalty(full_prompt)
            self._last_full_prompt = full_prompt
            self._logits_dirty = False

        def GenerateUnconstrainedChunk(self, input_prefix, maxNewTokens, openSpanToken, eosToken):
            self._check_runtime_deadline()
            self._check_answer_early_stop(input_prefix)
            max_new_tokens = int(maxNewTokens)
            if max_new_tokens <= 0:
                return self._build_unconstrained_chunk_result([], openSpanToken, eosToken, 0)

            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text
            self._begin_generation_transaction(input_prefix)
            id_list = self._full_input_ids(input_prefix)
            if len(id_list) > self._max_input_len:
                id_list = id_list[-self._max_input_len :]
            input_ids = torch.tensor([id_list], dtype=torch.long, device=self._input_device)
            attn = torch.ones_like(input_ids)
            inputs = {"input_ids": input_ids, "attention_mask": attn}
            prompt_len = input_ids.shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            token_ids = outputs[0, prompt_len:].tolist()
            return self._build_unconstrained_chunk_result(token_ids, openSpanToken, eosToken, max_new_tokens)

    return HuggingFaceLM(model, tokenizer, tokens_dafny, token_ids, input_device)


def create_vllm_lm(
    model_name: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 16384,
    enforce_eager: bool = True,
):
    """Create a vLLM-backed LM wrapper with tensorized logits capture."""
    if not device.startswith("cuda"):
        raise ValueError("vLLM runtime currently requires a CUDA device in this project.")

    _configure_vllm_multiprocessing()
    from vllm import SamplingParams

    tensor_parallel_size = resolve_vllm_tensor_parallel_size(tensor_parallel_size)
    vllm_kwargs = _get_vllm_quantization_kwargs(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    llm, tokenizer = _get_cached_vllm_engine(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        vllm_kwargs=vllm_kwargs,
    )

    if token_ids is None:
        token_ids = list(range(len(tokenizer)))

    tokens_dafny = _build_tokens_dafny(_dafny, tokenizer, token_ids)

    class VllmLM(_TensorizedLMBase, VerifiedDecoderAgent.LM):
        def __init__(self, engine, tok, tokens, tids):
            VerifiedDecoderAgent.LM.__init__(self)
            _TensorizedLMBase.__init__(self, _dafny, tok, tokens, tids, logits_device=torch.device(device))
            self.engine = engine

        def GenerateLogits(self, input_prefix):
            self._check_runtime_deadline()
            self._check_answer_early_stop(input_prefix)
            with _timed("GenerateLogits.total"):
                with _timed("GenerateLogits.prefix_text"):
                    prefix_text = self._prefix_text(input_prefix)
                    full_prompt = self.instruction_text + prefix_text
                    self._begin_generation_transaction(input_prefix)

                # Prefix-cache short-circuit
                if full_prompt == self._last_full_prompt and not self._logits_dirty:
                    self._cache_hits += 1
                    return

                self._generate_count += 1
                if self._generate_count % 10 == 0:
                    print(f"    [PROGRESS] GenerateLogits call #{self._generate_count}, prefix length: {len(input_prefix)}, cache_hits={self._cache_hits}")

                with _timed("GenerateLogits.sampling_params_construct"):
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=VLLM_TOPK_LOGPROBS,
                        detokenize=False,
                    )

                with _timed("GenerateLogits.engine_generate"):
                    outputs = self.engine.generate([full_prompt], sampling_params=sampling_params, use_tqdm=False)

                if not outputs or not outputs[0].outputs:
                    raise RuntimeError("vLLM returned no generation outputs for logits capture.")

                with _timed("GenerateLogits.extract_logprobs"):
                    logprob_steps = outputs[0].outputs[0].logprobs
                    if not logprob_steps:
                        raise RuntimeError("vLLM did not return next-token logprobs.")

                with _timed("GenerateLogits.finalize_from_dict"):
                    self._finalize_from_logprob_dict(logprob_steps[0])

                self._apply_recurrence_penalty(full_prompt)
                self._last_full_prompt = full_prompt
                self._logits_dirty = False

                if self._generate_count % _TIMINGS_PRINT_EVERY == 0:
                    _print_timings_breakdown(header=f"after {self._generate_count} GenerateLogits calls")

        def GenerateUnconstrainedChunk(self, input_prefix, maxNewTokens, openSpanToken, eosToken):
            self._check_runtime_deadline()
            self._check_answer_early_stop(input_prefix)
            max_new_tokens = int(maxNewTokens)
            if max_new_tokens <= 0:
                return self._build_unconstrained_chunk_result([], openSpanToken, eosToken, 0)

            prefix_text = self._prefix_text(input_prefix)
            full_prompt = self.instruction_text + prefix_text
            self._begin_generation_transaction(input_prefix)
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                detokenize=False,
            )
            outputs = self.engine.generate([full_prompt], sampling_params=sampling_params, use_tqdm=False)
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned no generation outputs for unconstrained chunk capture.")

            token_ids = outputs[0].outputs[0].token_ids
            return self._build_unconstrained_chunk_result(token_ids, openSpanToken, eosToken, max_new_tokens)

    return VllmLM(llm, tokenizer, tokens_dafny, token_ids)


def create_runtime_lm(
    model_name: str,
    backend: str,
    device: str,
    VerifiedDecoderAgent,
    _dafny,
    token_ids=None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vllm_tensor_parallel_size: int | None = None,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int = 16384,
    vllm_enforce_eager: bool = True,
):
    """Create the requested runtime LM backend."""
    if backend == "vllm":
        return create_vllm_lm(
            model_name=model_name,
            device=device,
            VerifiedDecoderAgent=VerifiedDecoderAgent,
            _dafny=_dafny,
            token_ids=token_ids,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            tensor_parallel_size=vllm_tensor_parallel_size,
            pipeline_parallel_size=vllm_pipeline_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            enforce_eager=vllm_enforce_eager,
        )

    return create_huggingface_lm(
        model_name=model_name,
        device=device,
        VerifiedDecoderAgent=VerifiedDecoderAgent,
        _dafny=_dafny,
        token_ids=token_ids,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
