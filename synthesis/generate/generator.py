"""
Strategy generator for CSD synthesis.

Supports HuggingFace, vLLM, OpenAI Chat Completions, and Amazon Bedrock Converse.
"""

import hashlib
import fcntl
import logging
import os
import json
import queue
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import torch

from synthesis.evaluate.benchmarks.common.model_utils import _configure_vllm_multiprocessing

from .prompts import (
    build_initial_prompt,
    build_verification_error_prompt,
    build_runtime_error_prompt,
    build_compilation_error_prompt,
    build_format_repair_prompt,
    build_evaluation_failure_prompt,
)
from .rationale import extract_rationale
from .provider_names import normalize_generation_backend
from .pi_oauth import (
    PI_MODEL,
    PiBridgeFailure,
    PiBridgeTimeout,
    pi_oauth_probe,
    run_pi_bridge,
)
from ..run_constants import ANTHROPIC_EFFORT, ANTHROPIC_THINKING_DISPLAY, VLLM_ENFORCE_EAGER
from ..safe_logging import safe_logging_enabled, text_metadata


LOGGER = logging.getLogger(__name__)

CLAUDE_CODE_MODEL = "claude-opus-5"
CLAUDE_ACCESS_ERROR_MARKER = "[claude-author-access]"
CODEX_MODEL = "gpt-5.6-sol"
CODEX_ACCESS_ERROR_MARKER = "[codex-author-access]"


class ClaudeTransientError(RuntimeError):
    """A temporary Claude transport failure that must not consume an attempt."""


class CodexTransientError(RuntimeError):
    """A temporary Pi/OpenAI-Codex transport failure that consumes no attempt."""


class StrategyGenerator:
    """
    Generates Dafny CSD strategies.

    Supports local HuggingFace inference, local vLLM inference, OpenAI-hosted
    models, or Amazon Bedrock Converse (e.g. Claude where configured).
    """

    # Default model - can be overridden
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Path to the template file
    TEMPLATE_PATH = Path(__file__).parent.parent / "verify" / "library" / "GeneratedCSD.dfy"

    # Marker in template to replace
    STRATEGY_MARKER = "// QWEN_INSERT_STRATEGY_HERE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "huggingface",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vllm_gpu_memory_utilization: float = 0.8,
        vllm_max_model_len: int = 4096,
        reasoning_budget_tokens: int = 4096,
        claude_executable: Optional[str] = None,
        claude_config_dir: Optional[str] = None,
        claude_expected_account: Optional[str] = None,
        claude_timeout_seconds: Optional[float] = None,
        claude_idle_timeout_seconds: Optional[float] = None,
        claude_emergency_timeout_seconds: Optional[float] = None,
        claude_max_retries: Optional[int] = None,
        claude_retry_delay_seconds: Optional[float] = None,
        claude_telemetry_dir: Optional[str] = None,
        claude_author_lock_file: Optional[str] = None,
        pi_node_executable: Optional[str] = None,
        pi_bridge_path: Optional[str] = None,
        pi_auth_path: Optional[str] = None,
        codex_timeout_seconds: Optional[float] = None,
        codex_max_retries: Optional[int] = None,
        codex_retry_delay_seconds: Optional[float] = None,
        codex_author_lock_file: Optional[str] = None,
    ):
        """
        Initialize the strategy generator.

        Args:
            model_name: HuggingFace model name (default: Qwen2.5-Coder-7B-Instruct)
            backend: Inference backend, including "claude" for Claude Code Max
                and "claude-bedrock" for AWS Bedrock
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            torch_dtype: Torch dtype for model (default: auto based on device)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            vllm_gpu_memory_utilization: GPU memory fraction reserved by vLLM
            vllm_max_model_len: Max context length passed to vLLM
            reasoning_budget_tokens: Provider-agnostic extended-thinking budget
                for hosted Claude authors (budget_tokens on Anthropic/Bedrock).
                Thinking is always ON — API keys and base URLs come from the
                environment (.env), never from parameters (BYOD).
        """
        self.backend = normalize_generation_backend(backend)
        self.model_name = model_name or (
            CLAUDE_CODE_MODEL if self.backend == "claude" else (
                CODEX_MODEL if self.backend == "codex" else self.DEFAULT_MODEL
            )
        )
        if self.backend == "claude" and self.model_name != CLAUDE_CODE_MODEL:
            raise ValueError(
                f"Claude Code synthesis requires model {CLAUDE_CODE_MODEL!r}; "
                f"got {self.model_name!r}"
            )
        if self.backend == "codex" and self.model_name != CODEX_MODEL:
            raise ValueError(
                f"Codex synthesis requires model {CODEX_MODEL!r}; "
                f"got {self.model_name!r}"
            )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.api_base_url = self._default_api_base_url(self.backend)
        self.api_key = self._default_api_key(self.backend)
        # Extended thinking is always on for hosted Claude authors
        # (hard-coded 2026-07-17; effort/display in synthesis/run_constants.py).
        self.anthropic_thinking = "always-on"
        self.reasoning_budget_tokens = int(reasoning_budget_tokens)
        self.anthropic_effort = ANTHROPIC_EFFORT
        self.anthropic_thinking_display = ANTHROPIC_THINKING_DISPLAY
        self.claude_executable = (
            claude_executable or os.environ.get("CSD_CLAUDE_EXECUTABLE") or "claude"
        )
        config_dir = claude_config_dir or os.environ.get("CSD_CLAUDE_CONFIG_DIR")
        expected_account = claude_expected_account or os.environ.get(
            "CSD_CLAUDE_EXPECTED_ACCOUNT"
        )
        self.claude_config_dir = Path(config_dir).expanduser().resolve() if config_dir else None
        self.claude_expected_account = expected_account
        if claude_timeout_seconds is None:
            claude_timeout_seconds = float(
                os.environ.get("CSD_CLAUDE_TIMEOUT_SECONDS", "1800")
            )
        self.claude_timeout_seconds = float(claude_timeout_seconds)
        if claude_idle_timeout_seconds is None:
            claude_idle_timeout_seconds = float(
                os.environ.get(
                    "CSD_CLAUDE_IDLE_TIMEOUT_SECONDS",
                    str(self.claude_timeout_seconds),
                )
            )
        if claude_emergency_timeout_seconds is None:
            claude_emergency_timeout_seconds = float(
                os.environ.get("CSD_CLAUDE_EMERGENCY_TIMEOUT_SECONDS", "7200")
            )
        if claude_max_retries is None:
            claude_max_retries = int(os.environ.get("CSD_CLAUDE_MAX_RETRIES", "2"))
        if claude_retry_delay_seconds is None:
            claude_retry_delay_seconds = float(
                os.environ.get("CSD_CLAUDE_RETRY_DELAY_SECONDS", "30")
            )
        telemetry_dir = claude_telemetry_dir or os.environ.get(
            "CSD_CLAUDE_TELEMETRY_DIR"
        )
        lock_file = claude_author_lock_file or os.environ.get(
            "CSD_CLAUDE_AUTHOR_LOCK_FILE"
        )
        self.claude_idle_timeout_seconds = float(claude_idle_timeout_seconds)
        self.claude_emergency_timeout_seconds = float(claude_emergency_timeout_seconds)
        self.claude_max_retries = int(claude_max_retries)
        self.claude_retry_delay_seconds = float(claude_retry_delay_seconds)
        self.claude_telemetry_dir = (
            Path(telemetry_dir).expanduser().resolve() if telemetry_dir else None
        )
        self.claude_author_lock_file = (
            Path(lock_file).expanduser().resolve()
            if lock_file
            else (
                self.claude_config_dir / "csd-author.lock"
                if self.claude_config_dir
                else Path(tempfile.gettempdir()) / "csd-claude-author.lock"
            )
        )
        if self.claude_idle_timeout_seconds <= 0:
            raise ValueError("claude_idle_timeout_seconds must be positive")
        if self.claude_emergency_timeout_seconds <= 0:
            raise ValueError("claude_emergency_timeout_seconds must be positive")
        if self.claude_max_retries < 0:
            raise ValueError("claude_max_retries must be non-negative")
        self._claude_account_verified = False
        self.pi_node_executable = pi_node_executable or os.environ.get(
            "CSD_PI_NODE_EXECUTABLE"
        )
        self.pi_bridge_path = pi_bridge_path or os.environ.get("CSD_PI_BRIDGE_PATH")
        self.pi_auth_path = pi_auth_path or os.environ.get("CSD_PI_AUTH_PATH")
        if codex_timeout_seconds is None:
            codex_timeout_seconds = float(os.environ.get("CSD_CODEX_TIMEOUT_SECONDS", "1800"))
        if codex_max_retries is None:
            codex_max_retries = int(os.environ.get("CSD_CODEX_MAX_RETRIES", "2"))
        if codex_retry_delay_seconds is None:
            codex_retry_delay_seconds = float(os.environ.get("CSD_CODEX_RETRY_DELAY_SECONDS", "30"))
        self.codex_timeout_seconds = float(codex_timeout_seconds)
        self.codex_max_retries = int(codex_max_retries)
        self.codex_retry_delay_seconds = float(codex_retry_delay_seconds)
        self.codex_author_lock_file = Path(
            codex_author_lock_file
            or os.environ.get("CSD_CODEX_AUTHOR_LOCK_FILE")
            or Path(tempfile.gettempdir()) / "csd-codex-author.lock"
        ).expanduser().resolve()
        if self.codex_timeout_seconds <= 0:
            raise ValueError("codex_timeout_seconds must be positive")
        if self.codex_max_retries < 0:
            raise ValueError("codex_max_retries must be non-negative")
        if self.codex_retry_delay_seconds < 0:
            raise ValueError("codex_retry_delay_seconds must be non-negative")
        self._codex_account_verified = False
        self._codex_route_identity: dict[str, object] | None = None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Auto-detect dtype
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None
        self._client = None
        self._vllm = None
        self._current_task_description: Optional[str] = None
        self._current_start_inside_constrained: bool = False
        self._prompt_log_counter = 0
        self._synthesis_context: Optional[dict[str, str]] = None
        self._summary_client = None
        self._summary_anthropic_client = None

        # Load template
        self._template = self._load_template()

    @staticmethod
    def _default_api_base_url(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_BASE_URL")
        if backend == "claude-bedrock":
            return os.environ.get("BEDROCK_BASE_URL")
        if backend == "gemini":
            return os.environ.get(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            )
        if backend == "vertex":
            location = (
                os.environ.get("VERTEX_AI_LOCATION")
                or os.environ.get("GOOGLE_CLOUD_LOCATION")
                or os.environ.get("GOOGLE_VERTEX_LOCATION")
                or "global"
            )
            if location == "global":
                return os.environ.get("VERTEX_AI_BASE_URL", "https://aiplatform.googleapis.com/v1")
            return os.environ.get(
                "VERTEX_AI_BASE_URL",
                f"https://{location}-aiplatform.googleapis.com/v1",
            )
        return None

    @staticmethod
    def _bedrock_runtime_base_url() -> str:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        return os.environ.get(
            "BEDROCK_BASE_URL",
            f"https://bedrock-runtime.{region}.amazonaws.com",
        )

    @staticmethod
    def _default_api_key(backend: str) -> Optional[str]:
        if backend == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if backend == "claude-bedrock":
            return os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if backend == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if backend == "gemini":
            return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if backend == "vertex":
            return os.environ.get("VERTEX_AI_ACCESS_TOKEN")
        return None

    def author_route_identity(self) -> dict[str, object]:
        """Return non-secret identity for the author route actually in use."""
        if self.backend == "claude":
            return {
                "auth_mode": "claude_code_max",
                "config_dir": (
                    str(self.claude_config_dir) if self.claude_config_dir else None
                ),
                "expected_account": self.claude_expected_account,
                "account_verified": self._claude_account_verified,
            }
        if self.backend == "codex":
            if self._codex_route_identity is None:
                return {
                    "auth_mode": "chatgpt_codex_oauth",
                    "provider": "openai-codex",
                    "model": CODEX_MODEL,
                    "account_verified": False,
                }
            return dict(self._codex_route_identity)
        if self.backend == "gemini":
            key_sha256 = getattr(self, "_active_gemini_api_key_sha256", None)
            if key_sha256 is None and self.api_key:
                key_sha256 = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
            return {
                "auth_mode": "gemini_api_key",
                "api_key_sha256": key_sha256,
            }
        if self.backend == "vertex":
            adc = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
            adc_sha256 = None
            if adc.is_file():
                digest = hashlib.sha256()
                with adc.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                adc_sha256 = digest.hexdigest()
            uses_adc = not (
                self.api_key
                or os.environ.get("VERTEX_AI_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            return {
                "auth_mode": "adc" if uses_adc else "token_or_api_key",
                "vertex_project": self._vertex_project(),
                "location": self._vertex_location(),
                "adc_sha256": adc_sha256,
            }
        return {"auth_mode": self.backend}

    def _load_template(self) -> str:
        """Load the GeneratedCSD.dfy template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure GeneratedCSD.dfy exists in synthesis/verify/library/."
            )
        return self.TEMPLATE_PATH.read_text()

    def set_synthesis_context(
        self,
        eval_model: str,
        dataset: str,
        max_steps: int,
        step_token_budget: int,
    ) -> None:
        """Store runtime evaluation context for inclusion in synthesis prompts."""
        self._synthesis_context = {
            "eval_model": eval_model,
            "dataset": dataset,
            "max_steps": str(max_steps),
            "step_token_budget": str(step_token_budget),
        }

    def set_task_description(self, task_description: str) -> None:
        """Set the task used by every refinement prompt in this synthesis run."""
        if not task_description.strip():
            raise ValueError("task_description must not be empty")
        self._current_task_description = task_description

    def _synthesis_context_block(self) -> str:
        if not self._synthesis_context:
            return ""
        ctx = self._synthesis_context
        return (
            "\n\n## Runtime Context\n"
            f"- Evaluation model: {ctx['eval_model']}\n"
            f"- Dataset: {ctx['dataset']}\n"
            f"- maxSteps budget: {ctx['max_steps']}\n"
            f"- stepTokenBudget: {ctx['step_token_budget']}\n"
        )
    
    def _ensure_backend_loaded(self) -> None:
        """Lazy-load the selected backend."""
        if self.backend == "claude":
            self._verify_claude_account()
            return
        if self.backend == "codex":
            self._verify_codex_account()
            return

        if self.backend == "openai":
            if self._client is None:
                if not self.api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required when --generation-backend=openai"
                    )
                from openai import OpenAI

                client_kwargs = {"api_key": self.api_key}
                if self.api_base_url:
                    client_kwargs["base_url"] = self.api_base_url
                self._client = OpenAI(**client_kwargs)
            return

        if self.backend == "claude-bedrock":
            if not self.api_key:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK is required when "
                    "--generation-backend=claude-bedrock"
                )
            return

        if self.backend == "anthropic":
            if self._client is None:
                if not self.api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY is required when --generation-backend=anthropic"
                    )
                from anthropic import Anthropic
                client_kwargs = {"api_key": self.api_key}
                if self.api_base_url:
                    client_kwargs["base_url"] = self.api_base_url
                self._client = Anthropic(**client_kwargs)
            return

        if self.backend == "gemini":
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY is required when --generation-backend=gemini"
                )
            return

        if self.backend == "vertex":
            return

        if self.backend == "vllm":
            if self._vllm is None:
                if not self.device.startswith("cuda"):
                    raise ValueError("vLLM generation currently requires CUDA in this project.")

                _configure_vllm_multiprocessing()
                from vllm import LLM
                from vllm.transformers_utils.tokenizer import get_tokenizer

                from synthesis.evaluate.benchmarks.common.model_utils import (
                    resolve_vllm_tensor_parallel_size,
                )

                # Tensor/pipeline parallelism and eager mode are settled: every
                # recorded run fits on one GPU, so these are hard-coded rather
                # than exposed as constructor parameters (2026-07-18 bucket-1
                # audit; see planning/ws2-ws3-landed-audit.md).
                tensor_parallel_size = resolve_vllm_tensor_parallel_size(None)
                self._tokenizer = get_tokenizer(self.model_name, trust_remote_code=True)
                self._vllm = LLM(
                    model=self.model_name,
                    tokenizer=self.model_name,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_model_len=self.vllm_max_model_len,
                    enforce_eager=VLLM_ENFORCE_EAGER,
                )
            return

        if self.backend != "huggingface":
            raise ValueError(f"Unsupported generation backend: {self.backend}")

        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # No recorded run uses quantized loading (2026-07-18 bucket-1 audit
            # removed the load_in_4bit/8bit knobs from this constructor).
            quantization_config = None

            # Try loading on requested device, fallback to CPU on CUDA OOM
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device.startswith("cuda") else (self.device if self.device != "mps" else None),
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                )
                if self.device == "mps":
                    self._model = self._model.to(self.device)
                print(f"Model loaded on {self.device}")
            except RuntimeError as e:
                error_str = str(e).lower()
                if self.device in ["cuda", "mps"] and ("out of memory" in error_str or "cuda" in error_str):
                    print(f"⚠️  {self.device.upper()} out of memory: {e}")
                    print(f"   Falling back to CPU (this will be slower)...")

                    # Clear CUDA cache if available
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                    # Retry on CPU
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                    ).to(self.device)
                    print(f"Model loaded on {self.device} (CPU fallback)")
                else:
                    raise

    def _resolved_claude_executable(self) -> str:
        """Resolve the configured Claude executable without invoking a shell."""
        executable = shutil.which(self.claude_executable)
        if executable is None:
            raise ValueError(
                f"Claude Code executable not found: {self.claude_executable!r}"
            )
        return executable

    @staticmethod
    def _safe_codex_error(error: object, *, limit: int = 500) -> str:
        """Return a short category without exposing provider output or OAuth data."""
        text = re.sub(r"\s+", " ", str(error)).strip().lower()
        if any(marker in text for marker in ("login", "auth", "credential", "subscription")):
            return "authentication"
        if any(marker in text for marker in ("rate", "quota", "limit", "credit")):
            return "quota"
        return text[:limit] if text else "unknown"

    def _acquire_codex_author_lock(self):
        """Serialize calls made through the account-wide ChatGPT OAuth login."""
        lock_path = self.codex_author_lock_file
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        lock = os.fdopen(descriptor, "w")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return lock

    def _verify_codex_account(self) -> None:
        """Require Pi to resolve the exact ChatGPT/Codex OAuth account."""
        if self._codex_account_verified:
            return
        try:
            probe = pi_oauth_probe(
                node_executable=self.pi_node_executable,
                bridge_path=self.pi_bridge_path,
                auth_path=self.pi_auth_path,
                timeout_seconds=min(self.codex_timeout_seconds, 90),
            )
        except (PiBridgeFailure, PiBridgeTimeout, ValueError) as exc:
            LOGGER.error(
                "[codex] provider=openai-codex model=%s status=oauth-rejected "
                "error_category=%s",
                self.model_name,
                self._safe_codex_error(exc),
            )
            raise ValueError(
                f"{CODEX_ACCESS_ERROR_MARKER} ChatGPT/Codex OAuth is unavailable"
            ) from exc
        self._codex_route_identity = dict(probe["route"])
        self._codex_account_verified = True
        LOGGER.info(
            "[codex] provider=openai-codex model=%s status=oauth-verified "
            "duration_seconds=%.3f",
            self.model_name,
            float(probe["duration_seconds"]),
        )

    def _generate_codex(self, system_prompt: str, user_prompt: str) -> str:
        """Generate through Pi's provider layer with no coding-agent prompt or tools."""
        self._verify_codex_account()
        request = {
            "operation": "complete",
            "model": PI_MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning": "high",
            "tools": [],
            "tool_choice": "none",
            "previous_response_id": None,
            "conversation": None,
        }
        request_bytes = json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        request_hash = hashlib.sha256(request_bytes).hexdigest()
        LOGGER.info(
            "[codex] provider=openai-codex model=%s request_bytes=%d "
            "request_sha256=%s contract=system-instructions-single-user-no-tools-v1",
            self.model_name,
            len(request_bytes),
            request_hash,
        )
        last_error: CodexTransientError | None = None
        for retry_number in range(self.codex_max_retries + 1):
            lock = self._acquire_codex_author_lock()
            try:
                payload, duration = run_pi_bridge(
                    request,
                    node_executable=self.pi_node_executable,
                    bridge_path=self.pi_bridge_path,
                    auth_path=self.pi_auth_path,
                    timeout_seconds=self.codex_timeout_seconds,
                )
                output = payload.get("text")
                route = payload.get("route")
                if not isinstance(output, str) or not output.strip():
                    raise CodexTransientError("Pi provider returned an empty answer")
                if not isinstance(route, dict) or self._codex_route_identity is None:
                    raise CodexTransientError("Pi provider returned no route identity")
                for field in (
                    "auth_mode",
                    "provider",
                    "model",
                    "account_id_sha256",
                ):
                    if route.get(field) != self._codex_route_identity.get(field):
                        raise RuntimeError(
                            f"{CODEX_ACCESS_ERROR_MARKER} Pi OAuth account changed"
                        )
                output = output.strip()
                output_bytes = output.encode("utf-8")
                LOGGER.info(
                    "[codex] provider=openai-codex model=%s status=success "
                    "output_bytes=%d output_sha256=%s duration_seconds=%.3f retry=%d",
                    self.model_name,
                    len(output_bytes),
                    hashlib.sha256(output_bytes).hexdigest(),
                    duration,
                    retry_number,
                )
                return output
            except PiBridgeTimeout as exc:
                last_error = CodexTransientError("Pi provider request timed out")
                last_error.__cause__ = exc
                LOGGER.error(
                    "[codex] provider=openai-codex model=%s status=timeout retry=%d/%d",
                    self.model_name,
                    retry_number,
                    self.codex_max_retries,
                )
            except PiBridgeFailure as exc:
                if exc.category == "authentication":
                    raise RuntimeError(
                        f"{CODEX_ACCESS_ERROR_MARKER} ChatGPT/Codex OAuth failed"
                    ) from exc
                last_error = CodexTransientError(
                    f"Pi provider request failed: {exc.category}"
                )
                LOGGER.warning(
                    "[codex] provider=openai-codex model=%s status=retry "
                    "retry=%d/%d request_sha256=%s error_category=%s",
                    self.model_name,
                    retry_number,
                    self.codex_max_retries,
                    request_hash,
                    exc.category,
                )
            except CodexTransientError as exc:
                last_error = exc
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()
            if last_error is not None and retry_number < self.codex_max_retries:
                time.sleep(self.codex_retry_delay_seconds)
        assert last_error is not None
        raise last_error

    def _claude_environment(self, home: Path) -> dict[str, str]:
        """Build the small environment passed to the isolated Claude process."""
        if self.claude_config_dir is None:
            raise ValueError(
                "Claude Code synthesis requires --claude-config-dir or "
                "CSD_CLAUDE_CONFIG_DIR"
            )
        if not self.claude_config_dir.is_dir():
            raise ValueError(
                f"Claude Code config directory does not exist: {self.claude_config_dir}"
            )
        if not self.claude_expected_account:
            raise ValueError(
                "Claude Code synthesis requires --claude-expected-account or "
                "CSD_CLAUDE_EXPECTED_ACCOUNT"
            )

        allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
        environment = {name: os.environ[name] for name in allowed if name in os.environ}
        environment.update(
            {
                "HOME": str(home),
                "CLAUDE_CONFIG_DIR": str(self.claude_config_dir),
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
                "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
                "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
                "DISABLE_AUTO_COMPACT": "1",
                # Authoring calls with ~100KB prompts can spend more than the
                # CLI's default 32000 output tokens on thinking alone and die
                # before emitting any strategy text.
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
                # Without a thinking cap a call can spend the entire output
                # budget on thinking and emit no strategy text at all (GSM-2B
                # burned 128000 tokens / 0 text on 2026-07-15); 48000 leaves
                # at least 16000 tokens for the actual strategy.
                "MAX_THINKING_TOKENS": "48000",
                # Sonnet 4.6 only honors MAX_THINKING_TOKENS as a fixed
                # budget when adaptive reasoning is disabled; with adaptive
                # reasoning on, calls kept burning the whole 64000-token
                # max_tokens on thinking across interleaved blocks.
                "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
            }
        )
        return environment

    @staticmethod
    def _safe_claude_error(error: object, *, limit: int = 500) -> str:
        """Return one short log-safe error line."""
        text = re.sub(r"\s+", " ", str(error)).strip()
        text = re.sub(r"https?://\S+", "<url-redacted>", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\bBearer\s+\S+",
            "Bearer <redacted>",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|code)"
            r"\s*[:=]\s*[^,\s]+",
            lambda match: f"{match.group(1)}=<redacted>",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "<key-redacted>", text)
        return text[:limit]

    @staticmethod
    def _claude_access_error(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in (
                "not logged in",
                "authentication",
                "subscription",
                "usage limit",
                "weekly limit",
                "hit your limit",
                "limit reached",
                "rate limit",
                "seat",
            )
        )

    @staticmethod
    def _log_claude_failure(
        *,
        started: float,
        exit_status: object,
        category: str,
    ) -> None:
        """Write one uniform, secret-free Claude failure record."""
        LOGGER.error(
            "[claude] failure exit_status=%s category=%s duration_seconds=%.3f",
            exit_status,
            category,
            time.monotonic() - started,
        )

    @staticmethod
    def _stop_claude_process_group(process: subprocess.Popen) -> None:
        """Stop the Claude process and every child in its new process group."""
        process_group = process.pid

        def group_alive() -> bool:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return False
            return True

        if not group_alive():
            return
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not group_alive():
            if process.poll() is None:
                process.wait(timeout=1)
            return
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if group_alive():
            LOGGER.error(
                "[claude] process group remained after SIGKILL pgid=%s",
                process_group,
            )

    def _run_claude_process(
        self,
        argv: list[str],
        *,
        input_bytes: bytes,
        cwd: Path,
        home: Path,
    ) -> tuple[int, bytes, bytes]:
        """Run one isolated Claude command with bounded process cleanup."""
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=self._claude_environment(home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(
                input=input_bytes,
                timeout=self.claude_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self._stop_claude_process_group(process)
            self._log_claude_failure(
                started=started,
                exit_status="timeout",
                category="timeout",
            )
            raise TimeoutError(
                f"Claude Code generation timed out after {self.claude_timeout_seconds:g}s"
            ) from exc
        except BaseException:
            self._stop_claude_process_group(process)
            self._log_claude_failure(
                started=started,
                exit_status="interrupted",
                category="interrupted",
            )
            raise
        return process.returncode, stdout, stderr

    def _claude_telemetry_path(self, request_hash: str, retry_number: int) -> Path:
        """Create one owner-only raw stream file for an author request."""
        root = self.claude_telemetry_dir
        if root is None:
            root = Path(tempfile.gettempdir()) / "csd-claude-stream-telemetry"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = root / (
            f"claude-{stamp}-{request_hash[:16]}-retry{retry_number}-"
            f"{os.getpid()}-{time.time_ns()}.jsonl"
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        return path

    @staticmethod
    def _claude_timeline_record(
        payload: dict,
        *,
        started: float,
        stream_kind: str | None,
    ) -> dict:
        """Summarize when one raw stream event arrived without copying its text."""
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        event = payload.get("event")
        event_type = event.get("type") if isinstance(event, dict) else None
        record = {
            "observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "stream_kind": stream_kind or (
                "result" if payload.get("type") == "result" else "event"
            ),
            "payload_type": payload.get("type"),
            "event_type": event_type,
        }
        if usage:
            record["input_tokens"] = usage.get("input_tokens")
            record["output_tokens"] = usage.get("output_tokens")
        return record

    @staticmethod
    def _claude_stream_delta(payload: dict) -> tuple[str | None, str | None]:
        """Return a human-readable stream kind and its exact emitted text."""
        if payload.get("type") != "stream_event":
            return None, None
        event = payload.get("event")
        if not isinstance(event, dict):
            return None, None
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return None, None
        delta_type = delta.get("type")
        if delta_type == "thinking_delta" and isinstance(delta.get("thinking"), str):
            return "thinking", delta["thinking"]
        if delta_type == "text_delta" and isinstance(delta.get("text"), str):
            return "text", delta["text"]
        return None, None

    def _run_claude_stream(
        self,
        argv: list[str],
        *,
        input_bytes: bytes,
        cwd: Path,
        home: Path,
        request_hash: str,
        retry_number: int,
    ) -> tuple[int, dict | None, str, Path]:
        """Stream one Claude request, retaining every raw event and enforcing idle time."""
        started = time.monotonic()
        last_activity = started
        telemetry_path = self._claude_telemetry_path(request_hash, retry_number)
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=self._claude_environment(home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        messages: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

        def read_lines(name: str, stream) -> None:
            try:
                for line in iter(stream.readline, b""):
                    messages.put((name, line))
            finally:
                messages.put((name, None))

        threads = [
            threading.Thread(target=read_lines, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=read_lines, args=("stderr", process.stderr), daemon=True),
        ]

        def write_prompt() -> None:
            try:
                process.stdin.write(input_bytes)
                process.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                messages.put(("stderr", f"stdin write failed: {exc}\n".encode()))

        writer = threading.Thread(target=write_prompt, daemon=True)
        for thread in threads:
            thread.start()
        writer.start()

        open_streams = 2
        final_payload: dict | None = None
        stderr_parts: list[str] = []
        thinking_events = 0
        text_events = 0
        stderr_path = telemetry_path.with_suffix(".stderr.log")
        timeline_path = telemetry_path.with_suffix(".timeline.jsonl")
        stderr_descriptor = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        timeline_descriptor = os.open(
            timeline_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with (
                telemetry_path.open("ab", buffering=0) as telemetry,
                os.fdopen(stderr_descriptor, "ab", buffering=0) as stderr_file,
                os.fdopen(timeline_descriptor, "ab", buffering=0) as timeline_file,
            ):
                while open_streams:
                    now = time.monotonic()
                    idle_remaining = self.claude_idle_timeout_seconds - (now - last_activity)
                    emergency_remaining = self.claude_emergency_timeout_seconds - (now - started)
                    remaining = min(idle_remaining, emergency_remaining)
                    if remaining <= 0:
                        category = (
                            "idle-timeout" if idle_remaining <= emergency_remaining
                            else "emergency-timeout"
                        )
                        raise TimeoutError(category)
                    try:
                        stream_name, line = messages.get(timeout=min(0.25, remaining))
                    except queue.Empty:
                        continue
                    if line is None:
                        open_streams -= 1
                        continue
                    last_activity = time.monotonic()
                    text = line.decode("utf-8", "replace")
                    if stream_name == "stderr":
                        stderr_parts.append(text)
                        stderr_file.write(line)
                        LOGGER.warning(
                            "[claude-stream] stderr %s",
                            self._safe_claude_error(text.rstrip()),
                        )
                        continue
                    telemetry.write(line)
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        LOGGER.warning("[claude-stream] non-json stdout bytes=%d", len(line))
                        continue
                    kind, delta = self._claude_stream_delta(payload)
                    timeline_file.write(
                        json.dumps(
                            self._claude_timeline_record(
                                payload,
                                started=started,
                                stream_kind=kind,
                            ),
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    if kind and delta is not None:
                        if kind == "thinking":
                            thinking_events += 1
                        else:
                            text_events += 1
                        LOGGER.warning("[claude-stream] %s %s", kind, delta)
                    if payload.get("type") == "result" or "result" in payload:
                        final_payload = payload
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            LOGGER.warning(
                                "[claude-stream] usage input_tokens=%s output_tokens=%s",
                                usage.get("input_tokens"),
                                usage.get("output_tokens"),
                            )
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("process-exit-timeout") from exc
        except TimeoutError as exc:
            self._stop_claude_process_group(process)
            category = str(exc)
            stderr_tail = self._safe_claude_error("".join(stderr_parts)[-2_000:])
            self._log_claude_failure(
                started=started,
                exit_status="timeout",
                category=category,
            )
            LOGGER.error(
                "[claude] timeout request_sha256=%s retry=%d duration_seconds=%.3f "
                "telemetry=%s stderr_tail=%s",
                request_hash,
                retry_number,
                time.monotonic() - started,
                telemetry_path,
                stderr_tail or "<empty>",
            )
            raise ClaudeTransientError(
                f"Claude Code stream {category} after "
                f"{time.monotonic() - started:.1f}s; telemetry={telemetry_path}"
            ) from exc
        except BaseException:
            self._stop_claude_process_group(process)
            raise
        LOGGER.warning(
            "[claude-stream] complete telemetry=%s duration_seconds=%.3f "
            "thinking_events=%d text_events=%d exit_status=%d",
            telemetry_path,
            time.monotonic() - started,
            thinking_events,
            text_events,
            returncode,
        )
        return returncode, final_payload, "".join(stderr_parts), telemetry_path

    def _acquire_claude_author_lock(self):
        """Open and exclusively lock the account-wide author slot."""
        lock_path = self.claude_author_lock_file
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        lock = os.fdopen(descriptor, "w")
        LOGGER.warning("[claude-lock] waiting path=%s", lock_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} acquired={time.time()}\n")
        lock.flush()
        LOGGER.warning("[claude-lock] acquired path=%s active_calls=1", lock_path)
        return lock

    def _verify_claude_account(self) -> None:
        """Require the dedicated first-party Max account before generation."""
        if self._claude_account_verified:
            return
        executable = self._resolved_claude_executable()
        LOGGER.info(
            "[claude] configuration executable=%s config_dir=%s "
            "expected_account=%s timeout_seconds=%g",
            executable,
            self.claude_config_dir,
            self.claude_expected_account,
            self.claude_timeout_seconds,
        )
        with (
            tempfile.TemporaryDirectory(prefix="csd-claude-auth-cwd-") as cwd_name,
            tempfile.TemporaryDirectory(prefix="csd-claude-auth-home-") as home_name,
        ):
            returncode, stdout, stderr = self._run_claude_process(
                [executable, "auth", "status", "--json"],
                input_bytes=b"",
                cwd=Path(cwd_name),
                home=Path(home_name),
            )
        if returncode != 0:
            detail = self._safe_claude_error(stderr.decode("utf-8", "replace"))
            raise ValueError(
                f"{CLAUDE_ACCESS_ERROR_MARKER} Claude Code auth status failed: {detail}"
            )
        try:
            status = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{CLAUDE_ACCESS_ERROR_MARKER} Claude Code auth status was not valid JSON"
            ) from exc

        expected = {
            "loggedIn": True,
            "email": self.claude_expected_account,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        }
        mismatches = [
            f"{name}={status.get(name)!r} (expected {value!r})"
            for name, value in expected.items()
            if status.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                f"{CLAUDE_ACCESS_ERROR_MARKER} Claude Code must be logged in as "
                f"{self.claude_expected_account!r} through claude.ai, firstParty, Max; "
                + "; ".join(mismatches)
            )
        self._claude_account_verified = True
        LOGGER.info(
            "[claude] account verified account=%s model=%s",
            self.claude_expected_account,
            self.model_name,
        )

    def _generate_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Generate through the isolated, allowance-only Claude Code CLI."""
        self._verify_claude_account()
        executable = self._resolved_claude_executable()
        system_bytes = system_prompt.encode("utf-8")
        user_bytes = user_prompt.encode("utf-8")
        system_hash = hashlib.sha256(system_bytes).hexdigest()
        user_hash = hashlib.sha256(user_bytes).hexdigest()
        request_hash = hashlib.sha256(system_bytes + b"\0" + user_bytes).hexdigest()
        LOGGER.warning(
            "[claude] request model=%s system_bytes=%d user_bytes=%d "
            "system_sha256=%s user_sha256=%s",
            self.model_name,
            len(system_bytes),
            len(user_bytes),
            system_hash,
            user_hash,
        )
        last_error: ClaudeTransientError | None = None
        for retry_number in range(self.claude_max_retries + 1):
            last_error = None
            started = time.monotonic()
            lock = self._acquire_claude_author_lock()
            try:
                with (
                    tempfile.TemporaryDirectory(prefix="csd-claude-prompt-") as prompt_dir_name,
                    tempfile.TemporaryDirectory(prefix="csd-claude-cwd-") as cwd_name,
                    tempfile.TemporaryDirectory(prefix="csd-claude-home-") as home_name,
                ):
                    prompt_path = Path(prompt_dir_name) / "system-prompt.txt"
                    descriptor = os.open(
                        prompt_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    with os.fdopen(descriptor, "wb") as prompt_file:
                        prompt_file.write(system_bytes)
                    argv = [
                        executable,
                        "--print",
                        "--model",
                        CLAUDE_CODE_MODEL,
                        "--effort",
                        "high",
                        "--system-prompt-file",
                        str(prompt_path),
                        "--output-format",
                        "stream-json",
                        "--include-partial-messages",
                        "--verbose",
                        "--tools",
                        "",
                        "--disable-slash-commands",
                        "--strict-mcp-config",
                        "--setting-sources",
                        "",
                        "--no-session-persistence",
                        "--no-chrome",
                    ]
                    returncode, payload, stderr_text, telemetry_path = self._run_claude_stream(
                        argv,
                        input_bytes=user_bytes,
                        cwd=Path(cwd_name),
                        home=Path(home_name),
                        request_hash=request_hash,
                        retry_number=retry_number,
                    )
            except ClaudeTransientError as exc:
                last_error = exc
                LOGGER.warning(
                    "[claude] transient retry=%d/%d request_sha256=%s detail=%s",
                    retry_number,
                    self.claude_max_retries,
                    request_hash,
                    exc,
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()
                LOGGER.warning("[claude-lock] released path=%s", self.claude_author_lock_file)

            if last_error is not None:
                if retry_number < self.claude_max_retries:
                    time.sleep(max(0.0, self.claude_retry_delay_seconds))
                    continue
                raise last_error

            if returncode != 0:
                error_text = " ".join(
                    part
                    for part in (
                        stderr_text,
                        str(payload.get("result", "")) if payload else "",
                    )
                    if part
                )
                detail = self._safe_claude_error(error_text)
                category = "access" if self._claude_access_error(error_text) else "cli-exit"
                self._log_claude_failure(
                    started=started,
                    exit_status=returncode,
                    category=category,
                )
                if category == "access":
                    raise RuntimeError(
                        f"{CLAUDE_ACCESS_ERROR_MARKER} Claude Code subscription limit "
                        f"or authentication failure: {detail}"
                    )
                last_error = ClaudeTransientError(
                    f"Claude Code exited with status {returncode}: {detail}; "
                    f"telemetry={telemetry_path}"
                )
                if retry_number < self.claude_max_retries:
                    time.sleep(max(0.0, self.claude_retry_delay_seconds))
                    continue
                raise last_error
            if payload is None:
                self._log_claude_failure(
                    started=started,
                    exit_status=returncode,
                    category="invalid-json",
                )
                raise RuntimeError(
                    f"Claude Code output was not valid JSON; telemetry={telemetry_path}"
                )
            if payload.get("is_error"):
                detail = self._safe_claude_error(payload.get("result", "unknown error"))
                category = "access" if self._claude_access_error(detail) else "cli-result"
                self._log_claude_failure(
                    started=started,
                    exit_status=returncode,
                    category=category,
                )
                if category == "access":
                    raise RuntimeError(
                        f"{CLAUDE_ACCESS_ERROR_MARKER} Claude Code subscription limit "
                        f"or authentication failure: {detail}"
                    )
                raise RuntimeError(
                    f"Claude Code returned an error: {detail}; telemetry={telemetry_path}"
                )
            result = payload.get("result")
            if not isinstance(result, str) or not result.strip():
                self._log_claude_failure(
                    started=started,
                    exit_status=returncode,
                    category="missing-result",
                )
                raise RuntimeError(
                    f"Claude Code JSON output is missing a non-empty result; telemetry={telemetry_path}"
                )
            LOGGER.warning(
                "[claude] response model=%s output_bytes=%d duration_seconds=%.3f ",
                self.model_name,
                len(result.encode("utf-8")),
                time.monotonic() - started,
            )
            return result.strip()
        assert last_error is not None
        raise last_error

    def _is_opus47(self) -> bool:
        return "claude-opus-4-7" in self.model_name

    def _anthropic_thinking_kwargs(self) -> dict[str, object]:
        """Extended thinking is always on (hard-coded 2026-07-17).

        opus-4-7 only supports adaptive thinking (no manual budget); every
        other Claude model gets a manual budget from --synthesizer-reasoning-budget.
        """
        if self._is_opus47():
            return {
                "thinking": {
                    "type": "adaptive",
                    "display": self.anthropic_thinking_display,
                },
                "output_config": {"effort": self.anthropic_effort},
            }

        if self.reasoning_budget_tokens < 1024:
            raise ValueError("reasoning_budget_tokens must be at least 1024")
        if self.reasoning_budget_tokens >= self.max_new_tokens:
            raise ValueError(
                "reasoning_budget_tokens must be less than max_new_tokens"
            )
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.reasoning_budget_tokens,
                "display": self.anthropic_thinking_display,
            }
        }

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using Qwen.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Generated text
        """
        self._ensure_backend_loaded()

        system_prompt = system_prompt + self._synthesis_context_block()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.backend == "codex":
            return self._generate_codex(system_prompt, user_prompt)

        if self.backend == "claude":
            output = self._generate_claude(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "openai":
            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": self.max_new_tokens,
            }
            if not self.model_name.startswith("gpt-5"):
                request_kwargs.update(
                    {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                )
            response = self._client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            output = content.strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "claude-bedrock":
            output = self._generate_bedrock(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "anthropic":
            # Anthropic API takes `system` as a top-level arg, not a message.
            # Claude Opus 4.7 rejects custom sampling params and manual thinking
            # budgets, so extended thinking is enabled through adaptive thinking.
            request_kwargs = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": self.max_new_tokens,
            }
            request_kwargs.update(self._anthropic_thinking_kwargs())
            # Streaming required by the SDK for requests whose estimated
            # runtime exceeds 10 minutes (true with high max_tokens + adaptive
            # thinking). get_final_message() returns the same Message object
            # that non-streaming create() would have returned.
            with self._client.messages.stream(**request_kwargs) as stream:
                response = stream.get_final_message()
            # response.content is a list of content blocks; join text blocks.
            parts = []
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            output = "".join(parts).strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "gemini":
            output = self._generate_gemini(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "vertex":
            output = self._generate_vertex(system_prompt, user_prompt)
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        if self.backend == "vllm":
            from vllm import SamplingParams

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            outputs = self._vllm.generate([text], sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0].outputs[0].text.strip()
            self._log_prompt_io(system_prompt, user_prompt, output)
            return output

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)

        output = response.strip()
        self._log_prompt_io(system_prompt, user_prompt, output)
        return output

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
        max_retries: Optional[int] = None,
        retryable_statuses: Optional[set[int]] = None,
    ) -> dict:
        if max_retries is None:
            max_retries = int(os.environ.get("CSD_API_MAX_RETRIES", "5"))
        retry_base_seconds = float(os.environ.get("CSD_API_RETRY_BASE_SECONDS", "20"))
        if retryable_statuses is None:
            retryable_statuses = {408, 409, 429, 500, 502, 503, 504, 529}
        attempt = 0
        daily_quota_retry = 0
        while True:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Connection": "close", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = response.read().decode("utf-8")
                return json.loads(body)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                error_detail = (
                    "[redacted]"
                    if safe_logging_enabled()
                    else error_body[:1000]
                )
                is_bedrock_daily_quota = (
                    self.backend == "claude-bedrock"
                    and exc.code == 429
                    and "too many tokens per day" in error_body.lower()
                )
                if is_bedrock_daily_quota:
                    retry_seconds = float(
                        os.environ.get("CSD_DAILY_QUOTA_RETRY_SECONDS", "3600")
                    )
                    jitter_seconds = max(
                        0.0,
                        float(os.environ.get("CSD_DAILY_QUOTA_RETRY_JITTER_SECONDS", "300")),
                    )
                    sleep_seconds = max(
                        1.0,
                        random.uniform(
                            max(1.0, retry_seconds - jitter_seconds),
                            retry_seconds + jitter_seconds,
                        ),
                    )
                    daily_quota_retry += 1
                    next_retry = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(time.time() + sleep_seconds),
                    )
                    print(
                        f"[api-retry] {self.backend} HTTP 429 daily token quota; "
                        f"retry indefinitely count={daily_quota_retry} "
                        f"after {sleep_seconds:.1f}s next_retry={next_retry}: "
                        f"{error_detail[:300]}",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    continue
                if exc.code in retryable_statuses and attempt < max_retries:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                    print(
                        f"[api-retry] {self.backend} HTTP {exc.code}; "
                        f"retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s: "
                        f"{error_detail[:300]}",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"{self.backend} generation API returned HTTP {exc.code}: {error_detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Network-level failures (read timeout, connection reset) carry no
                # HTTP status, so the branch above never sees them. Retry them too.
                if attempt < max_retries:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                    print(
                        f"[api-retry] {self.backend} network error ({exc}); "
                        f"retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
                    attempt += 1
                    continue
                raise RuntimeError(
                    f"{self.backend} generation API failed at network level after "
                    f"{max_retries} retries: {exc}"
                ) from exc


    @staticmethod
    def _dedupe_nonempty(values: list[Optional[str]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _is_quota_exhausted_error(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        text = " ".join(
            str(part)
            for part in (
                exc,
                getattr(exc, "response_body", ""),
                getattr(exc, "body", ""),
            )
            if part
        ).lower()
        return status == 429 or "resource_exhausted" in text or "quota" in text

    def _gemini_api_keys(self, primary: Optional[str]) -> list[str]:
        backups = [
            os.environ.get(f"GEMINI_API_KEY_BACKUP_{idx}")
            for idx in range(1, 10)
        ]
        return self._dedupe_nonempty([
            primary,
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
            *backups,
        ])

    def _bedrock_thinking_fields(self) -> dict[str, object]:
        """Extended thinking for the Bedrock Converse API (wired 2026-07-17).

        Bedrock takes the raw Anthropic thinking block via
        additionalModelRequestFields. Only type + budget_tokens — the
        "display" key is an Anthropic-API-only extension. Before this,
        --anthropic-thinking was a silent no-op on the bedrock path: every
        bedrock-author run had effectively been running thinking-OFF.
        """
        if self.reasoning_budget_tokens < 1024:
            raise ValueError("reasoning_budget_tokens must be at least 1024")
        if self.reasoning_budget_tokens >= self.max_new_tokens:
            raise ValueError(
                "reasoning_budget_tokens must be less than max_new_tokens"
            )
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.reasoning_budget_tokens,
            }
        }

    def _generate_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "converse"):
            data = client.converse(
                modelId=self.model_name,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": self.max_new_tokens},
                additionalModelRequestFields=self._bedrock_thinking_fields(),
            )
            parts = data.get("output", {}).get("message", {}).get("content") or []
            return "".join(part.get("text", "") for part in parts).strip()

        base_url = (self.api_base_url or self._bedrock_runtime_base_url()).rstrip("/")
        model = urllib.parse.quote(self.model_name, safe="")
        url = f"{base_url}/model/{model}/converse"
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": self.max_new_tokens,
            },
            "additionalModelRequestFields": self._bedrock_thinking_fields(),
        }
        data = self._post_json(
            url,
            {
                "Authorization": f"Bearer {self.api_key or ''}",
                "Accept": "application/json",
            },
            payload,
        )
        parts = data.get("output", {}).get("message", {}).get("content") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.api_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        model = urllib.parse.quote(self.model_name, safe="")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_new_tokens,
            },
        }
        if self.model_name != "gemini-3.7-flash":
            payload["generationConfig"].update(
                {"temperature": self.temperature, "topP": self.top_p}
            )
        keys = self._gemini_api_keys(self.api_key)
        if not keys:
            keys = [""]
        last_exc: Optional[BaseException] = None
        for idx, api_key in enumerate(keys):
            url = f"{base_url}/models/{model}:generateContent"
            headers = {"x-goog-api-key": api_key}
            try:
                if len(keys) == 1:
                    data = self._post_json(url, headers, payload)
                else:
                    data = self._post_json(
                        url,
                        headers,
                        payload,
                        max_retries=0,
                        retryable_statuses=set(),
                    )
                self._active_gemini_api_key_sha256 = hashlib.sha256(
                    api_key.encode("utf-8")
                ).hexdigest()
                break
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(keys) and self._is_quota_exhausted_error(exc):
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _vertex_project(self) -> str:
        project = (
            os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        if not project:
            raise RuntimeError(
                "VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT is required for Vertex AI generation"
            )
        return project

    def _vertex_location(self) -> str:
        return (
            os.environ.get("VERTEX_AI_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GOOGLE_VERTEX_LOCATION")
            or "global"
        )

    def _vertex_access_token(self) -> str:
        token = self.api_key or os.environ.get("VERTEX_AI_ACCESS_TOKEN")
        if token:
            return token
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI generation requires VERTEX_AI_ACCESS_TOKEN or google-auth "
                "with Application Default Credentials."
            ) from exc
        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError("Application Default Credentials did not provide an access token")
        return token

    def _vertex_auth_headers(self) -> dict[str, str]:
        api_key = (
            os.environ.get("VERTEX_AI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if api_key:
            return {"x-goog-api-key": api_key}
        return {"Authorization": f"Bearer {self._vertex_access_token()}"}

    def _vertex_generate_content(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        base_url = (self.api_base_url or self._default_api_base_url("vertex") or "").rstrip("/")
        project = urllib.parse.quote(self._vertex_project(), safe="")
        location = urllib.parse.quote(self._vertex_location(), safe="")
        model = urllib.parse.quote(model_name, safe="")
        url = (
            f"{base_url}/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        headers = self._vertex_auth_headers()
        primary_key = headers.get("x-goog-api-key")
        if primary_key:
            keys = self._gemini_api_keys(primary_key)
            last_exc: Optional[BaseException] = None
            for idx, api_key in enumerate(keys):
                try:
                    if len(keys) == 1:
                        data = self._post_json(url, {"x-goog-api-key": api_key}, payload)
                    else:
                        data = self._post_json(
                            url,
                            {"x-goog-api-key": api_key},
                            payload,
                            max_retries=0,
                            retryable_statuses=set(),
                        )
                    break
                except Exception as exc:
                    last_exc = exc
                    if idx + 1 < len(keys) and self._is_quota_exhausted_error(exc):
                        continue
                    raise
            else:
                assert last_exc is not None
                raise last_exc
        else:
            data = self._post_json(url, headers, payload)
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts).strip()

    def _generate_vertex(self, system_prompt: str, user_prompt: str) -> str:
        return self._vertex_generate_content(
            self.model_name,
            system_prompt,
            user_prompt,
            self.max_new_tokens,
        )

    def _log_prompt_io(self, system_prompt: str, user_prompt: str, output: str) -> None:
        """Optionally persist exact prompt/response records for debugging."""
        log_dir = os.environ.get("CSD_PROMPT_LOG_DIR")
        if not log_dir:
            return

        self._prompt_log_counter += 1
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "index": self._prompt_log_counter,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "backend": self.backend,
            "model": self.model_name,
        }
        if safe_logging_enabled():
            record["redacted"] = True
            record["system_prompt_metadata"] = text_metadata(system_prompt)
            record["user_prompt_metadata"] = text_metadata(user_prompt)
            record["output_metadata"] = text_metadata(output)
        else:
            record.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "output": output,
                }
            )
        with (path / "prompt_io.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def summarize_rationale_claim(self, rationale: str) -> str:
        """Summarize an attempt rationale into one empirical branch claim.

        This uses a separate small hosted model so synthesis prompts get the
        rationale's causal claim without blunt text truncation. If the summary
        backend is unavailable, return the full rationale rather than silently
        losing context.
        """
        rationale = rationale.strip()
        if not rationale:
            return ""
        if self.backend == "codex":
            system_prompt, user_prompt = self._rationale_summary_messages(rationale)
            try:
                return self._clean_rationale_summary(
                    self._generate_codex(system_prompt, user_prompt)
                )
            except Exception as exc:
                LOGGER.warning(
                    "[codex] rationale summary failed status=fallback error_category=%s",
                    self._safe_codex_error(exc),
                )
                return rationale
        if self.backend == "claude":
            system_prompt, user_prompt = self._rationale_summary_messages(rationale)
            try:
                return self._clean_rationale_summary(
                    self._generate_claude(system_prompt, user_prompt)
                )
            except Exception as exc:
                LOGGER.warning(
                    "[claude] rationale summary failed; using full rationale: %s",
                    self._safe_claude_error(exc),
                )
                return rationale
        backend = os.environ.get("CSD_RATIONALE_SUMMARY_BACKEND", "anthropic").strip().lower()
        if backend in {"", "off", "none", "disabled"}:
            return rationale
        try:
            return self._summarize_rationale_claim_with_backend(rationale, backend, fallback=False)
        except Exception as exc:
            fallback_backend = os.environ.get(
                "CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND",
                "anthropic",
            ).strip().lower()
            if fallback_backend in {"", "off", "none", "disabled"}:
                print(f"[rationale-summary] using full rationale; summary failed: {exc}", flush=True)
                return rationale
            try:
                print(
                    f"[rationale-summary] primary {backend} summary failed; "
                    f"trying {fallback_backend}: {exc}",
                    flush=True,
                )
                return self._summarize_rationale_claim_with_backend(
                    rationale,
                    fallback_backend,
                    fallback=True,
                )
            except Exception as fallback_exc:
                print(
                    "[rationale-summary] using full rationale; "
                    f"primary failed: {exc}; fallback failed: {fallback_exc}",
                    flush=True,
                )
            return rationale

    def _rationale_summary_messages(self, rationale: str) -> tuple[str, str]:
        system_prompt = (
            "Summarize a CSD attempt rationale as a factual empirical branch claim. "
            "Preserve the causal hypothesis and concrete changed knobs. "
            "Do not add advice, evaluation, or facts not present in the rationale. "
            "Return one sentence, at most 35 words."
        )
        return system_prompt, f"Rationale:\n{rationale}"

    def _summarize_rationale_claim_with_backend(
        self,
        rationale: str,
        backend: str,
        *,
        fallback: bool,
    ) -> str:
        if backend == "openai":
            return self._summarize_rationale_claim_openai(rationale, fallback=fallback)
        if backend == "anthropic":
            return self._summarize_rationale_claim_anthropic(rationale, fallback=fallback)
        if backend in {"bedrock", "claude-bedrock"}:
            return self._summarize_rationale_claim_bedrock(rationale, fallback=fallback)
        if backend == "gemini":
            return self._summarize_rationale_claim_gemini(rationale, fallback=fallback)
        if backend == "vertex":
            return self._summarize_rationale_claim_vertex(rationale, fallback=fallback)
        raise ValueError(f"unsupported rationale summary backend: {backend}")

    def _summarize_rationale_claim_bedrock(self, rationale: str, *, fallback: bool = False) -> str:
        token = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not token:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set")
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        base_url = (
            (
                os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
                if fallback
                else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
            )
            or self._bedrock_runtime_base_url()
            or ""
        ).rstrip("/")
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        url = f"{base_url}/model/{urllib.parse.quote(model, safe='')}/converse"
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
            },
        }
        data = self._post_json(
            url,
            {"Authorization": f"Bearer {token}", "Accept": "application/json"},
            payload,
        )
        parts = data.get("output", {}).get("message", {}).get("content") or []
        content = "".join(part.get("text", "") for part in parts)
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_openai(self, rationale: str, *, fallback: bool = False) -> str:
        if fallback:
            raise ValueError("openai fallback is not configured by default")
        api_key = os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if self._summary_client is None:
            from openai import OpenAI

            self._summary_client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL"),
            )
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
            or os.environ.get("OPENAI_RATIONALE_SUMMARY_MODEL")
            or "chat-latest"
        )
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        response = self._summary_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        return self._clean_rationale_summary(response.choices[0].message.content or "")

    def _summarize_rationale_claim_anthropic(self, rationale: str, *, fallback: bool = False) -> str:
        api_key = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if self._summary_anthropic_client is None:
            from anthropic import Anthropic

            client_kwargs = {"api_key": api_key}
            base_url = (
                os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
                if fallback
                else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
            ) or os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                client_kwargs["base_url"] = base_url
            self._summary_anthropic_client = Anthropic(**client_kwargs)
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "claude-haiku-4-5"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        response = self._summary_anthropic_client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        content = "".join(
            getattr(block, "text", "") or ""
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_gemini(self, rationale: str, *, fallback: bool = False) -> str:
        primary_api_key = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_API_KEY")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_API_KEY")
        ) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        api_keys = self._gemini_api_keys(primary_api_key)
        if not api_keys:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "gemini-2.5-flash-lite"
        base_url = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_BASE_URL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_BASE_URL")
        ) or os.environ.get("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        model_path = urllib.parse.quote(model, safe="")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
            },
        }
        last_exc: Optional[BaseException] = None
        for idx, api_key in enumerate(api_keys):
            url = f"{base_url.rstrip('/')}/models/{model_path}:generateContent"
            headers = {"x-goog-api-key": api_key}
            try:
                if len(api_keys) == 1:
                    data = self._post_json(
                        url,
                        headers,
                        payload,
                    )
                else:
                    data = self._post_json(
                        url,
                        headers,
                        payload,
                        max_retries=0,
                        retryable_statuses=set(),
                    )
                break
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(api_keys) and self._is_quota_exhausted_error(exc):
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("empty Gemini rationale summary")
        parts = candidates[0].get("content", {}).get("parts") or []
        content = "".join(part.get("text", "") for part in parts)
        return self._clean_rationale_summary(content)

    def _summarize_rationale_claim_vertex(self, rationale: str, *, fallback: bool = False) -> str:
        model = (
            os.environ.get("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL")
            if fallback
            else os.environ.get("CSD_RATIONALE_SUMMARY_MODEL")
        ) or "gemini-2.5-flash-lite"
        system_prompt, user_prompt = self._rationale_summary_messages(rationale)
        output = self._vertex_generate_content(
            model,
            system_prompt,
            user_prompt,
            int(os.environ.get("CSD_RATIONALE_SUMMARY_MAX_TOKENS", "96")),
        )
        return self._clean_rationale_summary(output)

    @staticmethod
    def _clean_rationale_summary(content: str) -> str:
        summary = " ".join(content.strip().split())
        if not summary:
            raise RuntimeError("empty rationale summary")
        return summary

    def _extract_strategy(self, raw_output: str) -> str:
        """
        Extract the strategy expression from Qwen's output.

        Handles cases where Qwen includes extra text, code blocks, etc.

        Args:
            raw_output: Raw text from Qwen

        Returns:
            Cleaned strategy expression
        """
        # Remove markdown code blocks if present (handles both complete and truncated blocks)
        # First try complete code block
        code_block_pattern = r"```(?:dafny)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, raw_output)
        if match:
            raw_output = match.group(1)
        else:
            # Handle truncated code blocks (no closing fence due to token limit)
            truncated_pattern = r"^```(?:dafny)?\s*([\s\S]*)$"
            match = re.search(truncated_pattern, raw_output.strip())
            if match:
                raw_output = match.group(1)

        # Remove leading/trailing whitespace
        strategy = raw_output.strip()

        # Drop any prose preamble above the rationale block markers.
        marker_idx = strategy.find("// CSD_RATIONALE_BEGIN")
        if marker_idx > 0:
            strategy = strategy[marker_idx:]

        # Heuristic repair: Dafny uses `+` for sequence concatenation; `++` is invalid.
        # Replace `++` when it is used like an operator (surrounded by optional whitespace).
        # This avoids touching tokens like "C++" inside words.
        strategy = re.sub(r"\s*\+\+\s*", " + ", strategy)

        # If it looks like a full function/method definition, extract just the body.
        # We match the *final* '}' so nested braces inside the body are preserved.
        lowered = strategy.lower()
        if ("function" in lowered or "method" in lowered) and "{" in strategy:
            brace_match = re.search(r"\{([\s\S]*)\}\s*$", strategy)
            if brace_match:
                strategy = brace_match.group(1).strip()

        # Ensure the body ends in a reasonable terminator.
        # - Single statements should end with ';'
        # - Block bodies may end with '}' (e.g., if/else/while blocks)
        if strategy:
            last_char = strategy.rstrip()[-1]
            if last_char not in {";", "}"}:
                strategy = strategy.rstrip() + ";"

        return strategy

    def _ensure_rationale_block(
        self,
        strategy_body: str,
        *,
        max_repairs: int = 2,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Ensure the strategy body contains the required rationale markers.

        If missing, attempt a small number of "format repair" generations that rewrite
        the body into the required structure without changing semantics.
        """
        extracted = extract_rationale(strategy_body)
        if extracted.rationale is not None and extracted.has_markers:
            return strategy_body

        current = strategy_body
        for _ in range(max_repairs):
            system_prompt, user_prompt = build_format_repair_prompt(
                current,
                search_memory,
                allowed_helpers=allowed_helpers,
                start_inside_constrained=self._current_start_inside_constrained,
            )
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return repaired
            current = repaired

        return "// CSD_RATIONALE_BEGIN\n// (Auto-injected rationale)\n// CSD_RATIONALE_END\n" + current

    def generate_initial(
        self,
        task_description: str,
        allowed_helpers: list[str] | None = None,
        start_inside_constrained: bool = False,
    ) -> str:
        """
        Generate an initial strategy for the given task.

        Args:
            task_description: Description of what the strategy should accomplish
            start_inside_constrained: Whether this run's decoding surface starts
                already inside the constrained region (EnterObservedConstrainedSpan,
                no visible "<<") rather than outside it (OpenConstrainedSpan, which
                emits "<<"). Told to the author so it enters constrained mode the
                right way for this run instead of guessing.

        Returns:
            Strategy expression (Dafny code)
        """
        # set_task_description also rejects an empty task. It does not touch
        # _current_start_inside_constrained, so that stays a separate line --
        # without it the flag would keep its initial False and every later
        # prompt would enter constrained mode the wrong way.
        self.set_task_description(task_description)
        self._current_start_inside_constrained = start_inside_constrained
        system_prompt, user_prompt = build_initial_prompt(
            task_description,
            allowed_helpers=allowed_helpers,
            start_inside_constrained=start_inside_constrained,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy, allowed_helpers=allowed_helpers)

    def refine_after_verification_error(
        self,
        previous_strategy: str,
        error_message: str,
        behavioral_context: str = "",
        structured_feedback: str = "",
        error_history: str = "",
        strategy_context: str = "",
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after verification failure.

        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny verification error

        Returns:
            New strategy expression
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_verification_error_prompt(
            task_description,
            previous_strategy,
            error_message,
            behavioral_context,
            structured_feedback,
            error_history,
            strategy_context,
            search_memory,
            allowed_helpers=allowed_helpers,
            start_inside_constrained=self._current_start_inside_constrained,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_runtime_error(
        self,
        previous_strategy: str,
        error_traceback: str,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after runtime failure.

        Args:
            previous_strategy: The strategy that failed
            error_traceback: Python traceback

        Returns:
            New strategy expression
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy,
            error_traceback,
            task_description,
            search_memory,
            allowed_helpers=allowed_helpers,
            start_inside_constrained=self._current_start_inside_constrained,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_compilation_error(
        self,
        previous_strategy: str,
        error_message: str,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
    ) -> str:
        """
        Generate a refined strategy after compilation failure.

        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny compilation error

        Returns:
            New strategy expression
        """
        system_prompt, user_prompt = build_compilation_error_prompt(
            previous_strategy,
            error_message,
            search_memory,
            allowed_helpers=allowed_helpers,
            start_inside_constrained=self._current_start_inside_constrained,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def refine_after_evaluation_failure(
        self,
        previous_strategy: str,
        previous_accuracy: float,
        previous_syntax_rate: float,
        num_examples: int,
        goal_accuracy: float,
        goal_syntax_rate: float,
        evaluation_feedback: str,
        best_strategy: str | None = None,
        best_accuracy: float | None = None,
        best_syntax_rate: float | None = None,
        search_memory: str = "",
        allowed_helpers: list[str] | None = None,
        eval_max_seconds_per_example: float | None = None,
        mode_examples: str = "",
        attempt_outcome_ledger: str = "",
    ) -> str:
        """Generate a refined strategy after evaluation failure.

        The model sees the previous attempt's code and scores, the goal,
        the evaluation feedback, and — only when the previous attempt
        regressed from a better-scoring earlier attempt — the best-so-far
        strategy as a positive anchor. It may also see a compact empirical
        outcome ledger so it has global search context without replaying full
        prior strategy bodies.
        """
        task_description = self._current_task_description or "Unknown task"
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            task_description=task_description,
            previous_strategy=previous_strategy,
            previous_accuracy=previous_accuracy,
            previous_syntax_rate=previous_syntax_rate,
            num_examples=num_examples,
            goal_accuracy=goal_accuracy,
            goal_syntax_rate=goal_syntax_rate,
            evaluation_feedback=evaluation_feedback,
            best_strategy=best_strategy,
            best_accuracy=best_accuracy,
            best_syntax_rate=best_syntax_rate,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
            eval_max_seconds_per_example=eval_max_seconds_per_example,
            mode_examples=mode_examples,
            attempt_outcome_ledger=attempt_outcome_ledger,
            start_inside_constrained=self._current_start_inside_constrained,
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(
            strategy,
            search_memory=search_memory,
            allowed_helpers=allowed_helpers,
        )

    def inject_strategy(self, strategy: str) -> str:
        """
        Inject a strategy into the template.

        Args:
            strategy: Strategy expression to inject

        Returns:
            Complete Dafny source code
        """
        return self._template.replace(self.STRATEGY_MARKER, strategy)

    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template
