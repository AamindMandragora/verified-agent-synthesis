#!/usr/bin/env python3
"""Python launcher for the full CSD experiment matrix.

This mirrors run_all_tests.sh while keeping the control flow easier to inspect
and extend. It intentionally preserves the bash runner's public CLI, output
layout, matrix order, cache semantics, and MetaDecode target selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CALLER_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
DEFAULT_GSM_SPLIT_FILE = (
    ROOT_DIR / "environment" / "benchmark_splits" / "gsm_symbolic_crane_proportional.json"
)
DEFAULT_SPIDER_SPLIT_FILE = ROOT_DIR / "environment" / "benchmark_splits" / "spider_dev_proportional.json"

DEFAULT_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct,"
    "Qwen/Qwen2.5-Coder-7B-Instruct,"
    "Qwen/Qwen2.5-Coder-14B-Instruct,"
    "meta-llama/Llama-3.1-8B-Instruct"
)
DEFAULT_BENCHMARKS = "gsm,spider"
DEFAULT_STRATEGIES = "unconstrained,gcd,crane,itergen,metadecode,cars"
DEFAULT_TOKEN_BUDGETS = "1,2,4"
DEFAULT_SYNTH_ITERS = "3,5,10,30,40"
DEFAULT_MAIN_SYNTH_ITERS = "40"
DEFAULT_GEN_MODELS = "sonnet4.6,gpt5.6-sol,gemini"
DEFAULT_STEP_BUDGETS = "256,512,900,1024"
DEFAULT_GSM_MAX_STEPS = "900"
DEFAULT_GPU3_RETRY_QUEUE = ROOT_DIR / "outputs" / "gpu3_retry_queue.jsonl"
VALID_ABLATION_SECTIONS = ("A", "B", "C", "D", "E")
DEFAULT_ABLATION_SECTIONS = ",".join(VALID_ABLATION_SECTIONS)
DEFAULT_SMILES_CLASSES = "acrylates,chain_extenders,isocyanates"
CSD_TARGET_STRATEGIES = ("crane", "itergen")
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|CUDA out of memory|"
    r"CUDA error: out of memory|torch\.cuda\.OutOfMemoryError|"
    r"cumemAllocator|RESOURCE_EXHAUSTED|"
    r"Free memory on device|desired GPU memory utilization|"
    r"Engine core initialization failed",
    re.IGNORECASE,
)
# Quota / credit exhaustion on the AUTHOR-model API (OpenAI / Anthropic / Bedrock).
# These errors are NOT transient — retrying or moving GPUs won't help. We abort
# the whole matrix run so the user is forced to notice and fix the credit issue
# instead of letting every metadecode cell silently fail with empty output.
QUOTA_RE = re.compile(
    r"\[claude-author-access\]|"
    r"insufficient_quota|"
    r"RateLimitError|rate_limit_error|"
    r"credit balance is too low|"
    r"You exceeded your current quota|"
    r"quota.{0,30}exceeded|"
    r"BillingHardLimitReached|"
    r"You have exceeded your.{0,40}quota|"
    r"403 Forbidden.{0,100}billing|"
    r"401 Unauthorized.{0,100}(invalid_api_key|authentication)|"
    r"AccessDeniedException.{0,40}Bedrock",
    re.IGNORECASE,
)


from synthesis.env_utils import load_env_file
from synthesis.evaluate.benchmarks.smiles.dataset import normalize_smiles_classes


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_strategies(value: str) -> list[str]:
    strategies = csv_list(value)
    if not strategies:
        raise SystemExit("No runnable strategies specified.")
    return strategies


def normalize_ablation_sections(value: str) -> set[str]:
    sections = {item.upper() for item in csv_list(value)}
    if not sections:
        raise SystemExit(
            "At least one ablation section is required; choose from "
            f"{','.join(VALID_ABLATION_SECTIONS)}"
        )
    invalid = sorted(sections - set(VALID_ABLATION_SECTIONS))
    if invalid:
        raise SystemExit(
            f"Invalid ablation section(s): {','.join(invalid)}. "
            f"Expected one or more of: {','.join(VALID_ABLATION_SECTIONS)}"
        )
    return sections


def slugify(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def normalize_benchmark(value: str) -> str:
    return "gsm_symbolic" if value == "gsm" else value


def command_text(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def line_count(path: Path) -> int:
    try:
        return len(path.read_text().splitlines())
    except Exception:
        return 0


def canonical(path: Path) -> str:
    return str(path.resolve())


def maybe_int(value: object) -> int | object:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


def maybe_float(value: object) -> float | object:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return value


@dataclass
class Config:
    models: list[str]
    benchmarks: list[str]
    strategies: list[str]
    token_budgets: list[str]
    synth_iters: list[str]
    gen_models: list[str]
    step_budgets: list[str]
    ablation_sections: set[str]
    smiles_classes: list[str]
    eval_backend: str
    device: str
    generation_sample_size: str
    eval_sample_size: str
    gsm_generation_sample_size: str
    gsm_eval_sample_size: str
    eval_max_steps: str
    eval_max_steps_gsm: str
    eval_max_steps_smiles: str
    eval_max_seconds_per_example: str
    accuracy_win_margin: float
    synthesis_max_tokens: str
    vllm_gpu_memory_utilization: str
    vllm_tensor_parallel_size: int
    dafny_path: str
    generated_output_dir: Path
    baseline_output_dir: Path
    ablation_output_dir: Path
    baseline_cache_mode: str
    gsm_split_file: str = ""
    spider_split_file: str = ""
    dry_run: bool = False
    skip_main: bool = False
    skip_ablations: bool = False
    conda_env_path: Path = Path("/apps/conda/advayth2/envs/advayth2")
    cuda_devices: str = "auto"
    cuda_oom_fallback: str = "auto"
    free_gpu_max_used_mb: int = 1024
    gpu_wait_seconds: int = 60
    gpu_wait_timeout_seconds: int = 0
    main_synthesis_iterations: str = DEFAULT_MAIN_SYNTH_ITERS
    gpu3_retry_queue: Path = DEFAULT_GPU3_RETRY_QUEUE
    gpu3_retry_enabled: bool = True

@dataclass
class Runner:
    config: Config
    env: dict[str, str]
    prepared_baselines: set[tuple[str, str, str, str, str]] = field(default_factory=set)
    last_failure_was_author_access: bool = False

    def configure_cuda_devices(self) -> bool:
        from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

        selected = self.resolve_cuda_visible_devices("primary", ())
        if selected:
            selected = limit_cuda_visible_devices(selected) or selected
            self.env["CUDA_VISIBLE_DEVICES"] = selected
            os.environ["CUDA_VISIBLE_DEVICES"] = selected
            if self.config.cuda_devices == "auto" and not CALLER_CUDA_VISIBLE_DEVICES:
                print(
                    f"[env] CUDA_VISIBLE_DEVICES={selected} "
                    f"(auto-selected; max used <= {self.config.free_gpu_max_used_mb} MiB)"
                )
            else:
                print(f"[env] CUDA_VISIBLE_DEVICES={selected}")
        else:
            self.env.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if self.config.dry_run:
                print("[env] CUDA_VISIBLE_DEVICES=<auto> (dry-run; no idle GPU selected)")
            else:
                print("[error] Could not select a CUDA device.", file=sys.stderr)
                return False

        if self.config.cuda_oom_fallback:
            print(
                f"[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK={self.config.cuda_oom_fallback} "
                "(OOM retry; set empty to disable)"
            )
        else:
            print("[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK unset/disabled")
        return True

    def cuda_free_gpu_candidates(self) -> list[str]:
        nvidia_smi = shutil.which("nvidia-smi", path=self.env.get("PATH"))
        if not nvidia_smi:
            return []
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            env=self.env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return []

        candidates: list[tuple[int, int, int, str]] = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4 or not all(fields[:4]):
                continue
            try:
                index = fields[0]
                used = int(float(fields[1]))
                free = int(float(fields[2]))
                util = int(float(fields[3]))
            except ValueError:
                continue
            if used <= self.config.free_gpu_max_used_mb:
                candidates.append((used, util, 1_000_000 - free, index))
        return [candidate[3] for candidate in sorted(candidates)]

    def select_free_cuda_device(self, skip: tuple[str, ...]) -> str | None:
        skip_set = set(skip)
        for gpu in self.cuda_free_gpu_candidates():
            if gpu not in skip_set:
                return gpu
        return None

    def wait_for_free_cuda_device(self, skip: tuple[str, ...]) -> str | None:
        start = time.monotonic()
        while True:
            selected = self.select_free_cuda_device(skip)
            if selected:
                return selected
            if self.config.dry_run:
                return None
            if not shutil.which("nvidia-smi", path=self.env.get("PATH")):
                print(
                    "[error] RUN_ALL_TESTS_CUDA_DEVICES=auto requires nvidia-smi; "
                    "set RUN_ALL_TESTS_CUDA_DEVICES explicitly to override.",
                    file=sys.stderr,
                )
                return None
            if self.config.gpu_wait_timeout_seconds > 0:
                elapsed = int(time.monotonic() - start)
                if elapsed >= self.config.gpu_wait_timeout_seconds:
                    print(
                        f"[error] No GPU became idle within "
                        f"{self.config.gpu_wait_timeout_seconds}s.",
                        file=sys.stderr,
                    )
                    return None
            print(
                f"[env] No GPU with <= {self.config.free_gpu_max_used_mb} MiB used; "
                f"waiting {self.config.gpu_wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(self.config.gpu_wait_seconds)

    def resolve_cuda_visible_devices(self, role: str, skip: tuple[str, ...]) -> str | None:
        requested = (
            self.config.cuda_oom_fallback
            if role == "fallback"
            else self.config.cuda_devices
        )
        if not requested:
            return None
        if requested != "auto":
            return requested
        if role != "fallback" and CALLER_CUDA_VISIBLE_DEVICES:
            from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

            return limit_cuda_visible_devices(CALLER_CUDA_VISIBLE_DEVICES)
        selected = self.wait_for_free_cuda_device(skip)
        if selected:
            from synthesis.evaluate.benchmarks.common.model_utils import limit_cuda_visible_devices

            return limit_cuda_visible_devices(selected)
        return None

    def generation_sample_size(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return self.config.gsm_generation_sample_size
        return self.config.generation_sample_size

    def evaluation_sample_size(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return self.config.gsm_eval_sample_size
        return self.config.eval_sample_size

    def ensure_split_manifests(self) -> None:
        """Require tracked stratified manifests under environment/benchmark_splits/."""
        normalized = {normalize_benchmark(benchmark) for benchmark in self.config.benchmarks}
        if "gsm_symbolic" in normalized:
            gsm_path = Path(self.config.gsm_split_file)
            if not gsm_path.is_file():
                raise SystemExit(
                    f"GSM split manifest not found: {gsm_path}\n"
                    "Regenerate tracked splits with:\n"
                    "  python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits"
                )

        if "spider" in normalized:
            spider_path = Path(self.config.spider_split_file)
            if not spider_path.is_file():
                raise SystemExit(
                    f"Spider split manifest not found: {spider_path}\n"
                    "Regenerate tracked splits with:\n"
                    "  python -m synthesis.evaluate.benchmarks.write_fixed_benchmark_splits"
                )

    def gsm_split_name_for_role(self, role: str) -> str:
        """
        Map generation/evaluation roles to manifest keys.

        The default CRANE proportional manifest has train_size=0; use the eval
        pool for synthesis as well so metadecode tunes on the same fixed subset.
        """
        path = Path(self.config.gsm_split_file)
        if not path.is_file():
            return "eval" if role != "train" else "train"
        manifest = json.loads(path.read_text())
        if role == "train" and not manifest.get("train_indices"):
            return "eval"
        return "train" if role == "train" else "eval"

    def add_generation_split_flags(self, cmd: list[str], benchmark: str) -> None:
        if self.config.gsm_split_file and benchmark == "gsm_symbolic":
            cmd += [
                "--gsm-split-file",
                self.config.gsm_split_file,
                "--gsm-split-name",
                self.gsm_split_name_for_role("train"),
            ]
        if self.config.spider_split_file and benchmark == "spider":
            cmd += ["--spider-split-file", self.config.spider_split_file, "--spider-split-name", "train"]

    def add_evaluation_split_flags(self, cmd: list[str], benchmark: str) -> None:
        if self.config.gsm_split_file and benchmark == "gsm_symbolic":
            cmd += [
                "--gsm-split-file",
                self.config.gsm_split_file,
                "--gsm-split-name",
                self.gsm_split_name_for_role("eval"),
            ]
        if self.config.spider_split_file and benchmark == "spider":
            cmd += ["--spider-split-file", self.config.spider_split_file, "--spider-split-name", "eval"]

    def add_vllm_parallel_flags(self, cmd: list[str]) -> None:
        if self.config.eval_backend != "vllm":
            return
        cmd += [
            "--vllm-tensor-parallel-size",
            str(self.config.vllm_tensor_parallel_size),
        ]

    def split_metadata_for(self, benchmark: str, role: str) -> dict[str, str | None]:
        if benchmark == "gsm_symbolic":
            return {
                "split_file": self.config.gsm_split_file or None,
                "split_name": self.gsm_split_name_for_role(role),
            }
        if benchmark == "spider":
            return {
                "split_file": self.config.spider_split_file or None,
                "split_name": "train" if role == "train" else "eval",
            }
        return {"split_file": None, "split_name": None}

    def matrix_case_metadata(
        self,
        *,
        phase: str,
        strategy: str,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        command: list[str],
        synth_iter: str | None = None,
        gen_profile: str | None = None,
        generation_backend: str | None = None,
        generation_model: str | None = None,
        smiles_class: str = "",
        required_accuracy: float | None = None,
        required_syntax: float | None = None,
        target_accuracy_strategy: str | None = None,
        target_accuracy_path: str | None = None,
        target_syntax_strategy: str | None = None,
        target_syntax_path: str | None = None,
    ) -> dict:
        return {
            "phase": phase,
            "strategy": strategy,
            "benchmark": benchmark,
            "smiles_class": smiles_class or None,
            "eval_model": eval_model,
            "eval_backend": self.config.eval_backend,
            "generation_profile": gen_profile,
            "generation_backend": generation_backend,
            "generation_model": generation_model,
            "synthesis_iterations": maybe_int(synth_iter) if synth_iter is not None else None,
            "eval_step_token_budget": maybe_int(token_budget),
            "eval_max_steps": maybe_int(max_steps),
            "sample_sizes": {
                "synthesis_eval": maybe_int(self.generation_sample_size(benchmark)),
                "final_eval": maybe_int(self.evaluation_sample_size(benchmark)),
            },
            "thresholds": {
                "min_accuracy": required_accuracy,
                "min_syntax_rate": required_syntax,
                "accuracy_win_margin": self.config.accuracy_win_margin,
                "target_accuracy_strategy": target_accuracy_strategy,
                "target_accuracy_path": target_accuracy_path,
                "target_syntax_strategy": target_syntax_strategy,
                "target_syntax_path": target_syntax_path,
            },
            "runtime_controls": {
                "eval_max_seconds_per_example": maybe_float(self.config.eval_max_seconds_per_example),
                "vllm_gpu_memory_utilization": maybe_float(self.config.vllm_gpu_memory_utilization),
                "vllm_tensor_parallel_size": self.config.vllm_tensor_parallel_size,
            },
            "synthesis_controls": {
                "max_tokens": maybe_int(self.config.synthesis_max_tokens),
            },
            "splits": {
                benchmark: self.split_metadata_for(benchmark, "eval"),
            },
            "cache": {
                "baseline_cache_mode": self.config.baseline_cache_mode,
            },
            "command": command_text(command),
        }

    def annotate_result_json(self, path: Path, metadata: dict) -> None:
        if self.config.dry_run:
            return
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[warn] Could not annotate result JSON {path}: {exc}", file=sys.stderr)
            return
        payload["matrix_metadata"] = metadata
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def running_on_gpu3(self) -> bool:
        visible = (self.env.get("CUDA_VISIBLE_DEVICES") or CALLER_CUDA_VISIBLE_DEVICES or "").strip()
        if not visible:
            return False
        return [item.strip() for item in visible.split(",") if item.strip()] == ["3"]

    def enqueue_gpu3_retry(self, cmd: list[str], *, reason: str, case: dict[str, object]) -> None:
        if self.config.dry_run or not self.config.gpu3_retry_enabled or self.running_on_gpu3():
            return
        queue_path = self.config.gpu3_retry_queue
        record = {
            "id": hashlib.sha256(command_text(cmd).encode("utf-8")).hexdigest()[:16],
            "enqueued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "case": case,
            "cmd": [str(part) for part in cmd],
        }
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            with queue_path.open("a") as queue_file:
                try:
                    import fcntl

                    fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
                queue_file.write(line)
                queue_file.flush()
                try:
                    os.fsync(queue_file.fileno())
                except OSError:
                    pass
        except Exception as exc:
            print(f"[warn] Could not enqueue GPU3 retry in {queue_path}: {exc}", file=sys.stderr)
            return
        print(f"[retry-queue] queued GPU3 retry {record['id']} ({reason}) -> {queue_path}")

    def run_cmd(self, cmd: list[str], *, abort_on_quota: bool = True) -> bool:
        self.last_failure_was_author_access = False
        if self.config.dry_run:
            print(f"[dry-run] {command_text(cmd)}")
            return True

        primary = self.resolve_cuda_visible_devices("primary", ())
        if not primary:
            print(f"[error] Could not select a CUDA device for command: {command_text(cmd)}", file=sys.stderr)
            return False

        if not self.config.cuda_oom_fallback:
            print(f"[run] {command_text(cmd)}")
            return subprocess.run(cmd, env={**self.env, "CUDA_VISIBLE_DEVICES": primary}).returncode == 0

        print(f"[run] CUDA_VISIBLE_DEVICES={primary} {command_text(cmd)}")
        with tempfile.NamedTemporaryFile("w+", prefix="run_all_tests_cuda_try.", delete=False) as log:
            log_path = Path(log.name)
        try:
            proc = subprocess.Popen(
                cmd,
                env={**self.env, "CUDA_VISIBLE_DEVICES": primary},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            with log_path.open("w") as log_file:
                for line in proc.stdout:
                    print(line, end="")
                    log_file.write(line)
            return_code = proc.wait()
            if return_code == 0:
                return True
            log_text = log_path.read_text(errors="ignore")
            # Hard-stop the whole matrix on author-model quota / credit exhaustion.
            # These are NOT transient — retrying or switching GPUs won't recover.
            if QUOTA_RE.search(log_text):
                self.last_failure_was_author_access = True
                quota_excerpt = "\n".join(
                    ln for ln in log_text.splitlines()
                    if QUOTA_RE.search(ln)
                )[:1500]
                if not abort_on_quota:
                    print(
                        "\n[warn] Author-model quota / credit / auth error detected in "
                        "optional subprocess output; treating this cell as skipped/failed "
                        "and continuing the matrix.\n"
                        f"Matched lines (truncated):\n{quota_excerpt}\n",
                        file=sys.stderr,
                    )
                    return False
                print(
                    "\n[FATAL] Author-model quota / credit / auth error detected in subprocess output. "
                    "Aborting the matrix run so the user sees this immediately.\n"
                    f"Matched lines (truncated):\n{quota_excerpt}\n",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            if not OOM_RE.search(log_text):
                return False
        finally:
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass

        fallback = self.resolve_cuda_visible_devices("fallback", (primary,))
        if not fallback:
            print(
                f"[warn] CUDA OOM on CUDA_VISIBLE_DEVICES={primary}; "
                "no fallback CUDA device available",
                file=sys.stderr,
            )
            return False

        print(
            f"[warn] CUDA OOM on CUDA_VISIBLE_DEVICES={primary}; "
            f"retrying with CUDA_VISIBLE_DEVICES={fallback}",
            file=sys.stderr,
        )
        print(f"[run] CUDA_VISIBLE_DEVICES={fallback} {command_text(cmd)}")
        return subprocess.run(cmd, env={**self.env, "CUDA_VISIBLE_DEVICES": fallback}).returncode == 0

    def openai_generation_available(self, gen_profile: str) -> bool:
        if gen_profile != "gpt5.5":
            return True
        api_key = (self.env.get("OPENAI_API_KEY") or "").strip()
        if api_key:
            return True
        print(
            "[skip] OpenAI generation profile gpt5.5 skipped because OPENAI_API_KEY is not set.",
            file=sys.stderr,
        )
        return False

    def metadecode_task(self, benchmark: str) -> str:
        if benchmark == "gsm_symbolic":
            return "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters."
        if benchmark == "spider":
            return "Generate a single valid SQL query using only the provided schema context. Only output the SQL query."
        if benchmark == "smiles":
            return "Generate valid SMILES strings that match the requested molecular class while maintaining parser-valid output."
        return "Generate parser-valid benchmark answers."

    def resolve_gen_profile(self, profile: str) -> tuple[str, str]:
        anthropic_opus47 = self.env.get("ANTHROPIC_OPUS_MODEL", "claude-opus-4-7")
        anthropic_sonnet46 = self.env.get("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-6")
        bedrock_opus47 = (
            self.env.get("BEDROCK_OPUS_MODEL")
            or self.env.get("BEDROCK_GENERATION_MODEL")
            or self.env.get("AWS_BEDROCK_GENERATION_MODEL")
            or "us.anthropic.claude-opus-4-7"
        )
        bedrock_sonnet46 = (
            self.env.get("BEDROCK_SONNET_MODEL")
            or self.env.get("AWS_BEDROCK_SONNET_MODEL")
            or "us.anthropic.claude-sonnet-4-6"
        )
        openai_gpt = self.env.get("OPENAI_GENERATION_MODEL", "gpt-5.5")
        gemini_pro = self.env.get("GEMINI_GENERATION_MODEL", "gemini-3-pro-preview")
        vertex_gemini = (
            self.env.get("GEMINI_VERTEX_MODEL")
            or self.env.get("VERTEX_GEMINI_MODEL")
            or gemini_pro
        )
        if profile == "gpt5.5":
            return "openai", openai_gpt
        if profile == "gpt5.6-sol":
            return "codex", "gpt-5.6-sol"
        if profile == "opus4.7":
            if self.env.get("CSD_OPUS47_BACKEND", "").strip().lower() == "bedrock":
                raise ValueError(
                    "Bedrock generation profiles are disabled for the experimental matrix."
                )
            return "anthropic", anthropic_opus47
        if profile == "sonnet4.6":
            warnings.warn(
                "generation profile 'sonnet4.6' now uses Claude Code Max Opus 5; "
                "use 'anthropic-sonnet4.6' for the direct Anthropic API",
                FutureWarning,
                stacklevel=2,
            )
            return "claude", "claude-opus-5"
        if profile == "claude-sonnet4.6":
            return "claude", "claude-opus-5"
        if profile == "anthropic-sonnet4.6":
            return "anthropic", anthropic_sonnet46
        if profile in {"claude-bedrock-sonnet4.6", "bedrock-sonnet4.6"}:
            raise ValueError(
                "Bedrock generation profiles are disabled for the experimental matrix."
            )
        if profile == "bedrock-opus4.7" or profile == "bedrock":
            raise ValueError(
                "Bedrock generation profiles are disabled for the experimental matrix."
            )
        if profile.startswith("bedrock:"):
            model = profile.split(":", 1)[1].strip()
            if not model:
                raise ValueError("bedrock: profiles must include a Bedrock model id.")
            raise ValueError(
                "Bedrock generation profiles are disabled for the experimental matrix."
            )
        if profile == "gemini":
            if self.env.get("CSD_GEMINI_BACKEND", "").strip().lower() == "vertex":
                return "vertex", vertex_gemini
            return "gemini", gemini_pro
        if profile == "vertex-gemini" or profile == "vertex":
            return "vertex", vertex_gemini
        if profile.startswith("vertex:"):
            model = profile.split(":", 1)[1].strip()
            if not model:
                raise ValueError("vertex: profiles must include a Vertex AI model id.")
            return "vertex", model
        if profile == "gemini-pro":
            raise ValueError(
                "Bedrock-backed gemini-pro is disabled for the experimental matrix; "
                "use the direct gemini profile instead."
            )
        raise ValueError(
            f"Unknown generation profile: {profile}. "
            "Allowed profiles are sonnet4.6, opus4.7, gpt5.5, gpt5.6-sol, and gemini."
        )

    def baseline_case_key(
        self,
        strategy: str,
        model_slug: str,
        benchmark_key: str,
        token_budget: str,
        max_steps: str,
    ) -> tuple[str, str, str, str, str]:
        return strategy, model_slug, benchmark_key, token_budget, max_steps

    def baseline_json_complete(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        answers = payload.get("answers")
        if not isinstance(answers, list) or not answers:
            return False
        return all(isinstance(row, dict) and "generated_answer" in row for row in answers)

    def baseline_json_matches_strategy(self, path: Path, strategy: str) -> bool:
        if strategy != "crane":
            return True
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        return payload.get("metrics", {}).get("adapter") in (
            "crane_shared_evaluator",
            "crane_repo",
        )

    def baseline_json_usable(self, path: Path, strategy: str) -> bool:
        return (
            path.is_file()
            and path.stat().st_size > 20
            and self.baseline_json_complete(path)
            and self.baseline_json_matches_strategy(path, strategy)
        )

    def benchmark_key(self, benchmark: str, smiles_class: str = "") -> str:
        if benchmark == "smiles":
            return f"{benchmark}__class_{slugify(smiles_class)}"
        return benchmark

    def fixed_baseline_path(
        self,
        strategy: str,
        eval_model: str,
        benchmark: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> Path:
        model_slug = slugify(eval_model)
        key = self.benchmark_key(benchmark, smiles_class)
        return (
            self.config.baseline_output_dir
            / strategy
            / model_slug
            / f"{key}__tb{token_budget}__ms{max_steps}.json"
        )

    def best_csd_baseline_targets(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> tuple[float, str, str, str, float, str, str, str]:
        best_accuracy: tuple[float, str, str, str] | None = None
        best_syntax: tuple[float, str, str, str] | None = None
        for strategy in CSD_TARGET_STRATEGIES:
            path = self.fixed_baseline_path(
                strategy, eval_model, benchmark, token_budget, max_steps, smiles_class
            )
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            answers = payload.get("answers")
            if not isinstance(answers, list) or not answers:
                continue
            if not all(isinstance(row, dict) and "generated_answer" in row for row in answers):
                continue
            if strategy == "crane" and payload.get("metrics", {}).get("adapter") not in (
                "crane_shared_evaluator",
                "crane_repo",
            ):
                continue
            accuracy = payload.get("accuracy")
            if isinstance(accuracy, (int, float)):
                candidate = (float(accuracy), strategy, str(path), f"{float(accuracy):.1%}")
                if best_accuracy is None or candidate[0] > best_accuracy[0]:
                    best_accuracy = candidate
            syntax_rate = payload.get("syntax_rate")
            if isinstance(syntax_rate, (int, float)):
                candidate = (float(syntax_rate), strategy, str(path), f"{float(syntax_rate):.1%}")
                if best_syntax is None or candidate[0] > best_syntax[0]:
                    best_syntax = candidate

        if best_accuracy is None:
            best_accuracy = (0.0, "none", "", "0.0%")
        if best_syntax is None:
            best_syntax = (0.0, "none", "", "0.0%")
        else:
            # CRANE/IterGen hit near-100% syntax by grammar construction. Requiring an
            # unconstrained-decoded synth attempt to match that triggers spurious
            # threshold-impossible early stops in evaluator.py even when accuracy is
            # already beating CRANE. Clip the target so attempts can still win.
            SYNTAX_CEILING = 0.90
            if best_syntax[0] > SYNTAX_CEILING:
                clipped_pct = f"{SYNTAX_CEILING:.1%} (clipped from {best_syntax[0]:.1%})"
                best_syntax = (SYNTAX_CEILING, best_syntax[1], best_syntax[2], clipped_pct)
        return (*best_accuracy, *best_syntax)

    def accuracy_target_with_margin(self, baseline_accuracy: float, target_strategy: str) -> float:
        if target_strategy == "none":
            return 0.0
        if self.config.accuracy_win_margin <= 0:
            return math.nextafter(baseline_accuracy, 1.0)
        return min(1.0, baseline_accuracy + self.config.accuracy_win_margin)

    def ensure_csd_target_baselines(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> None:
        for strategy in CSD_TARGET_STRATEGIES:
            ok = self.run_fixed_strategy_case(
                strategy,
                benchmark,
                eval_model,
                token_budget,
                max_steps,
                smiles_class,
                phase="target_baseline",
            )
            if not ok:
                print(
                    f"[warn] Could not prepare {strategy} baseline for "
                    f"benchmark={benchmark} eval_model={eval_model} "
                    f"token_budget={token_budget} max_steps={max_steps} "
                    f"smiles_class={smiles_class or '<none>'}",
                    file=sys.stderr,
                )

    def run_fixed_strategy_case(
        self,
        strategy: str,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
        phase: str = "baseline",
    ) -> bool:
        if benchmark == "smiles" and not smiles_class:
            print("Internal error: SMILES fixed-strategy run requires a class.", file=sys.stderr)
            return False

        model_slug = slugify(eval_model)
        key = self.benchmark_key(benchmark, smiles_class)
        out_json = self.fixed_baseline_path(
            strategy, eval_model, benchmark, token_budget, max_steps, smiles_class
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        case_key = self.baseline_case_key(strategy, model_slug, key, token_budget, max_steps)
        allow_cache_reuse = (
            self.config.baseline_cache_mode == "reuse" or case_key in self.prepared_baselines
        )

        if allow_cache_reuse and self.baseline_json_usable(out_json, strategy):
            self.prepared_baselines.add(case_key)
            print(f"[skip] {out_json} already exists ({line_count(out_json)} lines). Delete it to re-run.")
            return True
        if out_json.exists():
            if self.config.baseline_cache_mode == "refresh" and case_key not in self.prepared_baselines:
                print(f"[rerun] {out_json} exists but --recompute-baselines was requested.")
            else:
                print(f"[rerun] {out_json} exists but is incomplete, corrupt, or from an obsolete adapter.")

        cmd = [
            "python",
            "-m",
            "synthesis.evaluate.run_legacy_fixed_strategy",
            "--strategy",
            strategy,
            "--dataset",
            benchmark,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--device",
            self.config.device,
            "--eval-sample-size",
            self.evaluation_sample_size(benchmark),
            "--eval-max-steps",
            max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--output-json",
            str(out_json),
        ]
        self.add_vllm_parallel_flags(cmd)
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]
        self.add_evaluation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += [
                "--smiles-classes",
                smiles_class,
                "--smiles-samples-per-class",
                self.evaluation_sample_size(benchmark),
            ]

        if self.run_cmd(cmd):
            self.prepared_baselines.add(case_key)
            self.annotate_result_json(
                out_json,
                self.matrix_case_metadata(
                    phase=phase,
                    strategy=strategy,
                    benchmark=benchmark,
                    eval_model=eval_model,
                    token_budget=token_budget,
                    max_steps=max_steps,
                    command=cmd,
                    smiles_class=smiles_class,
                ),
            )
            return True
        return False

    def run_fixed_strategy_cases(
        self,
        strategy: str,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        phase: str = "baseline",
    ) -> None:
        if benchmark == "smiles":
            for smiles_class in self.config.smiles_classes:
                self.run_fixed_strategy_case(
                    strategy,
                    benchmark,
                    eval_model,
                    token_budget,
                    max_steps,
                    smiles_class,
                    phase=phase,
                )
            return
        self.run_fixed_strategy_case(
            strategy, benchmark, eval_model, token_budget, max_steps, phase=phase
        )

    def metadecode_final_eval_command(
        self,
        compiled_module: Path,
        out_json: Path,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        max_steps: str,
        smiles_class: str = "",
    ) -> list[str]:
        cmd = [
            "python",
            "-m",
            "synthesis.scripts.reevaluate_compiled_csd",
            str(compiled_module),
            "--dataset",
            benchmark,
            "--eval-model",
            eval_model,
            "--eval-backend",
            self.config.eval_backend,
            "--device",
            self.config.device,
            "--sample-size",
            self.evaluation_sample_size(benchmark),
            "--max-steps",
            max_steps,
            "--step-token-budget",
            token_budget,
            "--vllm-gpu-memory-utilization",
            self.config.vllm_gpu_memory_utilization,
            "--output-json",
            str(out_json),
        ]
        self.add_vllm_parallel_flags(cmd)
        self.add_evaluation_split_flags(cmd, benchmark)
        if benchmark == "smiles":
            cmd += ["--smiles-classes", smiles_class]
        return cmd

    def run_metadecode_case(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        synth_iter: str,
        gen_profile: str,
        max_steps: str,
        smiles_class: str = "",
        phase: str = "metadecode",
    ) -> bool:
        if benchmark == "smiles" and not smiles_class:
            print("Internal error: SMILES metadecode run requires a class.", file=sys.stderr)
            return False

        if not self.openai_generation_available(gen_profile):
            return True

        backend, generation_model = self.resolve_gen_profile(gen_profile)
        model_slug = slugify(eval_model)
        gen_slug = slugify(gen_profile)
        class_suffix = f"_class_{slugify(smiles_class)}" if benchmark == "smiles" else ""
        key = self.benchmark_key(benchmark, smiles_class)
        run_name = (
            f"metadecode_{benchmark}_{model_slug}_{gen_slug}_"
            f"iter{synth_iter}_tb{token_budget}_ms{max_steps}{class_suffix}"
        )
        task = self.metadecode_task(benchmark)

        self.ensure_csd_target_baselines(benchmark, eval_model, token_budget, max_steps, smiles_class)
        (
            target_accuracy,
            target_strategy,
            _target_path,
            target_percent,
            target_syntax,
            target_syntax_strategy,
            _target_syntax_path,
            target_syntax_percent,
        ) = self.best_csd_baseline_targets(benchmark, eval_model, token_budget, max_steps, smiles_class)
        required_accuracy = self.accuracy_target_with_margin(target_accuracy, target_strategy)

        if target_strategy == "none" and target_syntax_strategy == "none":
            print(
                f"[target] metadecode {key}/{model_slug} tb{token_budget} ms{max_steps}: "
                "no valid CRANE/IterGen baseline found; passing --min-accuracy 0.0 --min-syntax-rate 0.0"
            )
        else:
            print(
                f"[target] metadecode {key}/{model_slug} tb{token_budget} ms{max_steps}: "
                f"best CSD baseline accuracy {target_strategy}={target_percent}; "
                f"accuracy target={required_accuracy:.1%} (+{self.config.accuracy_win_margin:.1%}); "
                f"syntax {target_syntax_strategy}={target_syntax_percent}; "
                f"passing --min-accuracy {required_accuracy:.12g} --min-syntax-rate {target_syntax:.12g}"
            )

        cmd = [
            "python",
            "-m",
            "synthesis.run_synthesis",
            "--task",
            task,
            "--dataset",
            benchmark,
            "--generation-model",
            generation_model,
            "--generation-backend",
            backend,
            "--eval-model",
            eval_model,
            "--max-iterations",
            synth_iter,
            "--min-accuracy",
            f"{required_accuracy:.12g}",
            "--min-syntax-rate",
            f"{target_syntax:.12g}",
            "--eval-sample-size",
            self.generation_sample_size(benchmark),
            "--eval-max-steps",
            max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--eval-max-seconds-per-example",
            self.config.eval_max_seconds_per_example,
            "--max-tokens",
            self.config.synthesis_max_tokens,
            "--device",
            self.config.device,
        ]
        if benchmark == "smiles":
            cmd += [
                "--smiles-samples-per-class",
                self.generation_sample_size(benchmark),
                "--smiles-classes",
                smiles_class,
            ]
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]

        retry_case = {
            "phase": phase,
            "benchmark": benchmark,
            "eval_model": eval_model,
            "token_budget": token_budget,
            "synth_iter": synth_iter,
            "gen_profile": gen_profile,
            "max_steps": max_steps,
            "smiles_class": smiles_class,
            "run_name": run_name,
        }
        if not self.run_cmd(cmd, abort_on_quota=False):
            if self.last_failure_was_author_access:
                print(
                    f"[skip] Metadecode author model unavailable for benchmark={benchmark} "
                    f"eval_model={eval_model} token_budget={token_budget} iter={synth_iter} "
                    f"gen={gen_profile} max_steps={max_steps}; continuing matrix.",
                    file=sys.stderr,
                )
                return True
            print(
                f"[warn] Metadecode synthesis failed for benchmark={benchmark} "
                f"eval_model={eval_model} token_budget={token_budget} iter={synth_iter} "
                f"gen={gen_profile} max_steps={max_steps}",
                file=sys.stderr,
            )
            self.enqueue_gpu3_retry(cmd, reason="synthesis_subprocess_failed", case=retry_case)
            return True

        out_json = (
            self.config.baseline_output_dir
            / "metadecode"
            / model_slug
            / f"{key}__tb{token_budget}__ms{max_steps}__gen{gen_slug}__iter{synth_iter}.json"
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if self.config.dry_run:
            final_cmd = self.metadecode_final_eval_command(
                Path(f"<{self.config.generated_output_dir}/.../python/{run_name}/GeneratedCSD.py>"),
                out_json,
                benchmark,
                eval_model,
                token_budget,
                max_steps,
                smiles_class,
            )
            print(f"[dry-run] {command_text(final_cmd)}")
            return True

        latest_file = self.config.generated_output_dir / "latest_run.txt"
        if not latest_file.is_file():
            print(f"[warn] No latest run file found after synthesis: {latest_file}", file=sys.stderr)
            self.enqueue_gpu3_retry(cmd, reason="missing_latest_run", case=retry_case)
            return True
        run_dir = Path(latest_file.read_text().strip())
        success_report = run_dir / "results" / "success_report.json"
        if not success_report.is_file():
            print(f"[warn] No success report found for run: {run_dir}", file=sys.stderr)
            self.enqueue_gpu3_retry(cmd, reason="missing_success_report", case=retry_case)
            return True
        try:
            report = json.loads(success_report.read_text())
        except Exception as exc:
            print(f"[warn] Could not read success report {success_report}: {exc}", file=sys.stderr)
            self.enqueue_gpu3_retry(cmd, reason="unreadable_success_report", case=retry_case)
            return True
        compiled_dir = report.get("compiled_dir")
        if not compiled_dir:
            print(f"[warn] Success report does not contain compiled_dir: {success_report}", file=sys.stderr)
            self.enqueue_gpu3_retry(cmd, reason="missing_compiled_dir", case=retry_case)
            return True
        compiled_module = Path(compiled_dir) / "GeneratedCSD.py"
        if not compiled_module.is_file():
            print(f"[warn] Compiled GeneratedCSD.py not found: {compiled_module}", file=sys.stderr)
            self.enqueue_gpu3_retry(cmd, reason="missing_compiled_module", case=retry_case)
            return True
        final_eval_cmd = self.metadecode_final_eval_command(
            compiled_module,
            out_json,
            benchmark,
            eval_model,
            token_budget,
            max_steps,
            smiles_class,
        )
        self.run_cmd(final_eval_cmd)
        self.annotate_result_json(
            out_json,
            self.matrix_case_metadata(
                phase=phase,
                strategy="metadecode",
                benchmark=benchmark,
                eval_model=eval_model,
                token_budget=token_budget,
                max_steps=max_steps,
                command=final_eval_cmd,
                synth_iter=synth_iter,
                gen_profile=gen_profile,
                generation_backend=backend,
                generation_model=generation_model,
                smiles_class=smiles_class,
                required_accuracy=required_accuracy,
                required_syntax=target_syntax,
                target_accuracy_strategy=target_strategy,
                target_accuracy_path=_target_path,
                target_syntax_strategy=target_syntax_strategy,
                target_syntax_path=_target_syntax_path,
            ),
        )
        return True

    def eval_max_steps_for(self, benchmark: str) -> str:
        # SMILES baselines spend the full step budget grinding through long invalid
        # molecules (`1C2C3C...C252C`). 750 covers the longest naturally-terminated
        # valid SMILES seen in unconstrained baselines (688) with headroom and cuts
        # ~17% of wasted compute on invalid runs.
        if benchmark == "gsm_symbolic":
            return self.config.eval_max_steps_gsm
        if benchmark == "smiles":
            return self.config.eval_max_steps_smiles
        return self.config.eval_max_steps

    def run_metadecode_cases(
        self,
        benchmark: str,
        eval_model: str,
        token_budget: str,
        synth_iter: str,
        gen_profile: str,
        max_steps: str,
        phase: str = "metadecode",
    ) -> None:
        if benchmark == "smiles":
            for smiles_class in self.config.smiles_classes:
                self.run_metadecode_case(
                    benchmark,
                    eval_model,
                    token_budget,
                    synth_iter,
                    gen_profile,
                    max_steps,
                    smiles_class,
                    phase=phase,
                )
            return
        self.run_metadecode_case(
            benchmark, eval_model, token_budget, synth_iter, gen_profile, max_steps, phase=phase
        )

    def print_matrix_header(self) -> None:
        print("=== run_all_tests matrix ===")
        print(f"models: {' '.join(self.config.models)}")
        print(f"benchmarks: {' '.join(self.config.benchmarks)}")
        print(f"strategies: {' '.join(self.config.strategies)}")
        print(f"token budgets: {' '.join(self.config.token_budgets)}")
        print(f"step budgets (ablation): {' '.join(self.config.step_budgets)}")
        print(
            "ablation sections: "
            f"{' '.join(section for section in VALID_ABLATION_SECTIONS if section in self.config.ablation_sections)}"
        )
        print(f"synthesis iters (metadecode): {' '.join(self.config.synth_iters)}")
        print(f"generation models (metadecode): {' '.join(self.config.gen_models)}")
        if "smiles" in {normalize_benchmark(benchmark) for benchmark in self.config.benchmarks}:
            print(f"SMILES classes: {' '.join(self.config.smiles_classes)}")
        print(
            f"eval max steps (main): {self.config.eval_max_steps} "
            f"(gsm: {self.config.eval_max_steps_gsm}, smiles: {self.config.eval_max_steps_smiles})"
        )
        print(
            "split policy: "
            f"GSM generation={self.config.gsm_generation_sample_size}/eval={self.config.gsm_eval_sample_size}; "
            f"other generation={self.config.generation_sample_size}/eval={self.config.eval_sample_size}"
            )
        print(f"accuracy win margin (metadecode): +{self.config.accuracy_win_margin:.1%}")
        print(f"main synthesis iterations (metadecode): {self.config.main_synthesis_iterations}")
        if self.config.gpu3_retry_enabled:
            print(f"GPU3 retry queue: {self.config.gpu3_retry_queue}")
        else:
            print("GPU3 retry queue: disabled")
        if self.config.gsm_split_file:
            print(f"GSM split file: {self.config.gsm_split_file}")
        if self.config.spider_split_file:
            print(f"Spider split file: {self.config.spider_split_file}")
        print(
            f"baseline cache mode: {self.config.baseline_cache_mode} "
            "(reuse=skip complete JSONs, refresh=recompute fixed baselines)"
        )
        print("")

    def run_main_matrix(self) -> None:
        if self.config.skip_main:
            return
        print("=== Phase 1: Main experiment matrix ===")
        for eval_model in self.config.models:
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for strategy in self.config.strategies:
                    if strategy == "metadecode":
                        self.run_metadecode_cases(
                            benchmark,
                            eval_model,
                            self.config.token_budgets[0],
                            self.config.main_synthesis_iterations,
                            self.config.gen_models[0],
                            self.eval_max_steps_for(benchmark),
                            phase="main_matrix",
                        )
                    else:
                        self.run_fixed_strategy_cases(
                            strategy,
                            benchmark,
                            eval_model,
                            self.config.token_budgets[0],
                            self.eval_max_steps_for(benchmark),
                            phase="main_matrix",
                        )
        print("=== Phase 1 complete ===")

    def run_ablation_e_case(
        self,
        benchmark: str,
        eval_model: str,
        beam_size: str,
        mask_flag: str,
        policy: str,
        smiles_class: str = "",
    ) -> None:
        task = self.metadecode_task(benchmark)
        class_suffix = f"_class_{slugify(smiles_class)}" if benchmark == "smiles" else ""
        run_name = f"ablat_beam{beam_size}_{'mask_off' if mask_flag == '--no-adaptive-helper-mask' else 'mask_on'}_{policy}_{benchmark}{class_suffix}"
        backend, generation_model = self.resolve_gen_profile("gpt5.5")
        if not self.openai_generation_available("gpt5.5"):
            return
        token_budget = self.config.token_budgets[0]
        max_steps = self.eval_max_steps_for(benchmark)
        self.ensure_csd_target_baselines(benchmark, eval_model, token_budget, max_steps, smiles_class)
        (
            target_accuracy,
            target_strategy,
            _target_path,
            target_percent,
            target_syntax,
            target_syntax_strategy,
            _target_syntax_path,
            target_syntax_percent,
        ) = self.best_csd_baseline_targets(benchmark, eval_model, token_budget, max_steps, smiles_class)
        required_accuracy = self.accuracy_target_with_margin(target_accuracy, target_strategy)

        if target_strategy == "none" and target_syntax_strategy == "none":
            print(
                f"[target] metadecode {benchmark}{class_suffix}/{slugify(eval_model)} "
                f"tb{token_budget} ms{max_steps}: no valid CRANE/IterGen baseline found; "
                "passing --min-accuracy 0.0 --min-syntax-rate 0.0"
            )
        else:
            print(
                f"[target] metadecode {benchmark}{class_suffix}/{slugify(eval_model)} "
                f"tb{token_budget} ms{max_steps}: best CSD baseline accuracy "
                f"{target_strategy}={target_percent}; "
                f"accuracy target={required_accuracy:.1%} (+{self.config.accuracy_win_margin:.1%}); "
                f"syntax {target_syntax_strategy}={target_syntax_percent}; "
                f"passing --min-accuracy {required_accuracy:.12g} --min-syntax-rate {target_syntax:.12g}"
            )

        cmd = [
            "python",
            "-m",
            "synthesis.run_synthesis",
            "--task",
            task,
            "--dataset",
            benchmark,
            "--generation-backend",
            backend,
            "--generation-model",
            generation_model,
            "--eval-model",
            eval_model,
            "--max-iterations",
            self.config.synth_iters[-1],
            "--min-accuracy",
            f"{required_accuracy:.12g}",
            "--min-syntax-rate",
            f"{target_syntax:.12g}",
            "--eval-sample-size",
            self.generation_sample_size(benchmark),
            "--eval-max-steps",
            max_steps,
            "--eval-step-token-budget",
            token_budget,
            "--eval-max-seconds-per-example",
            self.config.eval_max_seconds_per_example,
            "--max-tokens",
            self.config.synthesis_max_tokens,
            "--device",
            self.config.device,
        ]
        if benchmark == "smiles":
            cmd += [
                "--smiles-samples-per-class",
                self.generation_sample_size(benchmark),
                "--smiles-classes",
                smiles_class,
            ]
        if self.config.dafny_path:
            cmd += ["--dafny-path", self.config.dafny_path]
        self.run_cmd(cmd, abort_on_quota=False)
        out_json = (
            self.config.baseline_output_dir
            / "metadecode"
            / slugify(eval_model)
            / f"{self.benchmark_key(benchmark, smiles_class)}__tb{token_budget}__ms{max_steps}__gengpt5.5__iter{self.config.synth_iters[-1]}.json"
        )
        self.annotate_result_json(
            out_json,
            self.matrix_case_metadata(
                phase="ablation_helper_mask",
                strategy="metadecode",
                benchmark=benchmark,
                eval_model=eval_model,
                token_budget=token_budget,
                max_steps=max_steps,
                command=cmd,
                synth_iter=self.config.synth_iters[-1],
                gen_profile="gpt5.5",
                generation_backend=backend,
                generation_model=generation_model,
                smiles_class=smiles_class,
                required_accuracy=required_accuracy,
                required_syntax=target_syntax,
                target_accuracy_strategy=target_strategy,
                target_accuracy_path=_target_path,
                target_syntax_strategy=target_syntax_strategy,
                target_syntax_path=_target_syntax_path,
            ),
        )

    def run_ablations(self) -> None:
        if self.config.skip_ablations:
            return
        print("")
        print("=== Phase 2: Ablation studies ===")
        ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        sections = self.config.ablation_sections

        if "A" in sections:
            print("--- Ablation A: Step budget ---")
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for step_budget in self.config.step_budgets:
                    for strategy in ("gcd", "crane", "itergen", "metadecode"):
                        if strategy == "metadecode":
                            self.run_metadecode_cases(
                                benchmark,
                                ablation_model,
                                self.config.token_budgets[0],
                                self.config.synth_iters[-1],
                                self.config.gen_models[0],
                                step_budget,
                                phase="ablation_step_budget",
                            )
                        else:
                            self.run_fixed_strategy_cases(
                                strategy,
                                benchmark,
                                ablation_model,
                                self.config.token_budgets[0],
                                step_budget,
                                phase="ablation_step_budget",
                            )

        if "B" in sections:
            print("--- Ablation B: Synthesis iterations ---")
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for synth_iter in self.config.synth_iters:
                    self.run_metadecode_cases(
                        benchmark,
                        ablation_model,
                        self.config.token_budgets[0],
                        synth_iter,
                        self.config.gen_models[0],
                        self.eval_max_steps_for(benchmark),
                        phase="ablation_synthesis_iterations",
                    )

        if "C" in sections:
            print("--- Ablation C: Synthesizer model ---")
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for gen_profile in self.config.gen_models:
                    self.run_metadecode_cases(
                        benchmark,
                        ablation_model,
                        self.config.token_budgets[0],
                        self.config.synth_iters[-1],
                        gen_profile,
                        self.eval_max_steps_for(benchmark),
                        phase="ablation_synthesizer_model",
                    )

        if "D" in sections:
            print("--- Ablation D: Per-step token budget ---")
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for token_budget in self.config.token_budgets:
                    for strategy in ("gcd", "crane", "itergen", "metadecode"):
                        if strategy == "metadecode":
                            self.run_metadecode_cases(
                                benchmark,
                                ablation_model,
                                token_budget,
                                self.config.synth_iters[-1],
                                self.config.gen_models[0],
                                self.eval_max_steps_for(benchmark),
                                phase="ablation_token_budget",
                            )
                        else:
                            self.run_fixed_strategy_cases(
                                strategy,
                                benchmark,
                                ablation_model,
                                token_budget,
                                self.eval_max_steps_for(benchmark),
                                phase="ablation_token_budget",
                            )

        if "E" in sections:
            print("--- Ablation E: Adaptive helper masking (tool filtering on/off) ---")
            # Trimmed for deadline: only vary the mask_flag (tool filtering on/off),
            # holding beam_size=2 and UCB/bandit policy fixed. Original cross-product
            # (beam x mask x policy) was 60 cells; this is 10.
            for raw_benchmark in self.config.benchmarks:
                benchmark = normalize_benchmark(raw_benchmark)
                for mask_flag in ("--adaptive-helper-mask", "--no-adaptive-helper-mask"):
                    smiles_classes = self.config.smiles_classes if benchmark == "smiles" else [""]
                    for smiles_class in smiles_classes:
                        self.run_ablation_e_case(
                            benchmark, ablation_model, "2", mask_flag, "bandit", smiles_class
                        )
        print("=== Phase 2 complete ===")

    def run(self) -> int:
        self.config.generated_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.baseline_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.ablation_output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_split_manifests()
        if not self.configure_cuda_devices():
            return 1
        self.print_matrix_header()
        self.run_main_matrix()
        self.run_ablations()
        print("")
        print("All requested matrix jobs completed.")
        return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strategy x model x benchmark matrix, plus ablations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Outputs:\n"
            "- Synthesis runs: generated output dir\n"
            "- Baseline JSONs: baseline output dir\n"
            "- Ablation JSONs: ablation output dir\n"
        ),
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--token-budgets", default=DEFAULT_TOKEN_BUDGETS)
    parser.add_argument("--step-budgets", default=DEFAULT_STEP_BUDGETS)
    parser.add_argument("--synthesis-iterations", default=DEFAULT_SYNTH_ITERS)
    parser.add_argument("--generation-models", default=DEFAULT_GEN_MODELS)
    parser.add_argument(
        "--ablation-sections",
        default=DEFAULT_ABLATION_SECTIONS,
        help="Comma-separated ablation sections to run: A,B,C,D,E (default: all).",
    )
    parser.add_argument("--smiles-classes", "--smiles-class", default=DEFAULT_SMILES_CLASSES)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generation-sample-size", default="50")
    parser.add_argument("--eval-sample-size", default="100")
    parser.add_argument("--gsm-generation-sample-size", default="50")
    parser.add_argument("--gsm-eval-sample-size", default="50")
    parser.add_argument("--eval-max-steps", default="600")
    parser.add_argument(
        "--eval-max-steps-gsm",
        default=os.environ.get("CSD_GSM_EVAL_MAX_STEPS", DEFAULT_GSM_MAX_STEPS),
        help="Per-benchmark override for --eval-max-steps used on GSM-Symbolic (default: 900).",
    )
    parser.add_argument(
        "--eval-max-steps-smiles",
        default="400",
        help="Per-benchmark override for --eval-max-steps used on SMILES classes.",
    )
    parser.add_argument(
        "--eval-max-seconds-per-example",
        default="90",
        help="Per-example wall-clock timeout for synthesis evaluation (seconds). "
        "Wired through to synthesis.run_synthesis. Default: 90.",
    )
    parser.add_argument(
        "--accuracy-win-margin",
        type=float,
        default=float(os.environ.get("CSD_ACCURACY_WIN_MARGIN", "0.0")),
        help="Absolute accuracy margin added to the best matching legacy CSD baseline "
        "for MetaDecode success (default: 0.0, i.e. any strict baseline win counts).",
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        "--max-tokens",
        dest="synthesis_max_tokens",
        default=os.environ.get("CSD_SYNTHESIS_MAX_TOKENS", "32768"),
        help="Author-model token budget for each MetaDecode synthesis attempt (default: 32768).",
    )
    parser.add_argument(
        "--gsm-split-file",
        default=os.environ.get("CSD_GSM_SPLIT_FILE", str(DEFAULT_GSM_SPLIT_FILE)),
        help="Stratified GSM-Symbolic manifest (default: environment/benchmark_splits/gsm_symbolic_crane_proportional.json)",
    )
    parser.add_argument(
        "--spider-split-file",
        default=os.environ.get("CSD_SPIDER_SPLIT_FILE", str(DEFAULT_SPIDER_SPLIT_FILE)),
        help="Stratified Spider manifest (default: environment/benchmark_splits/spider_dev_proportional.json)",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        default=os.environ.get("VAS_VLLM_GPU_MEMORY_UTILIZATION", "0.80"),
        help="vLLM GPU memory fraction (default: 0.80; override via VAS_VLLM_GPU_MEMORY_UTILIZATION)",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VAS_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="vLLM tensor parallel size (default: 1; capped by VAS_MAX_CUDA_DEVICES)",
    )
    parser.add_argument("--dafny-path", default=os.environ.get("DAFNY_PATH", ""))
    parser.add_argument("--generated-output-dir", default=os.environ.get("CSD_OUTPUT_DIR", "outputs/generated"))
    parser.add_argument("--baseline-output-dir", default=os.environ.get("CSD_BASELINE_OUTPUT_DIR", "outputs/baselines"))
    parser.add_argument("--ablation-output-dir", default=os.environ.get("CSD_ABLATION_OUTPUT_DIR", "outputs/ablations"))
    parser.add_argument(
        "--recompute-baselines",
        dest="baseline_cache_mode",
        action="store_const",
        const="refresh",
    )
    parser.add_argument(
        "--reuse-baselines",
        dest="baseline_cache_mode",
        action="store_const",
        const="reuse",
    )
    parser.set_defaults(baseline_cache_mode="refresh")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument(
        "--main-synthesis-iterations",
        default=os.environ.get("CSD_MAIN_SYNTHESIS_ITERATIONS", DEFAULT_MAIN_SYNTH_ITERS),
        help="MetaDecode iteration count used by the main matrix (default: 40).",
    )
    parser.add_argument(
        "--gpu3-retry-queue",
        default=os.environ.get("CSD_GPU3_RETRY_QUEUE", str(DEFAULT_GPU3_RETRY_QUEUE)),
        help="JSONL queue where failed non-GPU3 MetaDecode runs are appended for GPU3 retry.",
    )
    parser.add_argument(
        "--no-gpu3-retry-queue",
        dest="gpu3_retry_enabled",
        action="store_false",
        help="Disable automatic GPU3 retry queue writes for failed non-GPU3 MetaDecode runs.",
    )
    parser.set_defaults(gpu3_retry_enabled=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_smiles_classes_for_cli(raw: str) -> list[str]:
    try:
        return normalize_smiles_classes(
            ",".join(csv_list(raw)),
            dedupe=True,
            require_non_empty=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def configure_conda_environment(root: Path) -> tuple[Path, dict[str, str]]:
    default_env = Path("/apps/conda/advayth2/envs/advayth2")
    conda_env_path = Path(
        os.environ.get("VAS_CONDA_ENV")
        or os.environ.get("VAS_RDKIT_CONDA_ENV")
        or str(default_env)
    )
    python_path = conda_env_path / "bin" / "python"
    if not python_path.exists():
        print(f"conda environment python not found: {python_path}", file=sys.stderr)
        raise SystemExit(1)

    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(conda_env_path)
    env["PATH"] = f"{conda_env_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    lib_dir = conda_env_path / "lib"
    if lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else str(lib_dir)
    env["PYTHONUNBUFFERED"] = "1"

    rdkit_check = subprocess.run(
        [str(python_path), "-c", "import rdkit"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    if rdkit_check.returncode != 0:
        sys.stderr.write(rdkit_check.stderr)
        print(f"failed to import rdkit in conda environment: {conda_env_path}", file=sys.stderr)
        raise SystemExit(rdkit_check.returncode)

    print(f"[env] using conda environment: {conda_env_path}")
    return conda_env_path, env


def build_config(args: argparse.Namespace, conda_env_path: Path) -> Config:
    from synthesis.evaluate.benchmarks.common.model_utils import resolve_vllm_tensor_parallel_size

    dafny_path = args.dafny_path
    if not dafny_path and (ROOT_DIR / "dafny" / "dafny").is_file():
        dafny_path = str(ROOT_DIR / "dafny" / "dafny")

    baseline_cache_mode = args.baseline_cache_mode
    if baseline_cache_mode not in {"reuse", "refresh"}:
        raise SystemExit(f"Invalid baseline cache mode: {baseline_cache_mode} (expected reuse or refresh)")

    return Config(
        models=csv_list(args.models),
        benchmarks=csv_list(args.benchmarks),
        strategies=normalize_strategies(args.strategies),
        token_budgets=csv_list(args.token_budgets),
        synth_iters=csv_list(args.synthesis_iterations),
        gen_models=csv_list(args.generation_models),
        step_budgets=csv_list(args.step_budgets),
        ablation_sections=normalize_ablation_sections(args.ablation_sections),
        smiles_classes=normalize_smiles_classes_for_cli(args.smiles_classes),
        eval_backend=args.eval_backend,
        device=args.device,
        generation_sample_size=str(args.generation_sample_size),
        eval_sample_size=str(args.eval_sample_size),
        gsm_generation_sample_size=str(args.gsm_generation_sample_size),
        gsm_eval_sample_size=str(args.gsm_eval_sample_size),
        eval_max_steps=str(args.eval_max_steps),
        eval_max_steps_gsm=str(args.eval_max_steps_gsm),
        eval_max_steps_smiles=str(args.eval_max_steps_smiles),
        eval_max_seconds_per_example=str(args.eval_max_seconds_per_example),
        accuracy_win_margin=float(args.accuracy_win_margin),
        synthesis_max_tokens=str(args.synthesis_max_tokens),
        vllm_gpu_memory_utilization=str(args.vllm_gpu_memory_utilization),
        vllm_tensor_parallel_size=resolve_vllm_tensor_parallel_size(args.vllm_tensor_parallel_size),
        dafny_path=dafny_path,
        generated_output_dir=Path(args.generated_output_dir),
        baseline_output_dir=Path(args.baseline_output_dir),
        ablation_output_dir=Path(args.ablation_output_dir),
        baseline_cache_mode=baseline_cache_mode,
        gsm_split_file=args.gsm_split_file,
        spider_split_file=args.spider_split_file,
        dry_run=args.dry_run,
        skip_main=args.skip_main,
        skip_ablations=args.skip_ablations,
        conda_env_path=conda_env_path,
        cuda_devices=os.environ.get("RUN_ALL_TESTS_CUDA_DEVICES", "auto"),
        cuda_oom_fallback=os.environ.get("RUN_ALL_TESTS_CUDA_OOM_FALLBACK", "auto"),
        free_gpu_max_used_mb=int(os.environ.get("RUN_ALL_TESTS_FREE_GPU_MAX_USED_MB", "1024")),
        gpu_wait_seconds=int(os.environ.get("RUN_ALL_TESTS_GPU_WAIT_SECONDS", "60")),
        gpu_wait_timeout_seconds=int(os.environ.get("RUN_ALL_TESTS_GPU_WAIT_TIMEOUT_SECONDS", "0")),
        main_synthesis_iterations=str(args.main_synthesis_iterations),
        gpu3_retry_queue=Path(args.gpu3_retry_queue),
        gpu3_retry_enabled=bool(args.gpu3_retry_enabled),
    )


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT_DIR)
    load_env_file(ROOT_DIR / "synthesis" / ".env")
    parser = make_parser()
    args = parser.parse_args(argv)
    conda_env_path, env = configure_conda_environment(ROOT_DIR)
    config = build_config(args, conda_env_path)
    return Runner(config=config, env=env).run()


if __name__ == "__main__":
    raise SystemExit(main())
