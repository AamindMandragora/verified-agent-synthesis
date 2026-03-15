"""
Qwen-based strategy generator for CSD synthesis.

Uses HuggingFace Transformers to load Qwen and generate Dafny strategy code.
"""

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompts import (
    build_initial_prompt,
    build_verification_error_prompt,
    build_runtime_error_prompt,
    build_compilation_error_prompt,
    build_format_repair_prompt,
    build_evaluation_failure_prompt,
)
from .rationale import extract_rationale


class StrategyGenerator:
    """
    Generates Dafny CSD strategies using Qwen.
    
    Loads the Qwen model via HuggingFace Transformers and provides methods
    for initial generation and error-based refinement.
    """
    
    # Default model - can be overridden
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Path to the template file
    TEMPLATE_PATH = Path(__file__).parent.parent / "dafny" / "GeneratedCSD.dfy"
    
    # Marker in template to replace
    STRATEGY_MARKER = "// QWEN_INSERT_STRATEGY_HERE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        generation_timeout: Optional[int] = None,
    ):
        """
        Initialize the strategy generator.
        
        Args:
            model_name: HuggingFace model name (default: Qwen2.5-Coder-7B-Instruct)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            torch_dtype: Torch dtype for model (default: auto based on device)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            generation_timeout: Max seconds per LLM call (None = no limit). Use to avoid unbounded hangs.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generation_timeout = generation_timeout  # seconds; None = no timeout
        
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
            if device is not None and (device == "cuda" or device.startswith("cuda:")):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        
        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None
        
        # Load template
        self._template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the GeneratedCSD.dfy template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure GeneratedCSD.dfy exists in the dafny/ directory."
            )
        return self.TEMPLATE_PATH.read_text()
    
    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model and tokenizer. On CUDA OOM, try other GPUs before CPU."""
        if self._model is None:
            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            device_map = self.device if (self.device and self.device != "mps") else None
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
                if self.device == "mps":
                    self._model = self._model.to(self.device)
                print(f"Model loaded on {self.device}")
            except RuntimeError as e:
                error_str = str(e).lower()
                is_cuda_oom = (
                    self.device and
                    (self.device.startswith("cuda") or self.device == "mps") and
                    ("out of memory" in error_str or "cuda" in error_str)
                )
                if not is_cuda_oom:
                    raise

                print(f"⚠️  {self.device.upper()} out of memory: {e}")

                # If we were on a specific CUDA device (e.g. cuda:0 or auto's cuda), try other GPUs first
                n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                tried = set()
                if self.device.startswith("cuda:"):
                    try:
                        idx = int(self.device.split(":")[1])
                        tried.add(idx)
                    except (IndexError, ValueError):
                        tried.add(0)
                else:
                    tried.add(0)

                loaded = False
                for gpu_id in range(n_gpus):
                    if gpu_id in tried:
                        continue
                    cand = f"cuda:{gpu_id}"
                    try:
                        if self.device == "mps":
                            torch.cuda.empty_cache()
                        self.device = cand
                        self.torch_dtype = torch.bfloat16
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=self.torch_dtype,
                            device_map=self.device,
                            trust_remote_code=True
                        )
                        print(f"   Loaded on {self.device} instead.")
                        loaded = True
                        break
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        continue

                if not loaded:
                    print(f"   Falling back to CPU (this will be slower)...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True
                    ).to(self.device)
                    print(f"Model loaded on {self.device} (CPU fallback)")
    
    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using Qwen.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            
        Returns:
            Generated text
        """
        self._ensure_model_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        def _run_generate():
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(gen, skip_special_tokens=True)

        if self.generation_timeout is not None and self.generation_timeout > 0:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_generate)
                try:
                    response = fut.result(timeout=self.generation_timeout)
                except FuturesTimeoutError:
                    raise RuntimeError(
                        f"Generation timed out after {self.generation_timeout}s. "
                        "If the model is on CPU or slow GPU, try --generation-timeout 0 (disable) or lower --max-tokens."
                    ) from None
        else:
            response = _run_generate()

        return response.strip()
    
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

        # Heuristic repair: Dafny uses `+` for sequence concatenation; `++` is invalid.
        # Replace `++` when it is used like an operator (surrounded by optional whitespace).
        # This avoids touching tokens like "C++" inside words.
        strategy = re.sub(r"\s*\+\+\s*", " + ", strategy)

        # Heuristic repair: malformed for-loop "for i in 0 .. |prompt| - 1" (Python/Rust-style) -> Dafny "for i := 0 to |prompt| - 1"
        strategy = re.sub(
            r"for\s+(\w+)\s+in\s+0\s*\.\.\s*\|(\w+)\|\s*-\s*1\b",
            r"for \1 := 0 to |\2| - 1",
            strategy,
        )
        
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

    def _ensure_rationale_block(self, strategy_body: str, *, max_repairs: int = 2) -> str:
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
            system_prompt, user_prompt = build_format_repair_prompt(current)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return repaired
            current = repaired

        raise ValueError(
            "Generated strategy is missing required rationale block markers "
            "(// CSD_RATIONALE_BEGIN ... // CSD_RATIONALE_END)."
        )
    
    def generate_initial(self, task_description: str) -> str:
        """
        Generate an initial strategy for the given task.
        
        Args:
            task_description: Description of what the strategy should accomplish

        Returns:
            Strategy expression (Dafny code)
        """
        system_prompt, user_prompt = build_initial_prompt(task_description)
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
    def refine_after_verification_error(
        self,
        previous_strategy: str,
        error_message: str
    ) -> str:
        """
        Generate a refined strategy after verification failure.
        
        Args:
            previous_strategy: The strategy that failed
            error_message: Dafny verification error
            
        Returns:
            New strategy expression
        """
        system_prompt, user_prompt = build_verification_error_prompt(
            previous_strategy, error_message
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
    def refine_after_runtime_error(
        self,
        previous_strategy: str,
        error_traceback: str
    ) -> str:
        """
        Generate a refined strategy after runtime failure.
        
        Args:
            previous_strategy: The strategy that failed
            error_traceback: Python traceback
            
        Returns:
            New strategy expression
        """
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy, error_traceback
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)
    
    def refine_after_compilation_error(
        self,
        previous_strategy: str,
        error_message: str
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
            previous_strategy, error_message
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)

    def refine_after_evaluation_failure(
        self,
        previous_strategy: str,
        evaluation_feedback: str
    ) -> str:
        """
        Generate a refined strategy after evaluation failure.

        The strategy passed verification, compilation, and runtime testing,
        but performed poorly on actual dataset evaluation (low accuracy,
        format rate, syntax rate, or semantic rate).

        Args:
            previous_strategy: The strategy that failed evaluation
            evaluation_feedback: Feedback summary from the evaluator

        Returns:
            New strategy expression
        """
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy, evaluation_feedback
        )
        raw_output = self._generate_text(system_prompt, user_prompt)
        strategy = self._extract_strategy(raw_output)
        return self._ensure_rationale_block(strategy)

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

