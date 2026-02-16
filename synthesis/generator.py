"""
Qwen-based strategy generator for CSD synthesis.

Uses HuggingFace Transformers to load Qwen and generate Dafny strategy code.
"""

import re
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
    COST_CONTRACT_MARKER = "// QWEN_INSERT_COST_CONTRACT_HERE"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
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
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
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
        """Lazy-load the model and tokenizer with CUDA fallback."""
        if self._model is None:
            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Try loading on requested device, fallback to CPU on CUDA OOM
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device if self.device != "mps" else None,
                    trust_remote_code=True
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
                        trust_remote_code=True
                    ).to(self.device)
                    print(f"Model loaded on {self.device} (CPU fallback)")
                else:
                    raise
    
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
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
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
    
    def generate_initial(self, task_description: str, cost_contract: str = "") -> str:
        """
        Generate an initial strategy for the given task.
        
        Args:
            task_description: Description of what the strategy should accomplish
            cost_contract: Optional cost contract (e.g. "ensures helpers.cost <= 10")
            
        Returns:
            Strategy expression (Dafny code)
        """
        full_task = task_description
        if cost_contract:
            full_task += f"\n\nYour strategy MUST satisfy this cost contract:\n{cost_contract}"

        system_prompt, user_prompt = build_initial_prompt(full_task)
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

    def inject_strategy(self, strategy: str, cost_contract: str = "") -> str:
        """
        Inject a strategy and optional cost contract into the template.
        
        Args:
            strategy: Strategy expression to inject
            cost_contract: Optional cost contract (e.g. "ensures helpers.cost <= 10")
            
        Returns:
            Complete Dafny source code
        """
        code = self._template.replace(self.STRATEGY_MARKER, strategy)
        if cost_contract:
            code = code.replace(self.COST_CONTRACT_MARKER, cost_contract)
        else:
            code = code.replace(self.COST_CONTRACT_MARKER, "")
        return code
    
    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template

