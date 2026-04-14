"""
Qwen-based strategy generator for CSD synthesis.

Uses HuggingFace Transformers to load Qwen and generate Python strategy code
for insertion into `generation/csd/GeneratedAgentTemplate.py`.
"""

import ast
import re
import textwrap
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
    build_structure_repair_prompt,
)
from .rationale import extract_rationale


class StrategyGenerator:
    """
    Generates Python CSD strategies using Qwen.
    
    Loads the Qwen model via HuggingFace Transformers and provides methods
    for initial generation and error-based refinement.
    """
    
    # Default model - can be overridden
    DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Path to the template file
    TEMPLATE_PATH = Path(__file__).resolve().parent / "csd" / "GeneratedAgentTemplate.py"

    # Markers delimiting the hole to replace
    STRATEGY_BEGIN_MARKER = "    # QWEN_INSERT_STRATEGY_BEGIN"
    STRATEGY_END_MARKER = "    # QWEN_INSERT_STRATEGY_END"

    # Under this budget, Qwen often truncates before emitting a full rationale + loop body.
    MIN_STRATEGY_TOKENS = 192
    SEARCH_ATTEMPTS = 3
    ALLOWED_HELPER_METHODS = {
        "ConstrainedAnswerStep",
        "ExpressiveStep",
        "CompletedDelimitedAnswer",
        "DelimitedAnswerValid",
        "ConstrainedWindowValid",
        "GetDelimitedContent",
        "InsideDelimitedWindow",
        "LeftDelimiter",
        "RightDelimiter",
        "ContentIsValidInWindow",
        "ValidNextTokensInLM",
        "DelimitersInLM",
        "DelimitersInLMAlways",
        "FinalizeDelimitedAnswer",
        "EnterDelimitedWindow",
        "ExitDelimitedWindow",
        "GetDelimitedContentAppend",
        "InDelimitedWindowThenContentValid",
        "ConstrainedStepNextValid",
        "RollbackPreservesTokenInvariant",
    }
    ALLOWED_PARSER_METHODS = {
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextToken",
        "ValidNextTokens",
        "EmptyPrefixIsValid",
    }

    STARTER_STRATEGY = """\
# CSD_RATIONALE_BEGIN
# Fallback starter strategy: keep delimiter control explicit, spend a small controlled budget on expressive free-form text, then drive a separate constrained answer channel until it becomes complete.
# CSD_RATIONALE_END
phase = 0
preamble_tokens = 0
exploration_budget = 1
answer_tokens = 0
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps - 2
# invariant 0 <= preamble_tokens <= 1
# invariant 0 <= exploration_budget <= 1
# invariant 0 <= answer_tokens
# invariant helpers.ConstrainedWindowValid(generated)
# invariant parser.IsValidPrefix(answer)
# invariant |generated| + |answer| + stepsLeft <= maxSteps - 2
# invariant |answer| == 0 ==> exploration_budget < stepsLeft
# decreases stepsLeft
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    next_token = eosToken
    new_steps = stepsLeft
    spend_freeform = phase < 2 and exploration_budget > 0 and preamble_tokens < 1 and stepsLeft > 1
    if spend_freeform:
        next_token, new_steps = helpers.ExpressiveStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        preamble_tokens = preamble_tokens + 1
        exploration_budget = exploration_budget - 1
        if preamble_tokens >= 1:
            phase = 1
        if preamble_tokens >= 1 or stepsLeft <= 1:
            phase = 2
    else:
        next_token, new_steps = helpers.ConstrainedAnswerStep(prompt, generated, answer, stepsLeft)
        answer = answer + [next_token]
        stepsLeft = new_steps
        answer_tokens = answer_tokens + 1
        if answer_tokens >= 1:
            phase = 3
"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 800,
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
        self.last_raw_outputs: list[str] = []

        # Load template
        self._template = self._load_template()
    
    def _load_template(self) -> str:
        """Load the `generation/csd/GeneratedAgentTemplate.py` template."""
        if not self.TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Template not found at {self.TEMPLATE_PATH}. "
                "Make sure generation/csd/GeneratedAgentTemplate.py exists."
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
    
    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
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
                    max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
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

        response = response.strip()
        self.last_raw_outputs.append(response)
        self.last_raw_outputs = self.last_raw_outputs[-10:]
        return response
    
    def _extract_strategy(self, raw_output: str) -> str:
        """
        Extract the Python strategy body from Qwen's output.
        
        Handles cases where Qwen includes extra text, code blocks, etc.
        
        Args:
            raw_output: Raw text from Qwen
            
        Returns:
            Cleaned strategy body
        """
        code_block_pattern = r"```(?:python|py|dafny)?\s*([\s\S]*?)```"
        match = re.search(code_block_pattern, raw_output)
        if match:
            raw_output = match.group(1)
        else:
            truncated_pattern = r"^```(?:python|py|dafny)?\s*([\s\S]*)$"
            match = re.search(truncated_pattern, raw_output.strip())
            if match:
                raw_output = match.group(1)

        strategy = raw_output.strip()

        # If the model returned a full Python function, strip the signature and dedent the body.
        func_match = re.search(r"def\s+MyCSDStrategy\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n([\s\S]*)$", strategy)
        if func_match:
            strategy = textwrap.dedent(func_match.group(1)).strip()

        # Best-effort normalization from older Dafny-oriented outputs.
        strategy = re.sub(r"(?m)^(\s*)//", r"\1#", strategy)
        strategy = strategy.replace(":=", "=")
        strategy = strategy.replace("&&", " and ")
        strategy = strategy.replace("||", " or ")
        strategy = re.sub(r"(?<![=!])!(?!=)", " not ", strategy)
        strategy = re.sub(r"(?m)^(\s*)(invariant|decreases)\b", r"\1# \2", strategy)
        strategy = re.sub(r"(?m);\s*$", "", strategy)
        strategy = self._autofix_python_strategy(strategy)

        return strategy.strip()

    def _ensure_rationale_block(self, strategy_body: str, *, max_repairs: int = 2) -> str:
        """
        Ensure the strategy body contains the required rationale markers.

        If missing, attempt a small number of "format repair" generations that rewrite
        the body into the required structure without changing semantics.
        """
        extracted = extract_rationale(strategy_body)
        if extracted.rationale is not None and extracted.has_markers:
            return self._normalize_rationale_block(strategy_body)

        current = strategy_body
        for _ in range(max_repairs):
            system_prompt, user_prompt = build_format_repair_prompt(current)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            extracted = extract_rationale(repaired)
            if extracted.rationale is not None and extracted.has_markers:
                return self._normalize_rationale_block(repaired)
            current = repaired

        raise ValueError(
            "Generated strategy is missing required rationale block markers "
            "(# CSD_RATIONALE_BEGIN ... # CSD_RATIONALE_END)."
        )

    def _body_without_rationale(self, strategy_body: str) -> str:
        extracted = extract_rationale(strategy_body)
        return extracted.body_without_rationale if extracted.has_markers else strategy_body

    def _normalize_rationale_block(self, strategy_body: str) -> str:
        lines = strategy_body.splitlines()
        begin_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() in {"# CSD_RATIONALE_BEGIN", "// CSD_RATIONALE_BEGIN"}:
                begin_idx = i
                break
        if begin_idx is None:
            return strategy_body
        for j in range(begin_idx + 1, len(lines)):
            if lines[j].strip() in {"# CSD_RATIONALE_END", "// CSD_RATIONALE_END"}:
                end_idx = j
                break
        if end_idx is None:
            return strategy_body

        normalized = list(lines)
        for k in range(begin_idx + 1, end_idx):
            raw = normalized[k]
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            indent = raw[: len(raw) - len(raw.lstrip())]
            normalized[k] = f"{indent}# {stripped}"
        return "\n".join(normalized)

    def _autofix_python_strategy(self, strategy_body: str) -> str:
        lines = strategy_body.splitlines()
        fixed: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            fixed.append(line)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped.startswith("if ") and stripped.endswith(":"):
                branch_indent = " " * (indent + 4)
                if i + 3 < len(lines):
                    first_branch = lines[i + 1]
                    else_line = None
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith(" " * indent + "else:"):
                            else_line = j
                            break
                    if else_line is not None and else_line + 1 < len(lines):
                        branch_assign = re.match(
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedAnswerStep|ExpressiveStep|UnconstrainedStep)\(",
                            first_branch,
                        )
                        else_assign = re.match(
                            rf"^{re.escape(branch_indent)}([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*helpers\.(?:ConstrainedAnswerStep|ExpressiveStep|UnconstrainedStep)\(",
                            lines[else_line + 1],
                        )
                        if branch_assign and else_assign and branch_assign.groups() == else_assign.groups():
                            name1, name2 = branch_assign.groups()
                            prev_slice = "\n".join(fixed[:-1])
                            if not re.search(rf"(?m)^\s*{re.escape(name1)}\s*=", prev_slice):
                                fixed.insert(len(fixed) - 1, " " * indent + f"{name1} = eosToken")
                            if not re.search(rf"(?m)^\s*{re.escape(name2)}\s*=", prev_slice):
                                default_rhs = "stepsLeft"
                                fixed.insert(len(fixed) - 1, " " * indent + f"{name2} = {default_rhs}")
            i += 1
        return "\n".join(fixed)

    def _structural_issue(self, strategy_body: str) -> str | None:
        body = self._body_without_rationale(strategy_body)
        executable_lines = [
            line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not executable_lines:
            return "The body has no executable Python statements after the rationale block."

        try:
            wrapped = "def _strategy():\n" + textwrap.indent(body, "    ")
            tree = ast.parse(wrapped)
        except SyntaxError as exc:
            return f"The body is not valid Python: {exc.msg}."

        has_while = any(isinstance(node, ast.While) for node in ast.walk(tree))
        if not has_while:
            return "The body must contain a while loop that performs decoding steps."

        step_calls = 0
        constrained_answer_calls = 0
        expressive_calls = 0
        appends_generated = False
        appends_answer = False
        extra_state: set[str] = set()
        disallowed_calls: set[str] = set()
        unsupported_helper_calls: set[str] = set()
        unsupported_parser_calls: set[str] = set()
        parser_on_generated_methods: set[str] = set()
        answer_reasoning = False
        if_count = 0
        while_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if_count += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "helpers":
                    if node.func.attr not in self.ALLOWED_HELPER_METHODS and node.func.attr not in {
                        "UnconstrainedStep",
                        "ConstrainedStep",
                        "RollbackToValidPrefix",
                        "InsideDelimitedWindow",
                    }:
                        unsupported_helper_calls.add(node.func.attr)
                    if node.func.attr in {"ExpressiveStep", "ConstrainedAnswerStep"}:
                        step_calls += 1
                    if node.func.attr == "ConstrainedAnswerStep":
                        constrained_answer_calls += 1
                    if node.func.attr == "ExpressiveStep":
                        expressive_calls += 1
                    if node.func.attr in {
                        "UnconstrainedStep",
                        "ConstrainedStep",
                        "RollbackToValidPrefix",
                        "InsideDelimitedWindow",
                        "FinalizeDelimitedAnswer",
                    }:
                        disallowed_calls.add(node.func.attr)
                    if node.func.attr in {"ConstrainedAnswerStep", "CompletedDelimitedAnswer", "DelimitedAnswerValid"}:
                        answer_reasoning = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "parser":
                    if node.func.attr not in self.ALLOWED_PARSER_METHODS:
                        unsupported_parser_calls.add(node.func.attr)
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "generated"
                    ):
                        parser_on_generated_methods.add(node.func.attr)
            if isinstance(node, ast.Assign):
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "generated"
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Add)
                    and isinstance(node.value.left, ast.Name)
                    and node.value.left.id == "generated"
                    and isinstance(node.value.right, ast.List)
                ):
                    appends_generated = True
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "answer"
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Add)
                    and isinstance(node.value.left, ast.Name)
                    and node.value.left.id == "answer"
                    and isinstance(node.value.right, ast.List)
                ):
                    appends_answer = True
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                        extra_state.add(target.id)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
            if isinstance(node, ast.Name) and node.id == "answer":
                answer_reasoning = True

        if step_calls == 0:
            return "The body must call helper step methods."
        if unsupported_parser_calls:
            return (
                "The body calls parser methods that do not exist in the supported synthesis API: "
                + ", ".join(sorted(unsupported_parser_calls))
                + "."
            )
        if parser_on_generated_methods:
            return (
                "Do not call parser methods on generated; the parser governs answer, not the free-form output. "
                "Offending methods: "
                + ", ".join(sorted(parser_on_generated_methods))
                + "."
            )
        if constrained_answer_calls == 0:
            return "The body must use helpers.ConstrainedAnswerStep(...) to build the constrained answer segment."
        if expressive_calls == 0:
            return "The body must use helpers.ExpressiveStep(...) for expressive free-form output outside the constrained answer segment."
        if unsupported_helper_calls:
            return (
                "The body calls helper methods that do not exist in the supported synthesis API: "
                + ", ".join(sorted(unsupported_helper_calls))
                + "."
            )
        if disallowed_calls:
            return "The strategy still relies on disallowed basic patterns: " + ", ".join(sorted(disallowed_calls)) + "."
        if not appends_generated:
            return "The body must append produced tokens with generated = generated + [next_token]."
        if not appends_answer:
            return "The body must append constrained answer tokens with answer = answer + [next_token]."
        if len(extra_state) < 2:
            return "The body must maintain at least two extra local state variables so the strategy is not a trivial loop."
        if if_count < 2:
            return "The body needs richer control flow than a single top-level switch."
        if not answer_reasoning:
            return "The body must reason explicitly about the constrained answer channel."
        if len(while_nodes) == 1:
            while_body = while_nodes[0].body
            significant = [
                node for node in while_body
                if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
            ]
            if len(significant) <= 4:
                simple_window_switch = (
                    len(significant) == 3
                    and isinstance(significant[0], ast.If)
                ) or (
                    len(significant) == 4
                    and isinstance(significant[0], ast.Assign)
                    and isinstance(significant[1], ast.If)
                )
                if simple_window_switch:
                    return "The body is still a basic loop/switch pattern; it needs more novel control structure."
        return None

    def _ensure_nontrivial_strategy(self, strategy_body: str, *, max_repairs: int = 2) -> str:
        current = strategy_body
        for _ in range(max_repairs):
            issue = self._structural_issue(current)
            if issue is None:
                return current
            system_prompt, user_prompt = build_structure_repair_prompt(current, issue)
            repaired_raw = self._generate_text(system_prompt, user_prompt)
            repaired = self._extract_strategy(repaired_raw)
            current = self._ensure_rationale_block(repaired)

        issue = self._structural_issue(current)
        if issue is None:
            return current

        raise ValueError(
            "Generated strategy is structurally invalid. "
            f"It must contain executable decoding logic with a while loop and helper step calls. Last issue: {issue}"
        )

    def _novelty_score(self, strategy_body: str) -> int:
        body = self._body_without_rationale(strategy_body)
        try:
            wrapped = "def _strategy():\n" + textwrap.indent(body, "    ")
            tree = ast.parse(wrapped)
        except SyntaxError:
            return -10_000

        helper_calls: set[str] = set()
        extra_state: set[str] = set()
        if_count = 0
        while_count = 0
        bool_complexity = 0
        nested_if = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "helpers":
                    helper_calls.add(node.func.attr)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                        extra_state.add(target.id)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in {"generated", "answer", "stepsLeft", "next_token", "new_steps"}:
                    extra_state.add(node.target.id)
            if isinstance(node, ast.If):
                if_count += 1
                if isinstance(node.test, ast.BoolOp):
                    bool_complexity += max(0, len(node.test.values) - 1)
                if any(isinstance(inner, ast.If) for inner in node.body):
                    nested_if += 1
            if isinstance(node, ast.While):
                while_count += 1
                if isinstance(node.test, ast.BoolOp):
                    bool_complexity += max(0, len(node.test.values) - 1)

        score = 0
        score += 6 * len(extra_state)
        score += 5 * if_count
        score += 4 * while_count
        score += 3 * bool_complexity
        score += 4 * nested_if
        if "ExpressiveStep" in helper_calls:
            score += 10
        if "ConstrainedAnswerStep" in helper_calls:
            score += 10
        if helper_calls == {"ExpressiveStep", "ConstrainedAnswerStep"}:
            score += 8
        return score

    def _fallback_strategy(self, reason: str) -> str:
        print(f"  Warning: {reason}")
        print("  Falling back to a built-in starter strategy so the pipeline can keep moving.")
        return self.STARTER_STRATEGY

    def _generate_valid_strategy(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        fallback_reason: str,
        fallback_strategy: Optional[str] = None,
    ) -> str:
        budgets = [
            max(self.max_new_tokens, self.MIN_STRATEGY_TOKENS),
            max(self.max_new_tokens, 320),
            max(self.max_new_tokens, 384),
        ][: self.SEARCH_ATTEMPTS]
        temperatures = [
            max(self.temperature, 0.85),
            max(self.temperature, 0.65),
            min(self.temperature, 0.35),
        ][: self.SEARCH_ATTEMPTS]

        last_error: str | None = None
        current_system = system_prompt
        current_user = user_prompt
        valid_candidates: list[tuple[int, str]] = []

        for idx, (budget, temp) in enumerate(zip(budgets, temperatures), start=1):
            raw_output = self._generate_text(
                current_system,
                current_user,
                max_new_tokens=budget,
                temperature=temp,
            )
            strategy = self._extract_strategy(raw_output)
            try:
                strategy = self._ensure_rationale_block(strategy)
                strategy = self._ensure_nontrivial_strategy(strategy)
                valid_candidates.append((self._novelty_score(strategy), strategy))
                current_system, current_user = system_prompt, user_prompt
                continue
            except ValueError as exc:
                last_error = str(exc)
                current_system, current_user = build_structure_repair_prompt(
                    strategy or raw_output or "# CSD_RATIONALE_BEGIN\n# Empty output.\n# CSD_RATIONALE_END",
                    last_error,
                )
                print(
                    f"  Initial generation attempt {idx} produced an invalid body; "
                    f"retrying with a stricter repair prompt ({last_error})."
                )

        if valid_candidates:
            best_score, best_strategy = max(valid_candidates, key=lambda item: item[0])
            print(f"  Selected the most novel structurally valid candidate (score={best_score}).")
            return best_strategy

        if fallback_strategy is not None and self._structural_issue(fallback_strategy) is None:
            print(f"  Warning: {fallback_reason} ({last_error or 'invalid model output'}).")
            print("  Reusing the previous valid strategy.")
            return fallback_strategy

        return self._fallback_strategy(f"{fallback_reason} ({last_error or 'invalid model output'})")
    
    def generate_initial(self, task_description: str) -> str:
        """
        Generate an initial strategy for the given task.
        
        Args:
            task_description: Description of what the strategy should accomplish

        Returns:
            Strategy body (Python code)
        """
        system_prompt, user_prompt = build_initial_prompt(task_description)
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            fallback_reason="Qwen did not produce a usable initial strategy",
        )
    
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
            New strategy body
        """
        system_prompt, user_prompt = build_verification_error_prompt(
            previous_strategy, error_message
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            fallback_reason="Qwen did not produce a usable verification repair",
            fallback_strategy=previous_strategy,
        )
    
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
            New strategy body
        """
        system_prompt, user_prompt = build_runtime_error_prompt(
            previous_strategy, error_traceback
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            fallback_reason="Qwen did not produce a usable runtime repair",
            fallback_strategy=previous_strategy,
        )
    
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
            New strategy body
        """
        system_prompt, user_prompt = build_compilation_error_prompt(
            previous_strategy, error_message
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            fallback_reason="Qwen did not produce a usable compilation repair",
            fallback_strategy=previous_strategy,
        )

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
            New strategy body
        """
        system_prompt, user_prompt = build_evaluation_failure_prompt(
            previous_strategy, evaluation_feedback
        )
        return self._generate_valid_strategy(
            system_prompt,
            user_prompt,
            fallback_reason="Qwen did not produce a usable evaluation repair",
            fallback_strategy=previous_strategy,
        )

    def inject_strategy(self, strategy: str) -> str:
        """
        Inject a strategy into the template.

        Args:
            strategy: Strategy expression to inject

        Returns:
            Complete Python source code
        """
        body = textwrap.dedent(strategy).strip("\n")
        indented = textwrap.indent(body, "    ")
        start = self._template.find(self.STRATEGY_BEGIN_MARKER)
        end = self._template.find(self.STRATEGY_END_MARKER)
        if start == -1 or end == -1 or end < start:
            raise ValueError("Strategy hole markers not found in generation/csd/GeneratedAgentTemplate.py")
        end += len(self.STRATEGY_END_MARKER)
        return self._template[:start] + indented + self._template[end:]
    
    def get_template(self) -> str:
        """Get the raw template content."""
        return self._template
