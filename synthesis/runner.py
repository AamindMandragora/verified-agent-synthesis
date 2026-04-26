"""
Python runner for compiled Dafny CSD strategies.

Executes the compiled Python code and captures results/errors.

Supports both permissive testing mode and real JSON parsing mode.
Also supports running the original Python source directly (no Dafny compilation).
"""

import importlib.util
import sys
import traceback
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VAS_DIR = _PROJECT_ROOT / "generation" / "csd"

# Type for parser mode
ParserMode = Literal["permissive", "json"]


@dataclass
class RuntimeResult:
    """Result of running a compiled strategy."""
    success: bool
    output: Optional[list[str]] = None  # Generated tokens
    cost: int = 0  # Cost from Dafny strategy
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def get_error_summary(self) -> str:
        """Get a human-readable error summary."""
        if self.success:
            return f"Execution successful. Output: {self.output}"
        
        summary = f"Runtime error: {self.error_type}: {self.error_message}"
        if self.error_traceback:
            summary += f"\n\nTraceback:\n{self.error_traceback}"
        return summary


class StrategyRunner:
    """
    Executes compiled Dafny strategies in Python.
    
    Loads the generated Python module, injects runtime stubs,
    and executes the strategy with test inputs.
    
    Supports two parser modes:
    - "permissive": Accepts all tokens (for testing compilation)
    - "json": Uses real JSON prefix validation
    """
    
    def __init__(
        self,
        extern_functions_path: Optional[Path] = None,
        vocab_size: int = 1000,
        max_steps: int = 100,
        parser_mode: ParserMode = "permissive"
    ):
        """
        Initialize the runner.
        
        Args:
            extern_functions_path: Path to extern_functions.py (auto-detected if None)
            vocab_size: Vocabulary size for the LM stub
            max_steps: Maximum generation steps
            parser_mode: "permissive" for testing, "json" for real JSON validation
        """
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.parser_mode = parser_mode
        
        self.extern_functions_path = extern_functions_path
        self._externs_module = None
    
    def _load_externs(self) -> Any:
        """Load the Dafny extern functions module."""
        if self._externs_module is None:
            spec = importlib.util.spec_from_file_location(
                "extern_functions",
                self.extern_functions_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not load extern functions from {self.extern_functions_path}"
                )
            
            self._externs_module = importlib.util.module_from_spec(spec)
            sys.modules["extern_functions"] = self._externs_module
            spec.loader.exec_module(self._externs_module)
        
        return self._externs_module
    
    def _load_compiled_module(self, module_path: Path) -> Any:
        """
        Load a compiled Dafny-to-Python module.
        
        Args:
            module_path: Path to the main .py file
            
        Returns:
            Loaded module
        """
        # Add the module's parent directory to sys.path
        module_dir = module_path.parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        
        # Load the module
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def _create_dafny_test_environment(self, module_dir: Path) -> tuple[Any, Any, Any, int]:
        """
        Create a Dafny-compatible test environment.
        
        Returns:
            Tuple of (lm, parser, prompt, maxSteps)
        """
        # Add module dir to path for imports
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        
        # Import the Dafny runtime and compiled modules
        import _dafny
        import VerifiedDecoderAgent
        
        # Create a Dafny-compatible LM with extern implementations
        class TestLM(VerifiedDecoderAgent.LM):
            def __init__(self, vocab_size=100, json_mode=False):
                super().__init__()
                self.json_mode = json_mode
                # Initialize vocabulary
                tokens = self._create_vocabulary(vocab_size, json_mode)
                
                # Convert tokens to Dafny strings (seq<char>)
                import _dafny
                dafny_tokens = [_dafny.SeqWithoutIsStrInference(t) for t in tokens]
                
                self._Tokens = _dafny.SeqWithoutIsStrInference(dafny_tokens)
                self._Ids = _dafny.SeqWithoutIsStrInference(list(range(len(tokens))))
                self.Logits = _dafny.Array(None, len(tokens))
                for i in range(len(tokens)):
                    self.Logits[i] = _dafny.BigRational(0)
            
            def _create_vocabulary(self, vocab_size, json_mode):
                """Create vocabulary, optionally with JSON tokens."""
                if not json_mode:
                    tokens = ["<<", ">>"]
                    for i in range(vocab_size - 2):
                        if i == 0:
                            tokens.append("<EOS>")
                        elif i < 26:
                            tokens.append(chr(ord('a') + i - 1))
                        else:
                            tokens.append(f"<T{i}>")
                    return tokens
                
                # JSON-friendly vocabulary
                tokens = []
                
                # Hybrid delimiters
                tokens.extend(["<<", ">>"])
                
                # Structural tokens
                structural = ["{", "}", "[", "]", ":", ",", " ", "\n", "\t", '"']
                tokens.extend(structural)
                
                # Escape sequences
                escapes = ["\\", "\\n", "\\t", "\\r", '\\"', "\\\\"]
                tokens.extend(escapes)
                
                # Numbers
                tokens.extend(list("0123456789"))
                tokens.extend([".", "-", "+", "e", "E"])
                
                # JSON literals
                tokens.extend(["true", "false", "null"])
                
                # Letters
                for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    tokens.append(c)
                
                # Common keys
                tokens.extend(["name", "value", "id", "type", "data", "message"])
                
                # EOS
                tokens.append("<EOS>")
                
                # Fill to vocab_size
                while len(tokens) < vocab_size:
                    tokens.append(f"<T{len(tokens)}>")
                
                return tokens[:vocab_size]
            
            def _bigrational_to_float(self, br) -> float:
                """Convert BigRational to float."""
                s = str(br)
                if "/" in s:
                    num, denom = s.split("/")
                    return float(num) / float(denom)
                return float(s)
            
            def GenerateLogits(self, input_prefix):
                """Extern: Generate random logits for testing."""
                import random
                for i in range(self.Logits.length(0)):
                    self.Logits[i] = _dafny.BigRational(random.gauss(0, 1))
            
            def ChooseNextToken(self):
                """Extern: Choose highest logit token that isn't masked."""
                best_idx = 0
                best_logit = _dafny.BigRational('-1e10')
                masked_val = _dafny.BigRational('-1e9')
                
                for i in range(self.Logits.length(0)):
                    if self.Logits[i] != masked_val and self.Logits[i] > best_logit:
                        best_logit = self.Logits[i]
                        best_idx = i
                
                return self._Tokens[best_idx]

            def ChooseNextTokenUnconstrained(self):
                """Extern: Choose highest logit token regardless of masking."""
                best_idx = 0
                best_logit = _dafny.BigRational('-1e10')
                
                for i in range(self.Logits.length(0)):
                    if self.Logits[i] > best_logit:
                        best_logit = self.Logits[i]
                        best_idx = i
                
                return self._Tokens[best_idx]
            
            def MaskTokensExcept(self, valid_tokens):
                """Extern: Mask all tokens except those in valid_tokens."""
                masked_val = _dafny.BigRational('-1e9')
                # Convert valid_tokens (Dafny Seq) to a set for fast lookup
                valid_set = set(valid_tokens)
                
                for i in range(self.Logits.length(0)):
                    if self._Tokens[i] not in valid_set:
                        self.Logits[i] = masked_val

            def HasEOSToken(self):
                """Extern: Check if LM has an EOS token."""
                return True
            
            def get_eos_token(self):
                """Get the EOS token for the LM."""
                import _dafny
                return _dafny.SeqWithoutIsStrInference("<EOS>")
        
        # Create a Dafny-compatible Parser with extern implementations
        class TestParser(VerifiedDecoderAgent.Parser):
            def __init__(self, lm_tokens):
                super().__init__()
                self._lm_tokens = lm_tokens
                self._step_count = 0
                import _dafny
                self._eos_token = _dafny.SeqWithoutIsStrInference("<EOS>")
            
            def IsValidPrefix(self, prefix):
                """Extern: Always valid for testing."""
                return True
            
            def IsCompletePrefix(self, prefix):
                """Extern: Complete after some tokens or if ends with EOS."""
                if len(prefix) == 0:
                    return False
                # Check if last token is EOS
                last = prefix[len(prefix) - 1] if hasattr(prefix, '__getitem__') else list(prefix)[-1]
                if last == self._eos_token:
                    return True
                # Also complete after 10 steps for testing
                self._step_count += 1
                return self._step_count > 10
            
            def ValidNextTokens(self, prefix):
                """Extern: Return all LM tokens as valid."""
                return self._lm_tokens

            def IsPermissive(self, prefix):
                """Extern: All tokens valid => permissive."""
                return True

        # Create a JSON-aware parser
        class JsonParser(VerifiedDecoderAgent.Parser):
            """Parser that validates JSON structure."""
            
            def __init__(self, lm_tokens):
                super().__init__()
                self._lm_tokens = lm_tokens
                self._token_list = list(lm_tokens)
                
                # Import JSON validator using Lark grammar
                from utils.parsers.lark_parser import LarkGrammarParser
                import os
                grammar_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'grammars', 'json.lark')
                self._parser = LarkGrammarParser.from_grammar_file(grammar_path)
                self._is_valid_prefix = self._parser.is_valid_prefix
                self._is_complete = self._parser.is_complete
            
            def _tokens_to_text(self, tokens) -> str:
                """Convert token sequence to text."""
                # Handle various sequence types including Dafny Seq
                try:
                    # Try to iterate - works for lists, Dafny Seq, etc.
                    return "".join(str(t) for t in tokens)
                except TypeError:
                    # Fallback for non-iterable
                    return str(tokens)
            
            def IsValidPrefix(self, prefix) -> bool:
                """Check if prefix decodes to valid JSON prefix."""
                if len(prefix) == 0:
                    return True
                text = self._tokens_to_text(prefix)
                return self._is_valid_prefix(text)
            
            def IsCompletePrefix(self, prefix) -> bool:
                """Check if prefix decodes to complete JSON."""
                if len(prefix) == 0:
                    return False
                text = self._tokens_to_text(prefix)
                return self._is_complete(text)
            
            def ValidNextTokens(self, prefix):
                """Get tokens that can validly follow the prefix."""
                current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
                
                # If current prefix is invalid, return empty
                if current_text and not self._is_valid_prefix(current_text):
                    return _dafny.SeqWithoutIsStrInference([])
                
                # Filter tokens
                valid = []
                for token in self._token_list:
                    token_str = str(token)
                    if not token_str:
                        continue
                    extended = current_text + token_str
                    if self._is_valid_prefix(extended):
                        valid.append(token)
                
                return _dafny.SeqWithoutIsStrInference(valid)

            def IsPermissive(self, prefix):
                """Not permissive: only some tokens valid for JSON."""
                return False

        # Create instances based on parser mode
        is_json_mode = self.parser_mode == "json"
        lm = TestLM(vocab_size=self.vocab_size, json_mode=is_json_mode)
        
        if is_json_mode:
            parser = JsonParser(lm._Tokens)
            prompt = _dafny.SeqWithoutIsStrInference([])  # Empty prompt for JSON
        else:
            parser = TestParser(lm._Tokens)
            # Prompt must be a sequence of Dafny strings
            prompt = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference("<START>")])
        
        return lm, parser, prompt, self.max_steps
    
    def run(
        self,
        module_path: Path,
        prompt: Optional[list[str]] = None
    ) -> RuntimeResult:
        """
        Execute a compiled strategy module.
        
        Args:
            module_path: Path to the compiled Python module
            prompt: Optional custom prompt (default: test prompt)
            
        Returns:
            RuntimeResult with output or error information
        """
        import time
        
        start_time = time.time()
        
        try:
            # Load the compiled module first to set up paths
            compiled_module = self._load_compiled_module(module_path)
            module_dir = module_path.parent
            
            # Create Dafny-compatible test environment
            lm, parser, test_prompt, max_steps = self._create_dafny_test_environment(module_dir)
            
            # Find and run the MyCSDStrategy method from GeneratedCSD module
            csd_strategy_method = None
            
            # Try direct import of GeneratedCSD module (should be in path now)
            try:
                import GeneratedCSD
                if hasattr(GeneratedCSD, "default__"):
                    if hasattr(GeneratedCSD.default__, "MyCSDStrategy"):
                        csd_strategy_method = GeneratedCSD.default__.MyCSDStrategy
            except ImportError:
                pass
            
            # Also try through compiled_module
            if csd_strategy_method is None:
                if hasattr(compiled_module, "GeneratedCSD"):
                    csd_module = compiled_module.GeneratedCSD
                    if hasattr(csd_module, "default__"):
                        if hasattr(csd_module.default__, "MyCSDStrategy"):
                            csd_strategy_method = csd_module.default__.MyCSDStrategy
            
            if csd_strategy_method is None:
                return RuntimeResult(
                    success=False,
                    error_type="StrategyNotFound",
                    error_message=(
                        "Could not find MyCSDStrategy method in compiled module. "
                        f"Available attributes: {dir(compiled_module)}"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Call the strategy method - it returns (generated sequence, remainingSteps)
            sig = inspect.signature(csd_strategy_method)
            if len(sig.parameters) >= 5:
                eos_token = lm.get_eos_token()
                result = csd_strategy_method(lm, parser, test_prompt, max_steps, eos_token)
            else:
                result = csd_strategy_method(lm, parser, test_prompt, max_steps)
            
            # MyCSDStrategy returns (generated: Prefix, remainingSteps: nat)
            if isinstance(result, tuple) and len(result) == 2:
                output, remaining_steps = result
                cost = max_steps - remaining_steps  # steps consumed (for backward compat)
            elif isinstance(result, tuple) and len(result) > 0:
                output = result[0]
                cost = 0
            else:
                output = result
                cost = 0
            
            # Convert Dafny sequence to Python list for output
            if hasattr(output, '__iter__'):
                output = list(output)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Note: We no longer manually validate the output here.
            # The Dafny strategy is formally verified to maintain parser validity
            # and satisfy cost contracts.
            
            return RuntimeResult(
                success=True,
                output=output,
                cost=cost,
                execution_time_ms=execution_time
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return RuntimeResult(
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                execution_time_ms=execution_time
            )

    def run_python_native(self, python_file: Path) -> "RuntimeResult":
        """
        Run a generated strategy Python file directly without Dafny compilation.

        Loads VerifiedAgentSynthesis for the contract library, creates stub
        LM and Parser instances that use plain Python types (no Dafny runtime),
        then calls MyCSDStrategy from the generated file.
        """
        import random
        import time

        start_time = time.time()

        try:
            # Make VerifiedAgentSynthesis importable from generation/csd/
            vas_dir = str(_VAS_DIR)
            if vas_dir not in sys.path:
                sys.path.insert(0, vas_dir)

            import VerifiedAgentSynthesis as VAS

            class _TestLM(VAS.LM):
                def __init__(self, vocab_size: int = 100):
                    # Override base __init__ with a proper vocabulary
                    tokens = ["<<", ">>", "<EOS>"]
                    for i in range(vocab_size - 3):
                        if i < 26:
                            tokens.append(chr(ord("a") + i))
                        else:
                            tokens.append(f"<T{i}>")
                    self.Tokens = tokens
                    self.Ids = list(range(len(tokens)))
                    self.Logits = [0.0] * len(tokens)
                    self._Tokens = self.Tokens

                def GenerateLogits(self, input_prefix: list) -> None:
                    for i in range(len(self.Logits)):
                        self.Logits[i] = random.gauss(0, 1)

                def ChooseNextToken(self) -> str:
                    best_i = max(range(len(self.Logits)), key=lambda i: self.Logits[i])
                    return self.Tokens[best_i]

                def MaskTokensExcept(self, valid_tokens: list) -> None:
                    valid_set = set(valid_tokens)
                    for i in range(len(self.Logits)):
                        if self.Tokens[i] not in valid_set:
                            self.Logits[i] = -1e9

            class _TestParser(VAS.Parser):
                def __init__(self, tokens: list):
                    self._tokens = list(tokens)
                    self._step_count = 0

                def IsValidPrefix(self, prefix: list) -> bool:
                    return True

                def IsCompletePrefix(self, prefix: list) -> bool:
                    if len(prefix) == 0:
                        return False
                    self._step_count += 1
                    return self._step_count > 10

                def ValidNextTokens(self, prefix: list) -> list:
                    return self._tokens

            # Load the strategy module so its import of VerifiedAgentSynthesis resolves
            module_dir = str(python_file.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            spec = importlib.util.spec_from_file_location("generated_csd_native", python_file)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load strategy from {python_file}")
            strategy_module = importlib.util.module_from_spec(spec)
            sys.modules["generated_csd_native"] = strategy_module
            spec.loader.exec_module(strategy_module)

            lm = _TestLM(vocab_size=self.vocab_size)
            parser = _TestParser(lm.Tokens)
            prompt: list = []
            max_steps = self.max_steps
            eos_token = "<EOS>"

            result = strategy_module.MyCSDStrategy(lm, parser, prompt, max_steps, eos_token)

            if isinstance(result, tuple) and len(result) == 2:
                output, remaining_steps = result
                cost = max_steps - remaining_steps
            else:
                output = result
                cost = 0

            output_list = list(output) if hasattr(output, "__iter__") else []
            execution_time = (time.time() - start_time) * 1000

            return RuntimeResult(
                success=True,
                output=output_list,
                cost=cost,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return RuntimeResult(
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                execution_time_ms=execution_time,
            )
