"""
Python runner for compiled Dafny CSD strategies.

Executes the compiled Python code and captures results/errors.

Supports both permissive testing mode and real JSON parsing mode.
"""

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

# Type for parser mode
ParserMode = Literal["permissive", "json"]


@dataclass
class RuntimeResult:
    """Result of running a compiled strategy."""
    success: bool
    output: Optional[list[str]] = None  # Generated tokens
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
        runtime_stubs_path: Optional[Path] = None,
        vocab_size: int = 1000,
        max_steps: int = 100,
        parser_mode: ParserMode = "permissive"
    ):
        """
        Initialize the runner.
        
        Args:
            runtime_stubs_path: Path to runtime_stubs.py (auto-detected if None)
            vocab_size: Vocabulary size for the LM stub
            max_steps: Maximum generation steps
            parser_mode: "permissive" for testing, "json" for real JSON validation
        """
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.parser_mode = parser_mode
        
        # Auto-detect runtime stubs path
        if runtime_stubs_path is None:
            runtime_stubs_path = (
                Path(__file__).parent.parent / 
                "runtime" / 
                "runtime_stubs.py"
            )
        
        if not runtime_stubs_path.exists():
            raise FileNotFoundError(
                f"Runtime stubs not found at {runtime_stubs_path}. "
                "Make sure runtime_stubs.py exists in runtime/"
            )
        
        self.runtime_stubs_path = runtime_stubs_path
        self._stubs_module = None
    
    def _load_stubs(self) -> Any:
        """Load the runtime stubs module."""
        if self._stubs_module is None:
            spec = importlib.util.spec_from_file_location(
                "runtime_stubs",
                self.runtime_stubs_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not load runtime stubs from {self.runtime_stubs_path}"
                )
            
            self._stubs_module = importlib.util.module_from_spec(spec)
            sys.modules["runtime_stubs"] = self._stubs_module
            spec.loader.exec_module(self._stubs_module)
        
        return self._stubs_module
    
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
    
    def _create_test_environment(self) -> tuple[Any, Any, list[str], set[str]]:
        """
        Create a test environment with LM, Parser, prompt, and tokens.
        
        Returns:
            Tuple of (lm, parser, prompt, allTokens)
        """
        stubs = self._load_stubs()
        
        # Create LM with test vocabulary
        lm = stubs.LM(vocab_size=self.vocab_size)
        
        # Create parser (permissive for testing)
        parser = stubs.Parser()
        
        # Simple test prompt
        prompt = ["<START>"]
        
        # All tokens from LM vocabulary
        all_tokens = set(lm.Tokens)
        
        return lm, parser, prompt, all_tokens
    
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
                
                self._Tokens = _dafny.SeqWithoutIsStrInference(tokens)
                self._Ids = _dafny.SeqWithoutIsStrInference(list(range(len(tokens))))
                self.Logits = _dafny.Array(None, len(tokens))
                for i in range(len(tokens)):
                    self.Logits[i] = _dafny.BigRational(0)
            
            def _create_vocabulary(self, vocab_size, json_mode):
                """Create vocabulary, optionally with JSON tokens."""
                if not json_mode:
                    tokens = []
                    for i in range(vocab_size):
                        if i == 0:
                            tokens.append("<EOS>")
                        elif i < 26:
                            tokens.append(chr(ord('a') + i - 1))
                        else:
                            tokens.append(f"<T{i}>")
                    return tokens
                
                # JSON-friendly vocabulary
                tokens = []
                
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
            
            def GenerateLogits(self, input_prefix):
                """Extern: Generate random logits for testing."""
                import random
                for i in range(self.Logits.length(0)):
                    self.Logits[i] = _dafny.BigRational(random.gauss(0, 1))
                
                # Bias toward structural tokens in JSON mode
                if self.json_mode:
                    structural = {"{", "}", "[", "]", ":", ",", '"', " "}
                    for i, token in enumerate(self._Tokens):
                        if str(token) in structural:
                            current = float(str(self.Logits[i]))
                            self.Logits[i] = _dafny.BigRational(current + 2.0)
            
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
        
        # Create a Dafny-compatible Parser with extern implementations
        class TestParser(VerifiedDecoderAgent.Parser):
            def __init__(self, lm_tokens):
                super().__init__()
                self._lm_tokens = lm_tokens
                self._step_count = 0
            
            def IsValidPrefix(self, prefix):
                """Extern: Always valid for testing."""
                return True
            
            def IsCompletePrefix(self, prefix):
                """Extern: Complete after some tokens or if ends with EOS."""
                if len(prefix) == 0:
                    return False
                # Check if last token is EOS
                last = prefix[len(prefix) - 1] if hasattr(prefix, '__getitem__') else list(prefix)[-1]
                if last == "<EOS>":
                    return True
                # Also complete after 10 steps for testing
                self._step_count += 1
                return self._step_count > 10
            
            def ValidNextTokens(self, prefix):
                """Extern: Return all LM tokens as valid."""
                return self._lm_tokens
        
        # Create a JSON-aware parser
        class JsonParser(VerifiedDecoderAgent.Parser):
            """Parser that validates JSON structure."""
            
            def __init__(self, lm_tokens):
                super().__init__()
                self._lm_tokens = lm_tokens
                self._token_list = list(lm_tokens)
                
                # Import JSON validator
                from parsers.json_prefix import is_valid_json_prefix, is_complete_json
                self._is_valid_prefix = is_valid_json_prefix
                self._is_complete = is_complete_json
            
            def _tokens_to_text(self, tokens) -> str:
                """Convert token sequence to text."""
                if hasattr(tokens, '__iter__'):
                    return "".join(str(t) for t in tokens)
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
        
        # Create instances based on parser mode
        is_json_mode = self.parser_mode == "json"
        lm = TestLM(vocab_size=self.vocab_size, json_mode=is_json_mode)
        
        if is_json_mode:
            parser = JsonParser(lm._Tokens)
            prompt = _dafny.SeqWithoutIsStrInference([])  # Empty prompt for JSON
        else:
            parser = TestParser(lm._Tokens)
            prompt = _dafny.SeqWithoutIsStrInference(["<START>"])
        
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
            
            # Call the strategy method directly - it performs constrained decoding
            # and returns the generated sequence
            output = csd_strategy_method(lm, parser, test_prompt, max_steps)
            
            # Convert Dafny sequence to Python list for output
            if hasattr(output, '__iter__'):
                output = list(output)
            
            execution_time = (time.time() - start_time) * 1000
            
            return RuntimeResult(
                success=True,
                output=output,
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
    
    def _is_strategy_like(self, obj: Any) -> bool:
        """Check if an object looks like a Strategy."""
        stubs = self._load_stubs()
        
        # Check if it's an instance of any Strategy type
        strategy_types = (
            stubs.Window,
            stubs.TryK,
            stubs.Cascade,
            stubs.BestOfN,
            stubs.Constrained,
            stubs.Free
        )
        
        return isinstance(obj, strategy_types)
    
    def run_with_strategy(
        self,
        strategy: Any,
        prompt: Optional[list[str]] = None
    ) -> RuntimeResult:
        """
        Execute a strategy object directly (not from a compiled module).
        
        Useful for testing strategies constructed in Python.
        
        Args:
            strategy: A Strategy object from runtime_stubs
            prompt: Optional custom prompt
            
        Returns:
            RuntimeResult
        """
        import time
        
        start_time = time.time()
        
        try:
            stubs = self._load_stubs()
            lm, parser, test_prompt, all_tokens = self._create_test_environment()
            
            if prompt is not None:
                test_prompt = prompt
            
            output = stubs.Run(
                lm,
                strategy,
                parser,
                test_prompt,
                all_tokens,
                self.max_steps
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return RuntimeResult(
                success=True,
                output=output,
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
    
    def validate_strategy(self, strategy: Any) -> tuple[bool, str]:
        """
        Validate that a strategy guarantees valid output.
        
        Args:
            strategy: A Strategy object
            
        Returns:
            Tuple of (is_valid, message)
        """
        stubs = self._load_stubs()
        
        if stubs.GuaranteesValidOutput(strategy):
            return True, "Strategy guarantees valid output"
        else:
            return False, "Strategy does NOT guarantee valid output (e.g., uses Free without fallback)"

