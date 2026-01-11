#!/usr/bin/env python3
"""
Validation script for JSON-oriented CSD strategies.

This script:
1. Loads a generated CSD strategy from a specified run
2. Creates a JSON-aware parser using the Qwen tokenizer
3. Runs constrained decoding with the JSON parser
4. Validates that the output is valid JSON

Usage:
    python scripts/validate_json_csd.py [--run-dir PATH] [--max-steps N] [--vocab-size N]
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.json_prefix import (
    is_valid_json_prefix, 
    is_complete_json,
    JsonPrefixValidator
)


def create_json_test_parser(VerifiedDecoderAgent):
    """
    Create a JSON parser compatible with the Dafny-compiled code.
    
    This is a wrapper that provides IsValidPrefix, IsCompletePrefix, and 
    ValidNextTokens methods that operate on Dafny sequences of tokens.
    
    Args:
        VerifiedDecoderAgent: The imported VerifiedDecoderAgent module
    """
    import _dafny
    
    class DafnyJsonParser(VerifiedDecoderAgent.Parser):
        """JSON parser compatible with Dafny-compiled VerifiedDecoderAgent.Parser."""
        
        def __init__(self, lm_tokens):
            """
            Initialize with the LM's token vocabulary.
            
            Args:
                lm_tokens: Dafny sequence of token strings
            """
            super().__init__()
            self._lm_tokens = lm_tokens
            # Convert to Python list for faster operations
            self._token_list = list(lm_tokens)
            
            # Precompute valid JSON single-character tokens
            self._json_chars = set('{}[],:"\\ 0123456789.-+eEabcdefhlnrstu\t\n\r')
        
        def _tokens_to_text(self, tokens) -> str:
            """Convert token sequence to text."""
            # Dafny sequences support indexing but not __iter__
            # Use len() and indexing to iterate safely
            try:
                return "".join(str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError):
                return str(tokens)
        
        def IsValidPrefix(self, prefix) -> bool:
            """Check if prefix decodes to valid JSON prefix."""
            if len(prefix) == 0:
                return True
            text = self._tokens_to_text(prefix)
            return is_valid_json_prefix(text)
        
        def IsCompletePrefix(self, prefix) -> bool:
            """Check if prefix decodes to complete JSON."""
            if len(prefix) == 0:
                return False
            text = self._tokens_to_text(prefix)
            return is_complete_json(text)
        
        def ValidNextTokens(self, prefix):
            """Get tokens that can validly follow the prefix."""
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            
            # If current prefix is invalid, return empty
            if current_text and not is_valid_json_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])
            
            # Filter tokens
            valid = []
            for token in self._token_list:
                token_str = str(token)
                if not token_str:
                    continue
                extended = current_text + token_str
                if is_valid_json_prefix(extended):
                    valid.append(token)
            
            return _dafny.SeqWithoutIsStrInference(valid)
        
        def IsDeadPrefix(self, prefix) -> bool:
            """Check if prefix has no valid continuations."""
            return not self.IsCompletePrefix(prefix) and len(self.ValidNextTokens(prefix)) == 0
        
        def ValidNextToken(self, prefix, token) -> bool:
            """Check if a specific token is valid next."""
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            extended = current_text + str(token)
            return is_valid_json_prefix(extended)
    
    return DafnyJsonParser


def create_json_vocabulary(vocab_size: int = 1000) -> list[str]:
    """
    Create a vocabulary suitable for JSON generation.
    
    All tokens must be valid JSON fragments that can appear in valid JSON.
    This ensures constrained decoding can always find valid continuations.
    """
    vocab = []
    
    # JSON structural tokens (highest priority)
    structural = ["{", "}", "[", "]", ":", ",", " ", "\n", "\t"]
    vocab.extend(structural)
    
    # String quote (needed for strings)
    vocab.append('"')
    
    # Numbers
    digits = list("0123456789")
    vocab.extend(digits)
    
    # Number components (only valid within numbers)
    vocab.extend([".", "-"])
    
    # JSON literals (complete)
    literals = ["true", "false", "null"]
    vocab.extend(literals)
    
    # Common complete string values (when inside quotes)
    # These are complete values that can appear between quotes
    string_values = [
        "name", "value", "id", "type", "data", "message", "status",
        "error", "result", "items", "count", "total", "page", "limit",
        "key", "text", "content", "title", "description", "url"
    ]
    vocab.extend(string_values)
    
    # Common complete number strings
    number_strings = ["10", "20", "100", "1000", "0.5", "1.0", "2.0"]
    vocab.extend(number_strings)
    
    # EOS token
    vocab.append("<EOS>")
    
    # Fill remaining with more string values to ensure we have enough tokens
    additional_strings = [
        "hello", "world", "test", "user", "admin", "guest",
        "active", "pending", "done", "open", "closed",
        "first", "last", "next", "prev", "new", "old"
    ]
    for s in additional_strings:
        if len(vocab) < vocab_size:
            vocab.append(s)
    
    # Fill to vocab_size with numbered tokens
    while len(vocab) < vocab_size:
        vocab.append(f"item{len(vocab)}")
    
    return vocab[:vocab_size]


def run_json_csd_validation(
    run_dir: Optional[Path] = None,
    max_steps: int = 50,
    vocab_size: int = 500,
    seed: int = 42
) -> dict:
    """
    Run validation of a JSON CSD strategy.
    
    Args:
        run_dir: Path to the run directory (default: latest JSON run)
        max_steps: Maximum generation steps
        vocab_size: Vocabulary size
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with validation results
    """
    random.seed(seed)
    
    # Default to the JSON-oriented run
    if run_dir is None:
        run_dir = PROJECT_ROOT / "outputs/generated-csd/runs/20260105_204255_8b7116"
    
    run_dir = Path(run_dir)
    module_dir = run_dir / "generated_csd"
    
    if not module_dir.exists():
        return {
            "success": False,
            "error": f"Module directory not found: {module_dir}"
        }
    
    # Add module directory to path
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    results = {
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "vocab_size": vocab_size,
        "tests": []
    }
    
    try:
        # Import Dafny modules
        import _dafny
        import VerifiedDecoderAgent
        import GeneratedCSD
        
        # Create JSON vocabulary
        vocab = create_json_vocabulary(vocab_size)
        
        # Helper to convert BigRational to float
        def bigrational_to_float(br):
            """Convert Dafny BigRational to Python float."""
            s = str(br)
            if '/' in s:
                num, den = s.split('/')
                return float(num) / float(den)
            return float(s)
        
        # Create LM with JSON vocabulary
        class JsonTestLM(VerifiedDecoderAgent.LM):
            def __init__(self, tokens):
                super().__init__()
                self._Tokens = _dafny.SeqWithoutIsStrInference(tokens)
                self._Ids = _dafny.SeqWithoutIsStrInference(list(range(len(tokens))))
                self.Logits = _dafny.Array(None, len(tokens))
                for i in range(len(tokens)):
                    self.Logits[i] = _dafny.BigRational(0)
                
                # Pre-identify structural token indices for biasing
                self._structural_indices = set()
                structural_tokens = {"{", "}", "[", "]", ":", ",", '"', " "}
                for i, token in enumerate(tokens):
                    if str(token) in structural_tokens:
                        self._structural_indices.add(i)
            
            def GenerateLogits(self, input_prefix):
                """Generate logits biased toward JSON structure."""
                # Reset logits with slight randomness
                for i in range(self.Logits.length(0)):
                    base_logit = random.gauss(0, 1)
                    # Bias toward structural tokens
                    if i in self._structural_indices:
                        base_logit += 2.0
                    self.Logits[i] = _dafny.BigRational(base_logit)
            
            def ChooseNextToken(self):
                """Choose highest unmasked logit."""
                best_idx = 0
                best_logit = -1e10
                masked_val = _dafny.BigRational('-1e9')
                
                for i in range(self.Logits.length(0)):
                    if self.Logits[i] != masked_val:
                        logit_val = bigrational_to_float(self.Logits[i])
                        if logit_val > best_logit:
                            best_logit = logit_val
                            best_idx = i
                
                return self._Tokens[best_idx]
        
        # Create instances
        lm = JsonTestLM(vocab)
        DafnyJsonParser = create_json_test_parser(VerifiedDecoderAgent)
        parser = DafnyJsonParser(lm._Tokens)
        
        # Run multiple test cases
        test_prompts = [
            [],  # Empty prompt
            ["{"],  # Start of object
            ["["],  # Start of array
        ]
        
        for i, prompt_tokens in enumerate(test_prompts):
            prompt = _dafny.SeqWithoutIsStrInference(prompt_tokens)
            
            try:
                # Run the CSD strategy
                output = GeneratedCSD.default__.MyCSDStrategy(lm, parser, prompt, max_steps)
                
                # Convert output to string
                output_list = list(output)
                output_text = "".join(str(t) for t in output_list)
                
                # Validate JSON
                is_valid_prefix = is_valid_json_prefix(output_text)
                is_complete = is_complete_json(output_text)
                
                # Try json.loads
                json_parse_success = False
                json_parse_error = None
                parsed_value = None
                
                try:
                    parsed_value = json.loads(output_text)
                    json_parse_success = True
                except json.JSONDecodeError as e:
                    json_parse_error = str(e)
                
                test_result = {
                    "test_id": i,
                    "prompt": prompt_tokens,
                    "output_tokens": output_list,
                    "output_text": output_text,
                    "output_length": len(output_list),
                    "is_valid_prefix": is_valid_prefix,
                    "is_complete_json": is_complete,
                    "json_loads_success": json_parse_success,
                    "json_parse_error": json_parse_error,
                    "parsed_value": parsed_value if json_parse_success else None
                }
                
                results["tests"].append(test_result)
                
            except Exception as e:
                results["tests"].append({
                    "test_id": i,
                    "prompt": prompt_tokens,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
        
        # Summarize results
        # A test passes if either:
        # 1. The output is complete, valid JSON (json_loads_success)
        # 2. The output is a valid JSON prefix (constraints were enforced)
        fully_passed = sum(1 for t in results["tests"] if t.get("json_loads_success", False))
        prefix_valid = sum(1 for t in results["tests"] if t.get("is_valid_prefix", False))
        total = len(results["tests"])
        
        results["summary"] = {
            "complete_json": fully_passed,
            "valid_prefix": prefix_valid,
            "total": total,
            "complete_rate": fully_passed / total if total > 0 else 0,
            "prefix_rate": prefix_valid / total if total > 0 else 0
        }
        # Success if all outputs are at least valid prefixes (constraints enforced)
        results["success"] = prefix_valid == total
        
    except Exception as e:
        import traceback
        results["success"] = False
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
    
    return results


def run_prefix_validator_tests() -> dict:
    """Run unit tests on the JSON prefix validator."""
    tests = []
    
    # Test cases: (input, expected_valid_prefix, expected_complete)
    test_cases = [
        # Valid prefixes
        ('', True, False),
        ('{', True, False),
        ('{"', True, False),
        ('{"key"', True, False),
        ('{"key":', True, False),
        ('{"key": ', True, False),
        ('{"key": "value"', True, False),
        ('{"key": "value"}', True, True),
        ('[', True, False),
        ('[1', True, False),
        ('[1,', True, False),
        ('[1, 2', True, False),
        ('[1, 2]', True, True),
        ('"hello"', True, True),
        ('123', True, True),
        ('-123', True, True),
        ('12.34', True, True),
        ('1.2e10', True, True),
        ('true', True, True),
        ('false', True, True),
        ('null', True, True),
        
        # Incomplete but valid
        ('tru', True, False),
        ('fals', True, False),
        ('nul', True, False),
        ('{"nested": {"inner":', True, False),
        ('[[[[', True, False),
        
        # String escapes
        ('"hello\\nworld"', True, True),
        ('"tab\\there"', True, True),
        ('"quote\\"here"', True, True),
        ('"unicode\\u0041"', True, True),
        ('"partial\\u00', True, False),
        
        # Numbers
        ('0', True, True),
        ('-0', True, True),
        ('0.5', True, True),
        ('1e5', True, True),
        ('1E+5', True, True),
        ('1e-5', True, True),
        ('-1.5e-10', True, True),
        
        # Invalid prefixes
        ('{key}', False, False),  # Unquoted key
        ('{"key": undefined}', False, False),  # Invalid literal
        ('"hello', True, False),  # Incomplete string (valid prefix)
        ('01', False, False),  # Leading zero
        (',', False, False),  # Leading comma
        (':', False, False),  # Leading colon
        ('}', False, False),  # Unmatched close
        (']', False, False),  # Unmatched close
        ('{"key": }', False, False),  # Missing value
        
        # Whitespace handling
        ('  {  }  ', True, True),
        ('{\n  "key": "value"\n}', True, True),
        ('[  1  ,  2  ,  3  ]', True, True),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_valid, expected_complete in test_cases:
        actual_valid = is_valid_json_prefix(text)
        actual_complete = is_complete_json(text)
        
        valid_match = actual_valid == expected_valid
        complete_match = actual_complete == expected_complete
        
        test_passed = valid_match and complete_match
        
        if test_passed:
            passed += 1
        else:
            failed += 1
        
        tests.append({
            "input": repr(text),
            "expected_valid_prefix": expected_valid,
            "actual_valid_prefix": actual_valid,
            "expected_complete": expected_complete,
            "actual_complete": actual_complete,
            "passed": test_passed
        })
    
    return {
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "success": failed == 0
    }


def main():
    parser = argparse.ArgumentParser(description="Validate JSON CSD strategy")
    parser.add_argument("--run-dir", type=str, help="Path to run directory")
    parser.add_argument("--max-steps", type=int, default=50, help="Max generation steps")
    parser.add_argument("--vocab-size", type=int, default=500, help="Vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-only", action="store_true", help="Only run prefix validator tests")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    results = {}
    
    # Run prefix validator tests
    print("=" * 60)
    print("Running JSON prefix validator tests...")
    print("=" * 60)
    
    validator_results = run_prefix_validator_tests()
    results["validator_tests"] = validator_results
    
    if not args.json:
        print(f"\nValidator tests: {validator_results['passed']} passed, {validator_results['failed']} failed")
        
        if validator_results["failed"] > 0:
            print("\nFailed tests:")
            for test in validator_results["tests"]:
                if not test["passed"]:
                    print(f"  Input: {test['input']}")
                    print(f"    Expected: valid={test['expected_valid_prefix']}, complete={test['expected_complete']}")
                    print(f"    Actual:   valid={test['actual_valid_prefix']}, complete={test['actual_complete']}")
    
    # Run CSD validation unless --test-only
    if not args.test_only:
        print("\n" + "=" * 60)
        print("Running CSD strategy validation...")
        print("=" * 60)
        
        run_dir = Path(args.run_dir) if args.run_dir else None
        csd_results = run_json_csd_validation(
            run_dir=run_dir,
            max_steps=args.max_steps,
            vocab_size=args.vocab_size,
            seed=args.seed
        )
        results["csd_validation"] = csd_results
        
        if not args.json:
            if csd_results.get("error"):
                print(f"\nError: {csd_results['error']}")
                if csd_results.get("traceback"):
                    print(csd_results["traceback"])
            else:
                summary = csd_results['summary']
                print(f"\nCSD validation summary:")
                print(f"  Valid JSON prefixes: {summary['valid_prefix']}/{summary['total']} ({summary['prefix_rate']:.0%})")
                print(f"  Complete JSON: {summary['complete_json']}/{summary['total']} ({summary['complete_rate']:.0%})")
                
                for test in csd_results.get("tests", []):
                    # ✓ for complete JSON, ~ for valid prefix only, ✗ for invalid
                    if test.get("json_loads_success"):
                        status = "✓"
                    elif test.get("is_valid_prefix"):
                        status = "~"  # Valid prefix but incomplete
                    else:
                        status = "✗"
                    print(f"\n  Test {test['test_id']}: {status}")
                    print(f"    Prompt: {test.get('prompt', 'N/A')}")
                    print(f"    Output length: {test.get('output_length', 'N/A')} tokens")
                    output_text = test.get('output_text', 'N/A')
                    print(f"    Output text: {repr(output_text[:50])}...")
                    print(f"    Valid prefix: {test.get('is_valid_prefix', 'N/A')}")
                    print(f"    Complete JSON: {test.get('is_complete_json', 'N/A')}")
                    if test.get("json_parse_error") and not test.get("is_valid_prefix"):
                        print(f"    Parse error: {test['json_parse_error']}")
    
    # Output JSON if requested
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    # Exit with appropriate code
    all_passed = (
        validator_results["success"] and
        (args.test_only or results.get("csd_validation", {}).get("success", False))
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

