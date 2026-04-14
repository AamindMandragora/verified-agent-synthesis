#!/usr/bin/env python3
"""
Run a compiled CSD strategy with a specific grammar.

This script allows you to:
1. Take a compiled CSD strategy (from a successful synthesis run)
2. Specify a grammar file (.lark) for constraint validation
3. Run constrained generation with real grammar enforcement

Usage:
    # With a .lark grammar file
    python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/XXXXX --grammar grammars/json.lark
    
    # With a built-in format
    python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/XXXXX --format json
    
    # Custom vocabulary from HuggingFace tokenizer
    python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/XXXXX --format json --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import sys
import json
import random
import re
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from reorganized modules
from evaluations.common.parser_utils import create_lark_dafny_parser, get_builtin_grammar
from evaluations.common.generation import dafny_seq_to_str


def resolve_compiled_module_dir(run_dir: Path) -> Path:
    """Return the directory containing GeneratedCSD.py."""
    run_dir = run_dir.resolve()
    if (run_dir / "GeneratedCSD.py").exists():
        return run_dir
    candidates = [
        run_dir / "generated_csd",
        run_dir / "folio_csd",
        run_dir / "gsm_crane_csd",
        run_dir / "fol_csd",
        run_dir / "pddl_csd",
        run_dir / "sygus_slia_csd",
    ]
    for d in candidates:
        if d.exists() and (d / "GeneratedCSD.py").exists():
            return d
    found = list(run_dir.glob("*/GeneratedCSD.py"))
    if found:
        return found[0].parent
    raise FileNotFoundError(f"No compiled CSD module found in {run_dir}")


def extract_final_constrained_segment(output_text: str) -> str:
    """Return the last << >> segment when present, else the full output."""
    matches = re.findall(r"<<\s*([\s\S]+?)\s*>>", output_text)
    if matches:
        return matches[-1]
    return output_text


def ensure_special_tokens(vocab: list[str], size: int) -> list[str]:
    """Ensure the vocabulary contains delimiter and EOS tokens expected by the template."""
    specials = ["<<", ">>", "<EOS>"]
    cleaned: list[str] = []
    for tok in specials + vocab:
        if tok and tok not in cleaned:
            cleaned.append(tok)
    return cleaned[:size]


def create_vocabulary(vocab_type: str = "default", tokenizer_name: Optional[str] = None, size: int = 500) -> list[str]:
    """Create vocabulary for the LM."""
    if vocab_type == "tokenizer" and tokenizer_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        vocab = []
        for i in range(min(len(tokenizer), size)):
            try:
                token = tokenizer.decode([i])
                if token:
                    vocab.append(token)
            except:
                pass
        return ensure_special_tokens(vocab, size)
    
    # Default vocabulary with generic tokens
    vocab = list('{}[]():,."\'+-*/=<>!&|^~%@#$_\\;? \t\n')
    vocab.extend(list('0123456789'))
    vocab.extend(list('abcdefghijklmnopqrstuvwxyz'))
    vocab.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    vocab.extend(["<<", ">>", "<EOS>"])
    
    while len(vocab) < size:
        vocab.append(f'<T{len(vocab)}>')
    
    return ensure_special_tokens(vocab, size)


def run_csd_with_grammar(
    run_dir: Path,
    grammar_source: str,
    max_steps: int = 50,
    vocab_size: int = 500,
    tokenizer_name: Optional[str] = None,
    seed: int = 42,
    start_rule: str = "start"
) -> dict:
    """
    Run a compiled CSD strategy with a specific grammar.
    
    Args:
        run_dir: Path to the run directory containing compiled CSD
        grammar_source: Grammar file path or built-in format name
        max_steps: Maximum generation steps
        vocab_size: Vocabulary size
        tokenizer_name: HuggingFace tokenizer for vocabulary
        seed: Random seed
        start_rule: Grammar start rule
        
    Returns:
        Dictionary with results
    """
    random.seed(seed)
    
    run_dir = Path(run_dir)
    try:
        module_dir = resolve_compiled_module_dir(run_dir)
    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc)}
    
    # Add module dir to path
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    
    try:
        import _dafny
        import VerifiedDecoderAgent
        import GeneratedCSD
        
        # Determine grammar source
        if Path(grammar_source).exists():
            grammar = Path(grammar_source).read_text()
        else:
            grammar = get_builtin_grammar(grammar_source)
        
        # Create vocabulary
        vocab = create_vocabulary(
            vocab_type="tokenizer" if tokenizer_name else "default",
            tokenizer_name=tokenizer_name,
            size=vocab_size
        )
        
        # Helper for BigRational
        def bigrational_to_float(br):
            s = str(br)
            if '/' in s:
                num, den = s.split('/')
                return float(num) / float(den)
            return float(s)
        
        # Create LM
        class TestLM(VerifiedDecoderAgent.LM):
            def __init__(self, tokens):
                super().__init__()
                dafny_tokens = [_dafny.SeqWithoutIsStrInference(t) for t in tokens]
                self._Tokens = _dafny.SeqWithoutIsStrInference(dafny_tokens)
                self._Ids = _dafny.SeqWithoutIsStrInference(list(range(len(tokens))))
                self.Logits = _dafny.Array(None, len(tokens))
                for i in range(len(tokens)):
                    self.Logits[i] = _dafny.BigRational(0)
            
            def GenerateLogits(self, input_prefix):
                for i in range(self.Logits.length(0)):
                    self.Logits[i] = _dafny.BigRational(random.gauss(0, 1))
            
            def ChooseNextToken(self):
                best_idx = 0
                best_logit = -1e10
                masked_val = _dafny.BigRational('-1e9')
                for i in range(self.Logits.length(0)):
                    if self.Logits[i] != masked_val:
                        val = bigrational_to_float(self.Logits[i])
                        if val > best_logit:
                            best_logit = val
                            best_idx = i
                return self._Tokens[best_idx]
        
        # Create parser from grammar
        LarkDafnyParser = create_lark_dafny_parser(grammar, VerifiedDecoderAgent, _dafny, start_rule)
        
        lm = TestLM(vocab)
        parser = LarkDafnyParser(lm._Tokens)
        prompt = _dafny.SeqWithoutIsStrInference([])
        eos_token = _dafny.SeqWithoutIsStrInference("<EOS>")
        
        # Run the CSD strategy
        result = GeneratedCSD.default__.MyCSDStrategy(lm, parser, prompt, max_steps, eos_token)
        if isinstance(result, tuple):
            output, _remaining_steps = result
        else:
            output = result
        output_list = [dafny_seq_to_str(t) for t in output]
        output_text = "".join(output_list)
        
        # Validate the constrained segment rather than any free-form prefix outside delimiters.
        constrained_text = extract_final_constrained_segment(output_text)
        is_valid = parser._is_valid_prefix(constrained_text)
        is_complete = parser._is_complete(constrained_text)
        
        return {
            "success": True,
            "output_tokens": output_list,
            "output_text": output_text,
            "constrained_text": constrained_text,
            "output_length": len(output_list),
            "is_valid_prefix": is_valid,
            "is_complete": is_complete,
            "grammar_source": str(grammar_source),
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run a compiled CSD strategy with a specific grammar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a .lark grammar file
  python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 --grammar grammars/json.lark
  
  # Use a built-in format
  python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 --format json
  
  # With HuggingFace tokenizer vocabulary
  python scripts/run_csd_with_grammar.py --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 --format json --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
"""
    )
    
    parser.add_argument("--run-dir", "-r", type=Path, required=True,
                        help="Path to the run directory containing compiled CSD")
    
    grammar_group = parser.add_mutually_exclusive_group(required=True)
    grammar_group.add_argument("--grammar", "-g", type=str,
                               help="Path to .lark grammar file")
    grammar_group.add_argument("--format", "-f", type=str, choices=["json", "sql", "math"],
                               help="Built-in format name")
    
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum generation steps (default: 50)")
    parser.add_argument("--vocab-size", type=int, default=500,
                        help="Vocabulary size (default: 500)")
    parser.add_argument("--tokenizer", "-t", type=str, default=None,
                        help="HuggingFace tokenizer for vocabulary")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--start-rule", type=str, default="start",
                        help="Grammar start rule (default: start)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    
    args = parser.parse_args()
    
    # Determine grammar source
    grammar_source = args.grammar if args.grammar else args.format
    
    print(f"Running CSD with grammar: {grammar_source}")
    print(f"Run directory: {args.run_dir}")
    print()
    
    results = run_csd_with_grammar(
        run_dir=args.run_dir,
        grammar_source=grammar_source,
        max_steps=args.max_steps,
        vocab_size=args.vocab_size,
        tokenizer_name=args.tokenizer,
        seed=args.seed,
        start_rule=args.start_rule
    )
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if results["success"]:
            print(f"✓ Generation successful")
            print(f"  Output length: {results['output_length']} tokens")
            print(f"  Output: {repr(results['output_text'][:80])}...")
            print(f"  Constrained segment: {repr(results['constrained_text'][:80])}...")
            print(f"  Valid prefix: {results['is_valid_prefix']}")
            print(f"  Complete: {results['is_complete']}")
        else:
            print(f"✗ Generation failed: {results.get('error', 'Unknown error')}")
            if results.get("traceback"):
                print(results["traceback"])
    
    sys.exit(0 if results["success"] and results.get("is_valid_prefix") else 1)


if __name__ == "__main__":
    main()
