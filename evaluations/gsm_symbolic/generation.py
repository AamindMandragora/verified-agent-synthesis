"""
Generation methods for GSM-Symbolic evaluation.

Provides CSD (Constrained Decoding Strategy) generation:
- CRANE-CSD: CRANE-style with CSD strategy for constrained windows
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

from evaluations.gsm_symbolic.prompts import make_chatml_instruction


def dafny_seq_to_str(seq) -> str:
    """
    Convert a Dafny Seq to a Python string.

    Dafny.Seq objects have __len__ and __getitem__ but NOT __iter__,
    so ''.join(seq) fails. We must use index-based iteration.
    """
    try:
        # First try direct join (works for Python strings/lists)
        return ''.join(seq)
    except TypeError:
        # Dafny.Seq doesn't support iteration - use index-based access
        try:
            return ''.join(seq[i] for i in range(len(seq)))
        except (TypeError, AttributeError, IndexError):
            return str(seq)


def run_crane_csd(
    env: dict,
    prompt_text: str,
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
    dynamic_parser=None,
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    Unified generation using Dafny-verified HybridGeneration strategy.
    
    This replaces the manual Python-side orchestration with a single call to the
    verified Dafny strategy, as requested by the user.
    """
    _dafny = env["_dafny"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]

    lm.instruction_text = make_chatml_instruction(prompt_text)
    start_time = time.time()

    # Determine EOS token for the model
    # For Qwen2.5, it's <|im_end|> but we use the tokenizer's eos_token
    eos_token_str = lm.tokenizer.eos_token or "<|im_end|>"
    eos_token_dafny = _dafny.Seq(eos_token_str)

    # Call the verified strategy.
    # It handles both unconstrained reasoning and constrained math windows.
    # Returns (generated_tokens, total_cost)
    result = GeneratedCSD.default__.MyCSDStrategy(
        lm, parser, _dafny.SeqWithoutIsStrInference([]), max_steps, eos_token_dafny
    )
    
    # Handle Dafny multiple return values
    if isinstance(result, tuple):
        csd_output, total_cost = result
    else:
        csd_output = result
        total_cost = 0

    result_tokens = [dafny_seq_to_str(t) for t in csd_output]
    output_text = "".join(result_tokens)
    execution_time = time.time() - start_time

    # Extract constrained segments for metrics/reporting
    # We still want to know which parts were valid for evaluation purposes.
    constrained_segments = []
    
    # Use the same logic as before to find segments between << and >>
    import re
    matches = re.findall(r'<<(.*?)>>', output_text, re.DOTALL)
    
    from parsers.lark_parser import LarkGrammarParser
    math_validator = LarkGrammarParser.from_grammar_file(str(grammar_file), start='any_expr')
    
    for segment in matches:
        is_valid = math_validator.is_valid_prefix(segment.strip())
        constrained_segments.append((segment, is_valid))

    if debug_delimiters:
        print(f"  [DEBUG] Generation finished in {execution_time:.2f}s. Cost: {total_cost}. Tokens: {len(result_tokens)}")

    return output_text, len(result_tokens), execution_time, constrained_segments


def run_unconstrained(
    env: dict,
    prompt_text: str,
    max_steps: int,
    debug: bool = False,
) -> Tuple[str, int, float]:
    """
    Run unconstrained generation without CSD.
    
    This is the baseline CRANE approach - generates freely without
    grammar constraints in the << >> windows.
    
    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text
        max_steps: Maximum generation steps
        debug: Whether to print debug output
        
    Returns:
        Tuple of (output_text, token_count, time_seconds)
    """
    _dafny = env["_dafny"]
    lm = env["lm"]
    
    result_tokens: List[str] = []
    
    lm.instruction_text = make_chatml_instruction(prompt_text)
    start_time = time.time()
    
    eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
    min_tokens_before_eos = 20
    
    # Repetition detection
    recent_text_window = []
    
    for step in range(max_steps):
        dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
        lm.GenerateLogits(dafny_prefix)
        token = lm.ChooseNextTokenUnconstrained()
        token_str = dafny_seq_to_str(token)
        result_tokens.append(token_str)
        
        # Check for EOS
        if token_str in eos_tokens and step >= min_tokens_before_eos:
            break
        
        # Check for #### termination (GSM-specific)
        joined = "".join(result_tokens[-30:])
        if "####" in joined:
            # Continue for a few more tokens to capture the answer
            for _ in range(min(15, max_steps - step - 1)):
                dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
                lm.GenerateLogits(dafny_prefix)
                token = lm.ChooseNextTokenUnconstrained()
                token_str = dafny_seq_to_str(token)
                result_tokens.append(token_str)
                if token_str in eos_tokens or token_str == "\n":
                    break
            break
        
        # Detect repetitive loops
        recent_text_window.append(token_str)
        if len(recent_text_window) > 100:
            recent_text_window.pop(0)
        
        if step > 20:
            for pattern_len in [5, 10, 15, 20]:
                if len(result_tokens) >= pattern_len * 3:
                    last_pattern = result_tokens[-pattern_len:]
                    prev_pattern = result_tokens[-pattern_len*2:-pattern_len]
                    prev_prev_pattern = result_tokens[-pattern_len*3:-pattern_len*2]
                    if last_pattern == prev_pattern == prev_prev_pattern:
                        if debug:
                            pattern_text = "".join(last_pattern)
                            print(f"  [DEBUG] WARNING: Detected repetitive loop at step {step}")
                        break
            else:
                continue
            break
    
    end_time = time.time()
    output_text = "".join(result_tokens)
    
    return output_text, len(result_tokens), end_time - start_time
