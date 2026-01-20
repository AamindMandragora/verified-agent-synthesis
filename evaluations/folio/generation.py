"""
Generation methods for FOLIO evaluation.

Provides CSD (Constrained Decoding Strategy) generation:
- CRANE-CSD: CRANE-style with CSD strategy for constrained FOL windows
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

from evaluations.folio.prompts import (
    CONSTRAINT_START, 
    CONSTRAINT_END, 
    ANSWER_PREFIX,
)


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


def make_chatml_instruction(prompt_text: str) -> str:
    """
    Create a ChatML instruction for the model.
    """
    return f"""<|im_start|>system
You are a precise first-order logic parser. You MUST output FOL formulas in EXACTLY this format:

FOL Solution:
Predicates:
PredicateName(x) ::: description

Premises:
<<{{forall}} x (Predicate(x) {{implies}} Other(x))>> ::: description

Conclusion:
<<FOL_FORMULA>> ::: description

Answer: True/False/Uncertain

CRITICAL: Every FOL formula MUST be enclosed in << >> delimiters. Use {{forall}}, {{exists}}, {{and}}, {{or}}, {{not}}, {{implies}}, {{iff}}, {{xor}} for logical operators.
<|im_end|>
<|im_start|>user
{prompt_text}
<|im_end|>
<|im_start|>assistant
FOL Solution:
Predicates:
"""


def run_crane_csd(
    env: dict,
    prompt_text: str,
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
    dynamic_parser=None,
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    CRANE-style generation with CSD for constrained FOL windows.

    Architecture:
      1. Generate unconstrained until << detected
      2. Run CSD strategy for constrained FOL expression
      3. Validate after >> delimiter
      4. Resume unconstrained until Answer: or EOS

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file for validation
        debug_delimiters: Whether to print delimiter debug output
        dynamic_parser: Optional per-question parser with restricted predicates/constants.
                        If provided, used for CSD instead of env["parser"].
        
    Returns:
        Tuple of (output_text, token_count, time_seconds, constrained_segments)
        constrained_segments: List of (segment_text, is_valid) tuples
    """
    _dafny = env["_dafny"]
    VerifiedDecoderAgent = env["VerifiedDecoderAgent"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    # Use dynamic parser if provided, otherwise fall back to static parser
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]

    # For validating FOL segments
    from parsers.lark_parser import LarkGrammarParser
    fol_validator = LarkGrammarParser.from_grammar_file(str(grammar_file))

    # The instruction ends with primed text to guide the model into the correct format
    # The primed prefix is already in the instruction, not in result_tokens
    # We track it separately to include in the final output
    PRIMED_PREFIX = "FOL Solution:\nPredicates:\n"
    result_tokens: List[str] = []
    constrained_segments: List[Tuple[str, bool]] = []
    mode = "unconstrained"
    current_window: List[str] = []

    lm.instruction_text = make_chatml_instruction(prompt_text)
    start_time = time.time()

    step = 0
    consecutive_unconstrained_without_delimiter = 0
    max_unconstrained_before_force = 80  # More tokens for FOL reasoning
    
    # EOS handling
    eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
    min_tokens_before_eos = 30
    
    # Repetition detection
    recent_text_window = []
    
    # Track last delimiter position
    last_delimiter_step = -100
    delimiter_cooldown = 30  # Don't detect delimiter again within 30 steps
    
    if debug_delimiters:
        print(f"  [DEBUG] Starting FOLIO generation with max_steps={max_steps}")
    
    while step < max_steps:
        if mode == "unconstrained":
            # Generate one token freely (unconstrained) from full vocabulary
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            lm.GenerateLogits(dafny_prefix)
            token = lm.ChooseNextTokenUnconstrained()
            token_str = dafny_seq_to_str(token)
            result_tokens.append(token_str)
            step += 1
            consecutive_unconstrained_without_delimiter += 1
            
            should_stop_on_eos = token_str in eos_tokens and step >= min_tokens_before_eos

            # Detect repetitive loops
            recent_text_window.append(token_str)
            if len(recent_text_window) > 100:
                recent_text_window.pop(0)

            # Check for repetition patterns
            should_break_outer = False
            if step > 20:
                for pattern_len in [5, 10, 15, 20]:
                    if len(result_tokens) >= pattern_len * 3:
                        last_pattern = result_tokens[-pattern_len:]
                        prev_pattern = result_tokens[-pattern_len*2:-pattern_len]
                        prev_prev_pattern = result_tokens[-pattern_len*3:-pattern_len*2]
                        if last_pattern == prev_pattern == prev_prev_pattern:
                            if debug_delimiters:
                                pattern_text = "".join(last_pattern)
                                print(f"  [DEBUG] WARNING: Detected {pattern_len}-token repetitive loop at step {step}!")
                                print(f"  [DEBUG] Repetitive pattern: {repr(pattern_text[:60])}")
                            should_break_outer = True
                            break
            
            if should_break_outer:
                break

            # Check for start delimiter <<
            joined = "".join(result_tokens[-20:])
            
            has_double_lt = CONSTRAINT_START in joined
            recent_5 = result_tokens[-5:]
            has_split_lt = len(recent_5) >= 2 and recent_5[-1] == "<" and recent_5[-2] == "<"
            
            if debug_delimiters and step % 10 == 0:
                print(f"  [DEBUG step {step}] Recent tokens: {repr(recent_5)} | Has <<: {has_double_lt}")
            
            steps_since_last_delimiter = step - last_delimiter_step
            if (has_double_lt or has_split_lt) and steps_since_last_delimiter > delimiter_cooldown and mode == "unconstrained":
                if debug_delimiters:
                    print(f"  [DEBUG] Detected << delimiter at step {step}, switching to constrained mode")
                mode = "constrained"
                current_window = []
                consecutive_unconstrained_without_delimiter = 0
                last_delimiter_step = step

            # Check for Answer: termination (FOLIO uses Answer: instead of ####)
            if ANSWER_PREFIX in joined:
                if debug_delimiters:
                    print(f"  [DEBUG] Found Answer: at step {step}, finishing answer extraction")
                # Continue generating for a few more tokens to get the answer
                for _ in range(min(15, max_steps - step)):
                    dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
                    lm.GenerateLogits(dafny_prefix)
                    token = lm.ChooseNextTokenUnconstrained()
                    token_str = dafny_seq_to_str(token)
                    result_tokens.append(token_str)
                    step += 1
                    # Stop on newline or EOS after Answer:
                    if token_str in eos_tokens or token_str == "\n":
                        break
                break
            elif should_stop_on_eos:
                if debug_delimiters:
                    print(f"  [DEBUG] EOS token '{token_str}' detected at step {step}, stopping")
                break

        elif mode == "constrained":
            # Run CSD strategy for constrained FOL generation
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            remaining_steps = max(1, min(30, max_steps - step))  # Allow more for FOL

            csd_output = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, dafny_prefix, remaining_steps
            )
            csd_tokens = [dafny_seq_to_str(t) for t in csd_output]
            
            current_window.extend(csd_tokens)
            result_tokens.extend(csd_tokens)
            step += len(csd_tokens)

            # Validate the constrained segment
            segment_text = "".join(current_window).strip()
            is_valid = fol_validator.is_complete(segment_text) if segment_text else False
            constrained_segments.append((segment_text, is_valid))

            # Switch back to unconstrained mode
            mode = "unconstrained"
            current_window = []
            last_delimiter_step = step
            
            # Generate >> delimiter in unconstrained mode
            for _ in range(min(5, max_steps - step)):
                dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
                lm.GenerateLogits(dafny_prefix)
                token = lm.ChooseNextToken()
                token_str = dafny_seq_to_str(token)
                result_tokens.append(token_str)
                step += 1
                
                joined = "".join(result_tokens[-5:])
                if CONSTRAINT_END in joined:
                    break
                if ANSWER_PREFIX in joined:
                    break
                elif step >= 100 and token_str in eos_tokens:
                    break

    end_time = time.time()
    # Include the primed prefix in the output for a complete result
    output_text = PRIMED_PREFIX + "".join(result_tokens)

    if step >= max_steps:
        if debug_delimiters:
            print(f"  [DEBUG] WARNING: Hit max_steps limit ({max_steps})")

    return output_text, len(result_tokens), end_time - start_time, constrained_segments


def run_unconstrained(
    env: dict,
    prompt_text: str,
    max_steps: int,
    debug: bool = False,
) -> Tuple[str, int, float]:
    """
    Run unconstrained generation without CSD.
    
    This is useful for baseline comparison.
    
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
    
    for step in range(max_steps):
        dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
        lm.GenerateLogits(dafny_prefix)
        token = lm.ChooseNextTokenUnconstrained()
        token_str = dafny_seq_to_str(token)
        result_tokens.append(token_str)
        
        # Check for termination
        if token_str in eos_tokens and step > 30:
            break
        
        # Check for Answer: followed by answer content
        joined = "".join(result_tokens[-30:])
        if ANSWER_PREFIX in joined:
            # Continue to get the answer
            for _ in range(min(15, max_steps - step - 1)):
                dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
                lm.GenerateLogits(dafny_prefix)
                token = lm.ChooseNextTokenUnconstrained()
                token_str = dafny_seq_to_str(token)
                result_tokens.append(token_str)
                if token_str in eos_tokens or token_str == "\n":
                    break
            break
    
    end_time = time.time()
    output_text = "".join(result_tokens)
    
    return output_text, len(result_tokens), end_time - start_time
