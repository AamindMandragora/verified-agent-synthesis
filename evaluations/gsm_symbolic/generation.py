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
    CRANE-style generation with CSD for constrained windows.

    Architecture:
      1. Generate unconstrained until << detected
      2. Run CSD strategy for constrained math expression
      3. Validate after >> delimiter
      4. Resume unconstrained until #### or EOS

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file for validation
        debug_delimiters: Whether to print delimiter debug output
        dynamic_parser: Optional per-question parser with restricted variables.
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

    # For validating math segments
    # Use 'any_expr' start rule to validate expressions (allows both symbolic and numeric)
    # The default 'start' rule requires s_expr (must contain variables), but generated
    # expressions may be pure numeric in some cases
    from parsers.lark_parser import LarkGrammarParser
    math_validator = LarkGrammarParser.from_grammar_file(str(grammar_file), start='any_expr')

    result_tokens: List[str] = []
    constrained_segments: List[Tuple[str, bool]] = []
    mode = "unconstrained"
    current_window: List[str] = []

    lm.instruction_text = make_chatml_instruction(prompt_text)
    start_time = time.time()

    step = 0
    consecutive_unconstrained_without_delimiter = 0
    max_unconstrained_before_force = 50  # Force CSD after 50 tokens if no << found
    
    # EOS handling
    eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
    min_tokens_before_eos = 20
    
    # Repetition detection
    recent_text_window = []  # Track recent text to detect loops
    
    # Track last delimiter position to avoid re-detecting the same delimiter
    last_delimiter_step = -100
    delimiter_cooldown = 25  # Don't detect delimiter again within 25 steps of last detection
    
    if debug_delimiters:
        print(f"  [DEBUG] Starting generation with max_steps={max_steps}")
    
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

            # Detect repetitive loops (model stuck generating same pattern)
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
            
            has_double_lt = "<<" in joined
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

            # Check for #### termination
            if "####" in joined:
                if debug_delimiters:
                    print(f"  [DEBUG] Found #### at step {step}, stopping")
                break
            elif should_stop_on_eos:
                if debug_delimiters:
                    print(f"  [DEBUG] EOS token '{token_str}' detected at step {step}, stopping")
                break

        elif mode == "constrained":
            # Run CSD strategy to generate a complete expression.
            #
            # CRITICAL FIX: Call MyCSDStrategy ONCE with a large batch size.
            # The Dafny MyCSDStrategy starts with `generated = []` each call.
            # If we call it multiple times with small batches, each call treats
            # the expression as starting fresh, breaking parser constraint accumulation.
            # For example:
            #   - First call (batch=1): generates 'f' (valid at START)
            #   - Second call (batch=1): generates '1' (ALSO valid at START!)
            #   - Result: 'f 1' which violates grammar constraints
            #
            # By calling ONCE with full max, the parser correctly constrains:
            #   - After 'f', only operators/>> are allowed, not numbers
            #
            # The MyCSDStrategy stops early when IsCompletePrefix returns True
            # (when >> is generated), so it won't waste steps.
            max_constrained_tokens = 50
            total_constrained_tokens = 0
            found_closing_delimiter = False
            llm_degenerate = False  # LLM producing pathological output (not CSD failure)

            # Run CSD strategy ONCE with the full max_constrained_tokens
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            remaining_steps = min(max_constrained_tokens, max_steps - step)

            if debug_delimiters:
                print(f"  [DEBUG] Running CSD with max_steps={remaining_steps}")

            csd_output = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, dafny_prefix, remaining_steps
            )
            csd_tokens = [dafny_seq_to_str(t) for t in csd_output]

            if not csd_tokens:
                # CSD returned nothing - no valid grammar continuations exist
                if debug_delimiters:
                    print(f"  [DEBUG] CSD returned no tokens at step {step} (no valid continuations)")
                llm_degenerate = True
            else:
                # Add CSD tokens to results
                current_window.extend(csd_tokens)
                result_tokens.extend(csd_tokens)
                step += len(csd_tokens)
                total_constrained_tokens = len(csd_tokens)

                # Check if we've generated >> (end of expression)
                window_so_far = "".join(current_window)
                if ">>" in window_so_far:
                    found_closing_delimiter = True
                    if debug_delimiters:
                        print(f"  [DEBUG] Found >> in CSD output at step {step}")

                # POST-GENERATION DEGENERACY DETECTION
                # Now that parser constraints are properly maintained, many of these
                # checks should not trigger (e.g., "a b" is blocked by parser).
                # These remain as safety checks for edge cases.
                open_parens = window_so_far.count('(')
                close_parens = window_so_far.count(')')
                op_count = len(re.findall(r'[+\-*/]', window_so_far))

                # Check 1: Nested open parens
                if re.search(r'\(\s*\(', window_so_far):
                    if debug_delimiters:
                        print(f"  [DEBUG] LLM degenerate: nested open parens in '{window_so_far[:30]}'")
                    llm_degenerate = True

                # Check 2: Paren imbalance
                elif open_parens > close_parens + 2:
                    if debug_delimiters:
                        print(f"  [DEBUG] LLM degenerate: {open_parens} open vs {close_parens} close parens")
                    llm_degenerate = True

                # Check 3: Repeated operators
                elif re.search(r'[+\-*/]\s*[+\-*/]', window_so_far):
                    if debug_delimiters:
                        print(f"  [DEBUG] LLM degenerate: repeated operators detected")
                    llm_degenerate = True

                # Check 4: Variables without operators (should be blocked by parser now)
                elif re.search(r'[a-zA-Z]\s+[a-zA-Z]', window_so_far):
                    if debug_delimiters:
                        print(f"  [DEBUG] LLM degenerate: adjacent variables in '{window_so_far[:40]}'")
                    llm_degenerate = True

                # Check 5: Adjacent multi-char letters (except 'int')
                else:
                    adjacent_letters = re.search(r'[a-zA-Z]{2,}', window_so_far)
                    if adjacent_letters and adjacent_letters.group() != 'int':
                        if debug_delimiters:
                            print(f"  [DEBUG] LLM degenerate: multi-char var '{adjacent_letters.group()}' in '{window_so_far[:40]}'")
                        llm_degenerate = True

                if debug_delimiters:
                    print(f"  [DEBUG CSD] Generated {total_constrained_tokens} tokens: {repr(window_so_far[:60])}")

            # Validate the constrained segment (strip >> from validation)
            segment_text = "".join(current_window).strip()
            expr_for_validation = segment_text.replace(">>", "").strip()

            # Check parenthesis balance
            open_count = expr_for_validation.count('(')
            close_count = expr_for_validation.count(')')
            parens_balanced = (open_count == close_count)

            # Only valid if grammar passes AND parens are balanced
            is_valid = False
            if expr_for_validation and parens_balanced:
                is_valid = math_validator.is_complete(expr_for_validation)

            # Only record non-empty, non-garbage, balanced segments
            if expr_for_validation and not llm_degenerate and parens_balanced:
                constrained_segments.append((expr_for_validation, is_valid))

            if debug_delimiters:
                print(f"  [DEBUG] Constrained segment ({total_constrained_tokens} tokens): valid={is_valid}, llm_degenerate={llm_degenerate}, balanced={parens_balanced}, expr={repr(expr_for_validation[:60])}")

            # If CSD didn't generate >>, force-close the expression
            if not found_closing_delimiter:
                result_tokens.append(">>")
                step += 1
                if debug_delimiters:
                    print(f"  [DEBUG] Force-closing with >>")

            # Switch back to unconstrained mode
            mode = "unconstrained"
            current_window = []
            last_delimiter_step = step

    end_time = time.time()
    output_text = "".join(result_tokens)
    
    if step >= max_steps:
        if debug_delimiters:
            print(f"  [DEBUG] WARNING: Hit max_steps limit ({max_steps})")
    
    return output_text, len(result_tokens), end_time - start_time, constrained_segments
