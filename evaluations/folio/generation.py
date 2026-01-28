"""
Generation methods for FOLIO evaluation.

Provides CSD (Constrained Decoding Strategy) generation:
- CRANE-CSD: CRANE-style with CSD strategy for constrained FOL windows
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

from evaluations.folio.prompts import ANSWER_PREFIX


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
    Create a ChatML instruction for the model matching CRANE paper format.

    The CRANE format includes grammar specification and worked examples to teach
    the model the exact FOL syntax expected.
    """
    return f"""<|im_start|>system
Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulas.
The grammar of the first-order logic formula is defined as follows:
1) logical conjunction of expr1 and expr2: expr1 {{and}} expr2
2) logical disjunction of expr1 and expr2: expr1 {{or}} expr2
3) logical exclusive disjunction of expr1 and expr2: expr1 {{xor}} expr2
4) logical negation of expr1: {{not}}expr1
5) expr1 implies expr2: expr1 {{implies}} expr2
6) expr1 if and only if expr2: expr1 {{iff}} expr2
7) logical universal quantification: {{forall}} x
8) logical existential quantification: {{exists}} x
These are the ONLY operators in the grammar.
------

Answer the question EXACTLY like the examples.

Problem:
All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
Question:
Based on the above information, is the following statement true, false, or uncertain? Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
###

We take three steps: first, we define the necessary predicates and premises, and finally, we encode the question 'Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.' in the conclusion. Now, we will write only the logic program, nothing else.
Predicates:
Dependent(x) ::: x is a person dependent on caffeine.
Drinks(x) ::: x regularly drinks coffee.
Jokes(x) ::: x jokes about being addicted to caffeine.
Unaware(x) ::: x is unaware that caffeine is a drug.
Student(x) ::: x is a student.
Premises:
{{forall}} x (Drinks(x) {{implies}} Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
{{forall}} x (Drinks(x) {{xor}} Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
{{forall}} x (Jokes(x) {{implies}} {{not}}Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.
(Student(rina) {{and}} Unaware(rina)) {{xor}} {{not}}(Student(rina) {{or}} Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.
Conclusion:
Jokes(rina) {{xor}} Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
------

Problem:
Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
Question:
Based on the above information, is the following statement true, false, or uncertain? Miroslav Venhoda loved music.
###

We take three steps: first, we define the necessary predicates and premises, and finally, we encode the question 'Miroslav Venhoda loved music.' in the conclusion. Now, we will write only the logic program, nothing else.
Predicates:
Czech(x) ::: x is a Czech person.
ChoralConductor(x) ::: x is a choral conductor.
Musician(x) ::: x is a musician.
Love(x, y) ::: x loves y.
Author(x, y) ::: x is the author of y.
Book(x) ::: x is a book.
Publish(x, y) ::: x is published in year y.
Specialize(x, y) ::: x specializes in y.
Premises:
Czech(miroslav) {{and}} ChoralConductor(miroslav) {{and}} Specialize(miroslav, renaissance) {{and}} Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
{{forall}} x (ChoralConductor(x) {{implies}} Musician(x)) ::: Any choral conductor is a musician.
{{exists}} x (Musician(x) {{and}} Love(x, music)) ::: Some musicians love music.
Book(methodOfStudyingGregorianChant) {{and}} Author(miroslav, methodOfStudyingGregorianChant) {{and}} Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.
Conclusion:
Love(miroslav, music) ::: Miroslav Venhoda loved music.
------
<|im_end|>
<|im_start|>user
Problem:
{prompt_text}
###
<|im_end|>
<|im_start|>assistant
We take three steps: first, we define the necessary predicates and premises, and finally, we encode the question in the conclusion. Now, we will write only the logic program, nothing else.
Predicates:
"""


def run_crane_csd(
    env: dict,
    prompt_text: str,
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
    dynamic_parser=None,
    debug_csd: bool = False,
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
    # CRANE format: starts with "We take three steps..." then "Predicates:"
    PRIMED_PREFIX = "We take three steps: first, we define the necessary predicates and premises, and finally, we encode the question in the conclusion. Now, we will write only the logic program, nothing else.\nPredicates:\n"
    result_tokens: List[str] = []
    constrained_segments: List[Tuple[str, bool]] = []
    mode = "unconstrained"
    current_window: List[str] = []

    # Track which section we're in for context-based CSD triggering
    # Sections: "predicates", "premises", "conclusion", "answer"
    current_section = "predicates"

    lm.instruction_text = make_chatml_instruction(prompt_text)
    start_time = time.time()

    step = 0
    consecutive_fol_tokens = 0  # Track tokens since last FOL region start
    max_fol_tokens = 50  # Max tokens for a single FOL formula

    # EOS handling
    eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
    min_tokens_before_eos = 30

    # Repetition detection
    recent_text_window = []

    # Track last FOL region
    last_fol_end_step = -100
    fol_cooldown = 5  # Min tokens between FOL regions

    if debug_delimiters:
        print(f"  [DEBUG] Starting FOLIO generation with max_steps={max_steps}")
    
    # Section markers for context-based FOL detection (CRANE format)
    PREMISES_MARKER = "Premises:"
    CONCLUSION_MARKER = "Conclusion:"
    FOL_END_MARKER = ":::"  # FOL formulas end when ::: is reached

    while step < max_steps:
        # Get current text for context detection
        joined = "".join(result_tokens[-50:]) if result_tokens else ""
        full_text = "".join(result_tokens)

        # Detect section transitions
        if PREMISES_MARKER in joined and current_section == "predicates":
            current_section = "premises"
            if debug_delimiters:
                print(f"  [DEBUG] Entered Premises section at step {step}")
            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:section_premises", "message": "Entered Premises section", "data": {"step": step, "recent_text": joined[-100:]}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1"}) + '\n')
            # #endregion
        elif CONCLUSION_MARKER in joined and current_section == "premises":
            current_section = "conclusion"
            if debug_delimiters:
                print(f"  [DEBUG] Entered Conclusion section at step {step}")
            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:section_conclusion", "message": "Entered Conclusion section", "data": {"step": step, "recent_text": joined[-100:]}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1"}) + '\n')
            # #endregion

        if mode == "unconstrained":
            # Generate one token freely (unconstrained) from full vocabulary
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            lm.GenerateLogits(dafny_prefix)
            token = lm.ChooseNextTokenUnconstrained()
            token_str = dafny_seq_to_str(token)
            result_tokens.append(token_str)
            step += 1

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

            # Context-based FOL region detection (CRANE format)
            # In Premises/Conclusion sections, detect ACTUAL FOL-starting patterns
            # FOL formula ends when ::: is reached
            steps_since_fol_end = step - last_fol_end_step

            # Detect FOL start: Only trigger on UNAMBIGUOUS FOL patterns
            # Don't trigger on single uppercase letters (could be English words)
            should_start_fol = False
            if current_section in ["premises", "conclusion"] and steps_since_fol_end > fol_cooldown:
                # Check text BEFORE current token to see if we're at line start
                text_before = "".join(result_tokens[-16:-1]) if len(result_tokens) > 1 else ""
                last_token = token_str.strip() if token_str else ""

                # STRICT FOL starters - only unambiguous patterns:
                # 1. '{' - FOL operator start (will become {forall}, {exists}, etc.)
                # 2. '(' at line start - parenthesized formula
                # 3. 'PredicateName(' pattern - predicate with open paren
                # Do NOT trigger on bare uppercase letters - too ambiguous!
                is_fol_starter = False
                if last_token.startswith('{'):
                    # FOL operator like {forall}, {exists}, {not}
                    is_fol_starter = True
                elif last_token == '(':
                    # Parenthesized formula at line start
                    is_fol_starter = True
                elif last_token.endswith('(') and len(last_token) > 1 and last_token[0].isupper():
                    # Predicate call like "Cat(" or "Performs("
                    is_fol_starter = True

                # Check if we're at a line start (position where FOL could begin)
                # CRITICAL: Token is at line start ONLY if immediately preceded by newline
                # The previous buggy check looked for ANY newline after :::, not immediate
                
                # Token is at line start only if:
                # 1. Nothing before it (very start)
                # 2. Immediately preceded by newline (text_before ends with \n or whitespace-only after \n)
                # 3. Immediately preceded by section marker
                
                # Check if text_before ends with newline (possibly followed by spaces/tabs only)
                text_after_last_newline = text_before.split('\n')[-1] if '\n' in text_before else text_before
                ends_with_newline = (text_after_last_newline.strip() == "") if text_before else False
                
                at_line_start = (
                    not text_before or  # Very start
                    ends_with_newline or  # Directly after newline (with optional whitespace)
                    text_before.rstrip().endswith(PREMISES_MARKER) or
                    text_before.rstrip().endswith(CONCLUSION_MARKER)
                )

                # Trigger FOL mode only if BOTH conditions met
                if at_line_start and is_fol_starter:
                    should_start_fol = True
                    # Include the current FOL-starting token in the window
                    current_window = [token_str]
                    # #region agent log
                    import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:fol_start_trigger", "message": "FOL region triggered", "data": {"step": step, "last_token": last_token, "text_before_end": text_before[-50:] if text_before else "", "text_after_last_newline": text_after_last_newline[:30] if text_after_last_newline else "", "ends_with_newline": ends_with_newline, "section": current_section, "at_line_start": at_line_start, "is_fol_starter": is_fol_starter}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1"}) + '\n')
                    # #endregion
                elif is_fol_starter and not at_line_start:
                    # #region agent log
                    import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:fol_not_at_line_start", "message": "FOL starter found but NOT at line start - correctly skipped", "data": {"step": step, "last_token": last_token, "text_before_end": text_before[-30:] if text_before else "", "text_after_last_newline": text_after_last_newline[:20] if text_after_last_newline else "", "ends_with_newline": ends_with_newline}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1_FIX"}) + '\n')
                    # #endregion

            if should_start_fol:
                if debug_delimiters:
                    print(f"  [DEBUG] Detected FOL start context at step {step}, section={current_section}")
                mode = "constrained"
                # CRITICAL: Include the trigger token in current_window so CSD continues from it
                # (don't overwrite the current_window = [token_str] set above)
                # current_window already contains [token_str] from line 297
                consecutive_fol_tokens = len(current_window)  # Count the trigger token

            if debug_delimiters and step % 20 == 0:
                print(f"  [DEBUG step {step}] Section: {current_section}, Recent: {repr(joined[-30:])}")

            # Check for Answer: termination
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
                    if token_str in eos_tokens or token_str == "\n":
                        break
                break
            elif should_stop_on_eos:
                if debug_delimiters:
                    print(f"  [DEBUG] EOS token '{token_str}' detected at step {step}, stopping")
                break

        elif mode == "constrained":
            # Run CSD strategy for constrained FOL generation
            # Pass full result_tokens for LM context, but track FOL window start
            # The parser will extract and validate only the FOL portion
            fol_window_start = len(result_tokens) - len(current_window)
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            remaining_steps = max(1, min(max_fol_tokens - consecutive_fol_tokens, max_steps - step))

            # NOTE: _fol_window_start is no longer used - CSD passes only 'generated' tokens to parser

            if debug_csd:
                print(f"  [CSD DEBUG] Starting constrained generation, remaining_steps={remaining_steps}")
                print(f"  [CSD DEBUG] FOL window start: {fol_window_start}, current window: {''.join(current_window)}")

            # Convert current_window tokens to Dafny sequence for partial prefix
            # This ensures CSD continues from the trigger token (e.g., '{')
            partial_prefix = _dafny.SeqWithoutIsStrInference(
                [_dafny.Seq(tok) for tok in current_window]
            ) if current_window else _dafny.SeqWithoutIsStrInference([])

            # Use CompletePrefix to continue from the trigger token(s)
            csd_output = VerifiedDecoderAgent.CSDHelpers.CompletePrefix(
                lm, parser, dafny_prefix, partial_prefix, remaining_steps
            )
            try:
                csd_tokens = [dafny_seq_to_str(t) for t in csd_output]
            except TypeError:
                csd_tokens = [dafny_seq_to_str(csd_output[i]) for i in range(len(csd_output))]

            # CSD output includes the partial prefix tokens - skip those to avoid duplicates
            # current_window already contains the trigger token(s)
            num_initial = len(current_window)
            new_tokens = csd_tokens[num_initial:] if len(csd_tokens) > num_initial else []

            # CRITICAL: Filter out special tokens and CSD delimiters from new_tokens
            # These should never appear in FOL output
            SPECIAL_TOKENS_TO_FILTER = {'>>', '<<', '[BEGIN_OF_TEXT]', '[END_OF_TEXT]',
                '<|begin_of_text|>', '<|end_of_text|>', '<|im_start|>', '<|im_end|>',
                '<EOS>', '</s>', '<s>'}
            filtered_tokens = []

            # First pass: normalize whitespace in tokens
            # The tokenizer may produce tokens like '}\n\n' - normalize to '} '
            normalized_tokens = []
            for tok in new_tokens:
                # Normalize excessive newlines to single space (keeps formatting cleaner)
                if '\n\n' in tok:
                    tok = tok.replace('\n\n', ' ')
                if '\n' in tok and not tok.strip():
                    # Pure whitespace token with newlines -> single space
                    tok = ' '
                normalized_tokens.append(tok)
            new_tokens = normalized_tokens

            for tok in new_tokens:
                # Skip if token is exactly a special token
                if tok.strip() in SPECIAL_TOKENS_TO_FILTER:
                    continue
                # Skip if token contains >> (CSD end delimiter)
                if '>>' in tok:
                    # Try to extract the part before >>
                    before_delim = tok.split('>>')[0]
                    if before_delim.strip():
                        filtered_tokens.append(before_delim)
                    continue
                # Skip tokens containing special control tokens
                has_special = any(st in tok for st in SPECIAL_TOKENS_TO_FILTER if len(st) > 2)
                if has_special:
                    continue
                filtered_tokens.append(tok)
            new_tokens = filtered_tokens

            if debug_csd:
                print(f"  [CSD DEBUG] CSD generated {len(csd_tokens)} tokens (skipping first {num_initial}): {new_tokens[:10]}{'...' if len(new_tokens) > 10 else ''}")

            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:csd_output", "message": "CSD constrained generation output", "data": {"step": step, "csd_tokens_raw": [str(t)[:30] for t in csd_tokens[:15]], "new_tokens": [str(t)[:30] for t in new_tokens[:15]], "current_window_before": "".join(current_window)[:50], "num_initial": num_initial}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H2"}) + '\n')
            # #endregion

            # If no new tokens were generated, CSD is stuck - force exit constrained mode
            if not new_tokens:
                if debug_delimiters:
                    print(f"  [DEBUG] CSD returned no new tokens at step {step}, forcing exit from constrained mode")
                # #region agent log
                import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:csd_no_tokens", "message": "CSD returned no new tokens - falling back to unconstrained", "data": {"step": step, "current_window": current_window, "csd_tokens_raw": csd_tokens[:10] if csd_tokens else [], "num_initial": num_initial, "partial_prefix_len": len(current_window)}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1"}) + '\n')
                # #endregion
                mode = "unconstrained"
                current_window = []
                last_fol_end_step = step
                consecutive_fol_tokens = 0
                continue

            current_window.extend(new_tokens)
            result_tokens.extend(new_tokens)
            step += len(new_tokens)
            consecutive_fol_tokens += len(new_tokens)

            # Validate the constrained segment
            segment_text = "".join(current_window).strip()
            is_valid = fol_validator.is_complete(segment_text) if segment_text else False
            constrained_segments.append((segment_text, is_valid))
            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:segment_validation", "message": "FOL segment validated", "data": {"segment_text": segment_text[:100], "is_valid": is_valid, "window_tokens": [str(t)[:20] for t in current_window[:10]]}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H4"}) + '\n')
            # #endregion

            # Check if FOL region should end (reached ::: or max tokens)
            # NOTE: We don't exit on grammar_complete because the model might want to
            # continue with {and}, {or}, etc. to build a larger formula
            recent_csd_text = "".join(csd_tokens)
            
            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:fol_region_check", "message": "FOL region completion check", "data": {"segment_text": segment_text[:50], "is_valid": is_valid, "has_end_marker": FOL_END_MARKER in recent_csd_text, "consecutive_fol_tokens": consecutive_fol_tokens}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H4"}) + '\n')
            # #endregion
            
            if FOL_END_MARKER in recent_csd_text or consecutive_fol_tokens >= max_fol_tokens:
                reason = ":::" if FOL_END_MARKER in recent_csd_text else "max_tokens"
                if debug_delimiters:
                    print(f"  [DEBUG] FOL region ended at step {step}, tokens={consecutive_fol_tokens}, reason={reason}")
                # #region agent log
                import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:fol_region_end", "message": "FOL region ended", "data": {"step": step, "reason": reason, "final_segment": "".join(current_window)[:80], "recent_csd_text": recent_csd_text[:60]}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3"}) + '\n')
                # #endregion
                mode = "unconstrained"
                current_window = []
                last_fol_end_step = step
                consecutive_fol_tokens = 0
            else:
                # Continue in constrained mode - update FOL window start
                # The window start should remain at the same position
                pass

    end_time = time.time()
    # Include the primed prefix in the output for a complete result
    output_text = PRIMED_PREFIX + "".join(result_tokens)

    if step >= max_steps:
        if debug_delimiters:
            print(f"  [DEBUG] WARNING: Hit max_steps limit ({max_steps})")

    # #region agent log
    import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:crane_csd_complete", "message": "CRANE-CSD generation complete", "data": {"total_tokens": len(result_tokens), "output_preview": output_text[:800], "num_constrained_segments": len(constrained_segments), "constrained_segments_preview": [(s[:60], v) for s, v in constrained_segments[:5]]}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H_FULL"}) + '\n')
    # #endregion

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
    
    # #region agent log
    import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "generation.py:generation_complete", "message": "Generation complete", "data": {"total_tokens": len(result_tokens), "output_preview": output_text[:500], "fol_segments_found": len([s for s in output_text.split('{') if 'forall}' in s or 'exists}' in s or 'not}' in s or 'and}' in s or 'or}' in s])}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H2"}) + '\n')
    # #endregion
    
    return output_text, len(result_tokens), end_time - start_time
