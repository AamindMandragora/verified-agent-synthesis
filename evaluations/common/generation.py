"""
Shared generation methods for CSD evaluation.

Provides dafny_seq_to_str, run_crane_csd, and run_unconstrained used by
both gsm_symbolic and folio evaluations. Delegates to the Dafny-verified
CSD strategy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple


def dafny_seq_to_str(seq) -> str:
    """
    Convert a Dafny Seq to a Python string.

    Dafny.Seq objects have __len__ and __getitem__ but NOT __iter__,
    so ''.join(seq) fails. We must use index-based iteration.
    """
    try:
        return ''.join(seq)
    except TypeError:
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
    debug_csd: bool = False,
    initial_prefix: Optional[List[str]] = None,
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    Run generation using the Dafny-verified CSD strategy.

    When initial_prefix is provided (e.g. tokenized gold conclusion), runs the same
    constrained loop in Python starting from that prefix so the CSD "reproduces" the
    prompt content and can then continue to a solution.

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text (set as lm.instruction_text)
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file for post-hoc segment validation (or interface compat)
        debug_delimiters: Whether to print debug output
        dynamic_parser: Optional per-question parser
        debug_csd: Whether to print CSD debug output
        initial_prefix: Optional list of token strings to start generation with (CSD then continues from there)

    Returns:
        Tuple of (output_text, token_count, time_seconds, constrained_segments)
    """
    _dafny = env["_dafny"]
    GeneratedCSD = env["GeneratedCSD"]
    VerifiedDecoderAgent = env["VerifiedDecoderAgent"]
    lm = env["lm"]
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]

    lm.instruction_text = prompt_text
    start_time = time.time()

    eos_token_str = lm.tokenizer.eos_token or "<|endoftext|>"
    eos_token_dafny = _dafny.Seq(eos_token_str)
    prompt_empty = _dafny.SeqWithoutIsStrInference([])

    if initial_prefix is not None and len(initial_prefix) > 0:
        # Run the same constrained loop in Python starting from initial_prefix (e.g. tokenized gold conclusion).
        # This lets the CSD "reproduce" the info from the prompt and then continue.
        generated = _dafny.SeqWithoutIsStrInference([
            _dafny.SeqWithoutIsStrInference(s) for s in initial_prefix
        ])
        steps_left = max_steps
        helpers = VerifiedDecoderAgent.CSDHelpers()
        helpers.ctor__()
        while steps_left > 0 and not parser.IsCompletePrefix(generated):
            next_tok, steps_left = helpers.ConstrainedStep(lm, parser, prompt_empty, generated, steps_left)
            generated = generated + _dafny.SeqWithoutIsStrInference([next_tok])
        csd_output = generated
        remaining_steps = steps_left
    else:
        result = GeneratedCSD.default__.MyCSDStrategy(
            lm, parser, prompt_empty, max_steps, eos_token_dafny
        )
        if isinstance(result, tuple):
            csd_output, remaining_steps = result
        else:
            csd_output = result
            remaining_steps = 0

    result_tokens = [dafny_seq_to_str(t) for t in csd_output]
    output_text = "".join(result_tokens)
    execution_time = time.time() - start_time

    constrained_segments: List[Tuple[str, bool]] = []

    if debug_delimiters or debug_csd:
        steps_used = max_steps - remaining_steps
        print(f"  [DEBUG] Generation finished in {execution_time:.2f}s. Steps used: {steps_used}. Tokens: {len(result_tokens)}")

    return output_text, len(result_tokens), execution_time, constrained_segments


def run_unconstrained(
    env: dict,
    prompt_text: str,
    max_steps: int,
    debug: bool = False,
) -> Tuple[str, int, float]:
    """
    Run unconstrained generation without CSD (baseline).

    Generates tokens freely until EOS or max_steps.

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text (set as lm.instruction_text)
        max_steps: Maximum generation steps
        debug: Whether to print debug output

    Returns:
        Tuple of (output_text, token_count, time_seconds)
    """
    _dafny = env["_dafny"]
    lm = env["lm"]

    result_tokens: List[str] = []

    lm.instruction_text = prompt_text
    start_time = time.time()

    eos_token_str = lm.tokenizer.eos_token or "<|endoftext|>"

    for step in range(max_steps):
        dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
        lm.GenerateLogits(dafny_prefix)
        token = lm.ChooseNextTokenUnconstrained()
        token_str = dafny_seq_to_str(token)
        result_tokens.append(token_str)

        if token_str == eos_token_str:
            break

    end_time = time.time()
    output_text = "".join(result_tokens)

    return output_text, len(result_tokens), end_time - start_time
