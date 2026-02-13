"""
Generation methods for evaluation.

Provides CSD (Constrained Decoding Strategy) generation by delegating
entirely to the Dafny-verified CSD strategy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple


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
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    Run generation using the Dafny-verified CSD strategy.

    Delegates entirely to the compiled Dafny strategy — no dataset-specific
    orchestration is performed here.

    Args:
        env: Environment dict with Dafny modules and model
        prompt_text: The prompt text (set as lm.instruction_text)
        max_steps: Maximum generation steps
        grammar_file: Path to grammar file (unused, kept for interface compat)
        debug_delimiters: Whether to print debug output
        dynamic_parser: Optional per-question parser
        debug_csd: Whether to print CSD debug output

    Returns:
        Tuple of (output_text, token_count, time_seconds, constrained_segments)
    """
    _dafny = env["_dafny"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    parser = dynamic_parser if dynamic_parser is not None else env["parser"]

    lm.instruction_text = prompt_text
    start_time = time.time()

    eos_token_str = lm.tokenizer.eos_token or "<|endoftext|>"
    eos_token_dafny = _dafny.Seq(eos_token_str)

    result = GeneratedCSD.default__.MyCSDStrategy(
        lm, parser, _dafny.SeqWithoutIsStrInference([]), max_steps, eos_token_dafny
    )

    if isinstance(result, tuple):
        csd_output, total_cost = result
    else:
        csd_output = result
        total_cost = 0

    result_tokens = [dafny_seq_to_str(t) for t in csd_output]
    output_text = "".join(result_tokens)
    execution_time = time.time() - start_time

    constrained_segments: List[Tuple[str, bool]] = []

    if debug_delimiters or debug_csd:
        print(f"  [DEBUG] Generation finished in {execution_time:.2f}s. Cost: {total_cost}. Tokens: {len(result_tokens)}")

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
