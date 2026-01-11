#!/usr/bin/env python3
"""
GSM-Symbolic Math Reasoning Evaluation with CRANE-Style CSD.

Dataset: apple/GSM-Symbolic (HuggingFace)
Metrics:
  - Answer accuracy (exact numeric match)
  - Syntax validity (math expressions inside << >> pass grammar validation)
  - Valid format rate (outputs contain #### <number>)

Architecture: Evaluation-level orchestration of CRANE-style windowing with CSD
  - Unconstrained reasoning until << detected
  - Run CSD strategy for constrained math expression
  - Validate after >> delimiter
  - Resume unconstrained until #### or EOS

Methods:
  - standard: unconstrained HF generate (baseline, no CRANE)
  - crane: CRANE-style windowing with grammar constraint per-token (no CSD)
  - crane-csd: CRANE-style with CSD strategy for constrained windows + validation

Example:
  python scripts/evaluate_gsm_symbolic.py --method crane-csd \
    --run-dir outputs/generated-csd/runs/20260109_XXXXXX \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --device cuda --limit 50
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_csd_with_grammar import create_lark_dafny_parser
from scripts.evaluate_csd_performance import create_huggingface_lm


# =============================================================================
# Dataset Loading
# =============================================================================

def _load_gsm_symbolic(config: str = "main", split: str = "test", limit: Optional[int] = None):
    """
    Load GSM-Symbolic dataset from HuggingFace.

    Args:
        config: Dataset configuration - "main", "p1", or "p2"
        split: Dataset split (usually "test")
        limit: Optional limit on number of examples to load (for efficiency)

    Returns:
        HuggingFace dataset object (limited if limit is specified)
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    valid_configs = ["main", "p1", "p2"]
    if config not in valid_configs:
        raise ValueError(f"Config must be one of {valid_configs}, got: {config}")

    limit_str = f" (limit={limit})" if limit else ""
    print(f"Loading GSM-Symbolic dataset (config={config}, split={split}{limit_str})...")
    try:
        ds = load_dataset("apple/GSM-Symbolic", name=config, split=split)
    except Exception as e:
        # Some datasets only have certain splits
        print(f"Failed to load split '{split}', trying 'test'...")
        try:
            ds = load_dataset("apple/GSM-Symbolic", name=config, split="test")
        except Exception:
            print(f"Failed to load 'test', trying 'train'...")
            ds = load_dataset("apple/GSM-Symbolic", name=config, split="train")

    # Apply limit if specified
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    print(f"Loaded {len(ds)} examples")
    return ds


# =============================================================================
# Prompt Formatting
# =============================================================================

def _make_gsm_prompt(question: str) -> str:
    """
    Format GSM question as CRANE-style prompt.
    Instructs the model to use << >> delimiters for calculations.
    
    IMPORTANT: The prompt must be explicit and include a concrete example showing:
    - How to structure the solution (reasoning + calculations)
    - How to use << >> delimiters for math expressions
    - How to end with #### followed by the final answer
    
    Without a clear example, models may generate repetitive or nonsensical text
    instead of following the expected format.
    """
    return (
        "Solve the math problem step by step. Use << >> for each calculation. End with #### and the final number.\n\n"
        "IMPORTANT: To find what percentage X is of Y, calculate: (X / Y) * 100\n\n"
        "Example:\n"
        "Question: A 100-foot whale has 5 remoras attached to it. Each remora is 24 inches long. What percentage of the whale's body length is covered by remoras?\n"
        "Solution:\n"
        "Step 1 - Convert whale length to inches: << 100 * 12 >> = 1200 inches\n"
        "Step 2 - Total remora length (5 remoras × 24 inches each): << 5 * 24 >> = 120 inches\n"
        "Step 3 - Percentage = (remora length / whale length) × 100: << 120 / 1200 * 100 >> = 10\n"
        "#### 10\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


def _make_chatml_instruction(prompt_text: str) -> str:
    """Format as ChatML instruction for Qwen models."""
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"


# =============================================================================
# Answer Extraction and Evaluation
# =============================================================================

def _extract_answer(text: str) -> Tuple[Optional[float], bool]:
    """
    Extract numeric answer from generated text.
    Uses multiple fallback strategies since models may not always use #### format.

    Returns:
        (answer_value, is_valid_format)
        - answer_value: The extracted numeric answer, or None if not found
        - is_valid_format: True if output contains #### <number> format OR has calculations
    """
    # Strategy 1: Look for #### followed by a number (preferred format)
    has_delimiter = "####" in text
    patterns = [
        # Pattern 1: #### directly followed by number (most common)
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # Pattern 2: #### on one line, number on next line
        r'####\s*\n\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # Pattern 3: #### with some text before number (e.g., "#### The answer is 5")
        r'####[^\d]*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        # Pattern 4: Number BEFORE #### (model might put answer before delimiter)
        r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*####',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            try:
                num_str = match.group(1).replace(',', '').strip()
                value = float(num_str)
                return value, True  # Found with #### format
            except ValueError:
                continue
    
    # Strategy 2: Extract from last calculation result in << >> blocks
    # Look for patterns like "<< ... = X>>" or "<< ... >> X" where X is a number
    calc_patterns = [
        r'<<[^>]*=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*>>',  # << ... = 5%>> or << ... = 5>>
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*[.\n]',  # >> 5% or >> 5.
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*$',  # >> 5% at end
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%',  # >> 5% (percentage answers)
    ]
    
    for pattern in calc_patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            # Take the last match (most recent calculation)
            last_match = matches[-1]
            try:
                num_str = last_match.group(1).replace(',', '').strip()
                value = float(num_str)
                # For percentage answers, extract just the number (e.g., "2.5%" -> 2.5)
                return value, False  # Found but not in #### format
            except ValueError:
                continue
    
    # Strategy 2b: Extract number right before #### (model might put answer before delimiter)
    # Look for patterns like "2.5% ####" or "20 ####"
    if has_delimiter:
        before_hash_patterns = [
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*####',  # 2.5% #### or 20 ####
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%\s*####',  # 2.5% #### (with %)
        ]
        for pattern in before_hash_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    num_str = match.group(1).replace(',', '').strip()
                    value = float(num_str)
                    return value, True  # Found with #### format (number before delimiter)
                except ValueError:
                    continue
    
    # Strategy 3: Extract last reasonable number in text (fallback)
    # Look for numbers that might be the final answer, especially percentages
    # Prefer numbers near the end and in percentage format
    all_numbers = list(re.finditer(r'\b([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\b', text))
    if all_numbers:
        # Look for percentage patterns first (e.g., "32.11%" or "= 2.5%")
        percent_patterns = [
            r'=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%',  # = 2.5%
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%\s*$',  # 2.5% at end
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%\s*\n',  # 2.5%\n
        ]
        for pattern in percent_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                last_match = matches[-1]
                try:
                    num_str = last_match.group(1).replace(',', '').strip()
                    value = float(num_str)
                    if 0 <= value <= 100:  # Reasonable percentage range
                        return value, False
                except ValueError:
                    continue
        
        # Fallback: take numbers from the last 30% of the text
        text_len = len(text)
        cutoff = int(text_len * 0.7)
        recent_numbers = [m for m in all_numbers if m.start() >= cutoff]
        
        # Prefer the last number, but filter by reasonableness
        candidates = recent_numbers if recent_numbers else all_numbers[-5:]  # Last 5 if no recent ones
        
        for match in reversed(candidates):  # Try from most recent
            try:
                num_str = match.group(1).replace(',', '').strip()
                value = float(num_str)
                # Filter: GSM answers are typically 0-100 for percentages, or small integers
                # But allow up to 1000 for edge cases
                if 0 <= value <= 1000:  # More reasonable range
                    return value, False  # Found but not in preferred format
            except ValueError:
                continue
    
    # No answer found
    # valid_format is True if #### exists OR if we have calculations (<< >>)
    has_calculations = "<<" in text and ">>" in text
    return None, (has_delimiter or has_calculations)


def _extract_gold_answer(answer_text: str) -> Optional[float]:
    """
    Extract gold answer from GSM-Symbolic answer field.
    Format: "... #### <number>"
    """
    pattern = r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, answer_text)
    if match:
        try:
            num_str = match.group(1).replace(',', '')
            return float(num_str)
        except ValueError:
            return None
    return None


def _is_answer_correct(pred: Optional[float], gold: Optional[float], tolerance: float = 1e-6) -> bool:
    """
    Check if predicted answer matches gold answer.
    Uses tolerance for floating point comparison.
    """
    if pred is None or gold is None:
        return False

    # Exact match for integers
    if abs(pred - gold) < tolerance:
        return True

    # 1% relative tolerance for larger numbers
    if gold != 0 and abs((pred - gold) / gold) < 0.01:
        return True

    return False


def _extract_constrained_segments(text: str) -> List[str]:
    """Extract all << ... >> segments from text."""
    pattern = r'<<\s*(.+?)\s*>>'
    return re.findall(pattern, text, re.DOTALL)


def _validate_math_segment(segment: str, parser) -> bool:
    """Check if a math segment is valid according to the grammar."""
    try:
        return parser.is_complete(segment.strip())
    except Exception:
        return False


# =============================================================================
# Token Vocabulary Selection
# =============================================================================

def _select_math_token_ids(tokenizer, max_tokens: int) -> List[int]:
    """
    Build a curated token set for math reasoning.
    Includes digits, operators, delimiters, and common words.
    """
    # Required tokens for CRANE-style math (HIGHEST PRIORITY)
    required_patterns = [
        "<<", ">>", "####", "###", "##", "#",
        "+", "-", "*", "/", "=", "(", ")", "%", "$",
        ".", ",", ":", ";",
        "\n", " ", "\t",
    ]

    # Add digits
    required_patterns.extend([str(i) for i in range(10)])

    found_ids: List[int] = []
    vocab = tokenizer.get_vocab()
    required_found = set()

    # First pass: find exact matches for required patterns (CRITICAL)
    for tok, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue

        if decoded in required_patterns:
            found_ids.append(tok_id)
            required_found.add(decoded)
    
    # Verify critical tokens are found
    critical_tokens = ["####", "<<", ">>"]
    missing = [t for t in critical_tokens if t not in required_found]
    if missing:
        # Try to find tokens that contain these patterns
        for tok, tok_id in vocab.items():
            try:
                decoded = tokenizer.decode([tok_id])
                for critical in missing:
                    if critical in decoded and tok_id not in found_ids:
                        found_ids.append(tok_id)
                        required_found.add(critical)
                        break
            except Exception:
                continue
    
    # Also check if << and >> exist as separate tokens (< and <, > and >)
    # Some tokenizers split << into two < tokens
    single_angle_brackets = ["<", ">"]
    for bracket in single_angle_brackets:
        if bracket in vocab and vocab[bracket] not in found_ids:
            found_ids.append(vocab[bracket])
            required_found.add(bracket)

    # Second pass: add safe alphanumeric tokens for reasoning
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    safe_punct = set(" ,.+-*/=%$#<>\n\t()[]")

    # Scan a large portion of vocabulary (up to 50k tokens) to ensure good coverage
    # This is much faster than scanning all tokens but gives better results than the buggy early-break
    scan_limit = min(50000, len(tokenizer))

    for tok_id in range(scan_limit):
        if tok_id in found_ids:
            continue
        # Early exit if we have enough tokens (optimization)
        if len(found_ids) >= max_tokens * 2:  # Collect 2x to have better selection
            break

        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded or len(decoded) > 15:
            continue

        # Allow tokens that are mostly safe characters
        stripped = decoded.strip()
        if not stripped:
            found_ids.append(tok_id)  # Whitespace tokens
            continue
        if all(c in safe_chars for c in stripped):
            found_ids.append(tok_id)
        elif all(c in safe_punct or c in safe_chars for c in decoded):
            found_ids.append(tok_id)

    return found_ids[:max_tokens]


# =============================================================================
# Generation Methods
# =============================================================================

def _run_standard(
    model_name: str,
    device: str,
    prompt_text: str,
    max_new_tokens: int
) -> Tuple[str, int, float]:
    """
    Standard unconstrained HuggingFace generation (baseline).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    instruction = _make_chatml_instruction(prompt_text)
    inputs = tok(instruction, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    end = time.time()

    gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
    return gen, tokens, end - start


def _load_compiled_modules(run_dir: Path):
    """Load compiled CSD modules from a synthesis run directory."""
    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        raise FileNotFoundError(f"Compiled module directory not found: {module_dir}")
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    import _dafny
    import VerifiedDecoderAgent
    import GeneratedCSD

    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def _setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int,
    grammar_file: Path,
):
    """
    Load model and setup Dafny environment once.
    Returns reusable objects for generation.
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = _load_compiled_modules(run_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    token_ids = _select_math_token_ids(tok, vocab_size)
    lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny, token_ids=token_ids)

    # Create grammar parser
    grammar_text = grammar_file.read_text()
    LarkDafnyParser = create_lark_dafny_parser(grammar_text, VerifiedDecoderAgent, _dafny, start="start")
    parser = LarkDafnyParser(lm._Tokens)

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": tok,
    }


def _run_crane_csd(
    env: dict,
    prompt_text: str,
    max_steps: int,
    grammar_file: Path,
    debug_delimiters: bool = False,
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    CRANE-style generation with CSD for constrained windows.

    Architecture:
      1. Generate unconstrained until << detected
      2. Run CSD strategy for constrained math expression
      3. Validate after >> delimiter
      4. Resume unconstrained until #### or EOS

    Returns:
        (output_text, token_count, time, constrained_segments)
        constrained_segments: List of (segment_text, is_valid) tuples
    """
    _dafny = env["_dafny"]
    VerifiedDecoderAgent = env["VerifiedDecoderAgent"]
    GeneratedCSD = env["GeneratedCSD"]
    lm = env["lm"]
    parser = env["parser"]

    # For validating math segments
    from parsers.lark_parser import LarkGrammarParser
    math_validator = LarkGrammarParser.from_grammar_file(str(grammar_file))

    result_tokens: List[str] = []
    constrained_segments: List[Tuple[str, bool]] = []
    mode = "unconstrained"
    current_window: List[str] = []

    lm.instruction_text = _make_chatml_instruction(prompt_text)
    start_time = time.time()

    step = 0
    consecutive_unconstrained_without_delimiter = 0
    max_unconstrained_before_force = 50  # Force CSD after 50 tokens if no << found
    
    # Repetition detection
    recent_text_window = []  # Track recent text to detect loops
    repetition_threshold = 100  # If we see the same pattern for 100+ tokens, it's stuck
    
    # Track last delimiter position to avoid re-detecting the same delimiter
    # Cooldown must be longer than window size (20 tokens) to prevent re-detection
    last_delimiter_step = -100
    delimiter_cooldown = 25  # Don't detect delimiter again within 25 steps of last detection
    
    if debug_delimiters:
        print(f"  [DEBUG] Starting generation with max_steps={max_steps}")
    
    while step < max_steps:
        if mode == "unconstrained":
            # Generate one token freely (unconstrained)
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            lm.GenerateLogits(dafny_prefix)
            token = lm.ChooseNextToken()
            token_str = str(token)
            result_tokens.append(token_str)
            step += 1
            consecutive_unconstrained_without_delimiter += 1

            # Detect repetitive loops (model stuck generating same pattern)
            recent_text_window.append(token_str)
            if len(recent_text_window) > 100:  # Increased window to catch longer patterns
                recent_text_window.pop(0)
            
            # Check for repetition after enough tokens accumulated
            if step > 100 and len(recent_text_window) > 50:
                recent_text = "".join(recent_text_window)
                if len(recent_text) > 30:
                    # Look for repeated patterns - check if a significant chunk repeats
                    # Check last 40 chars appearing multiple times
                    last_40 = recent_text[-40:]
                    if recent_text.count(last_40) >= 3:  # Same pattern appears 3+ times
                        if debug_delimiters:
                            print(f"  [DEBUG] WARNING: Detected repetitive loop at step {step}!")
                            print(f"  [DEBUG] Repetitive pattern: {repr(last_40[:60])}")
                            print(f"  [DEBUG] Model appears stuck - breaking generation early")
                        break  # Break out of loop to prevent infinite repetition
                    
                    # Also check for very similar patterns (e.g., "To find the number of remorahs" repeating)
                    # Split into chunks and check for duplicates
                    chunk_size = 30
                    chunks = [recent_text[i:i+chunk_size] for i in range(0, len(recent_text)-chunk_size, 10)]
                    if len(chunks) >= 3:
                        # Count how many times the most common chunk appears
                        from collections import Counter
                        chunk_counts = Counter(chunks)
                        most_common_count = chunk_counts.most_common(1)[0][1] if chunk_counts else 0
                        if most_common_count >= 4:  # Same chunk appears 4+ times
                            if debug_delimiters:
                                print(f"  [DEBUG] WARNING: Detected repetitive text chunks at step {step}!")
                                print(f"  [DEBUG] Most repeated chunk: {repr(chunk_counts.most_common(1)[0][0][:50])}")
                                print(f"  [DEBUG] Appears {most_common_count} times - breaking early")
                            break

            # Check for start delimiter << (check more tokens to handle multi-token delimiters)
            # Check last 20 tokens to catch delimiters that might be split across tokens
            joined = "".join(result_tokens[-20:])  # Increased from 10 to 20 for better detection
            
            # Check for << as a single token or split as < followed by <
            has_double_lt = "<<" in joined
            # Check if we have two consecutive < tokens (delimiter split across tokens)
            recent_5 = result_tokens[-5:]
            has_split_lt = len(recent_5) >= 2 and recent_5[-1] == "<" and recent_5[-2] == "<"
            
            if debug_delimiters and step % 10 == 0:
                print(f"  [DEBUG step {step}] Recent tokens: {repr(recent_5)} | Joined: {repr(joined[-50:])} | Has <<: {has_double_lt} | Has split <: {has_split_lt}")
            
            # Only detect delimiter if we haven't detected one recently (avoid re-detecting same delimiter)
            # Also check that we're not already in a constrained window (shouldn't happen, but safety check)
            steps_since_last_delimiter = step - last_delimiter_step
            if (has_double_lt or has_split_lt) and steps_since_last_delimiter > delimiter_cooldown and mode == "unconstrained":
                if debug_delimiters:
                    print(f"  [DEBUG] Detected << delimiter at step {step}, switching to constrained mode")
                mode = "constrained"
                current_window = []
                consecutive_unconstrained_without_delimiter = 0
                last_delimiter_step = step  # Track when we detected this delimiter

            # Check for termination
            # Only stop on EOS if we've generated a reasonable amount (at least 100 tokens)
            # This prevents premature stopping before the model has a chance to generate delimiters
            eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
            min_tokens_before_eos = 100  # Require at least 100 tokens before accepting EOS
            should_stop_on_eos = step >= min_tokens_before_eos and token_str in eos_tokens
            
            # When #### is found, check if number is complete before stopping
            if "####" in joined:
                hash_pos = joined.find("####")
                after_hash = joined[hash_pos + 4:]  # Don't strip - need to check for trailing chars
                
                # Look for a number after ####
                num_match = re.search(r'\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', after_hash)
                if num_match:
                    # Found a number - but is it complete?
                    # Check if there's a non-digit character after the number (space, newline, etc.)
                    number_end = num_match.end()
                    chars_after_number = after_hash[number_end:]
                    
                    # Number is complete if followed by whitespace, newline, %, or end of string with >3 chars
                    number_complete = (
                        len(chars_after_number) > 0 and (chars_after_number[0] in ' \n\t%') or
                        len(after_hash) > 5  # If we have many chars after ####, number is probably complete
                    )
                    
                    if number_complete:
                        if debug_delimiters:
                            print(f"  [DEBUG] Found #### with complete number '{num_match.group(1)}' at step {step}, stopping generation")
                        break
                    else:
                        # Number might be incomplete (e.g., "2" but "20" coming)
                        # Check how many tokens since #### - if many, assume complete
                        tokens_after_hash = 0
                        for i in range(len(result_tokens) - 1, max(0, len(result_tokens) - 20), -1):
                            if "####" in "".join(result_tokens[i:]):
                                tokens_after_hash = len(result_tokens) - i
                                break
                        
                        if tokens_after_hash >= 5:  # After 5 tokens, assume number is complete
                            if debug_delimiters:
                                print(f"  [DEBUG] Found #### with number '{num_match.group(1)}' at step {step} (5+ tokens), stopping")
                            break
                        else:
                            if debug_delimiters:
                                print(f"  [DEBUG] Found #### with partial number '{num_match.group(1)}' at step {step}, waiting for more")
                            continue
                else:
                    # #### found but no number yet - allow up to 10 more tokens for the number
                    tokens_after_hash = 0
                    for i in range(len(result_tokens) - 1, max(0, len(result_tokens) - 20), -1):
                        if "####" in "".join(result_tokens[i:]):
                            tokens_after_hash = len(result_tokens) - i
                            break
                    
                    if tokens_after_hash < 10 and step < max_steps - 5:
                        if debug_delimiters:
                            print(f"  [DEBUG] Found #### at step {step} but no number yet ({tokens_after_hash} tokens after), allowing more tokens")
                        continue
                    else:
                        if debug_delimiters:
                            print(f"  [DEBUG] Found #### at step {step} but no number after {tokens_after_hash} tokens, stopping")
                        break
            elif should_stop_on_eos:
                if debug_delimiters:
                    print(f"  [DEBUG] EOS token '{token_str}' detected at step {step} (after {step} tokens), stopping")
                break
            # Don't stop on EOS if we haven't generated enough tokens yet - continue generation
            elif token_str in eos_tokens and step < min_tokens_before_eos:
                if debug_delimiters:
                    print(f"  [DEBUG] Ignoring early EOS token '{token_str}' at step {step} (need {min_tokens_before_eos} tokens, continuing)")
                # Continue generation instead of stopping - the model might generate more

        elif mode == "constrained":
            # Run CSD strategy for constrained generation
            # CSD only generates math expressions (no << or >> delimiters)
            dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
            remaining_steps = max(1, min(20, max_steps - step))  # Reduced to 20 for shorter math expressions

            csd_output = GeneratedCSD.default__.MyCSDStrategy(
                lm, parser, dafny_prefix, remaining_steps
            )
            csd_tokens = [str(t) for t in csd_output]
            
            # CSD generates math expression only (no >> delimiter)
            # Add all CSD tokens to current window
            current_window.extend(csd_tokens)
            result_tokens.extend(csd_tokens)
            step += len(csd_tokens)

            # Validate the constrained segment
            segment_text = "".join(current_window).strip()
            is_valid = math_validator.is_complete(segment_text) if segment_text else False
            constrained_segments.append((segment_text, is_valid))

            # Switch back to unconstrained mode to generate >> delimiter
            # The model will naturally generate >> after the math expression
            mode = "unconstrained"
            current_window = []
            # Update last_delimiter_step to current step to prevent re-detecting same <<
            # The old << is still in the 20-token window, so we need to wait for cooldown
            last_delimiter_step = step
            
            # Generate >> delimiter in unconstrained mode
            # Limit to 5 tokens to find >>
            for _ in range(min(5, max_steps - step)):
                dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)
                lm.GenerateLogits(dafny_prefix)
                token = lm.ChooseNextToken()
                token_str = str(token)
                result_tokens.append(token_str)
                step += 1
                
                # Check if we found >> delimiter
                joined = "".join(result_tokens[-5:])
                if ">>" in joined:
                    break
                    
                # Check for termination (but be lenient - allow more tokens before stopping on EOS)
                eos_tokens = ["<EOS>", "<|im_end|>", "</s>"]
                if "####" in joined:
                    break
                elif step >= 100 and token_str in eos_tokens:
                    break
                # Continue if EOS is too early

    end_time = time.time()
    output_text = "".join(result_tokens)
    
    # Warn if we hit max_steps without completing
    if step >= max_steps:
        if debug_delimiters:
            print(f"  [DEBUG] WARNING: Hit max_steps limit ({max_steps}) without finding #### delimiter")
            print(f"  [DEBUG] Generated {len(result_tokens)} tokens total")
    
    # Warn if no delimiters were detected (model might not be following prompt format)
    if "<<<" not in output_text and "<<" not in output_text:
        if debug_delimiters:
            print(f"  [DEBUG] WARNING: No << delimiters detected in output! Model may not be following prompt format.")
            print(f"  [DEBUG] First 200 chars of output: {repr(output_text[:200])}")
    
    return output_text, len(result_tokens), end_time - start_time, constrained_segments


def _run_crane_simple(
    env: dict,
    prompt_text: str,
    max_steps: int,
    grammar_file: Path,
) -> Tuple[str, int, float, List[Tuple[str, bool]]]:
    """
    CRANE-style windowing with simple grammar constraint (no CSD).
    Uses per-token grammar masking inside << >> windows.
    """
    _dafny = env["_dafny"]
    VerifiedDecoderAgent = env["VerifiedDecoderAgent"]
    lm = env["lm"]
    parser = env["parser"]

    from parsers.lark_parser import LarkGrammarParser
    math_validator = LarkGrammarParser.from_grammar_file(str(grammar_file))

    result_tokens: List[str] = []
    constrained_segments: List[Tuple[str, bool]] = []
    mode = "unconstrained"
    current_window: List[str] = []

    lm.instruction_text = _make_chatml_instruction(prompt_text)
    start_time = time.time()

    for step in range(max_steps):
        dafny_prefix = _dafny.SeqWithoutIsStrInference(result_tokens)

        if mode == "unconstrained":
            # Generate one token freely
            lm.GenerateLogits(dafny_prefix)
            token = lm.ChooseNextToken()
        else:
            # Generate with grammar constraint
            lm.GenerateLogits(dafny_prefix)
            # Use constrained step from CSDHelpers
            token = VerifiedDecoderAgent.CSDHelpers.ConstrainedStep(
                lm, parser, dafny_prefix, _dafny.SeqWithoutIsStrInference(current_window)
            )

        token_str = str(token)
        result_tokens.append(token_str)

        # Check for delimiter transitions
        joined = "".join(result_tokens[-5:])

        if mode == "unconstrained" and "<<" in joined:
            mode = "constrained"
            current_window = []
        elif mode == "constrained":
            current_window.append(token_str)
            if ">>" in joined:
                # Validate segment
                segment_text = "".join(current_window[:-2]).strip()  # Exclude >>
                is_valid = math_validator.is_complete(segment_text) if segment_text else False
                constrained_segments.append((segment_text, is_valid))
                mode = "unconstrained"
                current_window = []

        # Check for termination
        if "####" in joined or token_str in ["<EOS>", "<|im_end|>", "</s>"]:
            break

    end_time = time.time()
    output_text = "".join(result_tokens)
    return output_text, len(result_tokens), end_time - start_time, constrained_segments


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class GSMMetrics:
    n: int = 0
    correct: int = 0
    valid_format: int = 0
    syntax_valid_segments: int = 0
    total_segments: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    def accuracy(self) -> float:
        return 100.0 * self.correct / max(1, self.n)

    def format_rate(self) -> float:
        return 100.0 * self.valid_format / max(1, self.n)

    def syntax_validity(self) -> float:
        return 100.0 * self.syntax_valid_segments / max(1, self.total_segments)

    def avg_tokens(self) -> float:
        return self.total_tokens / max(1, self.n)

    def avg_time(self) -> float:
        return self.total_time / max(1, self.n)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate GSM-Symbolic with CRANE-CSD")
    ap.add_argument("--method", choices=["standard", "crane", "crane-csd"], required=True,
                    help="Generation method")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Required for crane/crane-csd methods")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID (7B recommended for better instruction following)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", choices=["main", "p1", "p2"], default="main",
                    help="GSM-Symbolic difficulty level")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max examples to evaluate")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Max tokens for standard method")
    ap.add_argument("--max-steps", type=int, default=1024,
                    help="Max steps for crane/crane-csd methods (increased default to allow full solutions)")
    ap.add_argument("--vocab-size", type=int, default=2000,
                    help="Token vocabulary size limit")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark",
                    help="Grammar file for math validation")
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-example details")
    ap.add_argument("--debug-delimiters", action="store_true",
                    help="Debug delimiter detection (shows token details)")
    args = ap.parse_args()

    # Validation
    if args.method in {"crane", "crane-csd"} and args.run_dir is None:
        ap.error("--run-dir required for crane/crane-csd methods")

    # Load dataset (only load the examples we need)
    ds = _load_gsm_symbolic(args.config, limit=args.limit)
    n = len(ds)  # Already limited to args.limit
    metrics = GSMMetrics()

    # Setup environment for crane methods
    dafny_env = None
    if args.method in {"crane", "crane-csd"}:
        print(f"Setting up Dafny environment...")
        dafny_env = _setup_dafny_environment(
            run_dir=args.run_dir,
            model_name=args.model,
            device=args.device,
            vocab_size=args.vocab_size,
            grammar_file=args.grammar,
        )
        # Verify critical tokens are in vocabulary
        tok = dafny_env["tokenizer"]
        test_tokens = ["####", "<<", ">>"]
        vocab = tok.get_vocab()
        found_critical = []
        missing_critical = []
        for test_tok in test_tokens:
            if test_tok in vocab:
                found_critical.append(test_tok)
            else:
                # Check if any token contains it
                found_as_part = False
                for tok_str, tok_id in vocab.items():
                    if test_tok in tok_str:
                        found_critical.append(f"{test_tok} (as part of '{tok_str}')")
                        found_as_part = True
                        break
                if not found_as_part:
                    missing_critical.append(test_tok)
        
        if missing_critical:
            print(f"WARNING: Critical tokens missing from vocabulary: {missing_critical}")
            print(f"  This may cause the model to not generate delimiters correctly.")
            print(f"  Found tokens: {found_critical}")
            # Check if single angle brackets exist (delimiters might be split)
            if "<<" in missing_critical:
                if "<" in vocab:
                    print(f"  Note: Single '<' token exists, but '<<' does not. Delimiter may be split across tokens.")
                else:
                    print(f"  Note: Neither '<<' nor '<' found in vocabulary!")
        print("Model loaded. Starting evaluation...\n")

    # Track start time for ETA
    eval_start_time = time.time()

    for i in range(n):
        example = ds[i]
        question = example.get("question", "")
        gold_answer_text = example.get("answer", "")
        gold_answer = _extract_gold_answer(gold_answer_text)

        if not question:
            continue

        print(f"[{i+1}/{n}] Processing example...", flush=True)
        prompt = _make_gsm_prompt(question)

        # Generate based on method
        constrained_segments = []
        if args.method == "standard":
            out_text, tok_count, dt = _run_standard(
                args.model, args.device, prompt, args.max_new_tokens
            )
        elif args.method == "crane":
            out_text, tok_count, dt, constrained_segments = _run_crane_simple(
                dafny_env, prompt, args.max_steps, args.grammar
            )
        else:  # crane-csd
            out_text, tok_count, dt, constrained_segments = _run_crane_csd(
                dafny_env, prompt, args.max_steps, args.grammar, debug_delimiters=args.debug_delimiters
            )

        # Extract answer and evaluate
        pred_answer, valid_format = _extract_answer(out_text)
        is_correct = _is_answer_correct(pred_answer, gold_answer)

        # Update metrics
        metrics.n += 1
        metrics.correct += 1 if is_correct else 0
        metrics.valid_format += 1 if valid_format else 0
        metrics.total_tokens += tok_count
        metrics.total_time += dt

        # Count syntax validity from constrained segments
        for seg_text, is_valid in constrained_segments:
            metrics.total_segments += 1
            if is_valid:
                metrics.syntax_valid_segments += 1

        # Calculate ETA
        avg_time = metrics.total_time / metrics.n
        remaining = n - metrics.n
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"

        # Progress
        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | "
              f"Correct: {is_correct} | Format: {valid_format} | "
              f"Acc: {metrics.accuracy():.1f}% | ETA: {eta_str}", flush=True)

        if args.verbose or (not valid_format and i < 5) or (pred_answer is None and valid_format):  # Show first 5 failures, or cases where format is valid but extraction failed
            print(f"  Question: {question[:100]}...")
            print(f"  Pred: {pred_answer} | Gold: {gold_answer}")
            print(f"  Output (first 300 chars): {out_text[:300]}...")
            if constrained_segments:
                print(f"  Math segments: {constrained_segments[:3]}")
            if "####" not in out_text:
                print(f"  WARNING: Output does not contain '####' delimiter!")
            elif pred_answer is None and valid_format:
                # Show what comes after #### to debug extraction
                hash_idx = out_text.find("####")
                if hash_idx >= 0:
                    after_hash = out_text[hash_idx:min(hash_idx+100, len(out_text))]
                    print(f"  DEBUG: Found #### but couldn't extract number.")
                    print(f"  DEBUG: Text after #### (first 100 chars): {repr(after_hash)}")
                    # Try to find any numbers near ####
                    nearby_text = out_text[max(0, hash_idx-20):min(hash_idx+100, len(out_text))]
                    numbers_nearby = re.findall(r'\d+\.?\d*', nearby_text)
                    if numbers_nearby:
                        print(f"  DEBUG: Numbers found near ####: {numbers_nearby}")
            print()

    # Final results
    print("\n" + "=" * 60)
    print("GSM-SYMBOLIC RESULTS")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    if args.run_dir:
        print(f"CSD Run: {args.run_dir}")
    print(f"Grammar: {args.grammar}")
    print(f"Examples: {metrics.n}")
    print()
    print(f"Answer Accuracy (%): {metrics.accuracy():.1f}")
    print(f"Valid Format (%): {metrics.format_rate():.1f}")
    if metrics.total_segments > 0:
        print(f"Syntax Validity (%): {metrics.syntax_validity():.1f} ({metrics.syntax_valid_segments}/{metrics.total_segments} segments)")
    print(f"Avg Tokens: {metrics.avg_tokens():.1f}")
    print(f"Avg Time (s): {metrics.avg_time():.2f}")


if __name__ == "__main__":
    main()
