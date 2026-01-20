"""
Token vocabulary selection utilities for constrained decoding.

Provides functions to build curated token sets for different tasks,
ensuring that critical tokens (delimiters, operators) are included
while keeping the vocabulary manageable in size.
"""

from __future__ import annotations

from typing import List


def select_math_token_ids(tokenizer, max_tokens: int) -> List[int]:
    """
    Build a curated token set for math reasoning.
    
    Includes digits, operators, delimiters, and common words needed for
    CRANE-style math reasoning with << >> delimiters.
    
    Args:
        tokenizer: A HuggingFace tokenizer instance
        max_tokens: Maximum number of tokens to include
        
    Returns:
        List of token IDs for the constrained vocabulary
    """
    # Required tokens for CRANE-style math (HIGHEST PRIORITY)
    required_patterns = [
        "<<", ">>", "####", "###", "##", "#",
        "+", "-", "*", "/", "=", "(", ")", "%", "$",
        ".", ",", ":", ";",
        "{", "}",  # For FOL operators: {forall}, {and}, {implies}, etc.
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
    safe_punct = set(" ,.+-*/=%$#<>\n\t()[]{}")

    # Scan a large portion of vocabulary (up to 50k tokens) to ensure good coverage
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


def select_json_token_ids(tokenizer, max_tokens: int) -> List[int]:
    """
    Build a curated token set for JSON generation.
    
    Args:
        tokenizer: A HuggingFace tokenizer instance
        max_tokens: Maximum number of tokens to include
        
    Returns:
        List of token IDs for the constrained vocabulary
    """
    # Required tokens for JSON
    required_patterns = [
        "{", "}", "[", "]", ":", ",", '"',
        "true", "false", "null",
        ".", "-", "+",
        " ", "\n", "\t",
    ]
    
    # Add digits
    required_patterns.extend([str(i) for i in range(10)])
    
    found_ids: List[int] = []
    vocab = tokenizer.get_vocab()
    
    # First pass: find exact matches
    for tok, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue

        if decoded in required_patterns:
            found_ids.append(tok_id)
    
    # Second pass: add safe tokens
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    safe_punct = set(' {}[]:,".\n\t-+')
    
    scan_limit = min(50000, len(tokenizer))
    
    for tok_id in range(scan_limit):
        if tok_id in found_ids:
            continue
        if len(found_ids) >= max_tokens * 2:
            break
            
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded or len(decoded) > 20:
            continue
            
        stripped = decoded.strip()
        if not stripped:
            found_ids.append(tok_id)
            continue
        if all(c in safe_chars for c in stripped):
            found_ids.append(tok_id)
        elif all(c in safe_punct or c in safe_chars for c in decoded):
            found_ids.append(tok_id)
    
    return found_ids[:max_tokens]


def select_json_token_ids(tokenizer, max_tokens: int) -> List[int]:
    """
    Build a curated token set for JSON generation.
    
    Includes structural tokens, strings, numbers, and common JSON values.
    
    Args:
        tokenizer: A HuggingFace tokenizer instance
        max_tokens: Maximum number of tokens to include
        
    Returns:
        List of token IDs for the constrained vocabulary
    """
    # Required tokens for JSON structure (HIGHEST PRIORITY)
    required_patterns = [
        "{", "}", "[", "]", ":", ",",  # Structural
        '"',  # String delimiter
        "true", "false", "null",  # JSON literals
        "-", "+", ".", "e", "E",  # Number components
        " ", "\n", "\t",  # Whitespace
    ]
    
    # Add digits
    required_patterns.extend([str(i) for i in range(10)])
    
    found_ids: List[int] = []
    vocab = tokenizer.get_vocab()
    required_found = set()
    
    # First pass: find exact matches for structural tokens (CRITICAL)
    for tok, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue

        if decoded in required_patterns:
            found_ids.append(tok_id)
            required_found.add(decoded)
    
    # Verify critical structural tokens are found
    critical_tokens = ["{", "}", "[", "]", ":", ",", '"']
    for critical in critical_tokens:
        if critical not in required_found:
            # Try harder to find it
            for tok, tok_id in vocab.items():
                try:
                    decoded = tokenizer.decode([tok_id])
                    if decoded == critical and tok_id not in found_ids:
                        found_ids.append(tok_id)
                        required_found.add(critical)
                        break
                except Exception:
                    continue
    
    # Second pass: add string-safe characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    safe_punct = set(' {}[]":,.\n\t')
    
    scan_limit = min(50000, len(tokenizer))
    
    for tok_id in range(scan_limit):
        if tok_id in found_ids:
            continue
        if len(found_ids) >= max_tokens * 2:
            break
            
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded or len(decoded) > 30:
            continue
            
        stripped = decoded.strip()
        if not stripped:
            found_ids.append(tok_id)
            continue
        # Include tokens that are valid for JSON strings or numbers
        if all(c in safe_chars for c in stripped):
            found_ids.append(tok_id)
        elif all(c in safe_punct or c in safe_chars for c in decoded):
            found_ids.append(tok_id)
    
    return found_ids[:max_tokens]
