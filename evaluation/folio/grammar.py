"""
Dynamic grammar construction for FOLIO evaluation.

This module provides utilities for building dynamic FOL grammars that restrict
allowed predicates and constants based on the specific problem being solved.
"""

import re
from pathlib import Path
from typing import List, Set, Optional, Tuple


# Base grammar path
FOLIO_GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "utils" / "grammars" / "folio.lark"


def load_base_grammar() -> str:
    """Load the base FOLIO grammar file."""
    with open(FOLIO_GRAMMAR_PATH, 'r') as f:
        return f.read()


def extract_predicates_from_generation(text: str) -> List[Tuple[str, int]]:
    """
    Extract predicate names and arities from generated FOL text.
    
    Args:
        text: The generated text containing predicate definitions
        
    Returns:
        List of (predicate_name, arity) tuples
    """
    predicates = []
    
    # Pattern to match predicate calls like "PredicateName(x)" or "PredicateName(x, y)"
    predicate_pattern = r'([A-Z][a-zA-Z0-9]*)\(([^)]*)\)'
    
    for match in re.finditer(predicate_pattern, text):
        pred_name = match.group(1)
        args = match.group(2)
        # Count arguments by counting commas + 1
        arity = len([a.strip() for a in args.split(',') if a.strip()])
        predicates.append((pred_name, arity))
    
    return predicates


def extract_constants_from_generation(text: str) -> Set[str]:
    """
    Extract constant names from generated FOL text.
    
    Constants are lowercase identifiers used as arguments to predicates.
    
    Args:
        text: The generated text containing FOL formulas
        
    Returns:
        Set of constant names
    """
    constants = set()
    
    # Pattern to match predicate calls like Predicate(constant) or Predicate(x, constant)
    predicate_call_pattern = r'[A-Z][a-zA-Z0-9]*\(([^)]+)\)'
    
    for match in re.finditer(predicate_call_pattern, text):
        args = match.group(1)
        for arg in args.split(','):
            arg = arg.strip()
            # Constants are lowercase and longer than 1 char (variables are single letters)
            if arg and arg[0].islower() and len(arg) > 1:
                constants.add(arg)
    
    return constants


def build_dynamic_grammar(
    allowed_predicates: Optional[List[Tuple[str, int]]] = None,
    allowed_constants: Optional[Set[str]] = None,
    allowed_variables: Optional[Set[str]] = None,
) -> str:
    """
    Build a dynamic FOL grammar that restricts allowed symbols.
    
    Args:
        allowed_predicates: List of (predicate_name, arity) tuples. If None, allow any predicate.
        allowed_constants: Set of allowed constant names. If None, allow any constant.
        allowed_variables: Set of allowed variable names. If None, use default (single lowercase letters).
    
    Returns:
        Modified grammar string
    """
    grammar = load_base_grammar()
    
    # Modify PREDICATE_NAME rule if predicates are restricted
    if allowed_predicates is not None and len(allowed_predicates) > 0:
        pred_names = [p[0] for p in allowed_predicates]
        # Create alternation pattern
        pred_pattern = "|".join(f'"{name}"' for name in pred_names)
        # Replace the PREDICATE_NAME terminal
        # Grammar: PREDICATE_NAME: /[A-Z][a-zA-Z0-9]*/
        grammar = re.sub(
            r'PREDICATE_NAME:\s*/\[A-Z\]\[a-zA-Z0-9\]\*/',
            f'PREDICATE_NAME: {pred_pattern}',
            grammar
        )

    # Modify CONSTANT rule if constants are restricted
    if allowed_constants is not None and len(allowed_constants) > 0:
        const_pattern = "|".join(f'"{name}"' for name in sorted(allowed_constants))
        # Replace the CONSTANT terminal
        # Grammar: CONSTANT: /[a-z][a-zA-Z0-9]+/
        grammar = re.sub(
            r'CONSTANT:\s*/\[a-z\]\[a-zA-Z0-9\]\+/',
            f'CONSTANT: {const_pattern}',
            grammar
        )

    # Modify VARIABLE rule if variables are restricted
    if allowed_variables is not None and len(allowed_variables) > 0:
        var_pattern = "|".join(f'"{v}"' for v in sorted(allowed_variables))
        # Replace the VARIABLE terminal
        # Grammar: VARIABLE: /[a-z]/
        grammar = re.sub(
            r'VARIABLE:\s*/\[a-z\]/',
            f'VARIABLE: {var_pattern}',
            grammar
        )
    
    return grammar

