"""
Dynamic grammar construction for FOLIO evaluation.

This module provides utilities for building dynamic FOL grammars that restrict
allowed predicates and constants based on the specific problem being solved.
"""

import re
from pathlib import Path
from typing import List, Set, Optional, Tuple


# Base grammar path
FOLIO_GRAMMAR_PATH = Path(__file__).parent.parent.parent / "grammars" / "folio.lark"


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
    
    # Pattern to match predicate definitions like "PredicateName(x)" or "PredicateName(x, y)"
    # In the Predicates section
    predicate_pattern = r'([A-Z][a-zA-Z0-9]*)\(([^)]*)\)\s*:::'
    
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


def build_grammar_from_context(
    generated_so_far: str,
    problem_text: str,
    question_text: str,
) -> str:
    """
    Build a dynamic grammar based on the context of generation.
    
    This analyzes what predicates and constants have been defined so far
    and restricts future generation to use only those symbols.
    
    Args:
        generated_so_far: Text generated so far
        problem_text: The problem premises
        question_text: The question being answered
    
    Returns:
        Dynamic grammar string
    """
    # Extract predicates from the Predicates section if present
    predicates = extract_predicates_from_generation(generated_so_far)
    
    # Extract constants from usage
    constants = extract_constants_from_generation(generated_so_far)
    
    # If we haven't defined predicates yet, use unrestricted grammar
    if not predicates:
        return load_base_grammar()
    
    # Build restricted grammar
    return build_dynamic_grammar(
        allowed_predicates=predicates,
        allowed_constants=constants if constants else None,
        # Allow standard variables x, y, z
        allowed_variables={'x', 'y', 'z'},
    )


def is_in_predicates_section(text: str) -> bool:
    """Check if the generation is currently in the Predicates section."""
    # Find last occurrence of key markers
    pred_pos = text.rfind("Predicates:")
    prem_pos = text.rfind("Premises:")
    
    if pred_pos == -1:
        return False
    
    # We're in Predicates section if it's been started but Premises hasn't
    return prem_pos == -1 or pred_pos > prem_pos


def is_in_premises_section(text: str) -> bool:
    """Check if the generation is currently in the Premises section."""
    prem_pos = text.rfind("Premises:")
    conc_pos = text.rfind("Conclusion:")
    
    if prem_pos == -1:
        return False
    
    return conc_pos == -1 or prem_pos > conc_pos


def is_in_conclusion_section(text: str) -> bool:
    """Check if the generation is currently in the Conclusion section."""
    conc_pos = text.rfind("Conclusion:")
    ans_pos = text.rfind("Answer:")
    
    if conc_pos == -1:
        return False
    
    return ans_pos == -1 or conc_pos > ans_pos
