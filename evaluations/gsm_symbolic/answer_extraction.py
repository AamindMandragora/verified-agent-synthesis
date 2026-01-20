"""
Answer extraction and evaluation utilities for GSM-Symbolic.

Handles:
- Extracting symbolic expressions from << >> delimiters
- Evaluating symbolic expressions with variable substitution
- Extracting gold answers from dataset
- Multiple fallback strategies for answer extraction
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def extract_symbolic_expression(text: str) -> Optional[str]:
    """
    Extract symbolic expression from << >> delimiters (CRANE format).

    Args:
        text: Generated text containing << expression >>

    Returns:
        The symbolic expression string (e.g., "a + b - c") or None if not found
    """
    # Look for << expression >> pattern - use greedy match up to >>
    # This handles cases where expression might contain single > characters
    pattern = r'<<\s*(.+?)\s*>>'
    matches = list(re.finditer(pattern, text))

    if matches:
        # Take the last match (most recent expression)
        last_match = matches[-1]
        expr = last_match.group(1).strip()

        # Clean up malformed expressions
        # Remove trailing "= expression" (equations, not expressions)
        if ' = ' in expr:
            # Take just the left side of the equation
            expr = expr.split(' = ')[0].strip()

        # Remove trailing "= number" or similar
        expr = re.sub(r'\s*=\s*[+-]?\d+(\.\d+)?\s*$', '', expr)
        return expr

    return None


def is_expression_wellformed(expr: str) -> bool:
    """
    Check if expression uses only supported operators and syntax.

    Rejects:
    - ** (power) - not in grammar
    - = (equation) - not an expression
    - if/else (conditionals) - not in grammar
    - Unbalanced parentheses

    Returns:
        True if well-formed, False otherwise
    """
    if not expr:
        return False

    # Reject power operator (not in grammar)
    if '**' in expr:
        return False

    # Reject equations (= outside of comparisons)
    if '=' in expr:
        return False

    # Reject Python conditionals
    if ' if ' in expr or ' else ' in expr:
        return False

    # Check parenthesis balance
    if expr.count('(') != expr.count(')'):
        return False

    return True


def evaluate_symbolic_expression(
    expr: str,
    variable_values: Dict[str, float],
    debug: bool = False
) -> Optional[float]:
    """
    Safely evaluate a symbolic expression with given variable values.

    Args:
        expr: Symbolic expression string (e.g., "n1 * n2 / 12")
        variable_values: Dictionary mapping variable names to their values
        debug: If True, print debug information on failure

    Returns:
        Evaluated numerical result, or None if evaluation fails
    """
    try:
        # Replace variables with their values
        eval_expr = expr
        for var, val in variable_values.items():
            # Use word boundaries to avoid partial matches
            eval_expr = re.sub(r'\b' + re.escape(var) + r'\b', str(val), eval_expr)

        if debug:
            print(f"    [DEBUG] Original expr: {expr}")
            print(f"    [DEBUG] After substitution: {eval_expr}")
            print(f"    [DEBUG] Variable values: {variable_values}")

        # Allow int() function and basic arithmetic
        allowed_names = {"int": int}
        result = eval(eval_expr, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        if debug:
            print(f"    [DEBUG] Evaluation failed: {e}")
            print(f"    [DEBUG] Expression after substitution: {eval_expr}")
        return None


def is_symbolic_valid(expr: Optional[str], variable_mapping: Dict[str, str]) -> bool:
    """
    Check if a symbolic expression is valid.
    
    A valid expression:
    1. Is not None/empty
    2. Uses at least one variable from the variable mapping
    3. Contains only allowed operations: +, -, *, /, //, %, (), int(), and variable names
    
    Args:
        expr: The symbolic expression string
        variable_mapping: Dict mapping variable names to their values
    
    Returns:
        True if the expression is valid, False otherwise
    """
    if not expr or not expr.strip():
        return False
    
    expr = expr.strip()
    
    # Check if at least one variable from the mapping is used
    variables_used = [var for var in variable_mapping.keys() if var in expr]
    if not variables_used:
        return False
    
    # Check for valid characters (basic validation)
    # Allow: variable names (alphanumeric + underscore), numbers, operators, parentheses, spaces
    allowed_pattern = r'^[a-zA-Z0-9_+\-*/%(). ]+$'
    if not re.match(allowed_pattern, expr):
        # Allow int() function
        expr_without_int = expr.replace('int(', '(').replace('int ', ' ')
        if not re.match(allowed_pattern, expr_without_int):
            return False
    
    return True


def extract_constrained_segments(text: str) -> list[str]:
    """
    Extract all << ... >> segments from text.
    
    Args:
        text: Generated text
        
    Returns:
        List of segment contents (without delimiters)
    """
    pattern = r'<<\s*(.+?)\s*>>'
    return re.findall(pattern, text, re.DOTALL)


def validate_math_segment(segment: str, parser) -> bool:
    """
    Check if a math segment is valid according to the grammar.
    
    Args:
        segment: The math expression segment
        parser: A grammar parser with is_complete method
        
    Returns:
        True if valid, False otherwise
    """
    try:
        return parser.is_complete(segment.strip())
    except Exception:
        return False


def extract_answer(
    text: str,
    variable_mapping: Optional[Dict[str, str]] = None,
    grammar_parser=None,
    debug: bool = False
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Extract and evaluate symbolic expression answer from generated text.

    For CRANE format, the model outputs a symbolic expression like <<n1 * n2 / 12>>.
    We extract this expression, validate it against the grammar, and evaluate it numerically.

    Args:
        text: Generated text
        variable_mapping: Dict mapping variable names to their numeric values
        grammar_parser: Optional grammar parser for format validation
        debug: If True, print debug information on evaluation failure

    Returns:
        Tuple of:
        - evaluated_value: The numerical result of evaluating the expression, or None
        - is_valid_format: True if expression is syntactically valid according to grammar
        - symbolic_expr: The raw symbolic expression string
    """
    # Extract symbolic expression from << >> (CRANE format)
    symbolic_expr = extract_symbolic_expression(text)

    if not symbolic_expr:
        # No expression found at all
        return None, False, None

    # First check: well-formed (no invalid operators like **, =, if/else)
    if not is_expression_wellformed(symbolic_expr):
        return None, False, symbolic_expr

    # Check format validity using grammar parser if provided
    is_valid_format = True
    if grammar_parser:
        try:
            is_valid_format = grammar_parser.is_complete(symbolic_expr)
        except Exception:
            is_valid_format = False

    # Evaluate the expression numerically if we have variable mapping
    evaluated_value = None
    if variable_mapping:
        variable_values = {}
        for var, val_str in variable_mapping.items():
            try:
                variable_values[var] = float(val_str)
            except ValueError:
                pass

        if variable_values:
            evaluated_value = evaluate_symbolic_expression(symbolic_expr, variable_values, debug=debug)

    return evaluated_value, is_valid_format, symbolic_expr


def extract_gold_answer(answer_text: str) -> Optional[float]:
    """
    Extract gold answer from GSM-Symbolic answer field.
    
    The GSM-Symbolic dataset stores answers in format: "... #### <number>"
    
    Args:
        answer_text: The answer field from the dataset
        
    Returns:
        The numeric answer, or None if extraction fails
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


def extract_answer_with_fallbacks(text: str) -> Tuple[Optional[float], bool]:
    """
    Extract answer from generated text using multiple fallback strategies.
    
    This is a legacy function for when no variable mapping is available.
    Uses progressively less reliable extraction methods.
    
    Args:
        text: Generated text
        
    Returns:
        Tuple of (extracted_value, is_valid_format)
    """
    has_delimiter = "####" in text
    has_calculations = "<<" in text and ">>" in text
    
    # Strategy 1: Look for #### followed by a number (preferred format)
    patterns = [
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'####\s*\n\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'####[^\d]*?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*####',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            try:
                num_str = match.group(1).replace(',', '').strip()
                value = float(num_str)
                return value, True
            except ValueError:
                continue
    
    # Strategy 2: Extract from last calculation result in << >> blocks
    calc_patterns = [
        r'<<[^>]*=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*>>',
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*[.\n]',
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%?\s*$',
        r'>>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%',
    ]
    
    for pattern in calc_patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            last_match = matches[-1]
            try:
                num_str = last_match.group(1).replace(',', '').strip()
                value = float(num_str)
                return value, False
            except ValueError:
                continue
    
    # Strategy 3: Extract last reasonable number in text (fallback)
    all_numbers = list(re.finditer(r'\b([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\b', text))
    if all_numbers:
        # Look for percentage patterns first
        percent_patterns = [
            r'=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%',
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%\s*$',
            r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*%\s*\n',
        ]
        for pattern in percent_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                last_match = matches[-1]
                try:
                    num_str = last_match.group(1).replace(',', '').strip()
                    value = float(num_str)
                    if 0 <= value <= 100:
                        return value, False
                except ValueError:
                    continue
        
        # Fallback: take numbers from the last 30% of the text
        text_len = len(text)
        cutoff = int(text_len * 0.7)
        recent_numbers = [m for m in all_numbers if m.start() >= cutoff]
        candidates = recent_numbers if recent_numbers else all_numbers[-5:]
        
        for match in reversed(candidates):
            try:
                num_str = match.group(1).replace(',', '').strip()
                value = float(num_str)
                if 0 <= value <= 1000:
                    return value, False
            except ValueError:
                continue
    
    # No answer found
    return None, (has_delimiter or has_calculations)
