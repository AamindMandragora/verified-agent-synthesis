"""
Prompt formatting utilities for GSM-Symbolic evaluation.

Handles:
- Converting questions to symbolic form (replacing numbers with variables)
- Building CRANE-style prompts with few-shot examples
- ChatML instruction formatting for Qwen models
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# CRANE-style few-shot examples for GSM-Symbolic
# Uses context-aware variable naming like the original CRANE paper
# Variables are named based on what they represent in the problem
# IMPORTANT: Shows full reasoning with intermediate steps, not just final expression
# NOTE: Inside << >> we use plain variable names (a, b, c) NOT {a}, {b}, {c}
# CRITICAL: Keep expressions SIMPLE - avoid nested parens, use step-by-step decomposition
CRANE_FEW_SHOT_EXAMPLES = """Q: There are {t} trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be {f} trees. How many trees did the grove workers plant today?

A: Trees planted = final - initial = <<f - t>>
#### f - t

Q: If there are {c} cars in the parking lot and {n} more cars arrive, how many cars are in the parking lot?

A: Total = initial + arrivals = <<c + n>>
#### c + n

Q: Amy has ${m}. Amy bought {q} items for ${p} each. How much money does Amy have left?

A: Cost = <<q * p>>
Remaining = <<m - q * p>>
#### m - q * p

Q: Tom gave {g} marbles to his friend. His friend then gave Tom {b} marbles back. Tom now has {t} marbles. How many marbles did Tom start with?

A: Starting = final + given - received = <<t + g - b>>
#### t + g - b

Q: A rope is {f} feet long. How many inches is that? (1 foot = 12 inches)

A: Inches = <<f * 12>>
#### f * 12

Q: A person rents a car from {s} AM to {e} PM. They get {f} hours free. The first paid hour costs ${p} and each hour after that costs twice as much. How much do they pay?

A: Total hours = <<e - s>>
Paid hours = <<e - s - f>>
First hour = p
Second hour = <<p * 2>>
Third hour = <<p * 2 * 2>>
For 3 paid hours, cost = <<p + p * 2 + p * 2 * 2>>
#### p + p * 2 + p * 2 * 2

Q: A taxi charges ${b} base fare plus ${r} per mile. A {m}-mile trip with a ${t} tip costs how much total?

A: Mile cost = <<m * r>>
Before tip = <<b + m * r>>
Total = <<b + m * r + t>>
#### b + m * r + t

Q: A pool fills at {f} gallons per hour. After {h} hours, it has {c} gallons. How many gallons were there initially?

A: Added = <<f * h>>
Initial = <<c - f * h>>
#### c - f * h

Q: A worker earns ${w} per hour. They work {h} hours with {o} overtime hours at double pay. What is their total pay?

A: Regular = <<w * h>>
Overtime = <<w * 2 * o>>
Total = <<w * h + w * 2 * o>>
#### w * h + w * 2 * o
"""


def _extract_context_word(question: str, num_match_start: int, num_match_end: int) -> str:
    """
    Extract a SINGLE-LETTER context abbreviation near a number to use as variable name.
    
    CRITICAL: Must return ONLY single letters because multi-character tokens (like "ma", "cr")
    may not be in the restricted vocabulary for constrained decoding.
    Single letters a-z are always in the tokenizer vocabulary.
    
    Examples: t (trees), c (cars), m (money), p (price), q (quantity)
    
    Args:
        question: The full question text
        num_match_start: Start position of the number match
        num_match_end: End position of the number match
        
    Returns:
        A single letter variable name, or empty string if no good context found
    """
    # Look at text after the number
    after_text = question[num_match_end:num_match_end + 30]
    before_text = question[max(0, num_match_start - 30):num_match_start]
    
    # Pattern 1: hyphenated unit (e.g., "-foot", "-inch") -> first letter
    hyphen_match = re.match(r'-([a-zA-Z]+)', after_text)
    if hyphen_match:
        word = hyphen_match.group(1).lower()
        return word[0] if len(word) > 0 else ''
    
    # Pattern 2: word immediately after (e.g., " markers", " cars") -> FIRST LETTER ONLY
    word_after = re.match(r'\s+([a-zA-Z]+)', after_text)
    if word_after:
        word = word_after.group(1).lower()
        # Skip common non-descriptive words
        skip_words = {'each', 'the', 'a', 'an', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 
                      'more', 'less', 'than', 'from', 'to', 'for', 'with', 'at', 'by', 'on', 'in',
                      'which', 'that', 'this', 'these', 'those', 'what', 'how', 'when', 'where'}
        if word not in skip_words and len(word) > 1:
            # Return ONLY first letter for tokenizer compatibility
            return word[0]
    
    # Pattern 3: check if preceded by $ (dollar amount) -> 'm' for money or 'p' for price
    if before_text.rstrip().endswith('$'):
        # Check context for cost/price vs money
        if 'cost' in before_text.lower() or 'price' in before_text.lower():
            return 'p'
        return 'm'
    
    # Pattern 4: check context for common keywords
    context = (before_text + question[num_match_start:num_match_end] + after_text).lower()
    if 'cost' in context:
        return 'c'
    if 'price' in context:
        return 'p'
    if 'total' in context:
        return 't'
    
    return ''


def extract_numbers_with_context(question: str) -> List[Tuple[str, str, str]]:
    """
    Extract numbers from the question along with their context.

    Uses CRANE-style short variable naming (single letters) for tokenizer compatibility.
    Falls back to n1, n2, n3 for disambiguation.

    Args:
        question: The question text

    Returns:
        List of (number_str, variable_name, context) tuples

    Example:
        "16 markers which cost $8.5 each" ->
        [("16", "m", "...16 markers..."), ("8.5", "c", "...cost $8.5...")]
    """
    # Pattern to find all numbers (including decimals)
    number_pattern = r'\b(\d+(?:\.\d+)?)\b'
    
    found = []
    seen_positions = set()
    
    for match in re.finditer(number_pattern, question):
        num_str = match.group(1)
        pos = match.start()
        
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        
        # Get context word for variable name
        context_word = _extract_context_word(question, match.start(), match.end())
        
        # Get surrounding context for debugging
        start = max(0, match.start() - 20)
        end = min(len(question), match.end() + 20)
        context = question[start:end]
        
        found.append((num_str, context_word, context, pos))
    
    # Sort by position in text
    found.sort(key=lambda x: x[3])
    
    # Assign variable names - use context abbreviations when unique, otherwise cycle through alphabet
    # CRITICAL: Use ONLY single letters to ensure tokens are in vocabulary
    result = []
    used_names = set()
    
    # Alphabet for fallback variable names (excluding common confusing letters like i, l, o)
    fallback_letters = list('abcdefghjkmnpqrstuvwxyz')  # skip i, l, o
    fallback_idx = 0
    
    for i, (num_str, context_word, context, _) in enumerate(found):
        if context_word and context_word not in used_names:
            var_name = context_word
            used_names.add(var_name)
        else:
            # Use next available single letter from alphabet
            while fallback_idx < len(fallback_letters):
                candidate = fallback_letters[fallback_idx]
                fallback_idx += 1
                if candidate not in used_names:
                    var_name = candidate
                    used_names.add(var_name)
                    break
            else:
                # Exhausted alphabet, use n with digit (shouldn't happen with typical problem sizes)
                var_name = f"n{i + 1}"
                used_names.add(var_name)
        
        result.append((num_str, var_name, context))
    
    return result


def symbolize_question(question: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace numbers in the question with symbolic variable placeholders.
    
    Args:
        question: The original question text
        
    Returns:
        Tuple of:
        - symbolic_question: Question with numbers replaced by {var} placeholders
        - variable_mapping: Dict mapping variable names to their numeric values
    
    Example:
        Input: "A 210-foot whale has 7 72-inch remoras"
        Output: ("A {f}-foot whale has {a} {i}-inch remoras", 
                 {"f": "210", "a": "7", "i": "72"})
    """
    extractions = extract_numbers_with_context(question)
    
    # Build variable mapping
    variable_mapping = {}
    for num_str, var_name, _ in extractions:
        variable_mapping[var_name] = num_str
    
    # Replace numbers with placeholders
    # CRITICAL: Sort by length descending to replace longer numbers first
    # This prevents "5" from matching inside "11.5" or "20.5"
    sorted_extractions = sorted(extractions, key=lambda x: len(x[0]), reverse=True)
    
    symbolic_question = question
    
    for num_str, var_name, _ in sorted_extractions:
        # Use a pattern that doesn't match numbers within decimals
        # Match the number when it's not part of a larger decimal
        # Negative lookbehind for digit or decimal point, negative lookahead for decimal+digit
        pattern = rf'(?<![.\d])\b{re.escape(num_str)}\b(?!\d)'
        symbolic_question = re.sub(pattern, '{' + var_name + '}', symbolic_question)
    
    return symbolic_question, variable_mapping


def extract_variables(question: str) -> List[str]:
    """
    Extract variable names from a symbolic question.
    
    Args:
        question: Question text with {var} placeholders
        
    Returns:
        Sorted list of unique variable names found in {var} format
    """
    brace_pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    brace_matches = re.findall(brace_pattern, question)
    return sorted(list(set(brace_matches)))


def make_gsm_prompt(question: str, symbolic_question: Optional[str] = None) -> str:
    """
    Format GSM question as CRANE-style prompt for symbolic math solving.

    Uses the CRANE format where the model outputs symbolic expressions
    wrapped in << >> delimiters and final answer after ####.

    Args:
        question: Original question (used if symbolic_question is None)
        symbolic_question: Question with numbers replaced by {var} placeholders
        
    Returns:
        Formatted prompt string
    """
    q = symbolic_question if symbolic_question else question

    # Add unit conversion hint if question mentions feet/inches
    if "foot" in q.lower() and "inch" in q.lower():
        # Check if hint isn't already there
        if "convert feet to inches" not in q.lower():
            q = q.rstrip() + " Remember: convert feet to inches by multiplying by 12."

    return (
        "Solve math problems using symbolic variables.\n\n"
        "CRITICAL FORMAT:\n"
        "- ALWAYS wrap math in << >>: Result = <<a + b>>\n"
        "- Use plain letters inside << >>: <<a * b>> (correct), <<{a} * {b}>> (WRONG)\n"
        "- NEVER write {a} or {b} in your answer - only use plain a, b, c\n"
        "- End with #### and the symbolic expression\n\n"
        f"{CRANE_FEW_SHOT_EXAMPLES}\n"
        f"Q: {q}\n\n"
        f"A:"
    )


def make_chatml_instruction(prompt_text: str) -> str:
    """
    Format as ChatML instruction for Qwen models.
    
    Args:
        prompt_text: The prompt text to wrap
        
    Returns:
        ChatML-formatted instruction string
    """
    # Note: prompt_text ends with "A:" so assistant continues directly
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n "
