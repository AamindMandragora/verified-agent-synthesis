"""
Answer extraction utilities for FOLIO evaluation.

Extracts the final answer (True/False/Uncertain) from model output.
"""

import re
from typing import Optional, Tuple

from evaluations.folio.dataset import normalize_label


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from generated text.
    
    FOLIO answers are one of: True, False, or Uncertain
    
    Args:
        text: The generated text containing the answer
        
    Returns:
        The extracted answer (normalized) or None if not found
    """
    # Look for "Answer:" pattern
    answer_match = re.search(r'Answer:\s*(\w+)', text, re.IGNORECASE)
    if answer_match:
        # #region agent log
        import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "answer_extraction.py:extract_answer", "message": "Answer extracted via Answer: pattern", "data": {"raw_match": answer_match.group(0), "extracted": answer_match.group(1), "normalized": normalize_label(answer_match.group(1))}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3"}) + '\n')
        # #endregion
        return normalize_label(answer_match.group(1))
    
    # Alternative: look for standalone True/False/Uncertain at end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.lower() in ['true', 'false', 'uncertain']:
            # #region agent log
            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "answer_extraction.py:fallback_line", "message": "Answer extracted via standalone line", "data": {"line": line}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3"}) + '\n')
            # #endregion
            return normalize_label(line)
    
    # Try to find any occurrence of the answer words near the end
    last_chunk = text[-200:].lower() if len(text) > 200 else text.lower()
    
    # Check in order of priority
    if 'uncertain' in last_chunk:
        # Make sure it's not just part of explanation
        if re.search(r'\b(uncertain|unknown)\b', last_chunk):
            return "Uncertain"
    if 'false' in last_chunk:
        if re.search(r'\bfalse\b', last_chunk):
            return "False"
    if 'true' in last_chunk:
        if re.search(r'\btrue\b', last_chunk):
            return "True"
    
    return None


def extract_fol_sections(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract the FOL sections from generated text.
    
    Args:
        text: The generated text
        
    Returns:
        Tuple of (predicates_section, premises_section, conclusion_section)
        Each can be None if not found
    """
    predicates = None
    premises = None
    conclusion = None
    
    # Find Predicates section
    pred_match = re.search(r'Predicates:\s*(.*?)(?=Premises:|$)', text, re.DOTALL)
    if pred_match:
        predicates = pred_match.group(1).strip()
    
    # Find Premises section
    prem_match = re.search(r'Premises:\s*(.*?)(?=Conclusion:|$)', text, re.DOTALL)
    if prem_match:
        premises = prem_match.group(1).strip()
    
    # Find Conclusion section
    conc_match = re.search(r'Conclusion:\s*(.*?)(?=Answer:|$)', text, re.DOTALL)
    if conc_match:
        conclusion = conc_match.group(1).strip()
    
    return predicates, premises, conclusion


def extract_fol_expressions(text: str) -> list:
    """
    Extract all FOL expressions from constrained windows.
    
    Args:
        text: The generated text
        
    Returns:
        List of FOL expression strings
    """
    # Find all <<...>> windows
    pattern = r'<<([^>]+)>>'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip()]


def is_valid_fol_structure(text: str) -> bool:
    """
    Check if the generated text has valid FOL structure.
    
    This checks for the presence of required sections, not the validity
    of the FOL expressions themselves.
    
    Args:
        text: The generated text
        
    Returns:
        True if the structure is valid
    """
    has_predicates = 'Predicates:' in text
    has_premises = 'Premises:' in text
    has_conclusion = 'Conclusion:' in text
    has_answer = 'Answer:' in text
    
    return has_predicates and has_premises and has_conclusion and has_answer


def count_fol_expressions(text: str) -> int:
    """Count the number of FOL expressions in constrained windows."""
    return len(extract_fol_expressions(text))


def extract_defined_predicates(predicates_section: str) -> set:
    """
    Extract predicate names defined in the Predicates: section.
    
    Format: PredicateName(args) ::: description
    
    Args:
        predicates_section: The text of the Predicates section
        
    Returns:
        Set of predicate names (e.g., {'Cat', 'Mammal', 'Animal'})
    """
    if not predicates_section:
        return set()
    
    # Match patterns like "Cat(x)" or "Mammal(x, y)" at the start of lines
    # Pattern: Uppercase word followed by (
    pattern = r'^([A-Z][a-zA-Z0-9]*)\s*\('
    predicates = set()
    
    for line in predicates_section.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            predicates.add(match.group(1))
    
    return predicates


def extract_used_predicates(fol_text: str) -> set:
    """
    Extract predicate names used in FOL expressions.
    
    Args:
        fol_text: FOL expression text (e.g., "{forall} x (Cat(x) {implies} Mammal(x))")
        
    Returns:
        Set of predicate names used (e.g., {'Cat', 'Mammal'})
    """
    if not fol_text:
        return set()
    
    # Match uppercase words followed by (
    # This captures predicate calls like Cat(x), Mammal(x, y), etc.
    pattern = r'([A-Z][a-zA-Z0-9]*)\s*\('
    matches = re.findall(pattern, fol_text)
    return set(matches)


def validate_fol_predicates(text: str) -> Tuple[bool, set, set]:
    """
    Validate that FOL expressions only use predicates defined in the Predicates section.
    
    This is a SEMANTIC check - syntactically valid FOL like "Mall(x)" is INVALID
    if "Mall" was not defined in the Predicates section.
    
    Args:
        text: The full generated text
        
    Returns:
        Tuple of (is_valid, defined_predicates, undefined_predicates_used)
        - is_valid: True if all used predicates are defined
        - defined_predicates: Set of predicates defined in Predicates section
        - undefined_predicates_used: Set of predicates used but not defined
    """
    predicates_section, premises_section, conclusion_section = extract_fol_sections(text)
    
    # Extract defined predicates
    defined = extract_defined_predicates(predicates_section) if predicates_section else set()
    
    # Extract predicates used in Premises and Conclusion
    used = set()
    if premises_section:
        used.update(extract_used_predicates(premises_section))
    if conclusion_section:
        used.update(extract_used_predicates(conclusion_section))
    
    # Find undefined predicates
    undefined = used - defined
    
    is_valid = len(undefined) == 0
    
    return is_valid, defined, undefined


def is_valid_fol_semantics(text: str) -> bool:
    """
    Check if the FOL is semantically valid (uses only defined predicates).
    
    This is stricter than syntactic validation - it ensures the model
    actually used the predicates it defined, not garbage like "Mall" or "Mforall".
    
    Args:
        text: The full generated text
        
    Returns:
        True if all predicates used are properly defined
    """
    is_valid, defined, undefined = validate_fol_predicates(text)
    
    # #region agent log
    import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "answer_extraction.py:semantic_validation", "message": "FOL semantic validation", "data": {"is_valid": is_valid, "defined_predicates": list(defined), "undefined_predicates": list(undefined)}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H_SEMANTIC"}) + '\n')
    # #endregion
    
    return is_valid


def check_answer_correctness(predicted: Optional[str], gold: str) -> bool:
    """
    Check if the predicted answer matches the gold answer.
    
    Args:
        predicted: The predicted answer (can be None)
        gold: The gold/ground truth answer
        
    Returns:
        True if answers match
    """
    if predicted is None:
        return False
    return normalize_label(predicted) == normalize_label(gold)
