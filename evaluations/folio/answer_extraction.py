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
        return normalize_label(answer_match.group(1))
    
    # Alternative: look for standalone True/False/Uncertain at end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.lower() in ['true', 'false', 'uncertain']:
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
