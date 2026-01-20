"""
GSM-Symbolic evaluation module.

This package provides tools for evaluating CRANE-style CSD on the GSM-Symbolic
dataset (grade school math word problems with symbolic reasoning).

Modules:
- dataset: Dataset loading utilities
- prompts: Prompt formatting and variable extraction
- answer_extraction: Answer extraction and evaluation
- grammar: Dynamic grammar construction
- generation: CSD generation method (CRANE-CSD)
- environment: Dafny environment setup
- metrics: Evaluation metrics
- cli: Command-line interface

Usage:
    # As a command-line tool:
    python -m evaluations.gsm_symbolic.cli --run-dir ...
    
    # Or using the legacy script location:
    python scripts/evaluate_gsm_symbolic.py --run-dir ...
"""

from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
from evaluations.gsm_symbolic.prompts import (
    make_gsm_prompt,
    make_chatml_instruction,
    symbolize_question,
    extract_numbers_with_context,
    extract_variables,
    CRANE_FEW_SHOT_EXAMPLES,
)
from evaluations.gsm_symbolic.answer_extraction import (
    extract_answer,
    extract_gold_answer,
    extract_symbolic_expression,
    evaluate_symbolic_expression,
    is_symbolic_valid,
    extract_constrained_segments,
    validate_math_segment,
)
from evaluations.gsm_symbolic.grammar import build_dynamic_grammar, extract_variables_from_mapping
from evaluations.gsm_symbolic.metrics import GSMMetrics
from evaluations.gsm_symbolic.generation import (
    run_crane_csd,
    dafny_seq_to_str,
)
from evaluations.gsm_symbolic.environment import (
    load_compiled_modules,
    setup_dafny_environment,
    verify_critical_tokens,
)

__all__ = [
    # Dataset
    "load_gsm_symbolic",
    # Prompts
    "make_gsm_prompt",
    "make_chatml_instruction", 
    "symbolize_question",
    "extract_numbers_with_context",
    "extract_variables",
    "CRANE_FEW_SHOT_EXAMPLES",
    # Answer extraction
    "extract_answer",
    "extract_gold_answer",
    "extract_symbolic_expression",
    "evaluate_symbolic_expression",
    "is_symbolic_valid",
    "extract_constrained_segments",
    "validate_math_segment",
    # Grammar
    "build_dynamic_grammar",
    "extract_variables_from_mapping",
    # Metrics
    "GSMMetrics",
    # Generation
    "run_crane_csd",
    "dafny_seq_to_str",
    # Environment
    "load_compiled_modules",
    "setup_dafny_environment",
    "verify_critical_tokens",
]
