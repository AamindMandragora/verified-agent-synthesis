"""
GSM-Symbolic evaluation module.

Modules:
- dataset: Dataset loading utilities
- grammar: Dynamic grammar construction
- generation: CSD generation method
- environment: Dafny environment setup
- metrics: Evaluation metrics
- cli: Command-line interface
"""

from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
from evaluations.gsm_symbolic.grammar import build_dynamic_grammar, extract_variables_from_mapping
from evaluations.gsm_symbolic.metrics import GSMMetrics
from evaluations.gsm_symbolic.generation import (
    run_crane_csd,
    run_unconstrained,
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
    # Grammar
    "build_dynamic_grammar",
    "extract_variables_from_mapping",
    # Metrics
    "GSMMetrics",
    # Generation
    "run_crane_csd",
    "run_unconstrained",
    "dafny_seq_to_str",
    # Environment
    "load_compiled_modules",
    "setup_dafny_environment",
    "verify_critical_tokens",
]
