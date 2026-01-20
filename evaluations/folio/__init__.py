"""
FOLIO Evaluation Package for Dynamic CSD.

This package provides evaluation tools for the FOLIO first-order logic
reasoning dataset using CRANE-style constrained decoding strategies.

Modules:
    - dataset: Dataset loading utilities
    - prompts: Prompt templates and few-shot examples
    - grammar: Dynamic FOL grammar construction
    - generation: CRANE-CSD generation methods
    - answer_extraction: Answer extraction from outputs
    - metrics: Metrics tracking and reporting
    - cli: Command-line interface

Usage:
    python -m evaluations.folio.cli \\
        --run-dir outputs/generated-csd/runs/YOUR_RUN \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --limit 50
"""

from evaluations.folio.dataset import (
    FOLIOExample,
    load_folio,
    load_folio_from_json,
    create_synthetic_folio_examples,
    normalize_label,
)

from evaluations.folio.prompts import (
    make_folio_prompt,
    make_folio_prompt_no_cot,
    FOLIO_FEW_SHOT_EXAMPLES,
    FOL_GRAMMAR_DESCRIPTION,
    CONSTRAINT_START,
    CONSTRAINT_END,
)

from evaluations.folio.grammar import (
    build_dynamic_grammar,
    build_grammar_from_context,
    extract_predicates_from_generation,
    extract_constants_from_generation,
    load_base_grammar,
)

from evaluations.folio.generation import (
    run_crane_csd,
    run_unconstrained,
)

from evaluations.folio.answer_extraction import (
    extract_answer,
    extract_fol_sections,
    extract_fol_expressions,
    is_valid_fol_structure,
    check_answer_correctness,
)

from evaluations.folio.metrics import FOLIOMetrics

from evaluations.folio.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
    load_compiled_modules,
)


__all__ = [
    # Dataset
    "FOLIOExample",
    "load_folio",
    "load_folio_from_json",
    "create_synthetic_folio_examples",
    "normalize_label",
    # Prompts
    "make_folio_prompt",
    "make_folio_prompt_no_cot",
    "FOLIO_FEW_SHOT_EXAMPLES",
    "FOL_GRAMMAR_DESCRIPTION",
    "CONSTRAINT_START",
    "CONSTRAINT_END",
    # Grammar
    "build_dynamic_grammar",
    "build_grammar_from_context",
    "extract_predicates_from_generation",
    "extract_constants_from_generation",
    "load_base_grammar",
    # Generation
    "run_crane_csd",
    "run_unconstrained",
    # Answer extraction
    "extract_answer",
    "extract_fol_sections",
    "extract_fol_expressions",
    "is_valid_fol_structure",
    "check_answer_correctness",
    # Metrics
    "FOLIOMetrics",
    # Environment
    "setup_dafny_environment",
    "verify_critical_tokens",
    "load_compiled_modules",
]
