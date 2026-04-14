"""
FOLIO Evaluation Package.

Modules:
    - dataset: Dataset loading utilities
    - grammar: Dynamic FOL grammar construction
    - generation: CSD generation methods
    - metrics: Metrics tracking and reporting
    - cli: Command-line interface
"""

from evaluation.folio.dataset import (
    FOLIOExample,
    load_folio,
    load_folio_from_json,
    create_synthetic_folio_examples,
    normalize_label,
)

from evaluation.folio.grammar import (
    build_dynamic_grammar,
    extract_predicates_from_generation,
    extract_constants_from_generation,
    load_base_grammar,
)

from evaluation.folio.generation import (
    run_crane_csd,
    run_unconstrained,
)

from evaluation.folio.metrics import FOLIOMetrics

from evaluation.folio.environment import (
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
    # Grammar
    "build_dynamic_grammar",
    "extract_predicates_from_generation",
    "extract_constants_from_generation",
    "load_base_grammar",
    # Generation
    "run_crane_csd",
    "run_unconstrained",
    # Metrics
    "FOLIOMetrics",
    # Environment
    "setup_dafny_environment",
    "verify_critical_tokens",
    "load_compiled_modules",
]
