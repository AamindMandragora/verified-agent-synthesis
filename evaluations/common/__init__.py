"""
Common utilities shared across evaluation modules.

Includes:
- Model loading and management
- Parser creation utilities
- Generation (dafny_seq_to_str, run_crane_csd, run_unconstrained)
"""

from evaluations.common.model_utils import (
    create_huggingface_lm,
    get_model_input_device,
    get_max_input_length,
)
from evaluations.common.parser_utils import create_lark_dafny_parser
from evaluations.common.generation import (
    dafny_seq_to_str,
    run_crane_csd,
    run_unconstrained,
)
from evaluations.common.environment import (
    resolve_run_dir,
    load_compiled_modules,
    verify_critical_tokens,
    setup_dafny_environment,
)

__all__ = [
    "create_huggingface_lm",
    "get_model_input_device",
    "get_max_input_length",
    "create_lark_dafny_parser",
    "dafny_seq_to_str",
    "run_crane_csd",
    "run_unconstrained",
    "resolve_run_dir",
    "load_compiled_modules",
    "verify_critical_tokens",
    "setup_dafny_environment",
]
