"""
Common utilities shared across evaluation modules.

Includes:
- Model loading and management
- Parser creation utilities
- Generation (dafny_seq_to_str, run_crane_csd, run_unconstrained)
"""

from evaluation.common.model_utils import (
    create_huggingface_lm,
    get_model_input_device,
    get_max_input_length,
)
from evaluation.common.parser_utils import create_lark_dafny_parser
from evaluation.common.generation import (
    dafny_seq_to_str,
    run_crane_csd,
    run_unconstrained,
)
from evaluation.common.environment import (
    load_compiled_modules,
    verify_critical_tokens,
    setup_dafny_environment,
)
from evaluation.common.run_artifacts import (
    resolve_run_dir,
    find_compiled_module_dir,
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
    "find_compiled_module_dir",
    "load_compiled_modules",
    "verify_critical_tokens",
    "setup_dafny_environment",
]
