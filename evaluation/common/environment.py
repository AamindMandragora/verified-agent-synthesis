"""
Shared environment setup for CSD evaluation.

Provides resolve_run_dir, load_compiled_modules, verify_critical_tokens,
and setup_dafny_environment used by both gsm_symbolic and folio.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ImportError:
    torch = None

from evaluation.common.run_artifacts import find_compiled_module_dir, resolve_run_dir


def load_compiled_modules(run_dir: Path):
    """
    Load compiled CSD modules from a synthesis run directory.

    Args:
        run_dir: Path to the synthesis run directory

    Returns:
        Tuple of (_dafny, VerifiedDecoderAgent, GeneratedCSD) modules

    Raises:
        FileNotFoundError: If compiled modules are not found
    """
    run_dir = resolve_run_dir(run_dir)
    module_dir = find_compiled_module_dir(run_dir)

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    import _dafny
    import VerifiedDecoderAgent
    import GeneratedCSD

    return _dafny, VerifiedDecoderAgent, GeneratedCSD


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int,
    grammar_file: Path,
    start_rule: str = "start",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    add_fol_keyword_tokens: bool = False,
    add_gsm_delimiter_tokens: bool = False,
) -> Dict[str, Any]:
    """
    Load model and setup Dafny environment once.
    Returns reusable objects for generation.

    Args:
        run_dir: Path to the synthesis run directory
        model_name: HuggingFace model identifier
        device: Device to run on ("cuda" or "cpu")
        vocab_size: Size of constrained vocabulary
        grammar_file: Path to grammar file
        start_rule: Grammar start rule (e.g. "start" for FOLIO, "csd_start" for GSM)
        load_in_4bit: Whether to load model in 4-bit quantization
        load_in_8bit: Whether to load model in 8-bit quantization

    Returns:
        Environment dict with _dafny, VerifiedDecoderAgent, GeneratedCSD, lm, parser, tokenizer
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)

    from evaluation.common.model_utils import create_huggingface_lm
    from evaluation.common.parser_utils import create_lark_dafny_parser

    used_cpu_fallback = False
    try:
        lm = create_huggingface_lm(
            model_name,
            device,
            vocab_size,
            VerifiedDecoderAgent,
            _dafny,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_fol_keyword_tokens=add_fol_keyword_tokens,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        if not device.startswith("cuda") or torch is None:
            raise
        torch.cuda.empty_cache()
        print("  Evaluation: CUDA OOM, falling back to CPU for model load.")
        lm = create_huggingface_lm(
            model_name,
            "cpu",
            vocab_size,
            VerifiedDecoderAgent,
            _dafny,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            add_fol_keyword_tokens=add_fol_keyword_tokens,
            add_gsm_delimiter_tokens=add_gsm_delimiter_tokens,
        )
        used_cpu_fallback = True

    grammar_text = grammar_file.read_text()
    LarkDafnyParser = create_lark_dafny_parser(
        grammar_text, VerifiedDecoderAgent, _dafny, start=start_rule
    )
    parser = LarkDafnyParser(lm._Tokens)

    result = {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": lm.tokenizer,
    }
    if used_cpu_fallback:
        result["_eval_cpu_fallback"] = True
    return result
