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

# Fallback subdirs to search for compiled GeneratedCSD (order matters for discovery)
_COMPILED_MODULE_FALLBACK_SUBDIRS = ["gsm_crane_csd", "folio_csd", "fol_csd"]


def resolve_run_dir(run_dir: Path) -> Path:
    """
    Resolve a run directory path, handling 'latest' shortcut.

    If run_dir ends with 'latest' and doesn't exist as a directory,
    reads the actual path from 'latest_run.txt' in the parent directory.

    Args:
        run_dir: Path to the synthesis run directory (may be 'latest' shortcut)

    Returns:
        Resolved actual path to the run directory
    """
    if run_dir.name == "latest" and not run_dir.exists():
        latest_txt = run_dir.parent / "latest_run.txt"
        if latest_txt.exists():
            actual_path = Path(latest_txt.read_text().strip())
            if actual_path.exists():
                return actual_path
    return run_dir


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

    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        if (run_dir / "GeneratedCSD.py").exists():
            module_dir = run_dir
        else:
            for subdir in _COMPILED_MODULE_FALLBACK_SUBDIRS:
                candidate = run_dir / subdir
                if candidate.exists():
                    module_dir = candidate
                    break
            else:
                found = list(run_dir.glob("*/GeneratedCSD.py"))
                if found:
                    module_dir = found[0].parent
                else:
                    raise FileNotFoundError(f"Compiled module directory not found in {run_dir}")

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

    from transformers import AutoTokenizer
    from evaluations.common.model_utils import create_huggingface_lm
    from evaluations.common.parser_utils import create_lark_dafny_parser

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
        "tokenizer": tok,
    }
    if used_cpu_fallback:
        result["_eval_cpu_fallback"] = True
    return result
