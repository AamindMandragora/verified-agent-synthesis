"""
Environment setup utilities for FOLIO evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for constrained generation.
"""
# TODO: merge shared logic with gsm_symbolic/environment.py (resolve_run_dir, load_compiled_modules,
# verify_critical_tokens are near-identical; differences are in setup_dafny_environment signature
# and parser start rule)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


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
    # Resolve 'latest' shortcut if needed
    run_dir = resolve_run_dir(run_dir)
    
    module_dir = run_dir / "generated_csd"
    if not module_dir.exists():
        # Fallback to other possible directories
        for subdir in ["gsm_crane_csd", "folio_csd", "fol_csd"]:
            module_dir = run_dir / subdir
            if module_dir.exists():
                break
        else:
            # Try to find any directory that contains GeneratedCSD.py
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


def setup_dafny_environment(
    run_dir: Path,
    model_name: str,
    device: str,
    vocab_size: int,
    grammar_file: Path,
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
        
    Returns:
        Environment dict with:
        - "_dafny": Dafny runtime module
        - "VerifiedDecoderAgent": Dafny decoder agent module
        - "GeneratedCSD": Generated CSD module
        - "lm": Language model wrapper
        - "parser": Grammar parser
        - "tokenizer": HuggingFace tokenizer
    """
    _dafny, VerifiedDecoderAgent, GeneratedCSD = load_compiled_modules(run_dir)

    from transformers import AutoTokenizer
    from evaluations.common.model_utils import create_huggingface_lm
    from evaluations.common.parser_utils import create_lark_dafny_parser

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny)

    # Create grammar parser
    grammar_text = grammar_file.read_text()
    # Use 'start' rule for FOL grammar (full FOL statements)
    LarkDafnyParser = create_lark_dafny_parser(grammar_text, VerifiedDecoderAgent, _dafny, start="start")
    parser = LarkDafnyParser(lm._Tokens)

    return {
        "_dafny": _dafny,
        "VerifiedDecoderAgent": VerifiedDecoderAgent,
        "GeneratedCSD": GeneratedCSD,
        "lm": lm,
        "parser": parser,
        "tokenizer": tok,
    }


def verify_critical_tokens(tokenizer, verbose: bool = True) -> Dict[str, Any]:
    """No-op — critical token verification removed (was dataset-specific)."""
    return {"found": [], "missing": []}
