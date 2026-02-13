"""
Environment setup utilities for GSM-Symbolic evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for constrained generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


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
        # Check if GeneratedCSD.py is directly in run_dir
        if (run_dir / "GeneratedCSD.py").exists():
            module_dir = run_dir
        else:
            # Fallback to gsm_crane_csd or try to find the directory
            module_dir = run_dir / "gsm_crane_csd"
            if not module_dir.exists():
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
    # Use 'csd_start' rule so IsCompletePrefix returns True only when >> is generated
    # This ensures the CSD continues generating until the expression is properly closed
    LarkDafnyParser = create_lark_dafny_parser(grammar_text, VerifiedDecoderAgent, _dafny, start="csd_start")
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
