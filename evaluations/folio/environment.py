"""
Environment setup utilities for FOLIO evaluation.

Handles loading compiled CSD modules and setting up the Dafny environment
for CRANE-style FOL generation.
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


def select_fol_token_ids(tokenizer, vocab_size: int) -> List[int]:
    """
    Select tokens relevant for FOL generation.

    This includes:
    - All alphanumeric tokens (for predicates, variables, constants)
    - Logical operators in {keyword} format: {forall}, {exists}, {and}, {or}, {not}, {implies}, {iff}, {xor}
    - Parentheses, commas, colons
    - Section markers: Predicates:, Premises:, Conclusion:, Answer:
    - Constraint delimiters: <<, >>
    - Whitespace and newlines

    Args:
        tokenizer: HuggingFace tokenizer
        vocab_size: Maximum vocabulary size

    Returns:
        List of token IDs
    """
    # Critical FOL tokens that MUST be included
    critical_fol_tokens = [
        # Constraint delimiters
        "<<", ">>", "<", ">",
        # FOL operators
        "{forall}", "{exists}", "{and}", "{or}", "{not}", "{implies}", "{iff}", "{xor}",
        # Partial FOL operators (for incremental generation)
        "{", "}", "forall", "exists", "and", "or", "not", "implies", "iff", "xor",
        "{f", "{e", "{a", "{o", "{n", "{i", "{x",
        "forall}", "exists}", "and}", "or}", "not}", "implies}", "iff}", "xor}",
        # Section markers
        "Predicates:", "Premises:", "Conclusion:", "Answer:",
        "Predicates", "Premises", "Conclusion", "Answer",
        # Structural tokens
        "(", ")", ",", ":", ":::",
        # Whitespace
        " ", "\n", "\t", "  ",
    ]

    # Add individual characters for FOL operators
    for c in "forallexistsandornotimpliesiffxor":
        critical_fol_tokens.append(c)

    found_ids: List[int] = []
    vocab = tokenizer.get_vocab()
    found_tokens = set()

    # First pass: find critical FOL tokens (HIGHEST PRIORITY)
    for tok, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue

        # Check for exact matches
        if decoded in critical_fol_tokens:
            if tok_id not in found_ids:
                found_ids.append(tok_id)
                found_tokens.add(decoded)

        # Check for tokens containing critical patterns
        for critical in critical_fol_tokens:
            if critical in decoded and tok_id not in found_ids:
                found_ids.append(tok_id)
                found_tokens.add(decoded)
                break

    # Second pass: add alphanumeric tokens for predicates, variables, constants
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    safe_punct = set(" ,.:()\n\t{}<>")

    scan_limit = min(50000, len(tokenizer))

    for tok_id in range(scan_limit):
        if tok_id in found_ids:
            continue
        if len(found_ids) >= vocab_size * 2:
            break

        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded or len(decoded) > 20:
            continue

        stripped = decoded.strip()
        if not stripped:
            found_ids.append(tok_id)  # Whitespace tokens
            continue
        if all(c in safe_chars for c in stripped):
            found_ids.append(tok_id)
        elif all(c in safe_punct or c in safe_chars for c in decoded):
            found_ids.append(tok_id)

    return found_ids[:vocab_size]


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


    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    token_ids = select_fol_token_ids(tok, vocab_size)
    lm = create_huggingface_lm(model_name, device, vocab_size, VerifiedDecoderAgent, _dafny, token_ids=token_ids)

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
    """
    Verify that critical tokens for CRANE-FOL format can be produced.

    For FOL, many tokens like {forall} are tokenized as multiple sub-tokens.
    This function checks that the tokenizer can at least encode/decode them.

    Args:
        tokenizer: HuggingFace tokenizer
        verbose: Whether to print warnings

    Returns:
        Dict with "found" and "missing" lists
    """
    # Critical tokens for FOL generation
    test_tokens = [
        "<<", ">>",  # Constraint delimiters
        "{and}", "{or}", "{not}", "{implies}",  # Logical operators
        "{forall}", "{exists}",  # Quantifiers
        "Answer:", "Predicates:", "Premises:", "Conclusion:",  # Section markers
    ]

    found_critical = []
    missing_critical = []
    tokenization_info = []

    for test_tok in test_tokens:
        try:
            # Check how this token is encoded
            token_ids = tokenizer.encode(test_tok, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)

            if decoded.strip() == test_tok.strip():
                num_tokens = len(token_ids)
                found_critical.append(test_tok)
                tokenization_info.append(f"{test_tok} -> {num_tokens} token(s)")
            else:
                missing_critical.append(test_tok)
        except Exception as e:
            missing_critical.append(f"{test_tok} (error: {e})")

    if verbose:
        if missing_critical:
            print(f"WARNING: Critical tokens cannot be properly encoded: {missing_critical}")
            print(f"  This may affect FOL generation quality.")
        else:
            print(f"All critical FOL tokens can be encoded.")
        # Show tokenization info for first few
        if tokenization_info:
            print(f"Token encoding: {tokenization_info[:4]}...")

    return {"found": found_critical, "missing": missing_critical, "tokenization": tokenization_info}
