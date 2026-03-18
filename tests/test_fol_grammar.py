#!/usr/bin/env python3
"""
Test the FOL (FOLIO) grammar for correctness and harshness.

Checks:
- Valid prefixes are accepted, invalid rejected.
- Complete formulas are recognized.
- ValidNextTokens after a complete subformula (e.g. Pred1(x)) includes extension
  tokens like "{or}", "{and}" so we don't force early exit; and whether we get
  only whitespace (which would cause our wrapper to offer only "%").
- With the production tokenizer (FOL keywords added as special tokens), FOL
  keywords are single tokens and valid-next after "Alkane(mixture)" includes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (tests/ is one level below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_valid_prefix(parser, text: str) -> bool:
    """Match parser_utils LarkDafnyParser._is_valid_prefix."""
    if not text:
        return True
    try:
        parser.parse(text)
        return True
    except parser._UnexpectedEOF:
        return True
    except parser._UnexpectedToken as e:
        return e.token.type == "$END"
    except parser._UnexpectedCharacters:
        return False
    except Exception:
        return False


def _is_complete(parser, text: str) -> bool:
    """Match parser_utils LarkDafnyParser._is_complete."""
    if not text:
        return False
    try:
        parser.parse(text)
        return True
    except Exception:
        return False


def get_valid_next(parser, text: str, vocabulary: list[str]) -> list[str]:
    """Return token strings that can validly follow text (same logic as ValidNextTokens)."""
    if text and not _is_valid_prefix(parser, text):
        return []
    valid = []
    for tok in vocabulary:
        if not tok:
            continue
        if _is_valid_prefix(parser, text + tok):
            valid.append(tok)
    return valid


def main():
    from lark import Lark
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF

    grammar_path = PROJECT_ROOT / "grammars" / "folio.lark"
    grammar = grammar_path.read_text()
    parser = Lark(grammar, start="start", parser="lalr")
    parser._UnexpectedCharacters = UnexpectedCharacters
    parser._UnexpectedToken = UnexpectedToken
    parser._UnexpectedEOF = UnexpectedEOF

    # Vocabulary: FOL tokens + whitespace + delimiters + noise
    vocabulary = [
        # FOL keywords
        "{forall}", "{exists}", "{and}", "{or}", "{not}", "{implies}", "{iff}", "{xor}",
        # Punctuation
        "(", ")", ",", " ",
        # Delimiter and noise (to see if they appear as valid)
        "%", "!", "\n", "\t",
        # Predicates / atoms (grammar: PREDICATE_NAME, CONSTANT, VARIABLE)
        "Alkane", "Alkale", "mixture", "mix", "Dog", "P", "Q", "R",
        "x", "y", "z", "a", "b",
    ]

    # --- 1) Valid prefix / complete tests ---
    print("=== Valid prefix & complete ===")
    tests = [
        ("", True, False),
        ("Alkane(mixture)", True, True),
        ("Alkale(mix)", True, True),
        ("P(x)", True, True),
        ("P(x) {and}", True, False),
        ("P(x) {and} Q(x)", True, True),
        ("P(x) {and} Q(x) {or} R(x)", True, True),
        ("{not} P(x)", True, True),
        ("({and})", False, False),
        ("Alkane(mixture", True, False),
    ]
    for text, expect_valid, expect_complete in tests:
        v = _is_valid_prefix(parser, text)
        c = _is_complete(parser, text)
        ok = (v == expect_valid) and (c == expect_complete)
        print(f"  {repr(text):45} valid={v} (expect {expect_valid}) complete={c} (expect {expect_complete})  {'OK' if ok else 'MISMATCH'}")

    # --- 2) Valid next tokens after key prefixes ---
    print("\n=== Valid next tokens (extension vs whitespace-only) ===")
    prefixes = [
        "Alkane(mixture)",
        "P(x)",
        "P(x) ",
        "P(x) {and}",
        "P(x) {and} Q(x)",
        "Alkane(mixture) {and}",
    ]
    for prefix in prefixes:
        valid = get_valid_next(parser, prefix, vocabulary)
        complete = _is_complete(parser, prefix)
        # Split into formula-extending vs whitespace/noise
        ws_noise = [t for t in valid if t.strip() == "" or t in ("\n", "\t", "!", "%")]
        formula = [t for t in valid if t not in ws_noise]
        print(f"  Prefix: {repr(prefix)}")
        print(f"    complete={complete}  # valid next = {len(valid)}")
        print(f"    formula-extending: {sorted(set(formula))[:20]}")
        print(f"    whitespace/noise:  {sorted(set(ws_noise))}")
        if complete and formula:
            print(f"    -> Can extend (good); no forced early exit.")
        elif complete and not formula and ws_noise:
            print(f"    -> Only WS/noise valid after complete formula (we would offer only % in wrapper).")
        elif complete and not valid:
            print(f"    -> No valid next (grammar allows nothing after this).")
        print()

    # --- 3) After "Alkane(mixture)" do we get "{or}" / "{and}"? ---
    print("=== Critical: after single atom Alkane(mixture) ===")
    prefix = "Alkane(mixture)"
    valid = get_valid_next(parser, prefix, vocabulary)
    has_or = "{or}" in valid or " {or}" in valid
    has_and = "{and}" in valid or " {and}" in valid
    only_ws = all(t.strip() == "" or t in ("\n", "\t") for t in valid)
    print(f"  valid next ({len(valid)}): {sorted(valid)[:30]}")
    print(f"  has {{or}}: {has_or}, has {{and}}: {has_and}, only_ws: {only_ws}")
    if not has_or and not has_and and only_ws:
        print("  -> Grammar returns only WS after single atom; wrapper would offer only % (early exit).")
    else:
        print("  -> Grammar allows extension; no forced early exit.")

    # --- 3b) Inside predicate argument: can we continue a constant (e.g. P1(be) + "jing" -> P1(beijing)? ---
    print("\n=== Inside predicate: continue constant (e.g. P1(be) + jing)? ===")
    prefix = "P1(be"
    valid = get_valid_next(parser, prefix, vocabulary)
    has_close = ")" in valid
    continuation_candidates = [t for t in valid if t and t != ")" and t != "," and not t.isspace() and t not in ("{and}", "{or}")]
    print(f"  Prefix: {repr(prefix)}")
    print(f"    valid next (sample): {sorted(valid)[:25]}")
    print(f"    has ')': {has_close}, continuation-like (excl. ) , connectives): {continuation_candidates[:15]}")
    if not continuation_candidates and has_close:
        print("    -> Only ')' (and maybe ',') — cannot continue constant to e.g. 'beijing' with this vocabulary.")
    elif continuation_candidates:
        print("    -> Can continue constant (grammar allows it with this vocabulary).")

    # --- 4) Tokenizer: default (no FOL tokens) vs with FOL keywords added (production) ---
    print("\n=== Tokenizer: default vs with FOL keywords added (production) ===")
    try:
        from transformers import AutoTokenizer
        from evaluations.common.model_utils import FOL_KEYWORD_TOKENS

        model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Before: are FOL keywords single tokens?
        print("  Before add_tokens (default tokenizer):")
        for keyword in ["{and}", "{or}", "{not}", "%", " "]:
            ids = tok.encode(keyword, add_special_tokens=False)
            dec = [tok.decode([i]) for i in ids]
            print(f"    {repr(keyword):25} -> {len(ids)} token(s): {dec}")

        # Add FOL keywords the same way as production (create_huggingface_lm with add_fol_keyword_tokens=True)
        added = tok.add_tokens(FOL_KEYWORD_TOKENS, special_tokens=False)
        print(f"  add_tokens(FOL_KEYWORD_TOKENS): added {added} new token(s).")

        # After: recheck that FOL keywords are single tokens
        print("  After add_tokens (production tokenizer):")
        for keyword in FOL_KEYWORD_TOKENS:
            ids = tok.encode(keyword, add_special_tokens=False)
            dec = [tok.decode([i]) for i in ids]
            single = len(ids) == 1 and dec[0] == keyword
            print(f"    {repr(keyword):25} -> {len(ids)} token(s): {dec}  {'OK' if single else 'MISMATCH'}")

        # Valid next with vocab: mirror production (first vocab_size tokens + FOL keyword tokens)
        vocab_size = min(3000, tok.vocab_size if hasattr(tok, "vocab_size") else len(tok))
        token_list = [tok.decode([i]) for i in range(vocab_size)]
        for t in FOL_KEYWORD_TOKENS:
            tid = tok.convert_tokens_to_ids(t)
            if tid != getattr(tok, "unk_token_id", None):
                token_list.append(tok.decode([tid]))
        prefix_text = "Alkane(mixture)"
        valid_with_vocab = get_valid_next(parser, prefix_text, token_list)
        formula_like = [t for t in valid_with_vocab if t.strip() and t not in ("\n", "\t")]
        has_connective = any(c in " ".join(formula_like) for c in ["{and}", "{or}", "{not}"])
        print(f"  Valid next after '{prefix_text}' with production-like vocab: {len(valid_with_vocab)} total, {len(formula_like)} non-WS.")
        print(f"  Sample non-WS (first 25): {formula_like[:25]}")
        if has_connective or formula_like:
            print("  -> FOL keywords as single tokens allow formula extension (expected with add_fol_keyword_tokens=True).")
        else:
            print("  -> No formula-extending tokens in valid next (unexpected after add_tokens).")

        # P1(be: can we continue constant (e.g. to "beijing") or only ")"?
        prefix_inside = "P1(be"
        valid_inside = get_valid_next(parser, prefix_inside, token_list)
        close_only = [t for t in valid_inside if t.strip() in (")", ",")]
        continuation = [t for t in valid_inside if t.strip() and t not in (")", ",", "\n", "\t", " ")]
        continuation_constant = [t for t in continuation if t.isalpha() or (t and not any(c in t for c in "(){}$%"))]
        print(f"  Valid next after '{prefix_inside}' (production vocab): {len(valid_inside)} total.")
        print(f"    ')', ',' only: {len(close_only)}; other (continuation-like): {len(continuation_constant)} sample {continuation_constant[:12]}")
        if not continuation_constant and close_only:
            print("  -> Only ')' or ',' — constant cannot be continued to e.g. 'beijing' (tokenizer may split 'ijing' into sub-tokens not in valid set).")
        elif continuation_constant:
            print("  -> Can continue constant (e.g. toward 'beijing').")
    except Exception as e:
        print(f"  Skipped (install transformers?): {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
