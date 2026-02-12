"""
Parser creation utilities for Dafny-compatible grammar parsers.

Provides factory functions to create Lark-based parsers that conform to
the Dafny Parser interface used by CSD strategies.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


def create_lark_dafny_parser(
    grammar_source: str,
    VerifiedDecoderAgent,
    _dafny,
    start: str = "start"
):
    """
    Create a Dafny-compatible parser from a Lark grammar.
    
    This creates a parser class that implements the VerifiedDecoderAgent.Parser
    interface, allowing it to be used with CSD strategies compiled from Dafny.
    
    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name in the grammar
        
    Returns:
        A LarkDafnyParser class that can be instantiated with a token list
    """
    from lark import Lark
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
    
    # Load grammar - check if it's a file path (short string without newlines)
    if '\n' not in grammar_source and len(grammar_source) < 500:
        grammar_path = Path(grammar_source)
        if grammar_path.exists():
            grammar = grammar_path.read_text()
        else:
            grammar = grammar_source
    else:
        grammar = grammar_source
    
    # Create Lark parser
    lark_parser = Lark(grammar, start=start, parser='lalr')
    
    class LarkDafnyParser(VerifiedDecoderAgent.Parser):
        """Parser using Lark grammar, compatible with Dafny-compiled code."""

        def __init__(self, lm_tokens):
            super().__init__()
            self._lm_tokens = lm_tokens
            # Convert Dafny Seq to Python list using index-based access
            # (Dafny.Seq has __len__ and __getitem__ but NOT __iter__)
            try:
                self._token_list = list(lm_tokens)
            except TypeError:
                self._token_list = [lm_tokens[i] for i in range(len(lm_tokens))]
            self._lark = lark_parser
            self._grammar_source = grammar
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF
            # Cache the expr parser for performance
            self._expr_parser = None
            if grammar:
                try:
                    self._expr_parser = Lark(grammar, start='any_expr', parser='lalr')
                except Exception:
                    pass

            # FOL window start position for extracting FOL portion from full prefix
            # Set by generation code before CSD calls
            self._fol_window_start = 0

            # Detect if this is a MATH grammar vs FOL grammar
            # Use word boundaries to avoid substring matches (e.g., "quantified_expr" contains "n_expr")
            # Math grammars define rules like "n_expr:" or "s_expr:" at the start of a line
            # FOL grammars have {forall}/{exists} operators
            import re
            self._is_math_grammar = bool(re.search(r'\bn_expr\s*:', grammar) or
                                         re.search(r'\bs_expr\s*:', grammar) or
                                         re.search(r'\bn_term\s*:', grammar))
            self._is_fol_grammar = '{forall}' in grammar or '{exists}' in grammar

            import os
            self._debug = os.environ.get('CSD_MATH_DEBUG', '').lower() in ('1', 'true', 'yes')
            if self._debug:
                print(f"    [PARSER INIT] Grammar type: is_math={self._is_math_grammar}, is_fol={self._is_fol_grammar}")
        
        def _dafny_seq_to_str(self, seq) -> str:
            """Convert a Dafny Seq to a Python string.

            IMPORTANT: Dafny.Seq has __len__ and __getitem__ but NOT __iter__,
            so ''.join(seq) fails. We must use index-based iteration.
            """
            try:
                # First try direct join (works for Python strings/lists)
                return ''.join(seq)
            except TypeError:
                # Dafny.Seq doesn't support iteration - use index-based access
                try:
                    return ''.join(seq[i] for i in range(len(seq)))
                except (TypeError, AttributeError, IndexError):
                    # Last resort
                    return str(seq)
        
        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text.

            During CSD, the tokens passed here ARE the FOL tokens being generated
            (the CSD loop passes only 'generated' tokens, not the full LM context).
            So we simply convert all tokens to text without offset logic.
            """
            try:
                full_text = ''.join(self._dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                full_text = str(tokens)

            # NOTE: Removed _fol_window_start logic - during CSD, the tokens passed
            # are already just the FOL portion (from the CSD 'generated' sequence).
            # The offset was causing _tokens_to_text to return empty strings.

            return full_text
        
        def _expects_variable_next(self, text: str) -> bool:
            """Check if text ends in a state where a VARIABLE is expected next.

            This happens after quantifiers like '{forall}' or '{exists}'.
            """
            text = text.rstrip()
            # After a quantifier, we expect a variable
            for q in ["{forall}", "{exists}"]:
                if text.endswith(q):
                    return True
            return False

        def _expects_formula_start(self, text: str) -> bool:
            """Check if text ends in a state where a FORMULA START is expected.

            This happens after quantifier + variable (e.g., '{forall} x' or '{exists} y').
            In this state:
            - Predicates are valid (formula start)
            - '(' is valid (parenthesized formula)
            - {not} is valid (negated formula)
            - Binary operators ({and}, {or}, {implies}, etc.) are NOT valid
            - IMPORTANT: We DON'T allow {forall}/{exists} here to prevent
              nested quantifiers without formula bodies. The model should
              generate a formula like (Cat(x) {implies} Mammal(x)) first.
            """
            import re
            text = text.rstrip()
            # Pattern: quantifier followed by whitespace and a single variable letter
            # e.g., "{forall} x" or "{exists}y" or "{forall}  z"
            pattern = r'\{(forall|exists)\}\s*[a-z]$'
            return bool(re.search(pattern, text))

        def _is_inside_predicate_args(self, text: str) -> bool:
            """Check if we're inside predicate arguments (after PredicateName().

            Returns True if text ends with 'PredicateName(' or 'PredicateName(args,'
            where we're waiting for more arguments or close paren.

            Key insight: If '(' is preceded by an uppercase word (predicate name),
            we're in argument mode. If '(' is preceded by whitespace/operator/{not}/another (,
            it's formula grouping.
            """
            import re
            text = text.rstrip()
            if not text:
                return False

            # Find the last '(' and check what precedes it
            # We need to track unmatched parens to find if we're currently inside args
            paren_stack = []
            i = len(text) - 1

            while i >= 0:
                c = text[i]
                if c == ')':
                    paren_stack.append(')')
                elif c == '(':
                    if paren_stack:
                        paren_stack.pop()  # Matched a ')'
                    else:
                        # This is an unmatched '(' - check what precedes it
                        # Look backwards for the predicate name
                        before_paren = text[:i].rstrip()
                        if before_paren and re.search(r'[A-Z][a-zA-Z0-9]*$', before_paren):
                            # Preceded by uppercase word = predicate arguments
                            return True
                        else:
                            # Preceded by something else = formula grouping
                            return False
                i -= 1

            return False

        def _can_binary_operator_follow(self, text: str) -> bool:
            """Check if a binary operator can validly follow the current text.

            Binary operators ({and}, {or}, {xor}, {implies}, {iff}) can ONLY follow:
            - A closing paren ')'
            - A complete predicate call like 'Pred(x)' or 'Pred(x, y)'

            They CANNOT follow:
            - Empty/start of expression
            - '{forall}' or '{exists}' (expects variable)
            - '{forall} x' or '{exists} y' (expects formula body)
            - '(' (expects formula inside)
            - '{not}' (expects formula to negate)
            - ',' (expects argument)
            - Partial or complete binary operators (expects right operand)
            """
            text = text.rstrip()
            if not text:
                return False

            # Cannot follow quantifier (expects variable)
            if text.endswith('{forall}') or text.endswith('{exists}'):
                return False

            # Cannot follow quantifier+variable (expects formula body)
            if self._expects_formula_start(text):
                return False

            # Cannot follow '(' (expects formula inside)
            if text.endswith('('):
                return False

            # Cannot follow '{not}' (expects formula to negate)
            if text.endswith('{not}'):
                return False

            # Cannot follow ',' (expects argument)
            if text.endswith(','):
                return False

            # Cannot follow binary operators (expects right operand)
            binary_ops = ['{and}', '{or}', '{xor}', '{implies}', '{iff}']
            for op in binary_ops:
                if text.endswith(op):
                    return False

            # CAN follow ')' (closing a formula)
            if text.endswith(')'):
                return True

            # CAN follow a predicate call - check if we have 'Name(...)'
            # Simple heuristic: ends with ')' and has balanced parens with uppercase before
            import re
            # Match pattern like "Pred(x)" or "Pred(x, y)" at end
            if re.search(r'[A-Z][a-zA-Z0-9]*\([^()]*\)$', text):
                return True

            # For safety, if unclear, let the Lark parser decide
            return True

        def _strip_crane_delimiters(self, text: str) -> tuple:
            """Strip CRANE delimiters from text for FOL validation.

            CRANE paper format uses ::: to separate FOL from natural language description.
            FOL formula ends at ::: (or :: which might be partial :::).

            Returns:
                (stripped_text, has_end_marker) tuple
            """
            stripped = text
            has_end = False

            # CRANE format: FOL ends at ::: (separator before natural language)
            # Also check for :: which might be partial ::: or tokenizer split
            if ':::' in stripped:
                stripped = stripped.split(':::')[0].rstrip()
                has_end = True
            elif '::' in stripped:
                # Partial ::: - treat as end marker
                stripped = stripped.split('::')[0].rstrip()
                has_end = True
            elif stripped.rstrip().endswith(':'):
                # Single : at end might be start of :::
                # Don't mark as end yet, but be cautious
                pass

            # Legacy support: Strip trailing >> (end delimiter) if present
            if not has_end and stripped.rstrip().endswith('>>'):
                stripped = stripped.rstrip()[:-2].rstrip()
                has_end = True

            # Legacy support: Strip leading << (start delimiter) if present
            if stripped.lstrip().startswith('<<'):
                stripped = stripped.lstrip()[2:].lstrip()

            return stripped, has_end

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar.
            
            Uses an instance-level cache to avoid redundant Lark parses.
            """
            if not text:
                return True
                
            # Initialize cache if needed
            if not hasattr(self, '_prefix_validity_cache'):
                self._prefix_validity_cache = {}
                
            if text in self._prefix_validity_cache:
                return self._prefix_validity_cache[text]

            # For FOL grammars, handle CRANE format markers
            if self._is_fol_grammar:
                stripped, has_end = self._strip_crane_delimiters(text)
                if has_end:
                    # If we have ::: or >>, the FOL part before it should be complete
                    # Return True to allow this as valid (completion check is separate)
                    res = self._is_complete_fol(stripped) if stripped else False
                    self._prefix_validity_cache[text] = res
                    return res
                # Otherwise validate the stripped text
                text = stripped if stripped else text

            # Skip FOL handling for math grammars
            if not self._is_math_grammar:
                # Special handling for partial FOL operators
                # The Lark lexer rejects partial operators like "{for" even though
                # they're valid prefixes toward "{forall}". We handle this explicitly.

                # Operators that can START a formula (unary/quantifiers)
                start_operators = ["{forall}", "{exists}", "{not}"]
                # Binary operators that need something before them
                binary_operators = ["{and}", "{or}", "{xor}", "{implies}", "{iff}"]

                # Check if text ends with a partial FOL operator
                # First, check start operators (valid at beginning or after certain contexts)
                for op in start_operators:
                    for i in range(1, len(op)):
                        partial = op[:i]
                        if text.endswith(partial):
                            prefix_before = text[:-len(partial)]
                            if not prefix_before:
                                self._prefix_validity_cache[text] = True
                                return True  # Start operators are valid at the beginning
                            prefix_before_stripped = prefix_before.rstrip()
                            # NOT valid right after a quantifier (expects variable, not operator)
                            if self._expects_variable_next(prefix_before_stripped):
                                self._prefix_validity_cache[text] = False
                                return False
                            if self._is_valid_prefix(prefix_before_stripped):
                                self._prefix_validity_cache[text] = True
                                return True

                # For binary operators, check that they can validly follow
                for op in binary_operators:
                    for i in range(1, len(op)):
                        partial = op[:i]
                        if text.endswith(partial):
                            prefix_before = text[:-len(partial)].rstrip()
                            # Use the comprehensive check
                            if not self._can_binary_operator_follow(prefix_before):
                                self._prefix_validity_cache[text] = False
                                return False
                            # The prefix before must be a valid (possibly complete) formula
                            if self._is_valid_prefix(prefix_before):
                                self._prefix_validity_cache[text] = True
                                return True

            # The actual Lark parse logic
            try:
                self._lark.parse(text)
                res = True
            except self._UnexpectedEOF:
                res = True
            except self._UnexpectedToken as e:
                res = (e.token.type == '$END')
            except self._UnexpectedCharacters:
                res = False
            except Exception:
                res = False
                
            self._prefix_validity_cache[text] = res
            return res
        
        def _is_complete_fol(self, text: str) -> bool:
            """Check if text is a complete valid FOL parse (without delimiters)."""
            if not text:
                return False
            try:
                self._lark.parse(text)
                return True
            except Exception:
                return False

        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse.

            For CRANE format FOL:
            - Presence of ':::' signals end of FOL formula
            - Legacy: Presence of '>>' also signals completion
            """
            if not text:
                return False

            # For FOL grammars, handle CRANE format markers
            if self._is_fol_grammar:
                stripped, has_end = self._strip_crane_delimiters(text)
                if has_end:
                    # CRITICAL: If ::: or >> is present, the FOL region is DONE
                    # Return True to stop the CSD loop
                    return True
                # If no end marker, check the (possibly stripped) text
                text = stripped if stripped else text

            try:
                self._lark.parse(text)
                return True
            except Exception:
                return False
        
        def is_valid_prefix(self, text: str) -> bool:
            """Public method: Check if text is a valid prefix."""
            return self._is_valid_prefix(text)
        
        def is_complete(self, text: str) -> bool:
            """Public method: Check if text is complete."""
            return self._is_complete(text)
        
        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            if len(prefix) == 0:
                return True
            text = self._tokens_to_text(prefix)
            return self._is_valid_prefix(text)
        
        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete."""
            if len(prefix) == 0:
                return False
            text = self._tokens_to_text(prefix)
            return self._is_complete(text)
        
        def _get_partial_fol_op(self, text: str):
            """Check if text ends with a partial FOL operator and return continuation chars."""
            start_operators = ["{forall}", "{exists}", "{not}"]
            binary_operators = ["{and}", "{or}", "{xor}", "{implies}", "{iff}"]

            for op in start_operators + binary_operators:
                for i in range(1, len(op)):
                    partial = op[:i]
                    if text.endswith(partial):
                        return op[i]
            return None

        def _get_valid_fol_continuations(self, text: str):
            """Get all valid next characters when in the middle of an FOL operator.

            Context-aware: only allows operators that are grammatically valid at the position.
            - After quantifiers ({forall}, {exists}), we expect a VARIABLE, not another operator
            - Binary operators only valid after a formula
            """
            # Operators that can START a formula
            start_operators = ["{forall}", "{exists}", "{not}"]
            # Binary operators that need something before them
            binary_operators = ["{and}", "{or}", "{xor}", "{implies}", "{iff}"]

            continuations = set()

            # Check start operators (valid at beginning or certain contexts)
            for op in start_operators:
                for i in range(1, len(op)):
                    partial = op[:i]
                    if text.endswith(partial):
                        prefix_before = text[:-len(partial)]
                        if not prefix_before:
                            # At beginning - start operators are valid
                            continuations.add(op[i])
                        else:
                            prefix_before_stripped = prefix_before.rstrip()
                            # NOT valid right after a quantifier (expects variable)
                            if self._expects_variable_next(prefix_before_stripped):
                                continue
                            if self._is_valid_prefix(prefix_before_stripped):
                                continuations.add(op[i])

            # Check binary operators (only valid after a complete formula)
            for op in binary_operators:
                for i in range(1, len(op)):
                    partial = op[:i]
                    if text.endswith(partial):
                        prefix_before = text[:-len(partial)].rstrip()
                        # Use the comprehensive check
                        if not self._can_binary_operator_follow(prefix_before):
                            continue
                        if self._is_valid_prefix(prefix_before):
                            continuations.add(op[i])

            return continuations

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens.

            Dispatches to grammar-specific implementation.
            FOL grammar takes precedence if detected (more specific validation).
            """
            import os
            debug = os.environ.get('CSD_FOL_DEBUG', '').lower() in ('1', 'true', 'yes')

            if debug:
                print(f"    [PARSER] ValidNextTokens: is_math={self._is_math_grammar}, is_fol={self._is_fol_grammar}")

            # FOL grammar takes precedence - it has more specific validation
            if self._is_fol_grammar:
                return self._valid_next_tokens_fol(prefix)
            elif self._is_math_grammar:
                return self._valid_next_tokens_math(prefix)
            else:
                # Fallback to FOL for unknown grammars
                return self._valid_next_tokens_fol(prefix)

        def _valid_next_tokens_math(self, prefix):
            """Get valid next tokens for MATH grammars (GSM-Symbolic).

            Optimized version that pre-filters tokens to avoid expensive Lark calls.
            """
            import re
            import os

            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            
            if current_text and not self._is_valid_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])

            debug = self._debug
            if debug:
                print(f"    [MATH] ValidNextTokens for prefix: '{current_text}'")
            
            valid_tokens = []
            checked_count = 0
            
            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str: continue
                
                # Final grammar check
                checked_count += 1
                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    valid_tokens.append(token)

            if debug:
                print(f"    [MATH] Checked {checked_count} tokens against Lark. Valid: {len(valid_tokens)}")

            return _dafny.SeqWithoutIsStrInference(valid_tokens)

        def _valid_next_tokens_fol(self, prefix):
            """Get valid next tokens for FOL grammars (FOLIO).

            Uses the grammar to determine which tokens can validly follow the current prefix.
            Special handling for partial FOL operators to ensure proper operator completion.
            """
            import os
            debug_fol = os.environ.get('CSD_FOL_DEBUG', '').lower() in ('1', 'true', 'yes')

            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

            if debug_fol:
                print(f"    [FOL PARSER] ValidNextTokens called, current_text={repr(current_text[:50])}{'...' if len(current_text) > 50 else ''}")

            # CRITICAL: Check if text already contains ::: (CRANE end marker)
            # If so, the FOL expression is complete - no more tokens should be added
            stripped, has_end = self._strip_crane_delimiters(current_text)
            if has_end:
                if debug_fol:
                    print(f"    [FOL PARSER] Text contains :::, FOL complete, returning empty")
                return _dafny.SeqWithoutIsStrInference([])

            # Use stripped text (without delimiters) for validation
            validation_text = stripped if stripped else current_text

            # Special case: text ends with a quantifier like {forall} or {exists}
            # This is ALWAYS a valid prefix that expects a variable next
            # Don't reject it even if Lark parsing fails (LALR parser limitation)
            ends_with_quantifier = validation_text.rstrip().endswith('{forall}') or \
                                   validation_text.rstrip().endswith('{exists}')

            # Check if we're inside predicate arguments (after PredicateName()
            # This is DIFFERENT from formula grouping with ()
            inside_predicate_args = self._is_inside_predicate_args(validation_text)

            # Special case: text ends with '(' for FORMULA GROUPING (not predicate args)
            # Lark LALR parser may not handle this intermediate state correctly
            ends_with_open_paren = validation_text.rstrip().endswith('(') and not inside_predicate_args

            # Special case: text ends with '{not}' - always valid prefix expecting formula
            ends_with_not = validation_text.rstrip().endswith('{not}')

            # Special case: text ends with quantifier + variable (e.g., '{forall} x')
            # This expects a formula body - also a valid intermediate state
            # This also covers cases like '{not} {forall} x' which Lark can't parse
            expects_formula_body = self._expects_formula_start(validation_text.rstrip())

            # Skip validation for known valid intermediate states
            # BUT: Don't skip validation when inside predicate arguments
            skip_validation = (ends_with_quantifier or ends_with_open_paren or ends_with_not or expects_formula_body) and not inside_predicate_args

            if debug_fol:
                print(f"    [FOL PARSER] expects_formula_body={expects_formula_body}, inside_pred_args={inside_predicate_args}, skip_validation={skip_validation}")

            if validation_text and not skip_validation and not self._is_valid_prefix(validation_text):
                if debug_fol:
                    print(f"    [FOL PARSER] Current text is INVALID prefix, returning empty")
                return _dafny.SeqWithoutIsStrInference([])

            # Check if we expect a variable next (after quantifiers like {forall} or {exists})
            expects_variable = self._expects_variable_next(current_text.rstrip())

            # Check if we're in the middle of building an FOL operator
            fol_continuations = self._get_valid_fol_continuations(current_text)

            # Calculate current parenthesis nesting depth for sanity limits
            MAX_DEPTH = 6  # Maximum allowed nesting depth
            depth = 0
            for char in current_text:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1

            depth_limit_reached = depth >= MAX_DEPTH

            # Expression length limit - count operators as proxy for expression complexity
            MAX_EXPR_TOKENS = 15  # Maximum tokens before forcing closure
            MAX_OPERATORS = 8     # Maximum binary operators before forcing closure
            operator_count = sum(1 for c in current_text if c in '+-*/')
            token_count = len(prefix)

            # Check for repetitive patterns (e.g., "* 1 * 1 * 1" or "CarCarCar" or "{not}{not}{not}")
            # NOTE: Tightened to reduce false positives - require more significant repetition
            has_repetition = False
            if len(current_text) >= 15:  # Only check for longer texts
                window = current_text[-30:]  # Shorter window
                import re

                # Check for any 3-6 char sequence repeated 4+ times (tightened from 3+)
                for seq_len in range(3, 7):  # Start from 3, not 2
                    if len(window) >= seq_len * 4:  # Require 4 repetitions
                        for start in range(len(window) - seq_len * 4 + 1):
                            seq = window[start:start + seq_len]
                            if seq.isalpha() and window.count(seq) >= 4:  # 4+ times
                                has_repetition = True
                                break
                    if has_repetition:
                        break

                # CRITICAL: Check for repeated FOL operators (e.g., "{not} {not} {not}")
                # These won't be caught by isalpha() check above
                if not has_repetition:
                    fol_op_patterns = ['{not}', '{and}', '{or}', '{forall}', '{exists}', '{implies}']
                    for op in fol_op_patterns:
                        # Count how many times this operator appears in the window
                        op_count = window.count(op)
                        if op_count >= 3:  # 3+ consecutive operators is likely a loop
                            has_repetition = True
                            break

                # Check if current text is mostly one repeated pattern
                if not has_repetition and len(current_text) >= 20:  # Increased threshold
                    # If 60%+ of text is the same 3-6 char pattern, it's repetitive
                    for plen in range(3, 7):
                        if len(current_text) >= plen:
                            pattern = current_text[-plen:]
                            count = current_text.count(pattern)
                            if count * plen >= len(current_text) * 0.6:  # Tightened
                                has_repetition = True
                                break

            # For FOL: check for FOL operators instead of math operators
            fol_ops = ['{and}', '{or}', '{xor}', '{not}', '{implies}', '{iff}', '{forall}', '{exists}']
            has_fol_operator = any(op in current_text for op in fol_ops)
            has_variable = any(c.isalpha() for c in current_text)
            MIN_EXPR_LENGTH = 1
            meets_min_complexity = has_variable and len(current_text.strip()) >= MIN_EXPR_LENGTH

            valid_tokens = []
            gtgt_tokens = []

            # For FOL, we want to allow these keywords as part of operators
            # but ban Python/programming keywords that shouldn't appear in FOL
            BANNED_KEYWORDS = {
                # Python keywords
                'if', 'else', 'elif', 'while', 'in', 'is', 'for', 'For',
                'True', 'False', 'None', 'def', 'class', 'return', 'import', 'lambda',
                'try', 'except', 'finally', 'with', 'as', 'from', 'global', 'nonlocal',
                'assert', 'break', 'continue', 'del', 'pass', 'raise', 'yield',
                # Common programming words that aren't FOL
                'result', 'Result', 'let', 'Let', 'var', 'Var', 'val', 'Val',
                'function', 'Function', 'func', 'Func', 'fn', 'Fn',
                'int', 'Int', 'str', 'Str', 'bool', 'Bool', 'float', 'Float',
                'list', 'List', 'dict', 'Dict', 'set', 'Set', 'tuple', 'Tuple',
                'self', 'Self', 'this', 'This', 'new', 'New',
                'print', 'Print', 'input', 'Input', 'output', 'Output',
                # Common garbage tokens from model
                'get', 'Get', 'set', 'Set', 'put', 'Put', 'add', 'Add',
                'Car', 'car', 'Fort', 'fort', 'Impl', 'impl',
                'flies', 'Flies', 'Carse', 'carse', 'Lese', 'lese',
                'se', 'Se', 'le', 'Le', 're', 'Re', 'de', 'De',
                'the', 'The', 'an', 'An', 'is', 'Is', 'are', 'Are',
                'has', 'Has', 'have', 'Have', 'was', 'Was', 'were', 'Were',
                'be', 'Be', 'been', 'Been', 'being', 'Being',
                'do', 'Do', 'does', 'Does', 'did', 'Did', 'done', 'Done',
                'will', 'Will', 'would', 'Would', 'could', 'Could', 'should', 'Should',
                'can', 'Can', 'may', 'May', 'might', 'Might', 'must', 'Must',
            }
            BANNED_OPERATORS = {'**', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '='}

            # Special tokens that should NEVER appear in FOL output
            # These are tokenizer/model control tokens, not valid FOL
            SPECIAL_TOKENS = {
                # Common special tokens
                '[BEGIN_OF_TEXT]', '[END_OF_TEXT]', '<|begin_of_text|>', '<|end_of_text|>',
                '<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|pad|>',
                '<EOS>', '</s>', '<s>', '<pad>', '[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]',
                '<unk>', '<bos>', '<eos>', '\x00',
                # CSD delimiters - these are meta-markers, not FOL content
                '>>', '<<',
            }

            # Characters that are NOT valid in FOL grammar
            # FOL allows: letters, digits, {, }, (, ), comma, whitespace
            # Note: '.' (period) is also invalid in FOL expressions
            INVALID_FOL_CHARS = set('./*+-=!<>?@#$%^&~:;"\'|[]`_')

            # Check if binary operators can follow at current position
            can_binary_follow = self._can_binary_operator_follow(current_text.rstrip())

            # CRITICAL: Check if we're in a position where a FORMULA BODY is expected
            # This happens after {forall} x or {exists} y
            # In this state, we should NOT allow another quantifier to prevent
            # nested quantifiers without formula bodies (which causes garbage generation)
            # Use the already-computed value from skip_validation check
            formula_body_expected = expects_formula_body

            # If we're in the middle of building an FOL operator, prioritize continuation tokens
            fol_continuation_tokens = []

            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    continue

                stripped = token_str.strip()

                # CRITICAL: Filter out special tokens (tokenizer control tokens, CSD delimiters)
                # These should NEVER appear in FOL output
                if stripped in SPECIAL_TOKENS:
                    continue
                # Also check if any special token is contained within the token
                # (handles cases like " >>" or "\n[BEGIN_OF_TEXT]")
                has_special = any(st in token_str for st in SPECIAL_TOKENS if len(st) > 1)
                if has_special:
                    continue

                # Handle whitespace-only tokens
                # When formula_body_expected=True, we need whitespace to separate variable from predicate.
                # BUT: _expects_formula_start() uses rstrip(), so adding whitespace doesn't change state.
                # FIX: Allow ONE whitespace token (when no trailing space exists), then block more.
                if not stripped:
                    if formula_body_expected:
                        # Only allow whitespace if current text doesn't already end with whitespace
                        current_ends_with_ws = current_text and current_text[-1] in ' \t\n'
                        if not current_ends_with_ws:
                            valid_tokens.append(token)
                    continue

                # Reject banned keywords and operators
                if stripped in BANNED_KEYWORDS or stripped in BANNED_OPERATORS:
                    continue

                # CRITICAL: When inside predicate arguments (after PredicateName(),
                # only allow variables, constants, commas, and close paren.
                # This prevents FOL operators from being inserted into predicate args.
                if inside_predicate_args:
                    first_char = stripped[0] if stripped else ''

                    # Allow close paren to end arguments
                    if stripped == ')' or first_char == ')':
                        extended = current_text + token_str
                        if self._is_valid_prefix(extended):
                            valid_tokens.append(token)
                        continue

                    # Allow comma for argument separator
                    if stripped == ',' or first_char == ',':
                        extended = current_text + token_str
                        if self._is_valid_prefix(extended):
                            valid_tokens.append(token)
                        continue

                    # Allow lowercase (variables and constants)
                    if first_char.islower():
                        extended = current_text + token_str
                        if self._is_valid_prefix(extended):
                            valid_tokens.append(token)
                        continue

                    # Allow whitespace for spacing
                    if first_char.isspace():
                        extended = current_text + token_str
                        if self._is_valid_prefix(extended):
                            valid_tokens.append(token)
                        continue

                    # Block everything else (FOL operators, uppercase, etc.)
                    continue

                # CRITICAL: When a formula body is expected (after {forall} x),
                # We need proper separation from the variable - require whitespace or (
                # This prevents "xPredicate" concatenation and nested quantifiers
                if formula_body_expected:
                    first_raw_char = token_str[0] if token_str else ''

                    # BLOCK ::: tokens - can't end the expression without a formula body!
                    if ":::" in token_str or "::" in token_str:
                        continue

                    # Block tokens that would start a new quantifier
                    if stripped == '{' or stripped.startswith('{f') or stripped.startswith('{e'):
                        continue
                    # Block complete quantifier operators
                    if stripped in ['{forall}', '{exists}', ' {forall}', ' {exists}']:
                        continue
                    # Block tokens starting with space + { (unless it's {not})
                    if token_str.strip().startswith('{') and 'not' not in token_str:
                        if 'forall' in token_str or 'exists' in token_str or token_str.strip() == '{':
                            continue

                    # SPACING REQUIREMENT: After variable, we need separation.
                    # If current_text already ends with whitespace, predicates can follow directly.
                    # Otherwise, token must start with whitespace or (.
                    current_ends_with_space = current_text and current_text[-1] in ' \t\n'
                    if not current_ends_with_space:
                        # No trailing space - token must provide separation
                        if first_raw_char not in ' \t\n(' and stripped:
                            if debug_fol and token_str.startswith('('):
                                print(f"    [FOL PARSER] Spacing filter BLOCKING paren token {repr(token_str)}")
                            continue
                    # If current_text ends with space, allow predicates and ( directly

                # WHITELIST approach: At the START of FOL expression, only allow valid starters
                # Valid FOL expression starters: complete/partial FOL operators, single (, or short Predicate names
                is_at_start = not current_text.strip()
                if is_at_start and stripped:
                    # Must start with valid FOL operator prefix, single ( (for parens), or short uppercase (predicate)
                    first_char = stripped[0]
                    if first_char == '{':
                        # CRITICAL: Don't allow lone '{' - it leads to "{ { { { {" garbage
                        # Only allow tokens that are building valid FOL operators
                        valid_fol_prefixes = [
                            '{forall}', '{exists}', '{not}', '{and}', '{or}', '{xor}', '{implies}', '{iff}',
                            '{forall', '{exist', '{exi', '{ex', '{e',  # partial {exists}
                            '{foral', '{fora', '{for', '{fo', '{f',    # partial {forall}
                            '{no', '{n',                               # partial {not}
                        ]
                        if stripped not in valid_fol_prefixes and not any(stripped.startswith(p) for p in valid_fol_prefixes):
                            continue  # Reject - not a valid FOL operator start
                    elif stripped == '(':
                        pass  # OK - single open paren for parenthesized formula
                    elif first_char == '(' and len(stripped) > 1:
                        # Reject combined tokens like "(Get" - only allow single "("
                        continue
                    elif first_char.isupper():
                        # Only allow SHORT predicate names (1-4 chars) or tokens ending with (
                        # This prevents English words like "Every", "Some", "There" from being accepted
                        # Real predicates are typically short: Cat, Dog, P, Q, Mammal, etc.
                        if stripped.endswith('('):
                            pass  # OK - predicate with paren like "Cat("
                        elif len(stripped) <= 4 and stripped.isalpha():
                            pass  # OK - short predicate name like "Cat", "P", "Dog"
                        elif len(stripped) == 1:
                            pass  # OK - single letter like "P", "Q"
                        else:
                            # Reject longer words without ( - likely English words like "Every", "Some"
                            continue
                    else:
                        continue  # Reject lowercase/numeric starts

                # Prevent token repetition - if current_text ends with this token repeated,
                # don't allow it again (prevents "ForForForFor..." loops)
                if stripped and len(stripped) >= 2 and len(current_text) >= len(stripped) * 2:
                    # Check if this token already appears repeated at the end
                    repeat_check = stripped + stripped
                    if current_text.rstrip().endswith(repeat_check):
                        continue  # Don't allow third repetition

                # Reject tokens containing invalid FOL characters
                # (but allow ::: which is the CRANE end marker)
                if ":::" not in token_str and "::" not in token_str:
                    has_invalid = any(c in INVALID_FOL_CHARS for c in token_str)
                    if has_invalid:
                        continue

                # Limit excessive whitespace - don't allow whitespace-only tokens
                # if we already have trailing whitespace
                if not stripped and current_text.endswith(' '):
                    # Count trailing spaces
                    trailing_spaces = len(current_text) - len(current_text.rstrip())
                    if trailing_spaces >= 2:
                        continue  # Don't add more whitespace

                # Sanity limit: If depth limit reached, exclude opening parentheses
                if depth_limit_reached and '(' in token_str and ')' not in token_str:
                    continue

                # Limit consecutive open parens - prevent "( ( ( ( (" patterns
                # After 2 consecutive open parens, block more open parens
                consecutive_parens = 0
                for c in reversed(current_text):
                    if c == '(':
                        consecutive_parens += 1
                    elif not c.isspace():
                        break
                if consecutive_parens >= 2 and stripped == '(':
                    continue

                # Block quantifiers immediately after ( - should be predicate or {not}
                # This prevents patterns like "( {forall}" which lead to deeply nested quantifiers
                # But allow {not} which is valid for negated formulas like "({not} P(x))"
                # ALSO check for "( {" pattern - when we're building a quantifier after (
                import re
                # Check if we're right after an open paren (with optional whitespace)
                recently_opened_paren = bool(re.search(r'\(\s*$', current_text)) or \
                                       bool(re.search(r'\(\s*\{$', current_text))  # "( {" pattern
                
                if ends_with_open_paren or recently_opened_paren:
                    # Block quantifier starters (but not {not})
                    if stripped.startswith('{f') or stripped.startswith('{e'):
                        continue
                    if stripped in ['{forall}', '{exists}', ' {forall}', ' {exists}']:
                        continue
                    # Block tokens that continue quantifiers like 'forall' or 'exists'
                    if current_text.rstrip().endswith('{'):
                        if stripped in ['forall', 'exists', 'forall}', 'exists}']:
                            continue
                    # Block lone '{' after ( since it likely starts a quantifier
                    # (predicates don't start with {)
                    if ends_with_open_paren and stripped == '{':
                        continue
                    # CRITICAL: Block lone lowercase variables/letters after (
                    # After ( in formula context, we expect Predicates (uppercase) or {not}
                    # Not bare variables like 'x', 'y', etc.
                    if ends_with_open_paren and stripped and len(stripped) == 1 and stripped.islower():
                        continue
                    # Also block ' x' style tokens (space + single lowercase)
                    if ends_with_open_paren and token_str.strip() and len(token_str.strip()) == 1 and token_str.strip().islower():
                        continue

                # Don't allow ::: until we have at least a minimal expression AND
                # we're not waiting for a formula body (after {not}, {and}, quantifier+var, etc.)
                if ":::" in token_str or "::" in token_str:
                    # Block if we don't meet minimum complexity
                    if not meets_min_complexity:
                        continue
                    # Block if we're expecting a formula start (incomplete expression)
                    if formula_body_expected:
                        continue
                    # Block if we're right after an operator like {and}, {or}, {implies}, {not}
                    # Also block after incomplete quantifiers {forall}, {exists} without variable
                    text_stripped = current_text.rstrip()
                    ends_with_incomplete = any(text_stripped.endswith(op) for op in 
                        ['{and}', '{or}', '{implies}', '{iff}', '{xor}', '{not}',
                         '{forall}', '{exists}'])  # Added quantifiers
                    if ends_with_incomplete:
                        # #region agent log
                        import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:block_delimiter", "message": "Blocking ::: after incomplete operator", "data": {"text_end": text_stripped[-20:], "token": token_str}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H5"}) + '\n')
                        # #endregion
                        continue
                    # Block if text ends with incomplete predicate (uppercase word not followed by '(')
                    # e.g. "Mammal" or "M" without "(args)"
                    import re
                    incomplete_pred = re.search(r'[A-Z][a-zA-Z0-9]*\s*$', text_stripped)
                    if incomplete_pred and not text_stripped.rstrip().endswith(')'):
                        # #region agent log
                        import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:block_delimiter_pred", "message": "Blocking ::: after incomplete predicate", "data": {"text_end": text_stripped[-20:], "token": token_str, "match": incomplete_pred.group()}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H6"}) + '\n')
                        # #endregion
                        continue
                    # Block if we have unbalanced parentheses
                    open_parens = current_text.count('(') - current_text.count(')')
                    if open_parens > 0:
                        continue
                    # Block if we have unbalanced braces (partial FOL operator)
                    open_braces = current_text.count('{') - current_text.count('}')
                    if open_braces > 0:
                        continue

                # Special handling: if we're building an FOL operator, only allow continuations
                if fol_continuations:
                    # Check if this token can continue the partial operator
                    # Handle single chars, tokens with leading space, or multi-char continuations
                    token_content = token_str.lstrip()  # Remove leading whitespace
                    if token_content and token_content[0] in fol_continuations:
                        # Verify that adding this token creates a valid partial operator
                        extended = current_text + token_str
                        # Check if extended text still looks like a valid partial FOL operator
                        valid_partials = [
                            '{f', '{fo', '{for', '{fora', '{foral', '{forall', '{forall}',
                            '{e', '{ex', '{exi', '{exis', '{exist', '{exists', '{exists}',
                            '{n', '{no', '{not', '{not}',
                            '{a', '{an', '{and', '{and}',
                            '{o', '{or', '{or}',
                            '{x', '{xo', '{xor', '{xor}',
                            '{i', '{if', '{iff', '{iff}', '{im', '{imp', '{impl', '{impli', '{implie', '{implies', '{implies}',
                        ]
                        if any(extended.rstrip().endswith(p) for p in valid_partials):
                            fol_continuation_tokens.append(token)
                    continue  # Skip normal validation when in FOL operator mode

                # Special handling: when a VARIABLE is expected (after quantifiers)
                # Only allow single lowercase letters or whitespace
                if expects_variable:
                    # Don't allow FOL operators to start (they begin with '{')
                    if '{' in token_str:
                        continue
                    # Don't allow ::: when variable is expected
                    if ':::' in token_str or '::' in token_str:
                        continue
                    # Allow whitespace tokens
                    if not stripped:
                        valid_tokens.append(token)
                        continue
                    # Allow tokens that contain single lowercase letters (variables)
                    # The stripped content should be a single lowercase letter
                    if len(stripped) == 1 and stripped.islower():
                        valid_tokens.append(token)
                        continue
                    # Also allow tokens like " x" where the alpha part is a single lowercase
                    alpha_chars = [c for c in token_str if c.isalpha()]
                    if len(alpha_chars) == 1 and alpha_chars[0].islower():
                        # Don't need to verify with _is_valid_prefix - we know a variable is expected
                        # and this token contains exactly one lowercase letter
                        valid_tokens.append(token)
                    continue  # Skip normal validation when expecting variable

                # Special handling: reject binary operators when they can't follow
                # Binary operators can only follow complete formulas like Pred(x) or )
                if not can_binary_follow:
                    # Check if this token starts a binary operator
                    binary_op_starts = ['{a', '{o', '{x', '{i']  # {and}, {or}, {xor}, {implies}, {iff}
                    is_binary_op_start = any(stripped.startswith(bos) for bos in binary_op_starts)
                    if is_binary_op_start:
                        continue
                    # Also reject complete binary operators
                    binary_ops = ['{and}', '{or}', '{xor}', '{implies}', '{iff}']
                    if stripped in binary_ops:
                        continue

                # Special handling: if current text ends with a partial predicate name
                # (uppercase word without '('), only allow continuing the name or '('
                import re
                partial_pred_match = re.search(r'[A-Z][a-zA-Z0-9]*$', current_text.rstrip())
                if partial_pred_match:
                    partial_name = partial_pred_match.group()
                    MAX_PREDICATE_LENGTH = 12  # Max chars for predicate names

                    # CRITICAL: Block FOL keywords from being added to predicate names
                    # This prevents garbage like "Mforall" or "Catand" or "Personexists"
                    FOL_KEYWORD_PARTS = {'forall', 'exists', 'and', 'or', 'xor', 'not', 'implies', 'iff'}
                    token_lower = stripped.lower()
                    if token_lower in FOL_KEYWORD_PARTS:
                        # #region agent log
                        import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:block_fol_in_pred", "message": "Blocking FOL keyword in predicate name", "data": {"partial_name": partial_name, "blocked_token": stripped}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H_PRED_FIX"}) + '\n')
                        # #endregion
                        continue  # Reject - can't add FOL keyword to predicate name

                    # CRITICAL: If predicate name is already too long, ONLY allow '('
                    if len(partial_name) >= MAX_PREDICATE_LENGTH:
                        if stripped and stripped[0] == '(':
                            pass  # OK - must open arguments now
                        else:
                            continue  # Reject - predicate name is too long

                    # Only allow: more alphanumeric chars OR '('
                    if stripped:
                        first_char = stripped[0]
                        # Allow '(' to start predicate arguments
                        if first_char == '(':
                            pass  # OK
                        # Allow alphanumeric to continue predicate name (if not too long)
                        elif first_char.isalnum():
                            # Check if this would make the name too long
                            potential_name = partial_name + stripped
                            if len(potential_name) > MAX_PREDICATE_LENGTH:
                                continue  # Reject - would make predicate too long
                            pass  # OK
                        else:
                            continue  # Reject - can't have other chars after partial predicate
                    # Also reject if this would create a repeated predicate name
                    if stripped and stripped[0].isupper() and not stripped.startswith('('):
                        # This looks like starting a new predicate - reject if it would repeat
                        if current_text.rstrip().endswith(stripped):
                            continue

                extended = current_text + token_str

                # Special case: after '(' or '{not}' we should allow formula-starting tokens
                # without strict Lark validation (LALR parser has issues with intermediate states)
                if ends_with_open_paren or ends_with_not:
                    # After '(' or '{not}', allow: predicates, complete FOL operators, another (
                    first_char = stripped[0] if stripped else ''
                    if first_char.isupper():
                        # Predicate name - allow it
                        valid_tokens.append(token)
                        continue
                    elif first_char == '(':
                        # Nested paren or paren for formula - allow it
                        valid_tokens.append(token)
                        continue
                    elif first_char == '{':
                        # Only allow COMPLETE FOL operators, not partial ones like '{' or ' {'
                        # This prevents chains like "{not} {not} {not}" from partial operators
                        complete_ops = ['{not}', '{forall}', '{exists}']
                        if stripped in complete_ops:
                            valid_tokens.append(token)
                            continue
                        # Also allow space-prefixed complete operators
                        if stripped.lstrip() in complete_ops and token_str[0].isspace():
                            valid_tokens.append(token)
                            continue
                        # Don't allow partial operators - fall through to normal validation
                    elif first_char.islower() and len(stripped) == 1:
                        # Single variable (rare but valid in some contexts)
                        valid_tokens.append(token)
                        continue
                    # For other tokens, fall through to normal validation

                # Special case: when formula body is expected (after {forall} x or {exists} y)
                # Allow predicates and ( without strict Lark validation
                # Note: quantifiers are already blocked above when formula_body_expected is True
                if formula_body_expected:
                    first_char = stripped[0] if stripped else ''
                    first_raw_char = token_str[0] if token_str else ''
                    # Check if current_text already provides separation
                    has_trailing_space = current_text and current_text[-1] in ' \t\n'

                    # Case 1: Token starts with ( - always valid (parenthesized formula)
                    if first_raw_char == '(':
                        if debug_fol:
                            print(f"    [FOL PARSER] formula_body_expected: Adding paren token {repr(token_str)}")
                        valid_tokens.append(token)
                        continue

                    # Case 2: Token is pure whitespace - SKIP it
                    # Whitespace-only tokens cause infinite loops because _expects_formula_start()
                    # uses rstrip() before pattern matching. Formula body tokens should include
                    # their own leading whitespace if needed.
                    if not stripped:
                        continue

                    # Case 3: Token starts with whitespace - check what follows
                    if first_raw_char.isspace():
                        if first_char == '(':
                            # Space + ( - valid
                            valid_tokens.append(token)
                            continue
                        elif first_char.isupper():
                            # Space + Predicate - valid
                            valid_tokens.append(token)
                            continue
                        elif first_char == '{' and 'not' in stripped:
                            # Space + {not} - valid
                            valid_tokens.append(token)
                            continue

                    # Case 4: If current_text already ends with space, allow predicates directly
                    # This handles cases like "( {forall} x" where the ` x` token has leading space
                    if has_trailing_space:
                        if first_char.isupper():
                            # Predicate without leading space is OK when there's trailing space
                            valid_tokens.append(token)
                            continue
                        elif first_char == '{' and 'not' in stripped:
                            # {not} without leading space is OK when there's trailing space
                            valid_tokens.append(token)
                            continue

                    # Reject other tokens - they would create invalid concatenation like "xPredicate"
                    if debug_fol and len(valid_tokens) < 5:
                        print(f"    [FOL PARSER] formula_body_expected: REJECTING token {repr(token_str)}, first_raw={repr(first_raw_char)}, first={repr(first_char)}, has_space={has_trailing_space}")
                    continue

                if self._is_valid_prefix(extended):
                    # Separate ::: tokens so we can prioritize them when expression is complete
                    if ":::" in token_str or "::" in token_str:
                        # #region agent log
                        import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:gtgt_added", "message": "::: token added to gtgt_tokens", "data": {"text_end": current_text[-30:] if current_text else "", "token": token_str}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H7"}) + '\n')
                        # #endregion
                        gtgt_tokens.append(token)
                    elif token_str.strip():  # Non-whitespace
                        valid_tokens.append(token)

            # If we're building an FOL operator, return only the continuation tokens
            if fol_continuations and fol_continuation_tokens:
                if debug_fol:
                    print(f"    [FOL PARSER] Returning FOL continuations: {[self._dafny_seq_to_str(t) for t in fol_continuation_tokens]}")
                return _dafny.SeqWithoutIsStrInference(fol_continuation_tokens)

            if debug_fol:
                print(f"    [FOL PARSER] After token loop: valid_tokens={len(valid_tokens)}, gtgt_tokens={len(gtgt_tokens)}")

            # Check if current text looks like garbage (no FOL structure at all)
            has_fol_structure = any(c in current_text for c in '{}()')
            is_garbage = (
                len(current_text.strip()) > 10 and
                not has_fol_structure and
                not has_fol_operator
            )

            # CRITICAL: Force closure when expression is too long, has repetition, or is garbage
            # BUT: Never force closure when formula_body_expected - we need a complete formula first!
            should_force_closure = (
                meets_min_complexity and  # Have a meaningful expression
                not formula_body_expected and  # NOT waiting for a formula body
                (token_count >= MAX_EXPR_TOKENS or  # Too many tokens
                 operator_count >= MAX_OPERATORS or  # Too many operators
                 has_repetition or  # Detected repetitive pattern
                 is_garbage)  # Text has no FOL structure, just garbage
            )

            if should_force_closure:
                if debug_fol:
                    print(f"    [FOL PARSER] FORCE CLOSURE triggered! token_count={token_count}, has_repetition={has_repetition}, is_garbage={is_garbage}")
                    print(f"    [FOL PARSER] valid_tokens has {len(valid_tokens)} items, gtgt_tokens has {len(gtgt_tokens)} items")
                
                # CRITICAL: Before force closure, check if expression is actually incomplete
                # Don't return ::: if expression is clearly unfinished
                text_stripped_fc = current_text.rstrip()
                has_incomplete_pred_fc = bool(re.search(r'[A-Z][a-zA-Z0-9]*\s*$', text_stripped_fc)) and not text_stripped_fc.endswith(')')
                has_incomplete_op_fc = any(text_stripped_fc.endswith(op) for op in 
                    ['{and}', '{or}', '{implies}', '{iff}', '{xor}', '{not}', '{forall}', '{exists}'])
                open_parens_fc = current_text.count('(') - current_text.count(')')
                open_braces_fc = current_text.count('{') - current_text.count('}')
                is_incomplete_fc = has_incomplete_pred_fc or has_incomplete_op_fc or open_parens_fc > 0 or open_braces_fc > 0 or formula_body_expected
                
                # #region agent log
                import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:force_closure", "message": "Force closure triggered", "data": {"text_end": current_text[-30:] if current_text else "", "token_count": token_count, "has_repetition": has_repetition, "is_garbage": is_garbage, "gtgt_count": len(gtgt_tokens), "is_incomplete": is_incomplete_fc}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H8"}) + '\n')
                # #endregion
                
                # If expression is incomplete, DON'T force :::
                # Instead, return empty to let CSD exit gracefully
                if is_incomplete_fc:
                    return _dafny.SeqWithoutIsStrInference([])
                
                # Try to force ::: only when expression is complete
                if gtgt_tokens:
                    return _dafny.SeqWithoutIsStrInference(gtgt_tokens)
                else:
                    # Find any ::: token in the vocabulary and return it
                    # This is an emergency bail-out for garbage generation
                    for token in self._token_list:
                        token_str = self._dafny_seq_to_str(token)
                        if ":::" in token_str or "::" in token_str:
                            # #region agent log
                            import json; open('/home/aadivyar/csd-generation/.cursor/debug.log', 'a').write(json.dumps({"location": "parser_utils.py:force_closure_fallback", "message": "Force closure returning ::: fallback", "data": {"text_end": current_text[-30:] if current_text else "", "token": token_str}, "timestamp": __import__('time').time() * 1000, "sessionId": "debug-session", "hypothesisId": "H8"}) + '\n')
                            # #endregion
                            return _dafny.SeqWithoutIsStrInference([token])
                    # If no ::: token found, return empty to signal error
                    return _dafny.SeqWithoutIsStrInference([])

            # When ::: is grammatically valid, include it in the options
            # but don't force it (let the model decide when to close)
            # NOTE: gtgt_tokens should be empty when formula_body_expected due to earlier filtering
            all_valid = valid_tokens + gtgt_tokens

            # SAFETY: If no valid tokens found, try to return ::: to close gracefully
            # BUT: Only return ::: when the expression is actually complete!
            # Check for incomplete states that should NOT allow :::
            import re
            text_stripped = current_text.rstrip()
            has_incomplete_pred = bool(re.search(r'[A-Z][a-zA-Z0-9]*\s*$', text_stripped)) and not text_stripped.endswith(')')
            has_incomplete_op = any(text_stripped.endswith(op) for op in 
                ['{and}', '{or}', '{implies}', '{iff}', '{xor}', '{not}', '{forall}', '{exists}'])
            open_parens = current_text.count('(') - current_text.count(')')
            open_braces = current_text.count('{') - current_text.count('}')
            is_incomplete = formula_body_expected or has_incomplete_pred or has_incomplete_op or open_parens > 0 or open_braces > 0
            
            if not all_valid and not is_incomplete:
                for token in self._token_list:
                    token_str = self._dafny_seq_to_str(token)
                    if ":::" in token_str or "::" in token_str:
                        if debug_fol:
                            print(f"    [FOL PARSER] No valid tokens! Returning ::: to close gracefully")
                        return _dafny.SeqWithoutIsStrInference([token])

            if debug_fol:
                valid_strs = [self._dafny_seq_to_str(t) for t in all_valid[:10]]
                print(f"    [FOL PARSER] Returning {len(all_valid)} valid tokens: {valid_strs}{'...' if len(all_valid) > 10 else ''}")

            return _dafny.SeqWithoutIsStrInference(all_valid if all_valid else [])
    
    return LarkDafnyParser


def get_builtin_grammar(format_name: str) -> str:
    """
    Get built-in grammar for common formats.
    
    Args:
        format_name: One of "json", "sql", "math"
        
    Returns:
        Grammar string in Lark format
        
    Raises:
        ValueError: If format_name is not recognized
    """
    grammars = {
        "json": r'''
            start: value
            ?value: object | array | string | number | "true" -> true | "false" -> false | "null" -> null
            object: "{" [pair ("," pair)*] "}"
            pair: string ":" value
            array: "[" [value ("," value)*] "]"
            string: ESCAPED_STRING
            number: SIGNED_NUMBER
            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS
            %ignore WS
        ''',
        "sql": r'''
            start: select_stmt
            select_stmt: "SELECT"i columns "FROM"i table [where_clause]
            columns: "*" | column ("," column)*
            column: NAME
            table: NAME
            where_clause: "WHERE"i condition
            condition: NAME comp_op value
            comp_op: "=" | "!=" | "<" | ">" | "<=" | ">="
            value: NAME | NUMBER | STRING
            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.ESCAPED_STRING -> STRING
            %import common.WS
            %ignore WS
        ''',
        "math": r'''
            start: expr
            ?expr: term | expr "+" term | expr "-" term
            ?term: factor | term "*" factor | term "/" factor
            ?factor: NUMBER | "(" expr ")" | "-" factor
            %import common.NUMBER
            %import common.WS
            %ignore WS
        ''',
    }
    
    if format_name.lower() not in grammars:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(grammars.keys())}")
    
    return grammars[format_name.lower()]
