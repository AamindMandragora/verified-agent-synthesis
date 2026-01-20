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
            self._token_list = list(lm_tokens)
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

            # Detect if this is a MATH grammar vs FOL grammar
            # Math grammars have n_expr/s_expr rules, FOL has {forall}/{exists}
            self._is_math_grammar = 'n_expr' in grammar or 's_expr' in grammar or 'n_term' in grammar
            self._is_fol_grammar = '{forall}' in grammar or '{exists}' in grammar
        
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
            """Convert Dafny token sequence to text."""
            try:
                return "".join(self._dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError):
                return str(tokens)
        
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
            - {not}, {forall}, {exists} are valid (formula start)
            - '(' is valid (parenthesized formula)
            - Binary operators ({and}, {or}, {implies}, etc.) are NOT valid
            """
            import re
            text = text.rstrip()
            # Pattern: quantifier followed by whitespace and a single variable letter
            # e.g., "{forall} x" or "{exists}y" or "{forall}  z"
            pattern = r'\{(forall|exists)\}\s*[a-z]$'
            return bool(re.search(pattern, text))

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

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar."""
            if not text:
                return True

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
                                return True  # Start operators are valid at the beginning
                            prefix_before_stripped = prefix_before.rstrip()
                            # NOT valid right after a quantifier (expects variable, not operator)
                            if self._expects_variable_next(prefix_before_stripped):
                                return False
                            if self._is_valid_prefix(prefix_before_stripped):
                                return True

                # For binary operators, check that they can validly follow
                for op in binary_operators:
                    for i in range(1, len(op)):
                        partial = op[:i]
                        if text.endswith(partial):
                            prefix_before = text[:-len(partial)].rstrip()
                            # Use the comprehensive check
                            if not self._can_binary_operator_follow(prefix_before):
                                return False
                            # The prefix before must be a valid (possibly complete) formula
                            if self._is_valid_prefix(prefix_before):
                                return True

            try:
                self._lark.parse(text)
                return True
            except self._UnexpectedEOF:
                # Hit end of input while expecting more - valid prefix
                return True
            except self._UnexpectedToken as e:
                if e.token.type == '$END':
                    return True
                return False
            except self._UnexpectedCharacters:
                return False
            except Exception:
                return False
        
        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse."""
            if not text:
                return False
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
            """
            if self._is_math_grammar:
                return self._valid_next_tokens_math(prefix)
            else:
                return self._valid_next_tokens_fol(prefix)

        def _valid_next_tokens_math(self, prefix):
            """Get valid next tokens for MATH grammars (GSM-Symbolic).

            Simple grammar-based validation without FOL-specific filtering.
            Lets the Lark grammar do the heavy lifting.
            """
            import re
            import os

            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

            if current_text and not self._is_valid_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])

            # Basic sanity limits - reduced MAX_DEPTH to prevent deep nesting
            MAX_DEPTH = 3
            depth = sum(1 if c == '(' else -1 if c == ')' else 0 for c in current_text)
            depth_limit_reached = depth >= MAX_DEPTH

            # Expression complexity limits
            MAX_EXPR_TOKENS = 20
            MAX_OPERATORS = 10
            operator_count = sum(1 for c in current_text if c in '+-*/')
            token_count = len(prefix)

            # Check for minimal expression (need at least one variable/number)
            has_content = any(c.isalnum() for c in current_text)

            # Check if current text ends with a variable/number (operand)
            # In math expressions, operands must be followed by operators, not other operands
            current_stripped = current_text.rstrip()
            ends_with_operand = bool(re.search(r'[a-zA-Z0-9]$', current_stripped))
            ends_with_open_paren = current_stripped.endswith('(')
            ends_with_close_paren = current_stripped.endswith(')')
            ends_with_operator = bool(re.search(r'[+\-*/]$', current_stripped))

            # Get the last alphanumeric character for adjacent letter detection
            last_alnum_char = ''
            for c in reversed(current_stripped):
                if c.isalnum():
                    last_alnum_char = c
                    break
                elif not c.isspace():
                    break  # Hit a non-alnum, non-space char
            ends_with_letter = last_alnum_char.isalpha()

            if os.environ.get('CSD_PARSER_DEBUG'):
                print(f"    [PARSER] ValidNextTokens: prefix_len={len(prefix)}, text='{current_text[:20]}', ends_with_letter={ends_with_letter}")

            valid_tokens = []
            gtgt_tokens = []
            close_paren_tokens = []  # Track ')' tokens separately for paren balancing

            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    continue

                stripped = token_str.strip()

                # Limit excessive whitespace
                if not stripped and current_text.endswith('  '):
                    continue

                # === ADJACENT LETTER CHECK (FIX FOR ISSUE 1) ===
                # Prevent adjacent letters which would form multi-char variables
                # or "a b" patterns. Check BEFORE grammar validation since grammar
                # allows multi-char variables like "mc" but we want single-char only.
                if ends_with_letter and stripped:
                    # Find first non-whitespace character in token
                    first_content_char = ''
                    for c in token_str:
                        if not c.isspace():
                            first_content_char = c
                            break
                    # If token starts with a letter, reject (would create adjacent letters)
                    if first_content_char.isalpha():
                        import os
                        if os.environ.get('CSD_PARSER_DEBUG'):
                            print(f"    [PARSER] Rejecting '{repr(token_str)}' after '{current_text[-10:]}' (adjacent letters)")
                        continue

                # === PARENTHESIS BALANCE CHECKS ===

                # Don't allow ')' if no open parens to close
                if ')' in token_str and depth <= 0:
                    continue

                # Don't allow >> if there are unclosed parens (FIX FOR ISSUE 2)
                # We track close paren tokens separately to prioritize them when needed
                if ">>" in token_str and depth > 0:
                    continue

                # CRITICAL: Also catch when >> is formed by TWO separate > tokens
                # If current text ends with '>' and new token starts with '>',
                # together they form '>>' which should be blocked if parens unbalanced
                if depth > 0 and current_stripped.endswith('>'):
                    # Check if token starts with '>' (would complete >>)
                    first_char_of_token = ''
                    for c in token_str:
                        if not c.isspace():
                            first_char_of_token = c
                            break
                    if first_char_of_token == '>':
                        continue  # Reject - would form >> with unclosed parens

                # Also prevent single '>' when parens are unbalanced
                # '>' alone is not valid in math - it's only used as part of '>>'
                # If depth > 0, we shouldn't start building '>>' at all
                if depth > 0 and '>' in token_str and '>>' not in token_str:
                    continue  # Reject single '>' when parens unbalanced

                # After '(', only allow: variables, numbers, '(', '-' (unary minus), 'int'
                # NOT allowed: operators (+, *, /), ')', '>>'
                if ends_with_open_paren and stripped:
                    first_char = stripped[0]
                    # Reject binary operators (but allow '-' for negative numbers)
                    if first_char in '+*/':
                        continue
                    # Reject close paren right after open paren
                    if first_char == ')':
                        continue
                    # Reject >> right after open paren
                    if ">>" in token_str:
                        continue

                # After ')', only allow: operators, ')', '>>'
                # NOT allowed: variables, numbers, '('
                if ends_with_close_paren and stripped:
                    first_char = stripped[0]
                    if first_char.isalnum() or first_char == '(':
                        continue

                # === OPERAND SEQUENCE CHECKS ===

                # Depth limit - no more open parens
                if depth_limit_reached and '(' in token_str and ')' not in token_str:
                    continue

                # Don't allow >> until we have content
                if ">>" in token_str and not has_content:
                    continue

                # CRITICAL: If current text ends with an operand (variable/number),
                # the next token MUST start with an operator, close paren, or >>
                # This prevents "m c" (variable followed by variable)
                if ends_with_operand and stripped:
                    first_char = stripped[0]
                    # Allow: operators, close paren, >>
                    # Reject: variables, numbers, open paren (would need operator first)
                    if first_char.isalnum() or first_char == '(':
                        continue

                # After an operator, only allow: variables, numbers, '(', '-', 'int'
                # NOT allowed: other operators, ')', '>>'
                if ends_with_operator and stripped:
                    first_char = stripped[0]
                    # Reject close paren after operator
                    if first_char == ')':
                        continue
                    # Reject >> after operator (incomplete expression)
                    if ">>" in token_str:
                        continue
                    # Reject another operator (except '-' for negative)
                    if first_char in '+*/':
                        continue

                # Grammar-based validation
                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    if ">>" in token_str:
                        gtgt_tokens.append(token)
                    elif stripped == ')' or (')' in token_str and '(' not in token_str):
                        # Track close paren tokens for priority when balancing needed
                        close_paren_tokens.append(token)
                        valid_tokens.append(token)
                    else:
                        valid_tokens.append(token)

            # === FORCE CLOSURE LOGIC (FIX FOR ISSUE 2) ===
            # When expression is too long, we need to close it properly
            should_force_closure = has_content and (token_count >= MAX_EXPR_TOKENS or operator_count >= MAX_OPERATORS)

            if should_force_closure:
                # If we have unclosed parens, prioritize closing them first
                if depth > 0 and close_paren_tokens:
                    return _dafny.SeqWithoutIsStrInference(close_paren_tokens)
                # If parens are balanced, use >> to close
                if depth == 0 and gtgt_tokens:
                    return _dafny.SeqWithoutIsStrInference(gtgt_tokens)
                # Fallback: return close parens if available, then any valid tokens
                if close_paren_tokens:
                    return _dafny.SeqWithoutIsStrInference(close_paren_tokens)

            all_valid = valid_tokens + gtgt_tokens
            return _dafny.SeqWithoutIsStrInference(all_valid if all_valid else [])

        def _valid_next_tokens_fol(self, prefix):
            """Get valid next tokens for FOL grammars (FOLIO).

            Uses the grammar to determine which tokens can validly follow the current prefix.
            Special handling for partial FOL operators to ensure proper operator completion.
            """
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

            if current_text and not self._is_valid_prefix(current_text):
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

            # Check for repetitive patterns (e.g., "* 1 * 1 * 1" or "CarCarCar")
            has_repetition = False
            if len(current_text) >= 8:
                window = current_text[-40:]
                import re

                # Check for any 2-6 char sequence repeated 3+ times (catches "CarCarCar", "fliesflies")
                for seq_len in range(2, 7):
                    if len(window) >= seq_len * 3:
                        for start in range(len(window) - seq_len * 3 + 1):
                            seq = window[start:start + seq_len]
                            if seq.isalpha() and window.count(seq) >= 3:
                                has_repetition = True
                                break
                    if has_repetition:
                        break

                # Check if current text is mostly one repeated pattern
                if not has_repetition and len(current_text) >= 12:
                    # If 70%+ of text is the same 3-6 char pattern, it's repetitive
                    for plen in range(3, 7):
                        if len(current_text) >= plen:
                            pattern = current_text[-plen:]
                            count = current_text.count(pattern)
                            if count * plen >= len(current_text) * 0.5:
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

            # Characters that are NOT valid in FOL grammar
            # FOL allows: letters, digits, {, }, (, ), comma, whitespace
            # Note: '.' (period) is also invalid in FOL expressions
            INVALID_FOL_CHARS = set('./*+-=!<>?@#$%^&~:;"\'|[]`_')

            # Check if binary operators can follow at current position
            can_binary_follow = self._can_binary_operator_follow(current_text.rstrip())
            # For backwards compatibility, also set expects_formula_start
            expects_formula_start = not can_binary_follow and bool(current_text.strip())

            # If we're in the middle of building an FOL operator, prioritize continuation tokens
            fol_continuation_tokens = []

            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    continue

                # Reject banned keywords and operators
                stripped = token_str.strip()
                if stripped in BANNED_KEYWORDS or stripped in BANNED_OPERATORS:
                    continue

                # WHITELIST approach: At the START of FOL expression, only allow valid starters
                # Valid FOL expression starters: {, (, or Predicate(
                is_at_start = not current_text.strip()
                if is_at_start and stripped:
                    # Must start with: { (for FOL operators), ( (for parens), or uppercase (predicate)
                    first_char = stripped[0]
                    if first_char == '{':
                        pass  # OK - starting FOL operator
                    elif first_char == '(':
                        pass  # OK - parenthesized formula
                    elif first_char.isupper():
                        # Only allow if it's a predicate-like pattern (uppercase followed by more letters)
                        # Reject if it's just random garbage
                        if len(stripped) > 5 and not stripped.endswith('('):
                            # Long word without ( is suspicious - reject
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
                # (but allow >> which is the constraint delimiter)
                if ">>" not in token_str:
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

                # Don't allow >> until we have at least a minimal expression
                if ">>" in token_str and not meets_min_complexity:
                    continue

                # Special handling: if we're building an FOL operator, only allow continuations
                if fol_continuations:
                    # Check if this token can continue the partial operator
                    if len(token_str) == 1 and token_str in fol_continuations:
                        fol_continuation_tokens.append(token)
                    continue  # Skip normal validation when in FOL operator mode

                # Special handling: when a VARIABLE is expected (after quantifiers)
                # Only allow single lowercase letters or whitespace
                if expects_variable:
                    # Don't allow FOL operators to start (they begin with '{')
                    if '{' in token_str:
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
                        # Verify it's grammatically valid
                        extended = current_text + token_str
                        if self._is_valid_prefix(extended):
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
                    # Only allow: more alphanumeric chars OR '('
                    if stripped:
                        first_char = stripped[0]
                        # Allow '(' to start predicate arguments
                        if first_char == '(':
                            pass  # OK
                        # Allow alphanumeric to continue predicate name
                        elif first_char.isalnum():
                            pass  # OK
                        else:
                            continue  # Reject - can't have other chars after partial predicate
                    # Also reject if this would create a repeated predicate name
                    if stripped and stripped[0].isupper() and not stripped.startswith('('):
                        # This looks like starting a new predicate - reject if it would repeat
                        if current_text.rstrip().endswith(stripped):
                            continue

                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    # Separate >> tokens so we can prioritize them when expression is complete
                    if ">>" in token_str:
                        gtgt_tokens.append(token)
                    elif token_str.strip():  # Non-whitespace
                        valid_tokens.append(token)

            # If we're building an FOL operator, return only the continuation tokens
            if fol_continuations and fol_continuation_tokens:
                return _dafny.SeqWithoutIsStrInference(fol_continuation_tokens)

            # Check if current text looks like garbage (no FOL structure at all)
            has_fol_structure = any(c in current_text for c in '{}()')
            is_garbage = (
                len(current_text.strip()) > 10 and
                not has_fol_structure and
                not has_fol_operator
            )

            # CRITICAL: Force closure when expression is too long, has repetition, or is garbage
            should_force_closure = (
                meets_min_complexity and  # Have a meaningful expression
                (token_count >= MAX_EXPR_TOKENS or  # Too many tokens
                 operator_count >= MAX_OPERATORS or  # Too many operators
                 has_repetition or  # Detected repetitive pattern
                 is_garbage)  # Text has no FOL structure, just garbage
            )

            if should_force_closure:
                # Try to force >> even if not in gtgt_tokens
                if gtgt_tokens:
                    return _dafny.SeqWithoutIsStrInference(gtgt_tokens)
                else:
                    # Find any >> token in the vocabulary and return it
                    # This is an emergency bail-out for garbage generation
                    for token in self._token_list:
                        token_str = self._dafny_seq_to_str(token)
                        if ">>" in token_str:
                            return _dafny.SeqWithoutIsStrInference([token])
                    # If no >> token found, return empty to signal error
                    return _dafny.SeqWithoutIsStrInference([])

            # When >> is grammatically valid, include it in the options
            # but don't force it (let the model decide when to close)
            all_valid = valid_tokens + gtgt_tokens

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
