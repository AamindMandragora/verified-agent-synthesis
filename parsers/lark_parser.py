"""
Generic Lark-based parser for constrained decoding.

This module provides a grammar-agnostic parser that can validate prefixes
against ANY Lark grammar file. No custom code needed per grammar!

Usage:
    # For JSON
    parser = LarkGrammarParser.from_grammar_file("grammars/json.lark")
    
    # For Python  
    parser = LarkGrammarParser.from_grammar_file("grammars/python.lark")
    
    # For SQL
    parser = LarkGrammarParser.from_grammar_file("grammars/sql.lark")
"""

from pathlib import Path
from typing import Optional, Set
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class LarkGrammarParser:
    """
    Generic parser using Lark for any grammar.
    
    Uses Lark's incremental/interactive parsing to validate prefixes.
    This is the same approach used by Syncode.
    """
    
    def __init__(self, grammar: str, start: str = "start"):
        """
        Initialize with a Lark grammar string.
        
        Args:
            grammar: Lark grammar definition
            start: Start rule name
        """
        try:
            from lark import Lark
            from lark.exceptions import UnexpectedCharacters, UnexpectedToken
        except ImportError:
            raise ImportError(
                "Lark is required for grammar-based parsing. "
                "Install with: pip install lark"
            )
        
        self._grammar_str = grammar
        self._start = start
        self._UnexpectedCharacters = UnexpectedCharacters
        self._UnexpectedToken = UnexpectedToken
        
        # Create the parser
        # Use lalr for speed, with lexer='contextual' for better prefix handling
        self._parser = Lark(
            grammar,
            start=start,
            parser='lalr',
            lexer='contextual',
            propagate_positions=True,
        )
        
        # For interactive parsing
        self._interactive_parser = None
    
    @classmethod
    def from_grammar_file(cls, path: str | Path, start: str = "start") -> 'LarkGrammarParser':
        """
        Create parser from a grammar file.
        
        Args:
            path: Path to .lark grammar file
            start: Start rule name
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Grammar file not found: {path}")
        
        grammar = path.read_text()
        return cls(grammar, start)
    
    def is_valid_prefix(self, text: str) -> bool:
        """
        Check if text is a valid prefix of the grammar.
        
        Uses Lark's interactive parser to check if the input
        could be extended to a valid parse.
        """
        if not text:
            return True
        
        try:
            # Try to parse - if it succeeds, it's valid
            self._parser.parse(text)
            return True
        except self._UnexpectedCharacters:
            # Lexer error - invalid character sequence
            return False
        except self._UnexpectedToken as e:
            # Parser error - check if it's an incomplete parse
            # If the error is at the end and we're expecting more input,
            # it could still be a valid prefix
            if e.token.type == '$END':
                # Hit end of input while expecting more - valid prefix
                return True
            return False
        except Exception:
            return False
    
    def is_complete(self, text: str) -> bool:
        """Check if text is a complete valid parse."""
        if not text:
            return False
        
        try:
            self._parser.parse(text)
            return True
        except Exception:
            return False
    
    def get_valid_next_tokens(self, text: str, vocabulary: list[str]) -> list[str]:
        """
        Get tokens from vocabulary that can validly follow the text.
        
        Args:
            text: Current prefix
            vocabulary: List of possible next tokens
            
        Returns:
            Filtered list of valid next tokens
        """
        if not self.is_valid_prefix(text):
            return []
        
        valid = []
        for token in vocabulary:
            extended = text + token
            if self.is_valid_prefix(extended):
                valid.append(token)
        
        return valid
    
    @lru_cache(maxsize=10000)
    def _cached_is_valid_prefix(self, text: str) -> bool:
        """Cached version of is_valid_prefix."""
        return self.is_valid_prefix(text)


class InteractiveLarkParser:
    """
    More efficient parser using Lark's interactive parsing mode.
    
    This maintains parser state and is more efficient for
    incremental validation during token-by-token generation.
    """
    
    def __init__(self, grammar: str, start: str = "start"):
        """Initialize with Lark grammar."""
        try:
            from lark import Lark
            from lark.exceptions import UnexpectedCharacters, UnexpectedToken
        except ImportError:
            raise ImportError("Lark is required. Install with: pip install lark")
        
        self._parser = Lark(
            grammar,
            start=start,
            parser='lalr',
        )
        self._interactive = None
        self._UnexpectedCharacters = UnexpectedCharacters
        self._UnexpectedToken = UnexpectedToken
    
    def reset(self):
        """Reset to initial state."""
        self._interactive = self._parser.parse_interactive()
    
    def feed_token(self, token: str) -> bool:
        """
        Feed a token to the parser.
        
        Returns True if the token is accepted, False if rejected.
        """
        if self._interactive is None:
            self.reset()
        
        try:
            for char in token:
                self._interactive.feed_token(char)
            return True
        except (self._UnexpectedCharacters, self._UnexpectedToken):
            return False
    
    def accepts_token(self, token: str) -> bool:
        """
        Check if a token would be accepted without consuming it.
        
        Creates a copy of the parser state to test.
        """
        if self._interactive is None:
            self.reset()
        
        # Save current state
        import copy
        saved = copy.deepcopy(self._interactive)
        
        try:
            result = self.feed_token(token)
            return result
        finally:
            # Restore state
            self._interactive = saved


# =============================================================================
# Pre-built grammar parsers for common formats
# =============================================================================

def create_json_lark_grammar() -> str:
    """Return a Lark grammar for JSON."""
    return r'''
        start: value
        
        ?value: object
              | array
              | string
              | number
              | "true"  -> true
              | "false" -> false
              | "null"  -> null
        
        object: "{" [pair ("," pair)*] "}"
        pair: string ":" value
        
        array: "[" [value ("," value)*] "]"
        
        string: ESCAPED_STRING
        number: SIGNED_NUMBER
        
        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS
    '''


def create_python_lark_grammar() -> str:
    """Return a simplified Lark grammar for Python expressions."""
    return r'''
        start: stmt+
        
        ?stmt: simple_stmt
             | compound_stmt
        
        simple_stmt: (expr | assignment) NEWLINE?
        
        assignment: NAME "=" expr
        
        ?expr: term
             | expr "+" term
             | expr "-" term
        
        ?term: factor
             | term "*" factor
             | term "/" factor
        
        ?factor: atom
               | "(" expr ")"
               | "-" factor
        
        ?atom: NUMBER
             | STRING
             | NAME
             | "True" | "False" | "None"
        
        compound_stmt: if_stmt | for_stmt | func_def
        
        if_stmt: "if" expr ":" suite ("elif" expr ":" suite)* ["else" ":" suite]
        for_stmt: "for" NAME "in" expr ":" suite
        func_def: "def" NAME "(" [params] ")" ":" suite
        
        params: NAME ("," NAME)*
        suite: NEWLINE INDENT stmt+ DEDENT | simple_stmt
        
        %import common.CNAME -> NAME
        %import common.NUMBER
        %import common.ESCAPED_STRING -> STRING
        %import common.NEWLINE
        %import common.WS_INLINE
        %declare INDENT DEDENT
        %ignore WS_INLINE
    '''


def create_sql_lark_grammar() -> str:
    """Return a simplified Lark grammar for SQL SELECT statements."""
    return r'''
        start: select_stmt
        
        select_stmt: "SELECT" columns "FROM" table_ref [where_clause] [order_clause] [limit_clause]
        
        columns: "*" | column_list
        column_list: column ("," column)*
        column: NAME ["AS" NAME]
        
        table_ref: NAME ["AS" NAME]
        
        where_clause: "WHERE" condition
        
        ?condition: comparison
                  | condition "AND" condition
                  | condition "OR" condition
                  | "(" condition ")"
        
        comparison: expr comp_op expr
        
        ?comp_op: "=" | "!=" | "<" | ">" | "<=" | ">=" | "LIKE" | "IN"
        
        ?expr: NAME | NUMBER | STRING | "NULL"
        
        order_clause: "ORDER" "BY" order_list
        order_list: order_item ("," order_item)*
        order_item: NAME ["ASC" | "DESC"]
        
        limit_clause: "LIMIT" NUMBER ["OFFSET" NUMBER]
        
        %import common.CNAME -> NAME
        %import common.NUMBER
        %import common.ESCAPED_STRING -> STRING
        %import common.WS
        %ignore WS
    '''


# =============================================================================
# Convenience factory functions
# =============================================================================

def create_parser_for_format(format_name: str) -> LarkGrammarParser:
    """
    Create a parser for a common format.
    
    Args:
        format_name: One of "json", "python", "sql", "math"
        
    Returns:
        LarkGrammarParser configured for that format
    """
    grammars = {
        "json": create_json_lark_grammar(),
        "python": create_python_lark_grammar(),
        "sql": create_sql_lark_grammar(),
    }
    
    if format_name.lower() not in grammars:
        raise ValueError(
            f"Unknown format: {format_name}. "
            f"Available: {list(grammars.keys())}. "
            f"Or use LarkGrammarParser.from_grammar_file() for custom grammars."
        )
    
    return LarkGrammarParser(grammars[format_name.lower()])

