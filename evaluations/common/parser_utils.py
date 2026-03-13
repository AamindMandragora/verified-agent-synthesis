"""
Parser creation utilities for Dafny-compatible grammar parsers.

Provides factory functions to create Lark-based parsers that conform to
the Dafny Parser interface used by CSD strategies.
"""

from __future__ import annotations

from pathlib import Path

from evaluations.common.generation import dafny_seq_to_str


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
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF

        def _dafny_seq_to_str(self, seq) -> str:
            """Convert a Dafny Seq to a Python string."""
            return dafny_seq_to_str(seq)

        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            try:
                return ''.join(self._dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                return str(tokens)

        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar."""
            if not text:
                return True

            if not hasattr(self, '_prefix_validity_cache'):
                self._prefix_validity_cache = {}

            if text in self._prefix_validity_cache:
                return self._prefix_validity_cache[text]

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

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens.

            For each token in the vocabulary, checks whether appending it
            to the current prefix yields a valid grammar prefix.
            """
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""

            if current_text and not self._is_valid_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])

            valid_tokens = []

            for token in self._token_list:
                token_str = self._dafny_seq_to_str(token)
                if not token_str:
                    continue

                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    valid_tokens.append(token)

            return _dafny.SeqWithoutIsStrInference(valid_tokens)

        def IsPermissive(self, prefix) -> bool:
            """Dafny interface: True only when every token is valid (e.g. free-form sections). Strict grammar => False."""
            return False

    return LarkDafnyParser


def create_folio_wrapper_parser(
    VerifiedDecoderAgent,
    _dafny,
    fol_parser_instance,
    lm_tokens,
    tokenizer,
):
    """
    Parser for FOLIO: plain text, then " << ", then FOL (Prover9 grammar), then " >>", then plain text.
    The LLM's single CSD strategy runs over the whole output; structure is enforced by this parser.
    """
    try:
        token_list = list(lm_tokens)
    except TypeError:
        token_list = [lm_tokens[i] for i in range(len(lm_tokens))]

    fol_parser = fol_parser_instance
    open_marker = " << "
    close_marker = " >>"

    def tokens_to_text(prefix):
        if len(prefix) == 0:
            return ""
        try:
            return "".join(dafny_seq_to_str(prefix[i]) for i in range(len(prefix)))
        except (TypeError, AttributeError, IndexError):
            return str(prefix)

    class FOLIOWrapperParser(VerifiedDecoderAgent.Parser):
        def __init__(self):
            super().__init__()
            self._token_list = token_list
            self._fol_parser = fol_parser
            self._open = open_marker
            self._close = close_marker
            self._tokenizer = tokenizer

        def _dafny_seq_to_str(self, seq):
            return dafny_seq_to_str(seq)

        def _get_fol_section(self, full_text: str):
            if self._open not in full_text:
                return None, None, "intro"
            start = full_text.rfind(self._open) + len(self._open)
            after_open = full_text[start:]
            if self._close in after_open:
                end_idx = after_open.index(self._close)
                fol_text = after_open[:end_idx]
                return start, start + end_idx, "outro"
            return start, len(full_text), "fol"

        def _fol_tokens_from_prefix(self, prefix):
            """Return Dafny Seq of tokens that form the FOL part of prefix."""
            text = tokens_to_text(prefix)
            start, end, section = self._get_fol_section(text)
            if section != "fol" and section != "outro":
                return _dafny.SeqWithoutIsStrInference([])
            fol_text = text[start:end] if section == "outro" else text[start:]
            if section == "outro" and self._close in text[start:]:
                fol_text = text[start:].split(self._close)[0]
            if not fol_text.strip():
                return _dafny.SeqWithoutIsStrInference([])
            try:
                ids = self._tokenizer.encode(fol_text, add_special_tokens=False)
                token_strs = [self._tokenizer.decode([i]) for i in ids]
            except Exception:
                return _dafny.SeqWithoutIsStrInference([])
            return _dafny.SeqWithoutIsStrInference([
                _dafny.SeqWithoutIsStrInference(s) for s in token_strs
            ])

        def IsValidPrefix(self, prefix) -> bool:
            if len(prefix) == 0:
                return True
            text = tokens_to_text(prefix)
            _, _, section = self._get_fol_section(text)
            if section == "intro":
                return True
            fol_seq = self._fol_tokens_from_prefix(prefix)
            if len(fol_seq) == 0:
                return True
            return self._fol_parser.IsValidPrefix(fol_seq)

        def IsCompletePrefix(self, prefix) -> bool:
            if len(prefix) == 0:
                return False
            text = tokens_to_text(prefix)
            if self._open not in text or self._close not in text:
                return False
            start = text.rfind(self._open) + len(self._open)
            after_open = text[start:]
            if self._close not in after_open:
                return False
            close_idx = after_open.index(self._close)
            fol_text = after_open[:close_idx]
            if not fol_text.strip():
                return False
            # Require at least one character after " >>" so we don't stop as soon as " << formula >>" (avoids premature stop).
            after_close = after_open[close_idx + len(self._close) :].strip()
            if len(after_close) == 0:
                return False
            try:
                ids = self._tokenizer.encode(fol_text, add_special_tokens=False)
                token_strs = [self._tokenizer.decode([i]) for i in ids]
            except Exception:
                return False
            fol_seq = _dafny.SeqWithoutIsStrInference([
                _dafny.SeqWithoutIsStrInference(s) for s in token_strs
            ])
            return self._fol_parser.IsCompletePrefix(fol_seq)

        def _intro_delimiter_tokens(self):
            """Tokens that would start or continue '<<' / ' << ' with no prior content. Exclude these in empty/whitespace intro."""
            bad = {"<<", "<", " <<", " << "}
            out = []
            for i in range(len(token_list)):
                t = token_list[i]
                try:
                    s = self._dafny_seq_to_str(t)
                except (TypeError, AttributeError, IndexError):
                    s = str(t)
                if s not in bad:
                    out.append(t)
            if not out:
                out = [token_list[j] for j in range(len(token_list))]
            return out

        def ValidNextTokens(self, prefix):
            if len(prefix) == 0:
                # Empty intro: disallow any token that can produce "<< <<" (e.g. "<<", "<", " <<", " << ").
                allowed = self._intro_delimiter_tokens()
                return _dafny.SeqWithoutIsStrInference(allowed)
            text = tokens_to_text(prefix)
            _, _, section = self._get_fol_section(text)
            if section == "intro" or section == "outro":
                text_stripped = text.strip()
                if len(text_stripped) == 0:
                    # Still only whitespace — same exclusions so we don't get "<< <<".
                    allowed = self._intro_delimiter_tokens()
                    return _dafny.SeqWithoutIsStrInference(allowed)
                return _dafny.SeqWithoutIsStrInference(token_list)
            fol_seq = self._fol_tokens_from_prefix(prefix)
            return self._fol_parser.ValidNextTokens(fol_seq)

        def IsPermissive(self, prefix) -> bool:
            """True when any token is valid (intro/outro). Used by Dafny to maintain IsValidPrefix after UnconstrainedStep."""
            if len(prefix) == 0:
                return True
            text = tokens_to_text(prefix)
            _, _, section = self._get_fol_section(text)
            return section == "intro" or section == "outro"

    return FOLIOWrapperParser()


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
