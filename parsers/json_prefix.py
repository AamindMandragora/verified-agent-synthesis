"""
Streaming JSON prefix validator.

Provides functions to check if a string is:
- A valid JSON prefix (no contradiction so far, even if incomplete)
- A complete valid JSON value (parseable by json.loads with no trailing content)

Implementation uses a stack-based state machine that tracks:
- Context stack (object/array nesting)
- Expected next elements (key, colon, value, comma, end)
- String/escape/unicode states
- Number parsing states
"""

import json
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class Context(Enum):
    """JSON nesting context."""
    OBJECT = auto()
    ARRAY = auto()


class Expect(Enum):
    """What we expect next in the current context."""
    VALUE = auto()           # Expecting any value
    OBJECT_KEY_OR_END = auto()  # Expecting string key or }
    COLON = auto()           # Expecting : after key
    OBJECT_VALUE = auto()    # Expecting value after :
    COMMA_OR_OBJECT_END = auto()  # Expecting , or }
    COMMA_OR_ARRAY_END = auto()   # Expecting , or ]
    END = auto()             # Expecting end of input (after top-level value)


class StringState(Enum):
    """State within a string literal."""
    NONE = auto()            # Not in a string
    NORMAL = auto()          # In string, normal chars
    ESCAPE = auto()          # After backslash
    UNICODE_1 = auto()       # After \u, need 4 hex digits
    UNICODE_2 = auto()
    UNICODE_3 = auto()
    UNICODE_4 = auto()


class NumberState(Enum):
    """State within a number literal."""
    NONE = auto()            # Not in a number
    MINUS = auto()           # After leading -
    ZERO = auto()            # After leading 0
    INT_DIGITS = auto()      # In integer part (after 1-9)
    DOT = auto()             # After .
    FRAC_DIGITS = auto()     # In fraction part
    EXP = auto()             # After e/E
    EXP_SIGN = auto()        # After e+/e-
    EXP_DIGITS = auto()      # In exponent digits


@dataclass
class ParseState:
    """Complete parser state for JSON prefix validation."""
    context_stack: list[Context]
    expect: Expect
    string_state: StringState
    number_state: NumberState
    pos: int  # Current position in input
    
    def copy(self) -> 'ParseState':
        return ParseState(
            context_stack=self.context_stack.copy(),
            expect=self.expect,
            string_state=self.string_state,
            number_state=self.number_state,
            pos=self.pos
        )


def _is_hex(c: str) -> bool:
    return c in '0123456789abcdefABCDEF'


def _is_ws(c: str) -> bool:
    return c in ' \t\n\r'


class JsonPrefixValidator:
    """
    Stack-based streaming JSON prefix validator.
    
    Validates character by character, tracking state to determine
    if the input so far could be a prefix of a valid JSON document.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.state = ParseState(
            context_stack=[],
            expect=Expect.VALUE,
            string_state=StringState.NONE,
            number_state=NumberState.NONE,
            pos=0
        )
        self.error: Optional[str] = None
    
    def _fail(self, msg: str) -> bool:
        """Record error and return False."""
        self.error = f"Position {self.state.pos}: {msg}"
        return False
    
    def _in_string(self) -> bool:
        return self.state.string_state != StringState.NONE
    
    def _in_number(self) -> bool:
        return self.state.number_state != NumberState.NONE
    
    def _finish_number(self, transition_after: bool = True) -> bool:
        """
        Check if current number state can be terminated.
        Returns True if the number is valid to end here.
        
        Args:
            transition_after: If True, call _after_value() to update state
        """
        ns = self.state.number_state
        if ns == NumberState.NONE:
            return True
        # Valid ending states for a number
        if ns in (NumberState.ZERO, NumberState.INT_DIGITS, 
                  NumberState.FRAC_DIGITS, NumberState.EXP_DIGITS):
            self.state.number_state = NumberState.NONE
            if transition_after:
                return self._after_value()
            return True
        # Invalid to end here (e.g., after - or . or e)
        return False
    
    def _process_string_char(self, c: str) -> bool:
        """Process a character while in a string."""
        ss = self.state.string_state
        
        if ss == StringState.ESCAPE:
            # After backslash, must be valid escape char
            if c in '"\\/' + 'bfnrt':
                self.state.string_state = StringState.NORMAL
                return True
            elif c == 'u':
                self.state.string_state = StringState.UNICODE_1
                return True
            else:
                return self._fail(f"Invalid escape sequence: \\{c}")
        
        elif ss in (StringState.UNICODE_1, StringState.UNICODE_2, 
                    StringState.UNICODE_3, StringState.UNICODE_4):
            if not _is_hex(c):
                return self._fail(f"Invalid unicode escape: expected hex digit, got '{c}'")
            # Advance through unicode states
            if ss == StringState.UNICODE_1:
                self.state.string_state = StringState.UNICODE_2
            elif ss == StringState.UNICODE_2:
                self.state.string_state = StringState.UNICODE_3
            elif ss == StringState.UNICODE_3:
                self.state.string_state = StringState.UNICODE_4
            else:  # UNICODE_4
                self.state.string_state = StringState.NORMAL
            return True
        
        else:  # NORMAL
            if c == '"':
                # End of string
                self.state.string_state = StringState.NONE
                # Check if this was an object key (expect was COLON) - don't change expect
                # Otherwise it's a value and we should call _after_value
                if self.state.expect == Expect.COLON:
                    # This was an object key, expect remains COLON
                    return True
                else:
                    return self._after_value()
            elif c == '\\':
                self.state.string_state = StringState.ESCAPE
                return True
            elif ord(c) < 0x20:
                # Control characters must be escaped
                return self._fail(f"Unescaped control character in string: {repr(c)}")
            else:
                # Regular character
                return True
    
    def _process_number_char(self, c: str) -> bool:
        """Process a character while in a number."""
        ns = self.state.number_state
        
        if ns == NumberState.MINUS:
            if c == '0':
                self.state.number_state = NumberState.ZERO
                return True
            elif c in '123456789':
                self.state.number_state = NumberState.INT_DIGITS
                return True
            else:
                return self._fail(f"Expected digit after minus, got '{c}'")
        
        elif ns == NumberState.ZERO:
            # After leading 0, can have . or e/E or end
            if c == '.':
                self.state.number_state = NumberState.DOT
                return True
            elif c in 'eE':
                self.state.number_state = NumberState.EXP
                return True
            elif c in '0123456789':
                return self._fail("Leading zeros not allowed")
            else:
                # Number ends, process as non-number char
                if not self._finish_number():
                    return self._fail("Invalid number")
                return self._process_non_value_char(c)
        
        elif ns == NumberState.INT_DIGITS:
            if c in '0123456789':
                return True
            elif c == '.':
                self.state.number_state = NumberState.DOT
                return True
            elif c in 'eE':
                self.state.number_state = NumberState.EXP
                return True
            else:
                if not self._finish_number():
                    return self._fail("Invalid number")
                return self._process_non_value_char(c)
        
        elif ns == NumberState.DOT:
            if c in '0123456789':
                self.state.number_state = NumberState.FRAC_DIGITS
                return True
            else:
                return self._fail(f"Expected digit after decimal point, got '{c}'")
        
        elif ns == NumberState.FRAC_DIGITS:
            if c in '0123456789':
                return True
            elif c in 'eE':
                self.state.number_state = NumberState.EXP
                return True
            else:
                if not self._finish_number():
                    return self._fail("Invalid number")
                return self._process_non_value_char(c)
        
        elif ns == NumberState.EXP:
            if c in '+-':
                self.state.number_state = NumberState.EXP_SIGN
                return True
            elif c in '0123456789':
                self.state.number_state = NumberState.EXP_DIGITS
                return True
            else:
                return self._fail(f"Expected digit or sign after exponent, got '{c}'")
        
        elif ns == NumberState.EXP_SIGN:
            if c in '0123456789':
                self.state.number_state = NumberState.EXP_DIGITS
                return True
            else:
                return self._fail(f"Expected digit after exponent sign, got '{c}'")
        
        elif ns == NumberState.EXP_DIGITS:
            if c in '0123456789':
                return True
            else:
                if not self._finish_number():
                    return self._fail("Invalid number")
                return self._process_non_value_char(c)
        
        return self._fail("Invalid number state")
    
    def _after_value(self) -> bool:
        """Update state after completing a value."""
        if not self.state.context_stack:
            # Top-level value complete
            self.state.expect = Expect.END
            return True
        
        ctx = self.state.context_stack[-1]
        if ctx == Context.OBJECT:
            self.state.expect = Expect.COMMA_OR_OBJECT_END
        else:  # ARRAY
            self.state.expect = Expect.COMMA_OR_ARRAY_END
        return True
    
    def _process_non_value_char(self, c: str) -> bool:
        """Process a character that's not part of a value literal."""
        exp = self.state.expect
        
        # Skip whitespace (allowed almost anywhere)
        if _is_ws(c):
            return True
        
        if exp == Expect.END:
            # Already have complete top-level value, only whitespace allowed
            return self._fail(f"Unexpected character after value: '{c}'")
        
        elif exp == Expect.VALUE or exp == Expect.OBJECT_VALUE:
            # Expecting a value
            if c == '"':
                self.state.string_state = StringState.NORMAL
                return True
            elif c == '{':
                self.state.context_stack.append(Context.OBJECT)
                self.state.expect = Expect.OBJECT_KEY_OR_END
                return True
            elif c == '[':
                self.state.context_stack.append(Context.ARRAY)
                self.state.expect = Expect.VALUE
                return True
            elif c == '-':
                self.state.number_state = NumberState.MINUS
                return True
            elif c == '0':
                self.state.number_state = NumberState.ZERO
                return True
            elif c in '123456789':
                self.state.number_state = NumberState.INT_DIGITS
                return True
            elif c == 't':
                # Start of 'true'
                return self._start_keyword('true', 1)
            elif c == 'f':
                # Start of 'false'
                return self._start_keyword('false', 1)
            elif c == 'n':
                # Start of 'null'
                return self._start_keyword('null', 1)
            elif c == ']' and exp == Expect.VALUE and self.state.context_stack and self.state.context_stack[-1] == Context.ARRAY:
                # Empty array
                self.state.context_stack.pop()
                return self._after_value()
            else:
                return self._fail(f"Unexpected character when expecting value: '{c}'")
        
        elif exp == Expect.OBJECT_KEY_OR_END:
            if c == '"':
                self.state.string_state = StringState.NORMAL
                # After string completes, expect colon
                self.state.expect = Expect.COLON
                return True
            elif c == '}':
                self.state.context_stack.pop()
                return self._after_value()
            else:
                return self._fail(f"Expected string key or '}}', got '{c}'")
        
        elif exp == Expect.COLON:
            if c == ':':
                self.state.expect = Expect.OBJECT_VALUE
                return True
            else:
                return self._fail(f"Expected ':', got '{c}'")
        
        elif exp == Expect.COMMA_OR_OBJECT_END:
            if c == ',':
                self.state.expect = Expect.OBJECT_KEY_OR_END
                # But next must be a key, not }
                return True
            elif c == '}':
                self.state.context_stack.pop()
                return self._after_value()
            else:
                return self._fail(f"Expected ',' or '}}', got '{c}'")
        
        elif exp == Expect.COMMA_OR_ARRAY_END:
            if c == ',':
                self.state.expect = Expect.VALUE
                return True
            elif c == ']':
                self.state.context_stack.pop()
                return self._after_value()
            else:
                return self._fail(f"Expected ',' or ']', got '{c}'")
        
        return self._fail(f"Unexpected state: {exp}")
    
    def _start_keyword(self, keyword: str, pos: int) -> bool:
        """
        Start matching a keyword (true, false, null).
        Store the keyword and position to match remaining chars.
        """
        self._keyword = keyword
        self._keyword_pos = pos
        return True
    
    def _continue_keyword(self, c: str) -> bool:
        """Continue matching a keyword."""
        if self._keyword_pos < len(self._keyword):
            expected = self._keyword[self._keyword_pos]
            if c == expected:
                self._keyword_pos += 1
                if self._keyword_pos == len(self._keyword):
                    # Keyword complete
                    self._keyword = None
                    self._keyword_pos = 0
                    return self._after_value()
                return True
            else:
                return self._fail(f"Invalid keyword: expected '{expected}' in '{self._keyword}', got '{c}'")
        return self._fail("Keyword overflow")
    
    def _in_keyword(self) -> bool:
        return hasattr(self, '_keyword') and self._keyword is not None
    
    def feed(self, c: str) -> bool:
        """
        Feed a single character to the validator.
        
        Returns True if the prefix (including this char) is valid.
        Returns False if the prefix is invalid (contradicts JSON grammar).
        """
        if self.error:
            return False
        
        self.state.pos += 1
        
        # Handle keyword matching first
        if self._in_keyword():
            return self._continue_keyword(c)
        
        # Handle string state
        if self._in_string():
            return self._process_string_char(c)
        
        # Handle number state
        if self._in_number():
            return self._process_number_char(c)
        
        # Regular processing
        return self._process_non_value_char(c)
    
    def feed_string(self, s: str) -> bool:
        """
        Feed a string to the validator.
        
        Returns True if the prefix (including all chars) is valid.
        """
        for c in s:
            if not self.feed(c):
                return False
        return True
    
    def is_complete(self) -> bool:
        """
        Check if the current state represents a complete JSON value.
        
        A complete value means:
        - Not in the middle of a string/number/keyword
        - All containers are closed
        - Expect is END (or we have a complete value at top level)
        """
        if self.error:
            return False
        if self._in_string() or self._in_keyword():
            return False
        
        # Check if we're in a valid ending state for a number
        if self._in_number():
            ns = self.state.number_state
            if ns not in (NumberState.ZERO, NumberState.INT_DIGITS,
                         NumberState.FRAC_DIGITS, NumberState.EXP_DIGITS):
                return False
        
        # Must have no open containers
        if self.state.context_stack:
            return False
        
        # Must be in END state or have just completed a value
        return self.state.expect == Expect.END or (
            self.state.number_state != NumberState.NONE and
            self.state.number_state in (NumberState.ZERO, NumberState.INT_DIGITS,
                                        NumberState.FRAC_DIGITS, NumberState.EXP_DIGITS)
        )
    
    def can_continue(self) -> bool:
        """Check if more input could lead to a valid JSON."""
        return self.error is None


def is_valid_json_prefix(text: str) -> bool:
    """
    Check if text is a valid prefix of a JSON document.
    
    A valid prefix means the text could be extended to form valid JSON.
    This is true if no syntax error has been detected so far.
    
    Args:
        text: The string to check
        
    Returns:
        True if text is a valid JSON prefix, False otherwise
    """
    validator = JsonPrefixValidator()
    return validator.feed_string(text)


def is_complete_json(text: str) -> bool:
    """
    Check if text is a complete, valid JSON document.
    
    This means json.loads(text) would succeed and there's no
    trailing non-whitespace content.
    
    Args:
        text: The string to check
        
    Returns:
        True if text is complete valid JSON, False otherwise
    """
    # First check with our validator for the prefix
    validator = JsonPrefixValidator()
    if not validator.feed_string(text):
        return False
    
    if not validator.is_complete():
        return False
    
    # Double-check with json.loads (handles edge cases)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def get_validation_error(text: str) -> Optional[str]:
    """
    Get the validation error for an invalid JSON prefix.
    
    Args:
        text: The string to check
        
    Returns:
        Error message if invalid, None if valid
    """
    validator = JsonPrefixValidator()
    validator.feed_string(text)
    return validator.error


# =============================================================================
# Additional utilities for constrained decoding
# =============================================================================

def could_start_json_value(c: str) -> bool:
    """Check if a character could start a JSON value."""
    return c in '"{[0123456789-tfn' or _is_ws(c)


def get_valid_next_chars(text: str) -> set[str]:
    """
    Get the set of characters that could validly follow the given prefix.
    
    This is useful for constrained decoding at the character level.
    
    Args:
        text: Current JSON prefix
        
    Returns:
        Set of characters that would keep the prefix valid
    """
    if not is_valid_json_prefix(text):
        return set()
    
    # Test all possible next characters
    valid_chars = set()
    
    # Common JSON characters to test
    test_chars = (
        '{}[],:"\\'
        '0123456789'
        '.-+eE'
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ' \t\n\r'
        '_'
    )
    
    for c in test_chars:
        if is_valid_json_prefix(text + c):
            valid_chars.add(c)
    
    return valid_chars

