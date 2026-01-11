"""
Unit tests for the JSON prefix validator.

Tests cover:
- Valid and invalid JSON prefixes
- Complete JSON detection
- String escapes and unicode
- Number formats
- Nested structures
- Edge cases
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.json_prefix import (
    is_valid_json_prefix,
    is_complete_json,
    get_validation_error,
    get_valid_next_chars,
    JsonPrefixValidator,
)


class TestBasicValues:
    """Test basic JSON value types."""
    
    def test_empty_is_valid_prefix(self):
        assert is_valid_json_prefix("") is True
    
    def test_empty_is_not_complete(self):
        assert is_complete_json("") is False
    
    def test_null(self):
        assert is_valid_json_prefix("n") is True
        assert is_valid_json_prefix("nu") is True
        assert is_valid_json_prefix("nul") is True
        assert is_valid_json_prefix("null") is True
        assert is_complete_json("null") is True
    
    def test_true(self):
        assert is_valid_json_prefix("t") is True
        assert is_valid_json_prefix("tr") is True
        assert is_valid_json_prefix("tru") is True
        assert is_valid_json_prefix("true") is True
        assert is_complete_json("true") is True
    
    def test_false(self):
        assert is_valid_json_prefix("f") is True
        assert is_valid_json_prefix("fa") is True
        assert is_valid_json_prefix("fal") is True
        assert is_valid_json_prefix("fals") is True
        assert is_valid_json_prefix("false") is True
        assert is_complete_json("false") is True
    
    def test_invalid_keyword(self):
        assert is_valid_json_prefix("trux") is False
        assert is_valid_json_prefix("nope") is False


class TestStrings:
    """Test JSON string handling."""
    
    def test_simple_string(self):
        assert is_valid_json_prefix('"') is True
        assert is_valid_json_prefix('"h') is True
        assert is_valid_json_prefix('"hello') is True
        assert is_valid_json_prefix('"hello"') is True
        assert is_complete_json('"hello"') is True
    
    def test_empty_string(self):
        assert is_valid_json_prefix('""') is True
        assert is_complete_json('""') is True
    
    def test_escape_sequences(self):
        assert is_valid_json_prefix('"\\n"') is True
        assert is_complete_json('"\\n"') is True
        
        assert is_valid_json_prefix('"\\t"') is True
        assert is_complete_json('"\\t"') is True
        
        assert is_valid_json_prefix('"\\r"') is True
        assert is_complete_json('"\\r"') is True
        
        assert is_valid_json_prefix('"\\b"') is True
        assert is_complete_json('"\\b"') is True
        
        assert is_valid_json_prefix('"\\f"') is True
        assert is_complete_json('"\\f"') is True
        
        assert is_valid_json_prefix('"\\/"') is True
        assert is_complete_json('"\\/"') is True
        
        assert is_valid_json_prefix('"\\\\"') is True
        assert is_complete_json('"\\\\"') is True
        
        assert is_valid_json_prefix('"\\""') is True
        assert is_complete_json('"\\""') is True
    
    def test_unicode_escape(self):
        assert is_valid_json_prefix('"\\u') is True
        assert is_valid_json_prefix('"\\u0') is True
        assert is_valid_json_prefix('"\\u00') is True
        assert is_valid_json_prefix('"\\u004') is True
        assert is_valid_json_prefix('"\\u0041') is True
        assert is_valid_json_prefix('"\\u0041"') is True
        assert is_complete_json('"\\u0041"') is True
        
        # Invalid hex
        assert is_valid_json_prefix('"\\uGGGG') is False
        assert is_valid_json_prefix('"\\u00XY') is False
    
    def test_invalid_escape(self):
        assert is_valid_json_prefix('"\\x"') is False
        assert is_valid_json_prefix('"\\q"') is False
    
    def test_control_characters(self):
        # Unescaped control characters are invalid
        assert is_valid_json_prefix('"\x00"') is False
        assert is_valid_json_prefix('"\x1f"') is False


class TestNumbers:
    """Test JSON number handling."""
    
    def test_integers(self):
        assert is_valid_json_prefix("0") is True
        assert is_complete_json("0") is True
        
        assert is_valid_json_prefix("1") is True
        assert is_complete_json("1") is True
        
        assert is_valid_json_prefix("123") is True
        assert is_complete_json("123") is True
        
        assert is_valid_json_prefix("999999") is True
        assert is_complete_json("999999") is True
    
    def test_negative_integers(self):
        assert is_valid_json_prefix("-") is True
        assert is_complete_json("-") is False
        
        assert is_valid_json_prefix("-0") is True
        assert is_complete_json("-0") is True
        
        assert is_valid_json_prefix("-123") is True
        assert is_complete_json("-123") is True
    
    def test_decimals(self):
        assert is_valid_json_prefix("0.") is True
        assert is_complete_json("0.") is False
        
        assert is_valid_json_prefix("0.5") is True
        assert is_complete_json("0.5") is True
        
        assert is_valid_json_prefix("123.456") is True
        assert is_complete_json("123.456") is True
    
    def test_exponents(self):
        assert is_valid_json_prefix("1e") is True
        assert is_complete_json("1e") is False
        
        assert is_valid_json_prefix("1e5") is True
        assert is_complete_json("1e5") is True
        
        assert is_valid_json_prefix("1E5") is True
        assert is_complete_json("1E5") is True
        
        assert is_valid_json_prefix("1e+5") is True
        assert is_complete_json("1e+5") is True
        
        assert is_valid_json_prefix("1e-5") is True
        assert is_complete_json("1e-5") is True
        
        assert is_valid_json_prefix("1.5e10") is True
        assert is_complete_json("1.5e10") is True
    
    def test_invalid_numbers(self):
        # Leading zeros not allowed
        assert is_valid_json_prefix("01") is False
        assert is_valid_json_prefix("007") is False
        
        # Plus sign not allowed at start
        assert is_valid_json_prefix("+1") is False
        
        # Double minus not allowed
        assert is_valid_json_prefix("--1") is False


class TestObjects:
    """Test JSON object handling."""
    
    def test_empty_object(self):
        assert is_valid_json_prefix("{") is True
        assert is_valid_json_prefix("{}") is True
        assert is_complete_json("{}") is True
    
    def test_simple_object(self):
        assert is_valid_json_prefix('{"key"') is True
        assert is_valid_json_prefix('{"key":') is True
        assert is_valid_json_prefix('{"key": ') is True
        assert is_valid_json_prefix('{"key": "value"') is True
        assert is_valid_json_prefix('{"key": "value"}') is True
        assert is_complete_json('{"key": "value"}') is True
    
    def test_multiple_keys(self):
        assert is_valid_json_prefix('{"a": 1, "b": 2}') is True
        assert is_complete_json('{"a": 1, "b": 2}') is True
    
    def test_nested_objects(self):
        assert is_valid_json_prefix('{"outer": {"inner": 1}}') is True
        assert is_complete_json('{"outer": {"inner": 1}}') is True
    
    def test_object_with_array_value(self):
        assert is_valid_json_prefix('{"items": [1, 2, 3]}') is True
        assert is_complete_json('{"items": [1, 2, 3]}') is True
    
    def test_invalid_object(self):
        # Unquoted key
        assert is_valid_json_prefix('{key}') is False
        
        # Missing colon
        assert is_valid_json_prefix('{"key" "value"}') is False
        
        # Trailing comma
        assert is_valid_json_prefix('{"key": 1,}') is False
    
    def test_object_with_all_value_types(self):
        obj = '{"str": "hello", "num": 42, "float": 3.14, "bool": true, "null": null, "arr": [1], "obj": {}}'
        assert is_valid_json_prefix(obj) is True
        assert is_complete_json(obj) is True


class TestArrays:
    """Test JSON array handling."""
    
    def test_empty_array(self):
        assert is_valid_json_prefix("[") is True
        assert is_valid_json_prefix("[]") is True
        assert is_complete_json("[]") is True
    
    def test_simple_array(self):
        assert is_valid_json_prefix("[1") is True
        assert is_valid_json_prefix("[1,") is True
        assert is_valid_json_prefix("[1, 2") is True
        assert is_valid_json_prefix("[1, 2]") is True
        assert is_complete_json("[1, 2]") is True
    
    def test_string_array(self):
        assert is_valid_json_prefix('["a", "b", "c"]') is True
        assert is_complete_json('["a", "b", "c"]') is True
    
    def test_nested_arrays(self):
        assert is_valid_json_prefix("[[1, 2], [3, 4]]") is True
        assert is_complete_json("[[1, 2], [3, 4]]") is True
    
    def test_deeply_nested(self):
        deep = "[[[[[[1]]]]]]"
        assert is_valid_json_prefix(deep) is True
        assert is_complete_json(deep) is True
    
    def test_invalid_array(self):
        # Trailing comma
        assert is_valid_json_prefix("[1,]") is False
        
        # Missing comma
        assert is_valid_json_prefix("[1 2]") is False


class TestWhitespace:
    """Test whitespace handling."""
    
    def test_leading_whitespace(self):
        assert is_valid_json_prefix("  {") is True
        assert is_valid_json_prefix("  {}") is True
        assert is_complete_json("  {}") is True
    
    def test_trailing_whitespace(self):
        assert is_valid_json_prefix("{}  ") is True
        assert is_complete_json("{}  ") is True
    
    def test_internal_whitespace(self):
        assert is_valid_json_prefix('{ "key" : "value" }') is True
        assert is_complete_json('{ "key" : "value" }') is True
    
    def test_newlines_and_tabs(self):
        obj = '{\n  "key": "value",\n  "other": 123\n}'
        assert is_valid_json_prefix(obj) is True
        assert is_complete_json(obj) is True


class TestEdgeCases:
    """Test edge cases and tricky inputs."""
    
    def test_unmatched_brackets(self):
        assert is_valid_json_prefix("{") is True
        assert is_complete_json("{") is False
        
        assert is_valid_json_prefix("[") is True
        assert is_complete_json("[") is False
        
        # Unmatched close is invalid
        assert is_valid_json_prefix("}") is False
        assert is_valid_json_prefix("]") is False
    
    def test_mismatched_brackets(self):
        assert is_valid_json_prefix("{]") is False
        assert is_valid_json_prefix("[}") is False
    
    def test_multiple_top_level_values(self):
        # Second value after first complete one is invalid
        assert is_valid_json_prefix("{}{}") is False
        assert is_valid_json_prefix("1 2") is False
        assert is_valid_json_prefix('"a" "b"') is False
    
    def test_garbage_input(self):
        assert is_valid_json_prefix("hello") is False
        assert is_valid_json_prefix("undefined") is False
        assert is_valid_json_prefix("NaN") is False
        assert is_valid_json_prefix("Infinity") is False


class TestValidNextChars:
    """Test the get_valid_next_chars function."""
    
    def test_empty_prefix(self):
        chars = get_valid_next_chars("")
        assert "{" in chars
        assert "[" in chars
        assert '"' in chars
        assert "t" in chars  # start of true
        assert "f" in chars  # start of false
        assert "n" in chars  # start of null
        assert "-" in chars
        for d in "0123456789":
            assert d in chars
    
    def test_after_open_brace(self):
        chars = get_valid_next_chars("{")
        assert '"' in chars  # key must be string
        assert "}" in chars  # empty object
        assert " " in chars  # whitespace
        assert "a" not in chars  # unquoted key invalid
    
    def test_after_key(self):
        chars = get_valid_next_chars('{"key"')
        assert ":" in chars
        assert " " in chars
        assert "}" not in chars  # need value first
    
    def test_in_string(self):
        chars = get_valid_next_chars('"hello')
        assert '"' in chars  # end string
        assert "\\" in chars  # escape
        assert "a" in chars  # more chars


class TestJsonPrefixValidator:
    """Test the JsonPrefixValidator class directly."""
    
    def test_incremental_feeding(self):
        v = JsonPrefixValidator()
        assert v.feed("{") is True
        assert v.feed('"') is True
        assert v.feed("k") is True
        assert v.feed("e") is True
        assert v.feed("y") is True
        assert v.feed('"') is True
        assert v.feed(":") is True
        assert v.feed("1") is True
        assert v.feed("}") is True
        assert v.is_complete() is True
    
    def test_error_reporting(self):
        v = JsonPrefixValidator()
        v.feed("{")
        v.feed("x")  # Invalid - unquoted key
        assert v.error is not None
        assert "unexpected" in v.error.lower() or "expected" in v.error.lower()
    
    def test_can_continue(self):
        v = JsonPrefixValidator()
        assert v.can_continue() is True
        v.feed("{")
        assert v.can_continue() is True
        v.feed("x")  # Error
        assert v.can_continue() is False


class TestRealWorldExamples:
    """Test with real-world JSON examples."""
    
    def test_api_response(self):
        json_str = '''
        {
            "status": "success",
            "data": {
                "users": [
                    {"id": 1, "name": "Alice", "active": true},
                    {"id": 2, "name": "Bob", "active": false}
                ],
                "total": 2,
                "page": 1
            },
            "message": null
        }
        '''
        assert is_valid_json_prefix(json_str) is True
        assert is_complete_json(json_str) is True
    
    def test_geojson(self):
        json_str = '''
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [125.6, 10.1]
            },
            "properties": {
                "name": "Dinagat Islands"
            }
        }
        '''
        assert is_valid_json_prefix(json_str) is True
        assert is_complete_json(json_str) is True
    
    def test_package_json(self):
        json_str = '''
        {
            "name": "my-package",
            "version": "1.0.0",
            "dependencies": {
                "lodash": "^4.17.21"
            },
            "scripts": {
                "test": "jest"
            }
        }
        '''
        assert is_valid_json_prefix(json_str) is True
        assert is_complete_json(json_str) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

