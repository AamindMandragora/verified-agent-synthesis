# Parsers module for JSON and grammar-based validation
#
# Two approaches are supported:
#
# 1. GENERIC (recommended for most cases):
#    Use LarkGrammarParser with any .lark grammar file
#    
#    from parsers import LarkGrammarParser, create_parser_for_format
#    parser = create_parser_for_format("json")  # or "python", "sql"
#    parser = LarkGrammarParser.from_grammar_file("my_grammar.lark")
#
# 2. CUSTOM (for performance-critical cases):
#    Hand-coded parsers like JsonPrefixValidator
#    More efficient but requires custom implementation

from .json_prefix import (
    is_valid_json_prefix,
    is_complete_json,
    get_validation_error,
    get_valid_next_chars,
    JsonPrefixValidator,
)

from .model_token_parser import (
    ModelTokenJsonParser,
    CachedModelTokenJsonParser,
    create_json_parser,
)

from .lark_parser import (
    LarkGrammarParser,
    InteractiveLarkParser,
    create_parser_for_format,
    create_json_lark_grammar,
    create_python_lark_grammar,
    create_sql_lark_grammar,
)

from .schema_to_grammar import (
    json_schema_to_lark_grammar,
    create_schema_specific_grammar,
)

__all__ = [
    # Generic Lark-based parsing (recommended)
    "LarkGrammarParser",
    "InteractiveLarkParser", 
    "create_parser_for_format",
    "create_json_lark_grammar",
    "create_python_lark_grammar",
    "create_sql_lark_grammar",
    # Schema-to-grammar conversion
    "json_schema_to_lark_grammar",
    "create_schema_specific_grammar",
    # JSON-specific (hand-optimized)
    "is_valid_json_prefix",
    "is_complete_json",
    "get_validation_error",
    "get_valid_next_chars",
    "JsonPrefixValidator",
    # Model token parsing
    "ModelTokenJsonParser",
    "CachedModelTokenJsonParser",
    "create_json_parser",
]

