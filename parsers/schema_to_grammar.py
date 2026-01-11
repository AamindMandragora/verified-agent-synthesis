"""
JSON Schema to Lark Grammar Converter (Character-Level).

Converts a JSON Schema into a Lark grammar that:
1. Uses character-level string building (works with LLM tokenizers)
2. Enforces required field names by spelling them character-by-character
3. Allows any string content for values

This enables constrained decoding to generate schema-compliant JSON
with LLM vocabularies that have partial string tokens.

Usage:
    from parsers.schema_to_grammar import json_schema_to_lark_grammar
    
    schema = {"type": "object", "properties": {...}, "required": [...]}
    grammar = json_schema_to_lark_grammar(schema)
"""

from typing import Any, Dict, List, Optional, Set
import json


def json_schema_to_lark_grammar(schema: Dict[str, Any], start_rule: str = "start") -> str:
    """
    Convert a JSON Schema to a character-level Lark grammar.
    
    Args:
        schema: JSON Schema dictionary
        start_rule: Name of the start rule (default: "start")
        
    Returns:
        Lark grammar string with character-level string support
    """
    converter = SchemaToGrammarConverter()
    return converter.convert(schema, start_rule)


class SchemaToGrammarConverter:
    """
    Converts JSON Schema to character-level Lark grammar.
    
    Key features:
    - Strings are built character-by-character (works with LLM vocab)
    - Required property names are spelled out exactly
    - Values can be any valid JSON
    """
    
    def __init__(self):
        self._rules: List[str] = []
        self._rule_counter = 0
        self._defined_rules: Set[str] = set()
        self._key_rules: Dict[str, str] = {}  # Maps property name -> rule name
    
    def _new_rule_name(self, prefix: str = "rule") -> str:
        """Generate a unique rule name."""
        self._rule_counter += 1
        return f"{prefix}_{self._rule_counter}"
    
    def convert(self, schema: Dict[str, Any], start_rule: str = "start") -> str:
        """Convert schema to character-level grammar."""
        self._rules = []
        self._rule_counter = 0
        self._defined_rules = set()
        self._key_rules = {}
        
        # Generate the main value rule
        value_rule = self._convert_type(schema, "root_value")
        
        # Build the complete grammar
        grammar_parts = [
            "// Auto-generated character-level grammar from JSON Schema",
            "// Property names are spelled character-by-character for LLM compatibility",
            "",
            f"{start_rule}: {value_rule}",
            "",
        ]
        
        # Add generated rules
        grammar_parts.extend(self._rules)
        
        # Add character-level terminal definitions
        grammar_parts.extend(self._get_charwise_terminals())
        
        return "\n".join(grammar_parts)
    
    def _get_charwise_terminals(self) -> List[str]:
        """Return character-level terminal definitions."""
        return [
            "",
            "// Character-level string definitions",
            "// Allows incremental string building with LLM tokenizers",
            "",
            "// Generic string (any content)",
            "generic_string: DQUOTE string_chars DQUOTE",
            "string_chars: STRING_CHAR*",
            "",
            "// String character terminals",
            'DQUOTE: "\\""',
            "// Any printable ASCII except quote and backslash",
            r'STRING_CHAR: /[ !#-\[\]-~]/ | ESCAPE_SEQ',
            r'ESCAPE_SEQ: /\\[\"\\\/bfnrt]/ | /\\u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]/',
            "",
            "// Number definition",
            "number: SIGNED_NUMBER",
            "",
            "// Boolean and null",
            'BOOL_TRUE: "true"',
            'BOOL_FALSE: "false"', 
            'NULL: "null"',
            "",
            "%import common.SIGNED_NUMBER",
            "%import common.WS",
            "%ignore WS",
        ]
    
    def _make_charwise_key(self, prop_name: str) -> str:
        """
        Create a grammar rule that spells out a property name character-by-character.
        
        For "ssid", generates: key_ssid: DQUOTE "s" "s" "i" "d" DQUOTE
        
        This allows the grammar to match partial tokens from LLM vocabularies.
        """
        if prop_name in self._key_rules:
            return self._key_rules[prop_name]
        
        rule_name = f"key_{self._sanitize_name(prop_name)}"
        
        # Spell out each character
        char_literals = []
        for char in prop_name:
            if char == '"':
                char_literals.append(r'"\""')
            elif char == '\\':
                char_literals.append(r'"\\"')
            else:
                char_literals.append(f'"{char}"')
        
        char_sequence = " ".join(char_literals)
        self._rules.append(f'{rule_name}: DQUOTE {char_sequence} DQUOTE')
        
        self._key_rules[prop_name] = rule_name
        return rule_name
    
    def _convert_type(self, schema: Dict[str, Any], hint: str = "") -> str:
        """Convert a schema type to a grammar rule reference."""
        schema_type = schema.get("type")
        
        if schema_type == "object":
            return self._convert_object(schema, hint)
        elif schema_type == "array":
            return self._convert_array(schema, hint)
        elif schema_type == "string":
            return "generic_string"
        elif schema_type == "number" or schema_type == "integer":
            return "number"
        elif schema_type == "boolean":
            return "(BOOL_TRUE | BOOL_FALSE)"
        elif schema_type == "null":
            return "NULL"
        elif isinstance(schema_type, list):
            return self._convert_union(schema_type, schema, hint)
        elif "anyOf" in schema or "oneOf" in schema:
            return self._convert_any_of(schema, hint)
        elif "enum" in schema:
            return self._convert_enum(schema, hint)
        else:
            # Fallback to generic value
            self._ensure_generic_value()
            return "generic_value"
    
    def _convert_object(self, schema: Dict[str, Any], hint: str = "") -> str:
        """Convert an object schema to grammar rules with character-level keys."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        if not properties:
            # Empty object schema - allow any object
            rule_name = self._new_rule_name("object")
            self._ensure_generic_pair()
            self._rules.append(f'{rule_name}: "{{" [generic_pair ("," generic_pair)*] "}}"')
            return rule_name
        
        rule_name = self._new_rule_name("object")
        
        # Separate required and optional properties
        required_props = [(k, v) for k, v in properties.items() if k in required]
        optional_props = [(k, v) for k, v in properties.items() if k not in required]
        
        if not required_props:
            # No required properties - all optional
            pair_rules = []
            for prop_name, prop_schema in properties.items():
                pair_rule = self._make_pair_rule(prop_name, prop_schema)
                pair_rules.append(pair_rule)
            
            if pair_rules:
                optional_rule = " | ".join(pair_rules)
                self._rules.append(f'{rule_name}: "{{" [({optional_rule}) ("," ({optional_rule}))*] "}}"')
            else:
                self._rules.append(f'{rule_name}: "{{" "}}"')
            return rule_name
        
        # Build grammar that requires all required fields
        if len(required_props) <= 4:
            return self._convert_object_with_permutations(
                rule_name, required_props, optional_props
            )
        else:
            return self._convert_object_ordered(
                rule_name, required_props, optional_props
            )
    
    def _convert_object_with_permutations(
        self,
        rule_name: str,
        required_props: List[tuple],
        optional_props: List[tuple]
    ) -> str:
        """
        Generate grammar allowing required fields in any order.
        Uses permutation enumeration for small field counts.
        """
        from itertools import permutations
        
        # Create pair rules for each required property
        req_pair_rules = []
        for prop_name, prop_schema in required_props:
            pair_rule = self._make_pair_rule(prop_name, prop_schema)
            req_pair_rules.append(pair_rule)
        
        # Create pair rules for optional properties
        opt_pair_rules = []
        for prop_name, prop_schema in optional_props:
            pair_rule = self._make_pair_rule(prop_name, prop_schema)
            opt_pair_rules.append(pair_rule)
        
        # Optional fields suffix
        if opt_pair_rules:
            opt_combined = " | ".join(opt_pair_rules)
            opt_rule_name = self._new_rule_name("optional_fields")
            self._rules.append(f'{opt_rule_name}: ("," ({opt_combined}))*')
            opt_suffix = f" {opt_rule_name}"
        else:
            opt_suffix = ""
        
        # Generate all permutations of required fields
        if len(req_pair_rules) == 1:
            body = '"{" ' + req_pair_rules[0] + opt_suffix + ' "}"'
        else:
            perm_alternatives = []
            for perm in permutations(req_pair_rules):
                perm_str = ' "," '.join(perm)
                perm_alternatives.append(f'({perm_str})')
            
            alternatives = " | ".join(perm_alternatives)
            body = '"{" (' + alternatives + ')' + opt_suffix + ' "}"'
        
        self._rules.append(f'{rule_name}: {body}')
        return rule_name
    
    def _convert_object_ordered(
        self,
        rule_name: str,
        required_props: List[tuple],
        optional_props: List[tuple]
    ) -> str:
        """
        Generate grammar with required fields in a fixed order.
        Used for schemas with many required fields.
        """
        all_pairs = []
        for prop_name, prop_schema in required_props:
            pair_rule = self._make_pair_rule(prop_name, prop_schema)
            all_pairs.append(pair_rule)
        
        if len(all_pairs) == 1:
            required_part = all_pairs[0]
        else:
            required_part = ' "," '.join(all_pairs)
        
        if optional_props:
            opt_rules = []
            for prop_name, prop_schema in optional_props:
                opt_rules.append(self._make_pair_rule(prop_name, prop_schema))
            opt_combined = " | ".join(opt_rules)
            opt_part = f' ("," ({opt_combined}))*'
        else:
            opt_part = ""
        
        self._rules.append(f'{rule_name}: "{{" {required_part}{opt_part} "}}"')
        return rule_name
    
    def _make_pair_rule(self, prop_name: str, prop_schema: Dict[str, Any]) -> str:
        """Create a grammar rule for a key:value pair with character-level key."""
        # Create character-level key rule
        key_rule = self._make_charwise_key(prop_name)
        
        # Get the value type
        value_type = self._convert_type(prop_schema, f"prop_{prop_name}")
        
        # Create the pair rule
        pair_rule_name = self._new_rule_name(f"pair_{self._sanitize_name(prop_name)}")
        self._rules.append(f'{pair_rule_name}: {key_rule} ":" {value_type}')
        return pair_rule_name
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a property name for use in rule names."""
        return "".join(c.lower() if c.isalnum() else "_" for c in name)
    
    def _convert_array(self, schema: Dict[str, Any], hint: str = "") -> str:
        """Convert an array schema to grammar rules."""
        items_schema = schema.get("items", {})
        
        rule_name = self._new_rule_name("array")
        
        if items_schema:
            item_type = self._convert_type(items_schema, f"{hint}_item")
            self._rules.append(f'{rule_name}: "[" [{item_type} ("," {item_type})*] "]"')
        else:
            self._ensure_generic_value()
            self._rules.append(f'{rule_name}: "[" [generic_value ("," generic_value)*] "]"')
        
        return rule_name
    
    def _convert_union(self, types: List[str], schema: Dict[str, Any], hint: str) -> str:
        """Convert a union type to grammar alternatives."""
        alternatives = []
        for t in types:
            if t == "string":
                alternatives.append("generic_string")
            elif t == "number" or t == "integer":
                alternatives.append("number")
            elif t == "boolean":
                alternatives.append("(BOOL_TRUE | BOOL_FALSE)")
            elif t == "null":
                alternatives.append("NULL")
            elif t == "object":
                alternatives.append(self._convert_object(schema, hint))
            elif t == "array":
                alternatives.append(self._convert_array(schema, hint))
        
        if len(alternatives) == 1:
            return alternatives[0]
        
        rule_name = self._new_rule_name("union")
        self._rules.append(f'{rule_name}: {" | ".join(alternatives)}')
        return rule_name
    
    def _convert_any_of(self, schema: Dict[str, Any], hint: str) -> str:
        """Convert anyOf/oneOf to grammar alternatives."""
        options = schema.get("anyOf") or schema.get("oneOf", [])
        
        alternatives = []
        for i, opt_schema in enumerate(options):
            alt = self._convert_type(opt_schema, f"{hint}_opt{i}")
            alternatives.append(alt)
        
        if len(alternatives) == 1:
            return alternatives[0]
        
        rule_name = self._new_rule_name("anyof")
        self._rules.append(f'{rule_name}: {" | ".join(alternatives)}')
        return rule_name
    
    def _convert_enum(self, schema: Dict[str, Any], hint: str) -> str:
        """Convert enum to grammar with literal alternatives."""
        values = schema.get("enum", [])
        
        if not values:
            return "generic_string"
        
        alternatives = []
        for val in values:
            if isinstance(val, str):
                # For enum strings, spell out character-by-character
                char_literals = ['DQUOTE']
                for char in val:
                    if char == '"':
                        char_literals.append(r'"\""')
                    elif char == '\\':
                        char_literals.append(r'"\\"')
                    else:
                        char_literals.append(f'"{char}"')
                char_literals.append('DQUOTE')
                alternatives.append("(" + " ".join(char_literals) + ")")
            elif isinstance(val, bool):
                alternatives.append("BOOL_TRUE" if val else "BOOL_FALSE")
            elif val is None:
                alternatives.append("NULL")
            elif isinstance(val, (int, float)):
                alternatives.append(str(val))
        
        if len(alternatives) == 1:
            return alternatives[0]
        
        rule_name = self._new_rule_name("enum")
        self._rules.append(f'{rule_name}: {" | ".join(alternatives)}')
        return rule_name
    
    def _ensure_generic_value(self):
        """Ensure generic_value rule exists for unknown types."""
        if "generic_value" not in self._defined_rules:
            self._defined_rules.add("generic_value")
            self._rules.append(
                'generic_value: generic_string | number | BOOL_TRUE | BOOL_FALSE | NULL | generic_object | generic_array'
            )
            self._ensure_generic_object()
            self._ensure_generic_array()
    
    def _ensure_generic_object(self):
        """Ensure generic_object rule exists."""
        if "generic_object" not in self._defined_rules:
            self._defined_rules.add("generic_object")
            self._ensure_generic_pair()
            self._rules.append('generic_object: "{" [generic_pair ("," generic_pair)*] "}"')
    
    def _ensure_generic_array(self):
        """Ensure generic_array rule exists."""
        if "generic_array" not in self._defined_rules:
            self._defined_rules.add("generic_array")
            self._ensure_generic_value()
            self._rules.append('generic_array: "[" [generic_value ("," generic_value)*] "]"')
    
    def _ensure_generic_pair(self):
        """Ensure generic_pair rule exists."""
        if "generic_pair" not in self._defined_rules:
            self._defined_rules.add("generic_pair")
            self._rules.append('generic_pair: generic_string ":" generic_value')
            self._ensure_generic_value()


def create_schema_specific_grammar(schema: Dict[str, Any]) -> str:
    """
    Convenience function to create a schema-specific grammar.
    
    This is the main entry point for evaluation scripts.
    """
    return json_schema_to_lark_grammar(schema)


# ============================================================================
# SQL Schema to Grammar Converter (for Spider dataset)
# ============================================================================

def spider_schema_to_sql_grammar(
    table_names: list,
    column_names: list,
    base_grammar_path: str = None
) -> str:
    """
    Create a schema-aware SQL grammar that restricts table/column names
    to those from the actual database schema.
    
    Args:
        table_names: List of table names in the database
        column_names: List of column names (can include table.column format)
        base_grammar_path: Path to base SQL grammar (uses built-in if None)
        
    Returns:
        Lark grammar string with schema-specific name constraints
    """
    from pathlib import Path
    
    # Get base SQL grammar
    if base_grammar_path:
        base_grammar = Path(base_grammar_path).read_text()
    else:
        # Use the default sql_no_subquery.lark
        grammar_dir = Path(__file__).parent.parent / "grammars"
        base_grammar = (grammar_dir / "sql_no_subquery.lark").read_text()
    
    # Collect all valid identifiers (tables, columns, aliases)
    # IMPORTANT: SQLite is case-sensitive for identifiers, so we only include exact case
    # from the schema. We include original case + lowercase (common convention) for flexibility.
    all_names = set()
    
    # Add table names - prioritize exact case, but also allow lowercase (common SQL convention)
    for tname in table_names:
        all_names.add(tname)  # original case (required - SQLite is case-sensitive)
        if tname.lower() != tname:
            all_names.add(tname.lower())  # lowercase variant (common SQL convention)
    
    # Add column names (may be "col" or "table.col")
    for cname in column_names:
        if "." in cname:
            parts = cname.split(".")
            for p in parts:
                all_names.add(p)  # original case
                if p.lower() != p:
                    all_names.add(p.lower())  # lowercase variant
        else:
            all_names.add(cname)  # original case
            if cname.lower() != cname:
                all_names.add(cname.lower())  # lowercase variant
    
    # Add common SQL aliases (T1, T2, etc.) that models often generate
    for i in range(1, 11):
        all_names.add(f"T{i}")
        all_names.add(f"t{i}")
    
    # Add common aggregate aliases
    for alias in ["count", "avg", "sum", "min", "max", "cnt", "total"]:
        all_names.add(alias)
        all_names.add(alias.upper())
    
    # Build the name alternatives
    # Sort by length descending so longer matches are tried first
    sorted_names = sorted(all_names, key=lambda x: (-len(x), x))
    
    # Create case-SENSITIVE patterns for names (SQLite is case-sensitive for identifiers)
    # We include all case variants but match them exactly
    name_alternatives = []
    for n in sorted_names:
        if n:
            # Escape any special regex characters in the name
            escaped = n.replace('"', '\\"')
            # Use case-sensitive match (no 'i' flag) - SQLite requires exact case
            name_alternatives.append(f'"{escaped}"')
    
    # Replace the name rule in the grammar
    # Original: name: CNAME | ESCAPED_STRING
    # New: name: "table1" | "table2" | "col1" | ... | (CNAME for string literals only)
    
    # IMPORTANT: We still need CNAME and ESCAPED_STRING for string literals and quoted identifiers
    # in queries, but we prioritize exact schema names first.
    num_names = len(name_alternatives)
    # Increase limit to 1000 to include more schema names before falling back
    # This helps prevent CNAME fallback which allows invalid identifiers
    max_names = 1000
    new_name_rule = "name: " + " | ".join(name_alternatives[:max_names])
    if num_names > max_names:
        # Still include CNAME for string literals, but schema names are tried first
        new_name_rule += " | CNAME"
        # Note: truncated to max_names to keep grammar size manageable
    # Always include ESCAPED_STRING for quoted identifiers (e.g., "table name")
    new_name_rule += " | ESCAPED_STRING"
    
    # Replace the name rule in the grammar
    # Use more flexible regex to handle whitespace variations
    import re
    modified_grammar = re.sub(
        r'^name:\s*CNAME\s*\|\s*ESCAPED_STRING\s*$',
        new_name_rule,
        base_grammar,
        flags=re.MULTILINE
    )
    
    # If the above didn't match (e.g., different formatting), try a more flexible pattern
    if modified_grammar == base_grammar:
        # Try matching with optional whitespace around the pipe
        modified_grammar = re.sub(
            r'^name:\s*CNAME\s*\|\s*ESCAPED_STRING',
            new_name_rule,
            base_grammar,
            flags=re.MULTILINE
        )
    
    return modified_grammar


def create_spider_schema_grammar(db_meta: dict, base_grammar_path: str = None) -> str:
    """
    Create a schema-aware grammar from Spider's table metadata.
    
    Args:
        db_meta: Spider table metadata dict (from tables.json for one db)
            Expected keys: table_names_original, column_names_original
        base_grammar_path: Optional path to base SQL grammar
        
    Returns:
        Schema-specific Lark grammar string
    """
    table_names = db_meta.get("table_names_original", [])
    
    # column_names_original is list of [table_idx, column_name]
    column_names = []
    for table_idx, col_name in db_meta.get("column_names_original", []):
        if table_idx >= 0 and col_name != "*":
            column_names.append(col_name)
    
    # Log schema stats (only if verbose, but we'll keep it minimal)
    num_tables = len(table_names)
    num_columns = len(column_names)
    
    return spider_schema_to_sql_grammar(table_names, column_names, base_grammar_path)


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple schema
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        },
        "required": ["name", "age"]
    }
    
    grammar = json_schema_to_lark_grammar(test_schema)
    print("Generated Character-Level Grammar:")
    print("=" * 60)
    print(grammar)
    print("=" * 60)
    
    # Test parsing
    try:
        from lark import Lark
        parser = Lark(grammar, start="start", parser="lalr")
        
        # Test valid JSON
        test_json = '{"name": "John", "age": 30}'
        tree = parser.parse(test_json)
        print(f"\n✓ Valid: {test_json}")
        
        # Test with optional field
        test_json2 = '{"name": "Jane", "age": 25, "email": "jane@example.com"}'
        tree2 = parser.parse(test_json2)
        print(f"✓ Valid: {test_json2}")
        
        # Test different order
        test_json3 = '{"age": 40, "name": "Bob"}'
        tree3 = parser.parse(test_json3)
        print(f"✓ Valid (different order): {test_json3}")
        
        # Test invalid (missing required field)
        try:
            invalid_json = '{"name": "Bob"}'  # Missing 'age'
            parser.parse(invalid_json)
            print(f"✗ Should have failed: {invalid_json}")
        except Exception:
            print(f"✓ Correctly rejected (missing 'age'): {invalid_json}")
        
        # Test invalid (wrong field name)
        try:
            invalid_json2 = '{"nam": "Bob", "age": 30}'  # Wrong key 'nam' instead of 'name'
            parser.parse(invalid_json2)
            print(f"✗ Should have failed: {invalid_json2}")
        except Exception:
            print(f"✓ Correctly rejected (wrong key): {invalid_json2}")
        
        # Test prefix validity
        print("\nTesting prefix validity:")
        from lark.exceptions import UnexpectedEOF, UnexpectedToken, UnexpectedCharacters
        
        prefixes = ['{', '{"', '{"n', '{"na', '{"nam', '{"name', '{"name"', '{"name":']
        for p in prefixes:
            try:
                parser.parse(p)
                print(f"  {repr(p)}: complete")
            except UnexpectedEOF:
                print(f"  {repr(p)}: valid prefix (EOF)")
            except UnexpectedToken as e:
                if e.token.type == '$END':
                    print(f"  {repr(p)}: valid prefix")
                else:
                    print(f"  {repr(p)}: invalid")
            except UnexpectedCharacters:
                print(f"  {repr(p)}: invalid chars")
            
    except ImportError:
        print("(Lark not installed, skipping parse tests)")
