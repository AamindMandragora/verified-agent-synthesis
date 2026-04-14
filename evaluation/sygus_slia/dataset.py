"""
SyGuS SLIA dataset: embedded string-manipulation benchmark problems.

Each problem is defined by:
  - A natural-language description
  - A list of (input_bindings, expected_output) pairs
  - The variable name(s) in the expression

Evaluation: interpret the LM-generated S-expression with the given bindings
and check if the result matches expected_output.

Problems are inspired by the SyGuS-SLIA benchmark from the ASAp paper
(Geng et al., 2405.21047).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SLIAExample:
    """A single SyGuS SLIA problem instance."""
    problem_id: str
    description: str          # Natural language task description
    var_name: str             # Primary input variable name in the expression
    # Each entry: ({"var": value, ...}, expected_output_string)
    test_ios: List[Tuple[Dict[str, str], str]]
    # Single held-out eval pair shown to the model
    eval_input: Dict[str, str]
    eval_expected: str

    @property
    def question(self) -> str:
        """Format as a prompt-ready question string."""
        lines = [f"Task: {self.description}", ""]
        lines.append("Examples:")
        for bindings, expected in self.test_ios[:3]:
            args = ", ".join(f'{k}="{v}"' for k, v in bindings.items())
            lines.append(f"  f({args}) = \"{expected}\"")
        args = ", ".join(f'{k}="{v}"' for k, v in self.eval_input.items())
        lines.append(f"\nNow compute: f({args}) = ?")
        return "\n".join(lines)

    @property
    def answer(self) -> str:
        return self.eval_expected


# ---------------------------------------------------------------------------
# Embedded problem definitions
# ---------------------------------------------------------------------------

_PROBLEMS: List[dict] = [
    {
        "problem_id": "firstname",
        "description": (
            "Extract the first name from a full name string. "
            "The input is a full name in 'First Last' format."
        ),
        "var_name": "name",
        "test_ios": [
            ({"name": "John Smith"},   "John"),
            ({"name": "Alice Jones"},  "Alice"),
            ({"name": "Bob Brown"},    "Bob"),
            ({"name": "Carol White"},  "Carol"),
        ],
        "eval_input":    {"name": "David Lee"},
        "eval_expected": "David",
    },
    {
        "problem_id": "lastname",
        "description": (
            "Extract the last name from a full name string. "
            "The input is a full name in 'First Last' format."
        ),
        "var_name": "name",
        "test_ios": [
            ({"name": "John Smith"},   "Smith"),
            ({"name": "Alice Jones"},  "Jones"),
            ({"name": "Bob Brown"},    "Brown"),
            ({"name": "Carol White"},  "White"),
        ],
        "eval_input":    {"name": "David Lee"},
        "eval_expected": "Lee",
    },
    {
        "problem_id": "reverse-name",
        "description": (
            "Reformat a full name from 'First Last' to 'Last, First'."
        ),
        "var_name": "name",
        "test_ios": [
            ({"name": "John Smith"},  "Smith, John"),
            ({"name": "Alice Jones"}, "Jones, Alice"),
            ({"name": "Bob Brown"},   "Brown, Bob"),
        ],
        "eval_input":    {"name": "Carol White"},
        "eval_expected": "White, Carol",
    },
    {
        "problem_id": "email-user",
        "description": (
            "Extract the username (part before '@') from an email address."
        ),
        "var_name": "email",
        "test_ios": [
            ({"email": "alice@example.com"},  "alice"),
            ({"email": "bob@mail.org"},       "bob"),
            ({"email": "carol@test.net"},     "carol"),
            ({"email": "david@domain.io"},    "david"),
        ],
        "eval_input":    {"email": "eve@server.edu"},
        "eval_expected": "eve",
    },
    {
        "problem_id": "email-domain",
        "description": (
            "Extract the domain (part after '@') from an email address."
        ),
        "var_name": "email",
        "test_ios": [
            ({"email": "alice@example.com"},  "example.com"),
            ({"email": "bob@mail.org"},       "mail.org"),
            ({"email": "carol@test.net"},     "test.net"),
            ({"email": "david@domain.io"},    "domain.io"),
        ],
        "eval_input":    {"email": "eve@server.edu"},
        "eval_expected": "server.edu",
    },
    {
        "problem_id": "add-greeting",
        "description": (
            "Prepend 'Hello, ' to a name to form a greeting string."
        ),
        "var_name": "name",
        "test_ios": [
            ({"name": "Alice"},  "Hello, Alice"),
            ({"name": "Bob"},    "Hello, Bob"),
            ({"name": "Carol"},  "Hello, Carol"),
            ({"name": "David"},  "Hello, David"),
        ],
        "eval_input":    {"name": "Eve"},
        "eval_expected": "Hello, Eve",
    },
    {
        "problem_id": "uppercase",
        "description": (
            "Convert a string to all uppercase letters."
        ),
        "var_name": "s",
        "test_ios": [
            ({"s": "hello"},  "HELLO"),
            ({"s": "world"},  "WORLD"),
            ({"s": "alice"},  "ALICE"),
            ({"s": "bob"},    "BOB"),
        ],
        "eval_input":    {"s": "carol"},
        "eval_expected": "CAROL",
    },
    {
        "problem_id": "file-basename",
        "description": (
            "Extract the filename without extension from 'name.ext'."
        ),
        "var_name": "filename",
        "test_ios": [
            ({"filename": "report.pdf"},   "report"),
            ({"filename": "image.png"},    "image"),
            ({"filename": "data.csv"},     "data"),
            ({"filename": "notes.txt"},    "notes"),
        ],
        "eval_input":    {"filename": "script.py"},
        "eval_expected": "script",
    },
]


# ---------------------------------------------------------------------------
# S-expression interpreter
# ---------------------------------------------------------------------------

def eval_slia(tree_or_expr, bindings: Dict[str, str]) -> str:
    """
    Evaluate a parsed SyGuS SLIA expression (Lark Tree or raw string).

    Args:
        tree_or_expr: Either a Lark Tree produced by parsing with start="start",
                      or a plain string (interpreted as a variable reference or literal).
        bindings:     Dict mapping variable names to their string values.

    Returns:
        The result as a Python string or int.
    """
    from lark import Tree, Token

    def _eval(node) -> Any:
        if isinstance(node, Token):
            if node.type == "VAR":
                name = str(node)
                if name not in bindings:
                    raise KeyError(f"Unbound variable: {name!r}")
                return bindings[name]
            if node.type == "STR_LIT":
                # Strip surrounding quotes
                return str(node)[1:-1]
            if node.type == "INT_LIT":
                return int(str(node))
            return str(node)

        if not isinstance(node, Tree):
            return node

        data = node.data
        ch = node.children

        if data == "concat":
            return str(_eval(ch[0])) + str(_eval(ch[1]))
        if data == "at":
            s, i = str(_eval(ch[0])), int(_eval(ch[1]))
            return s[i] if 0 <= i < len(s) else ""
        if data == "substr":
            s, start, length = str(_eval(ch[0])), int(_eval(ch[1])), int(_eval(ch[2]))
            return s[start: start + length] if start >= 0 else ""
        if data == "replace":
            s, old, new = str(_eval(ch[0])), str(_eval(ch[1])), str(_eval(ch[2]))
            return s.replace(old, new, 1)
        if data == "upper":
            return str(_eval(ch[0])).upper()
        if data == "lower":
            return str(_eval(ch[0])).lower()
        if data == "strlen":
            return len(str(_eval(ch[0])))
        if data == "indexof":
            s, sub, start = str(_eval(ch[0])), str(_eval(ch[1])), int(_eval(ch[2]))
            idx = s.find(sub, start)
            return idx  # -1 if not found, consistent with SMT-LIB str.indexof
        if data == "int_add":
            return int(_eval(ch[0])) + int(_eval(ch[1]))
        if data == "int_sub":
            return int(_eval(ch[0])) - int(_eval(ch[1]))

        # Transparent wrapper rules (Lark ?-aliased rules are already inlined,
        # but handle any remaining tree nodes gracefully)
        if len(ch) == 1:
            return _eval(ch[0])

        raise ValueError(f"Unknown SLIA node type: {data!r}")

    result = _eval(tree_or_expr)
    return str(result)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_sygus_slia(
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> List[SLIAExample]:
    """
    Load embedded SyGuS SLIA problems.

    Args:
        limit:         Maximum number of problems to return.
        random_sample: If True, sample randomly (with seed) instead of taking first N.
        seed:          Random seed for sampling.

    Returns:
        List of SLIAExample instances.
    """
    examples = [SLIAExample(**p) for p in _PROBLEMS]

    if random_sample and limit is not None:
        rng = random.Random(seed)
        examples = rng.sample(examples, min(limit, len(examples)))
    elif limit is not None:
        examples = examples[:limit]

    return examples
