"""
Support utilities for the Python-to-Dafny transpiler.
"""

from .comment_stripping import remove_comments, remove_comments_and_docstrings
from .dynamic_type_resolution import is_builtin_type_string, resolve_type_from_string
from .mypy_type_checker import MypyTypeChecker, TypeCheck
from .result import Err, Ok, Result

__all__ = [
    "Err",
    "Ok",
    "Result",
    "MypyTypeChecker",
    "TypeCheck",
    "remove_comments",
    "remove_comments_and_docstrings",
    "is_builtin_type_string",
    "resolve_type_from_string",
]
