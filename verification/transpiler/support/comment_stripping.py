from __future__ import annotations

import ast
import io
import tokenize


def remove_comments_and_docstrings(source_code: str) -> str:
    """
    Remove comments and docstrings from Python source.
    """
    parsed = ast.parse(source_code)

    for node in ast.walk(parsed):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue

        expr = node.body[0]
        if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
            node.body.pop(0)
        elif hasattr(ast, "Str") and isinstance(expr.value, ast.Str):
            node.body.pop(0)

    return ast.unparse(parsed)


def remove_comments(source_code: str) -> str:
    """Remove comments while preserving the remaining Python tokens."""
    io_obj = io.BytesIO(source_code.encode("utf-8"))
    out_tokens = []

    for token in tokenize.tokenize(io_obj.readline):
        if token.type == tokenize.COMMENT:
            continue
        out_tokens.append(token)

    return tokenize.untokenize(out_tokens).decode("utf-8")
