import tokenize
import io

import ast
import sys

def remove_comments_and_docstrings(source_code: str) -> str:
    """
    Removes comments and docstrings from a Python source code string using the ast module.
    
    Args:
        source_code (str): The source code to clean.
        
    Returns:
        str: The source code with comments and docstrings removed.
    """
    # ast.parse() automatically ignores/removes comments immediately.
    parsed = ast.parse(source_code)
    
    # Walk the tree to find and remove docstrings
    for node in ast.walk(parsed):
        # Docstrings are only found in Modules, Classes, and Functions
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        
        # A docstring is always the first statement in the body
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            
            # Check if the expression is a constant string
            # (In Python 3.8+, string literals are ast.Constant with string values)
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                # It is a docstring, remove it from the body
                node.body.pop(0)
            # Fallback for older Python AST structures (though ast.unparse requires 3.9+)
            elif hasattr(ast, 'Str') and isinstance(expr.value, ast.Str):
                 node.body.pop(0)

    # ast.unparse reconstructs the source code from the tree (Python 3.9+)
    # It handles indentation and ensures valid syntax (e.g., inserting 'pass' if a body becomes empty)
    return ast.unparse(parsed)



def remove_comments(source_code: str) -> str:
    """
    Removes comments from a Python source code string using the tokenize module.
    
    Args:
        source_code (str): The source code to clean.
        
    Returns:
        str: The source code with comments removed.
    """
    # Convert string to bytes for tokenizer
    io_obj = io.BytesIO(source_code.encode('utf-8'))
    
    # Generator for tokens
    out_tokens = []
    last_lineno = -1
    last_col = 0

    # Iterate through tokens
    for token in tokenize.tokenize(io_obj.readline):
        token_type = token.type
        token_string = token.string
        start_line, start_col = token.start
        end_line, end_col = token.end

        # Skip comment tokens
        if token_type == tokenize.COMMENT:
            continue
            
        # Add token to list (preserving formatting is handled by untokenize, 
        # but we filter first)
        out_tokens.append(token)

    # untokenize reconstructs the code from the tokens
    return tokenize.untokenize(out_tokens).decode('utf-8')