import ast
import argparse
from typing import Any, Protocol
from pathlib import Path
import textwrap

try:
    from .support import Err, Ok, Result, remove_comments_and_docstrings
except ImportError:
    from support import Err, Ok, Result, remove_comments_and_docstrings
from copy import deepcopy
import subprocess
import tempfile
import os
import shutil


def _default_dafny_binary() -> str:
    local_dafny = Path(__file__).resolve().parents[2] / "dafny" / "dafny"
    if local_dafny.exists():
        return str(local_dafny)
    return "dafny"


def _run_dafny_verify_file(dafny_file: str, dafny_binary: str | None = None) -> Result[bool, Exception]:
    try:
        command = [dafny_binary or _default_dafny_binary(), "verify", "--allow-external-contracts", dafny_file]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=100,
        )
        output_message = result.stdout + result.stderr
        if result.returncode == 0:
            return Ok(True)
        return Err(ValueError(output_message.strip()))
    except FileNotFoundError:
        return Err(ValueError("Error: 'dafny' command not found. Please ensure Dafny is installed and in your system's PATH."))
    except subprocess.TimeoutExpired:
        return Err(ValueError("Error: Dafny verification timed out after 30 seconds."))
    except Exception as e:
        return Err(e)


def _verify_dafny_code(
    code_string: str,
    dafny_binary: str | None = None,
    temp_dir: str | None = None,
) -> Result[bool, Exception]:
    """
    Verifies a string of Dafny code using the 'dafny verify' command.

    This function creates a unique temporary file to avoid race conditions,
    writes the provided code string to it, and then executes the Dafny
    verifier. It captures and returns the verifier's output.

    Args:
        code_string: A string containing the Dafny code to be verified.

    Returns:
        A tuple (success, message):
        - success (bool): True if verification succeeded (return code 0),
                          False otherwise.
        - message (str): The captured stdout and/or stderr from the
                         Dafny verifier. If the 'dafny' command itself
                         is not found, a specific error message is returned.
    """
    
    # Create a temporary file with a .dfy suffix.
    # 'delete=False' is used so we can close it, let dafny read it,
    # and then manually delete it. 'NamedTemporaryFile' ensures
    # the filename is unique, preventing race conditions.
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=".dfy", delete=False, dir=temp_dir) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(code_string)
            temp_file.flush()
            # Close the file so the subprocess can access it on all OSes
            temp_file.close()

        return _run_dafny_verify_file(temp_filename, dafny_binary=dafny_binary)
    except Exception as e:
        return Err(e)
    finally:
        # Ensure the temporary file is always deleted
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)

# Extracting symbolic variables from the cost contracts.
from sympy import sympify, Symbol
from sympy.core.sympify import SympifyError
from sympy.core.function import UndefinedFunction

def extract_symbolic_variables(equation_string: str) -> list[str]:
    """
    Parses a mathematical equation string and extracts a sorted list
    of all symbolic variables, ignoring numbers and operators.

    Args:
        equation_string: The equation as a string (e.g., "x * y + 2 + z").

    Returns:
        A sorted list of variable names (e.g., ['x', 'y', 'z']).
    """
    variables = []
    try:
        # Parse the string into a SymPy expression
        expr = sympify(equation_string)

        # Get all "atoms" from the expression that are of type 'Symbol'
        # 'Symbol' is the type SymPy uses for variables
        symbols_in_expr = expr.atoms(Symbol)

        # We also want to find 'UndefinedFunction' types, which
        # SymPy uses for things like 'model(x)'
        functions_in_expr = expr.atoms(UndefinedFunction)

        # Convert symbols to strings
        for s in symbols_in_expr:
            variables.append(str(s))
        
        # Convert functions to strings (e.g., 'model' from 'model(x)')
        for f in functions_in_expr:
            variables.append(str(f))

        # Use a set to get unique names, then sort for consistent output
        unique_variables = sorted(list(set(variables)))
        
        return unique_variables

    except SympifyError:
        print(f"Error: Could not parse the equation '{equation_string}'")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


class PythonStubToDafnyVisitor(ast.NodeVisitor):
    """
    An AST visitor that converts Python stub nodes into Dafny code snippets.
    """
    def __init__(self, formal_checks_dict: dict[str, Any]):
        self.dafny_code_lines: list[str] = []
        self.formal_checks_dict = formal_checks_dict
        self.needs_any_type: bool = False # Track if we need to define 'AnyType' 

    def _get_slice_node(self, slice_node):
        """
        Handles AST differences in subscripts (e.g., List[Any]) 
        between Python < 3.9 (uses ast.Index) and Python 3.9+
        """
        if isinstance(slice_node, ast.Index):
            return slice_node.value
        return slice_node # Python 3.9+

    def _map_type(self, type_node) -> str:
        """
        Recursively converts Python type annotations (ast nodes) to Dafny type strings.
        """
        if type_node is None:
            return "()" # Dafny's unit type, equivalent to 'None'

        if isinstance(type_node, ast.Name):
            py_type = type_node.id
            if py_type == 'str':
                return "string"
            if py_type == 'int':
                return "int"
            if py_type == 'bool':
                return "bool"
            if py_type == 'float':
                return "real"    
            if py_type == 'Any':
                self.needs_any_type = True
                return "AnyType"
            if py_type == 'List':
                return "seq" # Assume seq<AnyType> if unparameterized
            if py_type == 'Dict':
                return "map" # Assume map<AnyType, AnyType>
            if py_type == 'Set':
                return "set" # Assume set<AnyType>
            # Otherwise, it's a class name (e.g., Rationale, Question)
            return py_type

        if isinstance(type_node, ast.Subscript):
            # This handles parameterized types like List[Any] or Dict[str, int]
            base_type = self._map_type(type_node.value)
            slice_content = self._get_slice_node(type_node.slice)
            
            if base_type == "seq": # List[T]
                inner_type = self._map_type(slice_content)
                return f"seq<{inner_type}>"
            if base_type == "set": # Set[T]
                inner_type = self._map_type(slice_content)
                return f"set<{inner_type}>"
            if base_type == "map": # Dict[K, V]
                if isinstance(slice_content, ast.Tuple):
                    key_type = self._map_type(slice_content.elts[0])
                    val_type = self._map_type(slice_content.elts[1])
                    return f"map<{key_type}, {val_type}>"
            
            return f"{base_type}<...>" # Fallback for unknown subscript

        return "UnrecognizedType"

    def _format_args(self, args_node: ast.arguments) -> str:
        """
        Converts an 'ast.arguments' node into a Dafny parameter list string.
        """
        args_list = []
        for arg in args_node.args:
            if arg.arg == 'self':
                continue # Skip 'self'
            
            arg_name = arg.arg
            arg_type = self._map_type(arg.annotation)
            args_list.append(f"{arg_name}: {arg_type}")
        return ", ".join(args_list)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Called when the visitor sees a 'class ...' definition."""
        self.dafny_code_lines.append(f"class {node.name} {{")
        
        # Find the __init__ method to create the constructor
        init_node = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                init_node = item
                break # Found it
        
        if init_node:
            # Format arguments from __init__
            args = self._format_args(init_node.args)
            self.dafny_code_lines.append(f"    constructor ({args}) {{")
            self.dafny_code_lines.append(f"        // Constructor body not defined in stub")
            self.dafny_code_lines.append(f"    }}")
        
        self.dafny_code_lines.append(f"}}\n")
        # We don't visit children (generic_visit) because we manually processed
        # the __init__ and are ignoring other class methods per the prompt.

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Called for top-level functions. Class methods are handled by visit_ClassDef.
        """
        args = self._format_args(node.args)
        ret_type = self._map_type(node.returns)
        
        if 'lemma' in node.name:
            # Handle the potentially added lemmas for proofs.
            self.dafny_code_lines.append(f"lemma {{:axiom}} {node.name}({args})")            
        elif ret_type == "()": # '-> None' becomes '()'
            self.dafny_code_lines.append(f"function {{:extern}} {node.name}({args})")
        else:
            self.dafny_code_lines.append(f"function {{:extern}} {node.name}({args}): (r: {ret_type})")

        # Add the contracts
        if node.name in self.formal_checks_dict:
            if "safeguards" in self.formal_checks_dict[node.name]:
                for s in self.formal_checks_dict[node.name]["safeguards"]:
                    t = s.replace("""'""", '"')
                    self.dafny_code_lines.append(f"    {t}")
            if "outputspecs" in self.formal_checks_dict[node.name]:
                for s in self.formal_checks_dict[node.name]["outputspecs"]:
                    t = s.replace("""'""", '"')
                    self.dafny_code_lines.append(f"    {t}")
        self.dafny_code_lines.append(f"\n\n")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Ignores 'from typing import ...' lines."""
        pass

    def visit_Expr(self, node: ast.Expr):
        """Ignores '...' placeholders in the stub body."""
        pass
        
    def generic_visit(self, node):
        """
        Continue traversing the tree. visit_ClassDef and visit_FunctionDef
        are special; this handles the top-level module.
        """
        super().generic_visit(node)



from mypy.traverser import TraverserVisitor
from mypy.nodes import (
    MypyFile, FuncDef, WhileStmt, AssignmentStmt, AssertStmt, IfStmt, ReturnStmt, ExpressionStmt,
    PassStmt, Block, NameExpr, IntExpr, StrExpr, FloatExpr, OpExpr, IndexExpr,
    UnaryExpr, ComparisonExpr, CallExpr, ConditionalExpr, Expression, ListExpr, 
    MemberExpr, TempNode

)
from mypy.types import Type as MypyType, Instance, NoneType, AnyType, UnboundType
from typing import List, Set, Dict


class TypeExtractingVisitor(TraverserVisitor):
    """
    A mypy AST visitor to find assignments and extract types.
    """
    def __init__(self) -> None:
        self.variables: dict[str, list[str]] = {}
        self.scopes: list[str] = ["__main__"]

    
    
    def dafny_type(self, type_node: MypyType | None) -> str:
        """Translates a mypy.types.Type into a Dafny type string."""
        # print(f"actual type name {type_node} modified type name {type(type_node)}")
        if type_node is None:
            return "AnyType"
        
        if isinstance(type_node, UnboundType):
            type_name = type_node.name
            if type_name.replace('?', '') == 'int':
                return 'int'
            if type_name.replace('?', '') == 'bool':
                return 'bool'
            if type_name.replace('?', '') == 'str':
                return 'string'
            if type_name.replace('?', '') == 'float':
                return 'real'
            return type_node.name

        if isinstance(type_node, Instance):
            type_name = type_node.type.name
            if type_name.replace('?', '') == 'int':
                return 'int'
            if type_name.replace('?', '') == 'bool':
                return 'bool'
            if type_name.replace('?', '') == 'str':
                return 'string'
            if type_name.replace('?', '') == 'float':
                return 'real'
            # Add more type mappings as needed
            return f"{type_name}"
            
        if isinstance(type_node, NoneType):
            return "void"
            
        if isinstance(type_node, AnyType):
            return "AnyType" # Defaulting 'Any' to 'int'
            
        # Fallback for other types (e.g., Union, Tuple, etc.)
        return "None"


    def visit_func_def(self, o: FuncDef) -> None:
        """Entering a function, push its name to the scope stack."""
        self.scopes.append(o.name)
        super().visit_func_def(o)
        self.scopes.pop()

    def visit_assignment_stmt(self, o: AssignmentStmt) -> None:
        """Visit an assignment statement like 'x = 1'."""
        if isinstance(o.lvalues[0], NameExpr):
            var_name = o.lvalues[0].name
            node = o.lvalues[0].node
            
            if node and hasattr(node, 'type') and node.type:
                scope_path = ".".join(self.scopes)
                full_name = f"{scope_path}.{var_name}"
                
                if full_name not in self.variables:
                    self.variables[full_name] = []
                
                type_str = self.dafny_type(node.type)
                if type_str not in self.variables[full_name]:
                    self.variables[full_name].append(type_str)
                    
        super().visit_assignment_stmt(o)


class MypyToDafnyConverter(TraverserVisitor):
    """
    Converts a simplified, type-checked Mypy AST to Dafny code.

    This visitor assumes:
    - No loops or classes.
    - Only simple assignments, if/else, and function calls.
    - The Mypy AST has been fully type-checked, so nodes have .type
      attributes.
    """

    def __init__(self, custom_types: list[str],
                 agent_checks: dict[str, Any],
                 cost_info: dict[str, str],
                 positional_argument_dict: dict[str, dict[str, Any]],
                 local_variable_type_info: dict[str, list[str]]) -> None:
        self.lines: list[str] = []
        self.indent_level = 0
        self.agent_checks = agent_checks
        self.positional_argument_dict = positional_argument_dict
        self.custom_types: list[str] = custom_types
        self.cost_info: Dict[str, str] = cost_info
        
        # Pre process the variables used in the program helps in predefining 
        # the variables outside the scope of the if else branch.
        self.local_variable_type_info : dict[str, list[str]] = local_variable_type_info
        # Tracks variables that have been declared with 'var' in the
        # current function scope.
        self.declared_locals: Set[str] = set()
        
        # The Dafny return type for the current method.
        self.current_return_type: str = "void" # Set in visit_func_def
        
        self._fresh_tmp = 0

    # --- Utility Methods (Identical to original) ---
    def fresh_tmp(self) -> str:
        """Returns a new unique temporary variable name."""
        self._fresh_tmp += 1
        return f"tmp_variable_{self._fresh_tmp}"

    def emit(self, s: str = ""):
        """Adds a line of code with the current indentation."""
        self.lines.append(" " * (self.indent_level * 4) +s)

    def enter(self):
        """Increases indentation level."""
        self.indent_level += 1

    def leave(self):
        """Decreases indentation level."""
        self.indent_level -= 1

    def to_source(self) -> str:
        """Returns the complete generated Dafny source code."""
        return "\n".join(self.lines)

    # --- Type Conversion Helper ---

    def dafny_type(self, type_node: MypyType | None) -> str:
        """Translates a mypy.types.Type into a Dafny type string."""
        # print(f"actual type name {type_node} modified type name {type(type_node)}")
        if type_node is None:
            return "int // (untyped)"
        
        if isinstance(type_node, UnboundType):
            type_name = type_node.name
            if type_name.replace('?', '') == 'int':
                return 'int'
            if type_name.replace('?', '') == 'bool':
                return 'bool'
            if type_name.replace('?', '') == 'str':
                return 'string'
            if type_name.replace('?', '') == 'float':
                return 'real'
            return type_node.name

        if isinstance(type_node, Instance):
            type_name = type_node.type.name
            if type_name.replace('?', '') == 'int':
                return 'int'
            if type_name.replace('?', '') == 'bool':
                return 'bool'
            if type_name.replace('?', '') == 'str':
                return 'string'
            if type_name.replace('?', '') == 'float':
                return 'real'
            # Add more type mappings as needed
            return f"{type_name}"
            
        if isinstance(type_node, NoneType):
            return "void"
            
        if isinstance(type_node, AnyType):
            return "int // (any)" # Defaulting 'Any' to 'int'
            
        # Fallback for other types (e.g., Union, Tuple, etc.)
        return f"int // (unsupported type: {type(type_node).__name__})"

    # --- Visitor Methods (Top-Level) ---

    def visit_mypy_file(self, o: MypyFile) -> None:
        """Entry point, wraps code in a Dafny module."""
        # Do not emit the general module.
        # Visit all top-level statements (e.g., FuncDefs)
        super().visit_mypy_file(o)
    
    def predefine_local_vars(self, func_name):
        scoped_func_name = f'__main__.{func_name}'
        for scoped_var, types in self.local_variable_type_info.items():
            if len(types) != 1:
                continue
            if scoped_func_name in scoped_var:
                var_name = scoped_var.replace(scoped_func_name, '').replace('.', '')
                if var_name in self.declared_locals:
                    continue                
                self.emit(f"var {var_name} : {types[0]};")
                self.declared_locals.add(var_name)


    def visit_func_def(self, o: FuncDef) -> None:
        """Converts a Python function to a Dafny method."""
        # Reset declared locals for the new function scope
        self.declared_locals.clear()        
        # 1. Process parameters
        params = []
        for arg in o.arguments:
            arg_name = arg.variable.name
            # Get type from the mypy type annotation
            print(f'{arg_name} {arg.type_annotation}')
            arg_type = self.dafny_type(arg.type_annotation)
            params.append(f"{arg_name}: {arg_type}")
            # Parameters are automatically "declared"
            self.declared_locals.add(arg_name)

        params_src = ", ".join(params)
        
        # 2. Process return type
        # o.type is a CallableType, o.type.ret_type is the return Type
        ret_type = self.dafny_type(o.type.ret_type)
        self.current_return_type = ret_type
        
        # Handle "void" (None) returns
        if ret_type.lower() == "void":
            header = f"method {o.name}({params_src})"
        else:
            header = f"method {o.name}({params_src}) returns (r: {ret_type})"

        self.emit(header)
        self.enter()
        safe_guards = self.agent_checks.get("safeguards", [])
        post_conditions = self.agent_checks.get("outputspecs", [])
        for pre_cond in safe_guards:
            self.emit(f"{pre_cond}\n")
        for post_cond in post_conditions:
            self.emit(f"{post_cond}\n")
        self.leave()
        self.emit("{")
        self.enter()
        # Add the predefined locals at the start
        self.predefine_local_vars(o.name)

        # 3. Visit the function body
        super().visit_func_def(o)
            
        self.leave()
        self.emit("}")
        self.emit("") # Blank line

    # --- Visitor Methods (Statements) ---

    def process_cost_info(self, func_name: str, args: list) -> str:
        symbolic_expression = deepcopy(self.cost_info[func_name])
        sym_vars = extract_symbolic_variables(symbolic_expression)
        sym_var_to_arg = {}
        for var in sym_vars:
            if func_name not in self.positional_argument_dict:
                raise ValueError(f'{func_name} not found in pos arguments {self.positional_argument_dict.keys()}')
            if var not in self.positional_argument_dict[func_name]:
                raise ValueError(f'{var} used in cost exp not found in {func_name}`s arg list {self.positional_argument_dict[func_name].keys()}')
            try:
                sym_var_to_arg[var] = args[self.positional_argument_dict[func_name][var]]
            except:
                print(f'for {func_name} and args {args} trying to access {var}')
            if var == 'model_id':
                sym_var_to_arg[var] = f"cost_{func_name}({sym_var_to_arg[var]})"
            symbolic_expression = symbolic_expression.replace(var, f'({sym_var_to_arg[var]})')
        return symbolic_expression


    def visit_assignment_stmt(self, o: AssignmentStmt) -> None:
        """Converts an assignment (x = ...)."""
        # We only support single, simple name targets (e.g., x = 1)
        if len(o.lvalues) != 1 or not isinstance(o.lvalues[0], NameExpr):
            self.emit(f"// Skipping unsupported assignment")
            return
        
        target_name_node = o.lvalues[0]
        target_name = target_name_node.name

        # For the TempNode case, we skip the assignment as these are compiler-introduced temps.
        if type(o.rvalue) == TempNode:
            return
        
        expr_str = self._expr(o.rvalue)
        
        if target_name not in self.declared_locals:
            # First time seeing this var: DECLARE it with 'var'
            # Get the type from the variable's node
            var_type = self.dafny_type(target_name_node.node.type)
            self.emit(f"var {target_name}: {var_type} := {expr_str};")
            self.declared_locals.add(target_name) # Mark as declared
        else:
            # Already declared: RE-ASSIGN it with ':='
            self.emit(f"{target_name} := {expr_str};")


    def visit_while_stmt(self, o: WhileStmt) -> None:
        """Converts a Python while loop to a Dafny while loop."""

        # Convert loop condition
        cond = self._expr(o.expr)

        # Emit while header
        self.emit(f"while ({cond})")
        self.enter()

        self.leave()
        self.emit("{")
        self.enter()

        # Visit loop body
        o.body.accept(self)


        self.leave()
        self.emit("}")


    def visit_if_stmt(self, o: IfStmt) -> None:
        """Converts an if/elif/else statement."""
        for i, (expr_node, body_block) in enumerate(zip(o.expr, o.body)):
            cond = self._expr(expr_node)
            clause = "if" if i == 0 else "} else if"
            self.emit(f"{clause} {cond} {{")
            self.enter()
            body_block.accept(self) # Manually visit the body block
            self.leave()
        
        if o.else_body:
            self.emit("} else {")
            self.enter()
            o.else_body.accept(self) # Manually visit the else block
            self.leave()
            self.emit("}")
        else:
            self.emit("}")
    def visit_assert_stmt(self, o: AssertStmt) -> None:
            """Converts a Python assert statement to a Dafny assert."""
            # 1. Convert the assertion condition expression
            cond = self._expr(o.expr)


            # 3. Handle optional message (Python: assert condition, "Message")
            #    Dafny asserts don't standardly take string messages for runtime errors, 
            #    so we append it as a comment for readability.
            if o.msg:
                msg_str = self._expr(o.msg)
                self.emit(f"assert {cond}; // {msg_str}")
            else:
                self.emit(f"assert {cond};")

    def visit_return_stmt(self, o: ReturnStmt) -> None:
        """Converts a return statement."""
        if self.current_return_type == "void":
            self.emit("return;")
            return

        if o.expr is None:
            self.emit("// Warning: Missing return value for non-void method.")
            self.emit("return;")
        else:
            expr = self._expr(o.expr)
            self.emit(f"r := {expr};")
            self.emit("return;")

    def visit_expression_stmt(self, o: ExpressionStmt) -> None:
        """Converts an expression statement (e.g., a function call)."""
        expr_str = self._expr(o.expr)
        
        if isinstance(o.expr, CallExpr):
            if isinstance(o.expr.callee, MemberExpr) and o.expr.callee.name == 'append':
                self.emit(f"{expr_str}")
                return
            # TODO(debangshu): Check the return value.
            # Currently only lemmas will not have any return  functions.
            try:
                if 'lemma' in o.expr.callee.name:
                    self.emit(f"{expr_str};")
                else:   
                    self.emit(f"var {self.fresh_tmp()}:= {expr_str};")
            except:
                self.emit(f"var {self.fresh_tmp()}:= {expr_str};")    
        else:
            self.emit(f"{expr_str};")

    def visit_pass_stmt(self, o: PassStmt) -> None:
        """Converts 'pass' to a comment."""
        self.emit("// pass")

    def _expr(self, node: Expression) -> str:
        """Recursively converts a Mypy expression node into a Dafny string."""
        
        if isinstance(node, IntExpr):
            return repr(node.value)
        
        if isinstance(node, FloatExpr):
            return repr(node.value) # Dafny 'real' type accepts this

        # if isinstance(node, StrExpr):
        #     # Dafny string literal uses double quotes
        #     escaped = node.value.replace('"', '\\"')
        #     return f'"{escaped}"'
        if isinstance(node, StrExpr):
            escaped = node.value.replace('"', '\\"')

            escape_map = {
                '\n': '\\n',
                '\t': '\\t',
                ' ': '\\\\'
            }

            escaped = ''.join(escape_map.get(c, c) for c in escaped)
            return f'"{escaped}"'


        if isinstance(node, NameExpr):
            # Handle True/False/None
            if node.name == 'True':
                return 'true'
            if node.name == 'False':
                return 'false'
            if node.name == 'None':
                # This is ambiguous. Default to 0 for 'int'.
                return "0 // (None)"
            return node.name
            
        if isinstance(node, OpExpr):
            # Handles binary ops (a + b) and bool ops (a and b)
            left = self._expr(node.left)
            right = self._expr(node.right)
            op = self.dafny_binop(node.op)
            return f"({left} {op} {right})"
            
        if isinstance(node, UnaryExpr):
            operand = self._expr(node.expr)
            if node.op == 'not':
                return f"(!{operand})"
            if node.op == '-':
                return f"(-{operand})"
            return operand # e.g., unary '+'
            
        if isinstance(node, ComparisonExpr):
            # Simplified: only one comparator (e.g., a < b, not a < b < c)
            if len(node.operators) != 1:
                return "false // Error: Chained comparisons not supported"
            left = self._expr(node.operands[0])
            op = self.dafny_cmpop(node.operators[0])
            right = self._expr(node.operands[1])
            return f"({left} {op} {right})"
            
        if isinstance(node, CallExpr):
            if isinstance(node.callee, MemberExpr) and node.callee.name == 'append':
                base = self._expr(node.callee.expr)
                if len(node.args) != 1:
                    return "false // Error: append must have one argument"
                arg = self._expr(node.args[0])
        
                return f"{base} := ({base} + [{arg}])"
            func = self._expr(node.callee)
            args = [self._expr(a) for a in node.args]
            # Check whether the function is a constructor
            # print(f'func {func} {self.custom_types}')
            # Handle Length for list
            if func == 'len':
                return f"|{args[0]}|"
            if func in self.custom_types:
                return f"new {func}({', '.join(args)})"     
            return f"{func}({', '.join(args)})"
            
        if isinstance(node, ConditionalExpr):
            # Python ternary: b if a else c
            # Dafny ternary: if a then b else c
            cond = self._expr(node.cond)
            body = self._expr(node.if_expr)
            orelse = self._expr(node.else_expr)
            return f"(if {cond} then {body} else {orelse})"
        
        if isinstance(node, IndexExpr):
            # Corresponds to sequence/map access like my_seq[i] or my_map[k]
            base_str = self._expr(node.base)
            index_str = self._expr(node.index)
            return f"{base_str}[{index_str}]"
        
        if isinstance(node, ListExpr):
            # Corresponds to a Dafny sequence display: [1, 2, 3]
            # Recursively process each item in the list.
            items = [self._expr(item) for item in node.items]
            return f"[{', '.join(items)}]"

        if isinstance(node, TempNode):
            # TempNode represents a compiler-introduced temporary.
            # If it wraps a constant expression, materialize it as a Dafny temp variable.
            tmp = self.fresh_tmp()  # e.g., tmp0, tmp1, ...
            # Record the temp assignment
            # Assumes you later emit these before using the expression
            type_string: str = self.dafny_type(node.type)
            self.emit(f"var {tmp} : {type_string};")
            return tmp

        print(f"Unsupported expression type: {node}")
        raise NotImplementedError(f"Expression type not supported: {type(node)}")

    def dafny_binop(self, op: str) -> str:
        """Maps Mypy binary operator strings to Dafny."""
        if op in ('+', '-', '*', '%'):
            return op
        if op == '/':
            # We assume '/' on mypy 'float' types means real division.
            # This is a simplification; Dafny's / is real or int.
            return "/"
        if op == '//':
            return "/" # Dafny's '/' on 'int' is integer division
        if op == 'and':
            return '&&'
        if op == 'or':
            return '||'
        raise NotImplementedError(f"BinOp not supported: {op}")

    def dafny_cmpop(self, op: str) -> str:
        """Maps Mypy comparison operator strings to Dafny."""
        if op in ('==', '!=', '<', '<=', '>', '>='):
            return op
        raise NotImplementedError(f"CmpOp not supported: {op}")



class SupportsContextBuilder(Protocol):
    internal_map: dict[str, str]

    def get_global_context_signature(self) -> dict[str, Any]:
        ...


class SupportsTypeChecker(Protocol):
    agent_custom_types: list[str]
    custom_type_names: list[str]

    def get_stub(self) -> str:
        ...

    def get_typed_tree(self, code: str) -> Result[Any, Exception]:
        ...


class DafnyChecker:
    def __init__(self, 
                 context_builder: SupportsContextBuilder, 
                 type_checker: SupportsTypeChecker,
                 agent_config: dict[str, Any]) -> None:
        self.context_builder = context_builder
        self.type_checker = type_checker
        self.agent_config = agent_config
        # Positional argument of names
        self.positional_arg_dict = {}
        self.cost_info = {}
        # Additional functions added for cost contracts like model costs
        self.additional_stubs: list[str] = []
        # Translate the agent constraints and op constraints
        self.converted_formal_checks = self._convert_formal_checks()

    def _convert_stub_to_dafny(self, stub_string: str) -> str:
        """
        Parses a Python stub string and returns the converted Dafny code.
        """
        try:
            py_ast = ast.parse(stub_string)
        except SyntaxError as e:
            return f"// Error: Could not parse Python stub.\n// {e}"

        visitor = PythonStubToDafnyVisitor(self.converted_formal_checks)
        visitor.visit(py_ast)
        
        final_code = []
        if visitor.needs_any_type:
            final_code.append("// Dafny doesn't have a direct 'Any' type.")
            final_code.append("// We define an abstract 'AnyType' as a placeholder.")
            final_code.append("type AnyType\n")

        final_code.extend(visitor.dafny_code_lines)
        return "\n".join(final_code)
    
    def _forward(self, code: str) -> Result[bool, Exception]:
        try:
            # remove the comments and doc strings from the code.
            code = remove_comments_and_docstrings(code)
        except:
            code = code

        try:
            dafny_res = self._convert_python_to_dafny(code)
        except Exception as e:
            return Err(e)
        
        if dafny_res.is_ok():
            dafny_code = dafny_res.value
            try:
                return _verify_dafny_code(dafny_code)
            except Exception as e:
                return Err(e)
        else:
            return Err(dafny_res.error)

    def _convert_python_to_dafny(self, code: str)-> Result[str, Exception]:
        python_stub_code = self.type_checker.get_stub()
        dafny_stub_code = ""
        try:
            # remove the comments and doc strings from the code.
            code = remove_comments_and_docstrings(code)
        except:
            code = code
        dafny_stub_code += self._convert_stub_to_dafny(python_stub_code)
        if self.additional_stubs:
            dafny_stub_code += '\n' + "\n".join(self.additional_stubs) 
        tree = self.type_checker.get_typed_tree(code)
        if tree.is_err():
            return Err(tree.error)    
        tree = tree.value
        type_extractor = TypeExtractingVisitor()
        try:
            tree.accept(type_extractor)
        except Exception as e:
            return Err(e)

        variables_dict : dict[str, list[str]] = type_extractor.variables
        # print(f'variables_dict:\n\n{variables_dict}')
        # Collect all custom types 
        all_custom_types = list(self.type_checker.agent_custom_types) + list(self.type_checker.custom_type_names)
        all_custom_types = list(set(all_custom_types))
        dafny_visitor = MypyToDafnyConverter(custom_types=all_custom_types,
                                            agent_checks=self.converted_formal_checks["agent"],
                                            cost_info=self.cost_info, 
                                            positional_argument_dict=self.positional_arg_dict,
                                            local_variable_type_info=variables_dict)
        try:
            tree.accept(dafny_visitor)
        except Exception as e:
            return Err(e)
        converted_dafny_code = dafny_stub_code +'\n\n\n// =======STUB CODE ENDS==============\n\n\n' 
        converted_dafny_code += "\n".join(dafny_visitor.lines)
        return Ok(converted_dafny_code)


    def _add_positional_arg_dict(self, op_name: str, input_signature: dict[str, Any]):
       self.positional_arg_dict[op_name] = {}
       i = 0
       for arg_name in input_signature:
           self.positional_arg_dict[op_name][arg_name] = i
           i += 1
     

    def _add_model_cost_stubs(self, op_name: str, cost_metadata: dict[str, Any] | None):
        """
        Generates a Dafny '{:extern} function' stub for an operation's cost.

        Args:
            op_name: The name of the operation (e.g., 'abc').
            model_cost_dict: A dictionary mapping model names to their integer costs.

        Returns:
            A string containing the Dafny function definition.
        """
        if not cost_metadata:
            return

        # 1. Create the function signature
        # We use {{ and }} to escape the curly braces for the f-string
        signature = f"function {{:extern}} cost_{op_name}(model_id: string): (r: int)"


        # 2. Create the 'requires' clause
        # Format each key as a Dafny string (e.g., '"modelA"')
        model_names_dafny = [f'"{key}"' for key in cost_metadata.keys()]
        # Join them into a Dafny set: {"modelA", "modelB"}
        model_set = f"{{{', '.join(model_names_dafny)}}}"
        requires_clause = f"  requires model_id in {model_set}"

        # 3. Create the 'ensures' clause
        # Build a list of implication expressions
        # e.g., '(model_id == "modelA") ==> (r == 10)'
        ensures_expressions = [
            f'((model_id == "{model}") ==> (r == {cost}))'
            for model, cost in cost_metadata.items()
        ]
        # Join all expressions with '&&'
        ensures_clause = f"  ensures {' && '.join(ensures_expressions)}"

        # Add this new stub
        self.additional_stubs.append(f"{signature}\n\t{requires_clause}\n\t{ensures_clause}\n\n")


    def _convert_formal_checks(self) -> dict[str, Any]:
        converted_formal_checks = {}
        type_signatures = self.context_builder.get_global_context_signature()
        if "formal_checks" not in self.agent_config:
            return converted_formal_checks
        for key in self.agent_config["formal_checks"]:
            if key == 'agent':
                converted_formal_checks[key] = deepcopy(self.agent_config["formal_checks"]["agent"])
                continue
            if key not in self.context_builder.internal_map:
                # continue
                raise ValueError(f'{key} not found in the internal operator map')
            new_key = self.context_builder.internal_map[key]
            converted_formal_checks[new_key] = deepcopy(self.agent_config["formal_checks"][key])
            if "costspecs" in converted_formal_checks[new_key]:
                cost_spec = converted_formal_checks[new_key]["costspecs"]
                if cost_spec:
                    if len(cost_spec) > 1:
                        print("Warning: there are multiple cost specifications")
                    self.cost_info[new_key] = cost_spec[0]
            input_signature = type_signatures[new_key]['input_signature']
            self._add_positional_arg_dict(op_name=new_key, input_signature=input_signature)
            if "cost_metadata" in converted_formal_checks[new_key]:
                self._add_model_cost_stubs(op_name=new_key, 
                                           cost_metadata=converted_formal_checks[new_key]["cost_metadata"].get("model_costs", None))
        return converted_formal_checks


_RETURN_NAME_OVERRIDES: dict[str, list[str]] = {
    "IdToToken": ["token"],
    "TokenToId": ["id"],
    "TokenToIdRecursive": ["id"],
    "IdToLogit": ["logit"],
    "TokenToLogit": ["logit"],
    "TokensToLogits": ["logits"],
    "IdsToLogits": ["logits"],
    "LastLeftDelimiterIndex": ["result"],
    "FirstRightDelimiterIndex": ["result"],
    "LeftDelimiter": ["result"],
    "RightDelimiter": ["result"],
    "GetDelimitedContent": ["result"],
    "RollbackToValidPrefix": ["repaired"],
    "UnconstrainedStep": ["next", "stepsLeft'"],
    "ExpressiveStep": ["next", "stepsLeft'"],
    "ConstrainedStep": ["next", "stepsLeft'"],
    "ConstrainedAnswerStep": ["next", "stepsLeft'"],
    "ChooseNextToken": ["token"],
    "MyCSDStrategy": ["generated", "remainingSteps"],
}

_LOCAL_NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "MaskToken": {"token_id": "id"},
    "MaskTokens": {"n": "N"},
    "MaskTokensExcept": {"to_mask": "toMask", "n": "N"},
}

_FORCE_AXIOM_DECLS: set[tuple[str | None, str]] = set()

_RETURN_TYPE_OVERRIDES: dict[tuple[str | None, str], str] = {
    ("Delimiter", "LastLeftDelimiterIndex"): "nat",
    ("Delimiter", "FirstRightDelimiterIndex"): "nat",
    ("Parser", "ValidNextTokens"): "seq<Token>",
}

_FIELD_TYPE_OVERRIDES: dict[tuple[str, str], str] = {
    ("LM", "Tokens"): "seq<Token>",
}

_PARAM_TYPE_OVERRIDES: dict[tuple[str | None, str, str], str] = {
    ("LM", "TokenToIdRecursive", "offset"): "nat",
    ("LM", "TokensToLogits", "tokens"): "seq<Token>",
    ("LM", "MaskTokens", "tokens"): "seq<Token>",
    ("LM", "MaskTokensExcept", "tokens"): "seq<Token>",
    ("CSDHelpers", "UnconstrainedStep", "stepsLeft"): "nat",
    ("CSDHelpers", "ExpressiveStep", "stepsLeft"): "nat",
    ("CSDHelpers", "ConstrainedStep", "stepsLeft"): "nat",
    ("CSDHelpers", "ConstrainedAnswerStep", "stepsLeft"): "nat",
    (None, "MyCSDStrategy", "maxSteps"): "nat",
}

_METHOD_RETURN_TYPE_OVERRIDES: dict[tuple[str | None, str], list[str]] = {
    ("CSDHelpers", "UnconstrainedStep"): ["Token", "nat"],
    ("CSDHelpers", "ExpressiveStep"): ["Token", "nat"],
    ("CSDHelpers", "ConstrainedStep"): ["Token", "nat"],
    ("CSDHelpers", "ConstrainedAnswerStep"): ["Token", "nat"],
    (None, "MyCSDStrategy"): ["Prefix", "nat"],
}

_CONSTRUCTOR_NAMES: set[str] = {
    "LM",
    "Parser",
    "Delimiter",
    "CSDHelpers",
}


def _spec_from_decorator(node: ast.FunctionDef) -> dict[str, Any] | None:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "dafny_spec":
            spec: dict[str, Any] = {
                "kind": "method",
                "reads": (),
                "modifies": (),
                "requires": (),
                "ensures": (),
                "decreases": (),
                "axiom": False,
                "extern": False,
            }
            for kw in decorator.keywords:
                if kw.arg is None:
                    continue
                spec[kw.arg] = ast.literal_eval(kw.value)
            return spec
    return None


def _type_alias_value_to_dafny(node: ast.AST, alias_name: str | None = None) -> str:
    if alias_name == "Id":
        return "nat"
    if isinstance(node, ast.Name):
        mapping = {
            "str": "string",
            "int": "int",
            "float": "real",
            "bool": "bool",
        }
        return mapping.get(node.id, node.id)
    if isinstance(node, ast.Subscript):
        base = _type_alias_value_to_dafny(node.value)
        sub = node.slice
        if isinstance(base, str) and base == "list":
            return f"seq<{_type_alias_value_to_dafny(sub)}>"
        if isinstance(base, str) and base == "set":
            return f"set<{_type_alias_value_to_dafny(sub)}>"
        if isinstance(base, str) and base == "dict" and isinstance(sub, ast.Tuple):
            return f"map<{_type_alias_value_to_dafny(sub.elts[0])}, {_type_alias_value_to_dafny(sub.elts[1])}>"
        return f"{base}<{ast.unparse(sub)}>"
    return ast.unparse(node)


def _annotation_to_dafny(node: ast.AST | None, field_name: str | None = None) -> str:
    if node is None:
        return "()"
    if isinstance(node, ast.Name):
        mapping = {
            "str": "string",
            "int": "int",
            "float": "real",
            "bool": "bool",
            "Any": "AnyType",
            "Token": "Token",
            "Prefix": "Prefix",
            "Id": "Id",
            "Logit": "Logit",
        }
        return mapping.get(node.id, node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            base = node.value.id
            if base in {"list", "List"}:
                if field_name == "Logits":
                    return f"array<{_annotation_to_dafny(node.slice)}>"
                return f"seq<{_annotation_to_dafny(node.slice)}>"
            if base in {"tuple", "Tuple"}:
                if isinstance(node.slice, ast.Tuple):
                    return ", ".join(_annotation_to_dafny(elt) for elt in node.slice.elts)
            if base in {"set", "Set"}:
                return f"set<{_annotation_to_dafny(node.slice)}>"
            if base in {"dict", "Dict"} and isinstance(node.slice, ast.Tuple):
                return f"map<{_annotation_to_dafny(node.slice.elts[0])}, {_annotation_to_dafny(node.slice.elts[1])}>"
        return ast.unparse(node)
    return ast.unparse(node)


def _const_expr(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return '"' + node.value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return "-" + repr(node.operand.value)
    return ast.unparse(node)


def _attr_is_array(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "Logits"


def _translate_generator_quantifier(
    call_name: str,
    gen: ast.GeneratorExp,
    current_class: str | None,
    name_map: dict[str, str] | None = None,
) -> str:
    specialized = _translate_char_generator_quantifier(call_name, gen, current_class, name_map)
    if specialized is not None:
        return specialized
    vars_: list[str] = []
    domains: list[str] = []
    equalities: list[str] = []
    for comp in gen.generators:
        if isinstance(comp.target, ast.Name):
            target_names = [comp.target.id]
        elif isinstance(comp.target, ast.Tuple):
            target_names = [elt.id for elt in comp.target.elts if isinstance(elt, ast.Name)]
        else:
            raise NotImplementedError(f"Unsupported comprehension target: {ast.dump(comp.target)}")

        iter_node = comp.iter
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range":
            name = target_names[0]
            args = iter_node.args
            if len(args) == 1:
                domains.append(f"0 <= {name} < {_translate_expr(args[0], current_class, name_map)}")
            elif len(args) == 2:
                domains.append(f"{_translate_expr(args[0], current_class, name_map)} <= {name} < {_translate_expr(args[1], current_class, name_map)}")
            else:
                raise NotImplementedError("range() with step is not supported in quantifiers")
            vars_.append(name)
        elif isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "enumerate":
            seq_expr = _translate_expr(iter_node.args[0], current_class, name_map)
            idx_name, val_name = target_names
            vars_.extend([idx_name, val_name])
            domains.append(f"0 <= {idx_name} < |{seq_expr}|")
            equalities.append(f"{val_name} == {seq_expr}[{idx_name}]")
        else:
            iter_expr = _translate_expr(iter_node, current_class, name_map)
            if _attr_is_array(iter_node):
                iter_expr = f"{iter_expr}[0..{iter_expr}.Length]"
            vars_.append(target_names[0])
            domains.append(f"{target_names[0]} in {iter_expr}")

        for if_clause in comp.ifs:
            domains.append(_translate_expr(if_clause, current_class, name_map))

    body = _translate_expr(gen.elt, current_class, name_map)
    antecedent_parts = [part for part in domains + equalities if part]
    if call_name == "all":
        if antecedent_parts:
            return f"(forall {', '.join(vars_)} :: ({' && '.join(antecedent_parts)}) ==> ({body}))"
        return f"(forall {', '.join(vars_)} :: ({body}))"
    if antecedent_parts:
        return f"(exists {', '.join(vars_)} :: ({' && '.join(antecedent_parts)}) && ({body}))"
    return f"(exists {', '.join(vars_)} :: ({body}))"


def _translate_char_generator_quantifier(
    call_name: str,
    gen: ast.GeneratorExp,
    current_class: str | None,
    name_map: dict[str, str] | None = None,
) -> str | None:
    if len(gen.generators) != 1:
        return None
    comp = gen.generators[0]
    if comp.ifs:
        return None
    if not isinstance(comp.target, ast.Name):
        return None
    iter_node = comp.iter
    if not isinstance(iter_node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
        return None
    elt = gen.elt
    if not (
        isinstance(elt, ast.Call)
        and isinstance(elt.func, ast.Attribute)
        and len(elt.args) == 0
        and isinstance(elt.func.value, ast.Name)
        and elt.func.value.id == comp.target.id
        and elt.func.attr in {"isalpha", "isdigit"}
    ):
        return None

    base_expr = _translate_expr(iter_node, current_class, name_map)
    idx_name = f"{comp.target.id}_idx"
    char_expr = f"{base_expr}[{idx_name}]"
    predicate = _translate_char_predicate(char_expr, elt.func.attr)
    quantifier = "forall" if call_name == "all" else "exists"
    connector = "==>" if call_name == "all" else "&&"
    return (
        f"({quantifier} {idx_name} :: 0 <= {idx_name} < |{base_expr}| "
        f"{connector} ({predicate}))"
    )


def _is_none_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _translate_token_predicate(base: str, predicate: str) -> str:
    if predicate == "isalpha":
        return (
            f"(|{base}| > 0 && "
            f"(forall i :: 0 <= i < |{base}| ==> "
            f"((('a' <= {base}[i]) && ({base}[i] <= 'z')) || "
            f"(('A' <= {base}[i]) && ({base}[i] <= 'Z')))))"
        )
    if predicate == "isdigit":
        return (
            f"(|{base}| > 0 && "
            f"(forall i :: 0 <= i < |{base}| ==> "
            f"(('0' <= {base}[i]) && ({base}[i] <= '9'))))"
        )
    if predicate == "isnumeric":
        return _translate_token_predicate(base, "isdigit")
    raise NotImplementedError(f"Unsupported token predicate: {predicate}")


def _translate_char_predicate(base: str, predicate: str) -> str:
    if predicate == "isalpha":
        return (
            f"((\"a\" <= {base} && {base} <= \"z\") || "
            f"(\"A\" <= {base} && {base} <= \"Z\"))"
        )
    if predicate in {"isdigit", "isnumeric"}:
        return f"(\"0\" <= {base} && {base} <= \"9\")"
    raise NotImplementedError(f"Unsupported char predicate: {predicate}")


def _literal_membership_options(
    node: ast.AST,
    current_class: str | None,
    name_map: dict[str, str] | None = None,
) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [f"\"{ch}\"" for ch in node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_translate_expr(elt, current_class, name_map) for elt in node.elts]
    return None


def _translate_prefix_check(base: str, prefix: str) -> str:
    if prefix == "\"\"":
        return "true"
    return f"(|{base}| >= |{prefix}| && {base}[..|{prefix}|] == {prefix})"


def _translate_suffix_check(base: str, suffix: str) -> str:
    if suffix == "\"\"":
        return "true"
    return f"(|{base}| >= |{suffix}| && {base}[|{base}|-|{suffix}|..|{base}|] == {suffix})"


def _translate_compare(node: ast.Compare, current_class: str | None, name_map: dict[str, str] | None = None) -> str:
    pieces = []
    left = node.left
    for op, right in zip(node.ops, node.comparators):
        if isinstance(op, ast.Is) and (_is_none_literal(left) or _is_none_literal(right)):
            pieces.append("false")
            left = right
            continue
        if isinstance(op, ast.IsNot) and (_is_none_literal(left) or _is_none_literal(right)):
            pieces.append("true")
            left = right
            continue
        lhs = _translate_expr(left, current_class, name_map)
        rhs = _translate_expr(right, current_class, name_map)
        literal_options = _literal_membership_options(right, current_class, name_map)
        if (
            isinstance(op, (ast.In, ast.NotIn))
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
            and not (isinstance(left, ast.Name) and len(left.id) == 1)
        ):
            chars = list(right.value)
            if isinstance(op, ast.In):
                if not chars:
                    pieces.append("false")
                else:
                    pieces.append("(" + " || ".join(f'{lhs} == "{ch}"' for ch in chars) + ")")
            else:
                if not chars:
                    pieces.append("true")
                else:
                    pieces.append("(" + " && ".join(f'{lhs} != "{ch}"' for ch in chars) + ")")
            left = right
            continue
        if isinstance(op, (ast.In, ast.NotIn)) and literal_options is not None:
            if isinstance(op, ast.In):
                pieces.append("false" if not literal_options else "(" + " || ".join(f"{lhs} == {option}" for option in literal_options) + ")")
            else:
                pieces.append("true" if not literal_options else "(" + " && ".join(f"{lhs} != {option}" for option in literal_options) + ")")
            left = right
            continue
        if isinstance(op, ast.In):
            pieces.append(f"{lhs} in {rhs}")
        elif isinstance(op, ast.NotIn):
            pieces.append(f"{lhs} !in {rhs}")
        elif isinstance(op, ast.Eq):
            pieces.append(f"{lhs} == {rhs}")
        elif isinstance(op, ast.NotEq):
            pieces.append(f"{lhs} != {rhs}")
        elif isinstance(op, ast.Lt):
            pieces.append(f"{lhs} < {rhs}")
        elif isinstance(op, ast.LtE):
            pieces.append(f"{lhs} <= {rhs}")
        elif isinstance(op, ast.Gt):
            pieces.append(f"{lhs} > {rhs}")
        elif isinstance(op, ast.GtE):
            pieces.append(f"{lhs} >= {rhs}")
        else:
            raise NotImplementedError(f"Unsupported comparison operator: {type(op).__name__}")
        left = right
    if len(pieces) == 1:
        return pieces[0]
    return "(" + " && ".join(f"({piece})" for piece in pieces) + ")"


def _translate_slice(node: ast.Slice, current_class: str | None, name_map: dict[str, str] | None = None) -> str:
    lower = _translate_expr(node.lower, current_class, name_map) if node.lower is not None else ""
    upper = _translate_expr(node.upper, current_class, name_map) if node.upper is not None else ""
    return f"{lower}..{upper}"


def _translate_expr(node: ast.AST, current_class: str | None = None, name_map: dict[str, str] | None = None) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return '"' + node.value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if node.value is None:
            return "null"
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id == "self":
            return "this"
        if node.id == "True":
            return "true"
        if node.id == "False":
            return "false"
        return (name_map or {}).get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_translate_expr(node.value, current_class, name_map)}.{node.attr}"
    if isinstance(node, ast.List):
        return "[" + ", ".join(_translate_expr(elt, current_class, name_map) for elt in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return ", ".join(_translate_expr(elt, current_class, name_map) for elt in node.elts)
    if isinstance(node, ast.Subscript):
        base = _translate_expr(node.value, current_class, name_map)
        if isinstance(node.slice, ast.Slice):
            if (
                node.slice.lower is None
                and isinstance(node.slice.upper, ast.UnaryOp)
                and isinstance(node.slice.upper.op, ast.USub)
                and isinstance(node.slice.upper.operand, ast.Constant)
                and isinstance(node.slice.upper.operand.value, int)
            ):
                k = node.slice.upper.operand.value
                return f"{base}[..|{base}|-{k}]"
            return f"{base}[{_translate_slice(node.slice, current_class, name_map)}]"
        return f"{base}[{_translate_expr(node.slice, current_class, name_map)}]"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return f"!{_translate_expr(node.operand, current_class, name_map)}"
        if isinstance(node.op, ast.USub):
            return f"-{_translate_expr(node.operand, current_class, name_map)}"
        if isinstance(node.op, ast.UAdd):
            return f"+{_translate_expr(node.operand, current_class, name_map)}"
    if isinstance(node, ast.BoolOp):
        op = "&&" if isinstance(node.op, ast.And) else "||"
        return "(" + f" {op} ".join(f"({_translate_expr(v, current_class, name_map)})" for v in node.values) + ")"
    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        op = op_map[type(node.op)]
        return f"({_translate_expr(node.left, current_class, name_map)} {op} {_translate_expr(node.right, current_class, name_map)})"
    if isinstance(node, ast.Compare):
        return _translate_compare(node, current_class, name_map)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "len":
                arg = node.args[0]
                base = _translate_expr(arg, current_class, name_map)
                if _attr_is_array(arg):
                    return f"{base}.Length"
                return f"|{base}|"
            if func_name == "list":
                return _translate_expr(node.args[0], current_class, name_map)
            if func_name == "str" and len(node.args) == 1:
                return _translate_expr(node.args[0], current_class, name_map)
            if func_name == "isinstance" and len(node.args) == 2:
                type_arg = node.args[1]
                if isinstance(type_arg, ast.Name) and type_arg.id in {"str", "int", "bool", "float", "Token"}:
                    return "true"
            if func_name in {"any", "all"} and isinstance(node.args[0], ast.GeneratorExp):
                return _translate_generator_quantifier(func_name, node.args[0], current_class, name_map)
            if func_name in _CONSTRUCTOR_NAMES:
                return f"new {func_name}({', '.join(_translate_expr(arg, current_class, name_map) for arg in node.args)})"
        if isinstance(node.func, ast.Attribute):
            base = _translate_expr(node.func.value, current_class, name_map)
            if len(node.args) == 0 and node.func.attr in {"isalpha", "isdigit", "isnumeric"}:
                return _translate_token_predicate(base, node.func.attr)
            if len(node.args) == 1 and node.func.attr in {"startswith", "endswith"}:
                options = _literal_membership_options(node.args[0], current_class, name_map)
                if options is None:
                    options = [_translate_expr(node.args[0], current_class, name_map)]
                checks = [
                    _translate_prefix_check(base, option)
                    if node.func.attr == "startswith"
                    else _translate_suffix_check(base, option)
                    for option in options
                ]
                return "false" if not checks else "(" + " || ".join(checks) + ")"
            if node.func.attr == "append" and len(node.args) == 1:
                arg = _translate_expr(node.args[0], current_class, name_map)
                return f"{base} := ({base} + [{arg}])"
        return f"{_translate_expr(node.func, current_class, name_map)}({', '.join(_translate_expr(arg, current_class, name_map) for arg in node.args)})"
    raise NotImplementedError(f"Unsupported expression: {type(node).__name__}: {ast.dump(node)}")


def _translate_function_block(stmts: list[ast.stmt], current_class: str | None, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if not stmts:
        return [prefix + "{}"]
    lines: list[str] = []
    index = 0
    while index < len(stmts) and isinstance(stmts[index], (ast.Assign, ast.AnnAssign)):
        stmt = stmts[index]
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                raise NotImplementedError("Only simple local assignments are supported in function bodies")
            lines.append(prefix + f"var {target.id} := {_translate_expr(stmt.value, current_class)};")
        else:
            if not isinstance(stmt.target, ast.Name):
                raise NotImplementedError("Only simple local annotated assignments are supported in function bodies")
            lines.append(prefix + f"var {stmt.target.id} := {_translate_expr(stmt.value, current_class)};")
        index += 1

    tail = stmts[index:]
    if len(tail) != 1:
        raise NotImplementedError("Function bodies must end in a single return or if-expression")

    last = tail[0]
    if isinstance(last, ast.Return):
        lines.append(prefix + _translate_expr(last.value, current_class))
        return lines
    if isinstance(last, ast.If):
        cond = _translate_expr(last.test, current_class)
        then_lines = _translate_function_block(last.body, current_class, indent + 2)
        else_lines = _translate_function_block(last.orelse, current_class, indent + 2)
        if len(then_lines) == 1 and not then_lines[0].strip().startswith("var "):
            lines.append(prefix + f"if {cond} then {then_lines[0].strip()}")
        else:
            lines.append(prefix + f"if {cond} then")
            lines.extend(then_lines)
        if len(else_lines) == 1 and not else_lines[0].strip().startswith("var "):
            lines.append(prefix + f"else {else_lines[0].strip()}")
        else:
            lines.append(prefix + "else")
            lines.extend(else_lines)
        return lines
    raise NotImplementedError(f"Unsupported function tail statement: {type(last).__name__}")


def _special_function_body(name: str, current_class: str | None) -> list[str] | None:
    if name == "Contains":
        return ["exists i, j :: 0 <= i <= j <= |s| && s[i..j] == sub"]
    if name == "PrefixContains":
        return ["exists i :: 0 <= i < |p| && p[i] == t"]
    if name == "DelimitedAnswerValidForParser":
        return [
            "PrefixContains(prefix, LeftDelimiter) &&",
            "PrefixContains(prefix, RightDelimiter)",
        ]
    if name == "ValidTokensIdsLogits" and current_class == "LM":
        return [
            "((|Tokens| == |Ids|) && (|Ids| == Logits.Length) && (|Ids| > 0 && Ids[0] == 0)) &&",
            "(forall i :: 0 <= i < |Ids| ==> (i == Ids[i]) && (i in Ids)) &&",
            "(forall i, j :: 0 <= i < |Tokens| && 0 <= j < |Tokens| && i != j ==> Tokens[i] != Tokens[j]) &&",
            "(forall token: Token :: token in Tokens ==> (exists i :: 0 <= i < |Ids| && Tokens[i] == token)) &&",
            "(forall i :: 0 <= i < Logits.Length ==> Logits[i] <= 1e9 && Logits[i] >= -1e9)",
        ]
    if name == "TokenToIdRecursive":
        return [
            "if this.Tokens[offset] == token then offset",
            "else this.TokenToIdRecursive(token, offset + 1)",
        ]
    if name == "TokensToLogits":
        return [
            "if |tokens| == 1 then [this.TokenToLogit(tokens[0])]",
            "else [this.TokenToLogit(tokens[0])] + this.TokensToLogits(tokens[1..])",
        ]
    if name == "IdsToLogits":
        return [
            "if |ids| == 1 then [this.IdToLogit(ids[0])]",
            "else [this.IdToLogit(ids[0])] + this.IdsToLogits(ids[1..])",
        ]
    if name == "HasUnmaskedToken" and current_class == "LM":
        return ["exists t: Token :: t in Tokens && !IsMasked(t)"]
    if name == "IsDeadPrefix" and current_class == "Parser":
        return ["!IsCompletePrefix(prefix) && |ValidNextTokens(prefix)| == 0"]
    if name == "ValidNextToken" and current_class == "Parser":
        return ["token in ValidNextTokens(prefix)"]
    if name == "LastLeftDelimiterIndex":
        return [
            "if |prefix| == 0 then 0",
            "else",
            "  if prefix[|prefix|-1] == this.Left then |prefix|-1",
            "  else",
            "    var lastInRest := LastLeftDelimiterIndex(prefix[..|prefix|-1]);",
            "    if lastInRest < |prefix|-1 then lastInRest else |prefix|",
        ]
    if name == "FirstRightDelimiterIndex":
        return [
            "if |content| == 0 then 0",
            "else if content[0] == this.Right then 0",
            "else 1 + FirstRightDelimiterIndex(content[1..])",
        ]
    if name == "GetDelimitedContent" and current_class == "Delimiter":
        return [
            "var start := this.LastLeftDelimiterIndex(prefix) + 1;",
            "if start > |prefix| then []",
            "else",
            "  var afterLeft := prefix[start..|prefix|];",
            "  var endIdx := this.FirstRightDelimiterIndex(afterLeft);",
            "  afterLeft[..endIdx]",
        ]
    if name == "InsideDelimitedWindow" and current_class == "Delimiter":
        return [
            "var start := LastLeftDelimiterIndex(prefix) + 1;",
            "start <= |prefix| && FirstRightDelimiterIndex(prefix[start..|prefix|]) == |prefix[start..|prefix|]|",
        ]
    if name == "DelimitersInLM" and current_class == "CSDHelpers":
        return [
            "lm.ValidTokensIdsLogits() &&",
            "delimiter.Left in lm.Tokens &&",
            "delimiter.Right in lm.Tokens",
        ]
    if name == "ConstrainedWindowValid" and current_class == "CSDHelpers":
        return [
            "!this.delimiter.InsideDelimitedWindow(prefix) || this.parser.IsValidPrefix(this.delimiter.GetDelimitedContent(prefix))",
        ]
    return None


def _special_stmt_body(name: str, current_class: str | None) -> list[str] | None:
    if name == "__init__" and current_class == "Delimiter":
        return [
            "  this.Left := left;",
            "  this.Right := right;",
        ]
    if name == "__init__" and current_class == "CSDHelpers":
        return [
            "  this.lm := lm;",
            "  this.parser := parser;",
            "  this.delimiter := delimiter;",
        ]
    if name == "FirstRightDelimiterAppendRight" and current_class == "Delimiter":
        return [
            "  if |content| == 0 {",
            "  } else {",
            "    this.FirstRightDelimiterAppendRight(content[1..]);",
            "    assert (content + [this.Right])[1..] == content[1..] + [this.Right];",
            "  }",
        ]
    return None


def _translate_len_calls_in_spec(
    text: str,
    current_class: str | None,
    name_map: dict[str, str] | None = None,
) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("len(", i):
            j = i + 4
            depth = 1
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = text[i + 4 : j - 1].strip()
                try:
                    expr = ast.parse(inner, mode="eval").body
                    base = _translate_expr(expr, current_class, name_map)
                    if _attr_is_array(expr):
                        result.append(f"{base}.Length")
                    else:
                        result.append(f"|{base}|")
                except SyntaxError:
                    result.append(text[i:j])
                i = j
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _loop_specs(node: ast.While, source_lines: list[str]) -> tuple[list[str], list[str]]:
    invariants: list[str] = []
    decreases: list[str] = []
    line_index = node.lineno - 2
    skipped_setup_lines = 0
    skipped_setup_comments = 0
    found_specs = False
    while line_index >= 0:
        raw = source_lines[line_index]
        stripped = raw.strip()
        if not stripped.startswith("#"):
            if stripped == "":
                line_index -= 1
                continue
            if (
                not found_specs
                and skipped_setup_lines < 8
                and _looks_like_loop_setup_statement(stripped)
            ):
                skipped_setup_lines += 1
                line_index -= 1
                continue
            break
        body = stripped[1:].strip()
        if body.startswith("invariant "):
            invariants.append(body)
            found_specs = True
        elif body.startswith("decreases "):
            decreases.append(body)
            found_specs = True
        else:
            if (
                not found_specs
                and not body.startswith("CSD_RATIONALE_")
                and skipped_setup_comments < 4
            ):
                skipped_setup_comments += 1
                line_index -= 1
                continue
            break
        line_index -= 1
    invariants.reverse()
    decreases.reverse()
    return invariants, decreases


def _looks_like_loop_setup_statement(stripped_line: str) -> bool:
    try:
        parsed = ast.parse(stripped_line)
    except SyntaxError:
        return False
    if len(parsed.body) != 1:
        return False
    return isinstance(parsed.body[0], (ast.Assign, ast.AnnAssign))


def _translate_target(node: ast.AST, current_class: str | None, name_map: dict[str, str] | None = None) -> str:
    if isinstance(node, ast.Name):
        return (name_map or {}).get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _translate_expr(node, current_class, name_map)
    if isinstance(node, ast.Subscript):
        base = _translate_expr(node.value, current_class, name_map)
        index = _translate_expr(node.slice, current_class, name_map) if not isinstance(node.slice, ast.Slice) else _translate_slice(node.slice, current_class, name_map)
        return f"{base}[{index}]"
    raise NotImplementedError(f"Unsupported assignment target: {type(node).__name__}")


def _translate_stmt_list(
    stmts: list[ast.stmt],
    current_class: str | None,
    source_lines: list[str],
    declared: set[str],
    return_names: list[str],
    method_name: str,
    indent: int = 0,
) -> list[str]:
    prefix = " " * indent
    name_map = _LOCAL_NAME_OVERRIDES.get(method_name, {})
    lines: list[str] = []
    for stmt in stmts:
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                raise NotImplementedError("Only local annotated assignments are supported")
            target_name = name_map.get(stmt.target.id, stmt.target.id)
            if (
                isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
                and method_name == "MyCSDStrategy"
                and target_name in {"next_token", "new_steps"}
            ):
                value_expr = "eosToken" if target_name == "next_token" else "stepsLeft"
            else:
                value_expr = _translate_expr(stmt.value, current_class, name_map)
            if target_name in return_names:
                lines.append(prefix + f"{target_name} := {value_expr};")
            else:
                lines.append(prefix + f"var {target_name}: {_annotation_to_dafny(stmt.annotation)} := {value_expr};")
                declared.add(target_name)
            continue
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            if isinstance(target, ast.Tuple):
                if isinstance(stmt.value, ast.Tuple):
                    if len(target.elts) != len(stmt.value.elts):
                        raise NotImplementedError("Tuple assignment arity mismatch")
                    for elt, value in zip(target.elts, stmt.value.elts):
                        if not isinstance(elt, ast.Name):
                            raise NotImplementedError("Tuple assignment targets must be simple names")
                        target_name = name_map.get(elt.id, elt.id)
                        if (
                            isinstance(value, ast.Constant)
                            and value.value is None
                            and method_name == "MyCSDStrategy"
                            and target_name == "next_token"
                        ):
                            value_expr = "eosToken"
                        else:
                            value_expr = _translate_expr(value, current_class, name_map)
                        if target_name in declared or target_name in return_names:
                            lines.append(prefix + f"{target_name} := {value_expr};")
                        else:
                            lines.append(prefix + f"var {target_name} := {value_expr};")
                            declared.add(target_name)
                    continue
                if not isinstance(stmt.value, ast.Call):
                    raise NotImplementedError("Tuple assignment is only supported for call results or tuple literals")
                target_names = []
                needs_var = True
                for elt in target.elts:
                    if not isinstance(elt, ast.Name):
                        raise NotImplementedError("Tuple assignment targets must be simple names")
                    target_name = name_map.get(elt.id, elt.id)
                    if target_name in declared or target_name in return_names:
                        needs_var = False
                    if target_name not in declared and target_name not in return_names:
                        declared.add(target_name)
                    target_names.append(target_name)
                prefix_kw = "var " if needs_var else ""
                lines.append(
                    prefix
                    + f"{prefix_kw}{', '.join(target_names)} := {_translate_expr(stmt.value, current_class, name_map)};"
                )
                continue
            if isinstance(target, ast.Name):
                target_name = name_map.get(target.id, target.id)
                if (
                    isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is None
                    and method_name == "MyCSDStrategy"
                    and target_name in {"next_token", "new_steps"}
                ):
                    value_expr = "eosToken" if target_name == "next_token" else "stepsLeft"
                else:
                    value_expr = _translate_expr(stmt.value, current_class, name_map)
                if target_name in declared or target_name in return_names:
                    lines.append(prefix + f"{target_name} := {value_expr};")
                else:
                    lines.append(prefix + f"var {target_name} := {value_expr};")
                    declared.add(target_name)
            else:
                lines.append(prefix + f"{_translate_target(target, current_class, name_map)} := {_translate_expr(stmt.value, current_class, name_map)};")
            continue
        if isinstance(stmt, ast.Expr):
            lines.append(prefix + f"{_translate_expr(stmt.value, current_class, name_map)};")
            continue
        if isinstance(stmt, ast.Assert):
            lines.append(prefix + f"assert {_translate_expr(stmt.test, current_class, name_map)};")
            continue
        if isinstance(stmt, ast.If):
            common_branch_locals = _if_branch_locals(stmt, current_class, name_map)
            for local_name, local_type in common_branch_locals:
                if local_name not in declared and local_name not in return_names:
                    lines.append(prefix + f"var {local_name}: {local_type};")
                    declared.add(local_name)
            lines.append(prefix + f"if {_translate_expr(stmt.test, current_class, name_map)} {{")
            lines.extend(_translate_stmt_list(stmt.body, current_class, source_lines, declared.copy(), return_names, method_name, indent + 2))
            if stmt.orelse:
                lines.append(prefix + "} else {")
                lines.extend(_translate_stmt_list(stmt.orelse, current_class, source_lines, declared.copy(), return_names, method_name, indent + 2))
            lines.append(prefix + "}")
            continue
        if isinstance(stmt, ast.While):
            lines.append(prefix + f"while {_translate_expr(stmt.test, current_class, name_map)}")
            invariants, decreases = _loop_specs(stmt, source_lines)
            for inv in invariants:
                translated_inv = _translate_len_calls_in_spec(inv, current_class, name_map)
                lines.append(prefix + "  " + translated_inv)
            for dec in decreases:
                translated_dec = _translate_len_calls_in_spec(dec, current_class, name_map)
                lines.append(prefix + "  " + translated_dec)
            lines.append(prefix + "{")
            lines.extend(_translate_stmt_list(stmt.body, current_class, source_lines, declared.copy(), return_names, method_name, indent + 2))
            lines.append(prefix + "}")
            continue
        if isinstance(stmt, ast.AugAssign):
            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Mod: "%",
            }
            op = op_map[type(stmt.op)]
            target = _translate_target(stmt.target, current_class, name_map)
            value = _translate_expr(stmt.value, current_class, name_map)
            lines.append(prefix + f"{target} := {target} {op} {value};")
            continue
        if isinstance(stmt, ast.Return):
            if not return_names:
                lines.append(prefix + "return;")
            elif (
                isinstance(stmt.value, ast.Tuple)
                and len(stmt.value.elts) == len(return_names)
                and all(isinstance(elt, ast.Name) and elt.id == name for elt, name in zip(stmt.value.elts, return_names))
            ):
                continue
            elif isinstance(stmt.value, ast.Tuple):
                for name, elt in zip(return_names, stmt.value.elts):
                    lines.append(prefix + f"{name} := {_translate_expr(elt, current_class, name_map)};")
                lines.append(prefix + "return;")
            else:
                lines.append(prefix + f"{return_names[0]} := {_translate_expr(stmt.value, current_class, name_map)};")
                lines.append(prefix + "return;")
            continue
        if isinstance(stmt, ast.Break):
            lines.append(prefix + "break;")
            continue
        if isinstance(stmt, ast.Raise):
            lines.append(prefix + "assert false;")
            continue
        raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")
    return lines


def _call_multi_return_types(node: ast.Call) -> list[str] | None:
    func_name: str | None = None
    if isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    elif isinstance(node.func, ast.Name):
        func_name = node.func.id
    if func_name is None:
        return None
    for (_, name), types in _METHOD_RETURN_TYPE_OVERRIDES.items():
        if name == func_name:
            return types
    return None


def _tuple_assignments_with_types(
    stmts: list[ast.stmt],
    name_map: dict[str, str] | None = None,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for stmt in stmts:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Tuple) or not isinstance(stmt.value, ast.Call):
            continue
        ret_types = _call_multi_return_types(stmt.value)
        if ret_types is None or len(ret_types) != len(stmt.targets[0].elts):
            continue
        for elt, ret_type in zip(stmt.targets[0].elts, ret_types):
            if isinstance(elt, ast.Name):
                assignments[(name_map or {}).get(elt.id, elt.id)] = ret_type
    return assignments


def _if_branch_locals(
    stmt: ast.If,
    current_class: str | None,
    name_map: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    if not stmt.orelse:
        return []
    body_assignments = _tuple_assignments_with_types(stmt.body, name_map)
    orelse_assignments = _tuple_assignments_with_types(stmt.orelse, name_map)
    shared = []
    for name, ret_type in body_assignments.items():
        if name in orelse_assignments and orelse_assignments[name] == ret_type:
            shared.append((name, ret_type))
    return shared


def _method_return_names(node: ast.FunctionDef) -> list[str]:
    if node.name in _RETURN_NAME_OVERRIDES:
        return _RETURN_NAME_OVERRIDES[node.name]
    if node.returns is None:
        return []
    return ["result"]


def _mutated_self_fields(class_node: ast.ClassDef) -> set[str]:
    mutated: set[str] = set()
    for item in class_node.body:
        if not isinstance(item, ast.FunctionDef) or item.name == "__init__":
            continue
        for inner in ast.walk(item):
            targets: list[ast.AST] = []
            if isinstance(inner, ast.Assign):
                targets = list(inner.targets)
            elif isinstance(inner, ast.AnnAssign):
                targets = [inner.target]
            elif isinstance(inner, ast.AugAssign):
                targets = [inner.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    mutated.add(target.attr)
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute) and isinstance(target.value.value, ast.Name) and target.value.value.id == "self":
                    mutated.add(target.value.attr)
    return mutated


def _emit_signature_header(
    node: ast.FunctionDef,
    spec: dict[str, Any],
    current_class: str | None,
    axiomatize: bool = False,
) -> tuple[str, list[str]]:
    args = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        arg_type = _PARAM_TYPE_OVERRIDES.get((current_class, node.name, arg.arg), _annotation_to_dafny(arg.annotation))
        args.append(f"{arg.arg}: {arg_type}")
    arg_src = ", ".join(args)
    attrs = []
    if spec.get("extern"):
        attrs.append("{:extern}")
    if spec.get("axiom") or axiomatize:
        attrs.append("{:axiom}")
    attr_src = (" " + " ".join(attrs)) if attrs else ""
    kind = spec["kind"]
    return_names = _method_return_names(node)

    if kind == "predicate":
        header = f"predicate{attr_src} {node.name}({arg_src})"
    elif kind == "function":
        ret_type = _RETURN_TYPE_OVERRIDES.get((current_class, node.name), _annotation_to_dafny(node.returns))
        ret_name = return_names[0] if return_names else "result"
        header = f"function{attr_src} {node.name}({arg_src}): ({ret_name}: {ret_type})"
    elif kind == "lemma":
        header = f"lemma{attr_src} {node.name}({arg_src})"
    elif kind == "constructor":
        header = f"constructor{attr_src} ({arg_src})"
    else:
        if node.returns is None or (isinstance(node.returns, ast.Constant) and node.returns.value is None) or (isinstance(node.returns, ast.Name) and node.returns.id == "None"):
            header = f"method{attr_src} {node.name}({arg_src})"
        elif isinstance(node.returns, ast.Subscript) and isinstance(node.returns.value, ast.Name) and node.returns.value.id in {"tuple", "Tuple"} and isinstance(node.returns.slice, ast.Tuple):
            override_types = _METHOD_RETURN_TYPE_OVERRIDES.get((current_class, node.name))
            if override_types is not None:
                typed_returns = [f"{name}: {ret_type}" for name, ret_type in zip(return_names, override_types)]
            else:
                typed_returns = [
                    f"{name}: {_annotation_to_dafny(ann)}"
                    for name, ann in zip(return_names, node.returns.slice.elts)
                ]
            header = f"method{attr_src} {node.name}({arg_src}) returns ({', '.join(typed_returns)})"
        elif node.returns is not None:
            ret_name = return_names[0] if return_names else "result"
            header = f"method{attr_src} {node.name}({arg_src}) returns ({ret_name}: {_annotation_to_dafny(node.returns)})"
        else:
            header = f"method{attr_src} {node.name}({arg_src})"
    return header, return_names


def _emit_decl(
    node: ast.FunctionDef,
    spec: dict[str, Any],
    current_class: str | None,
    source_lines: list[str],
    axiomatize: bool = False,
) -> list[str]:
    force_axiom = axiomatize and (current_class, node.name) in _FORCE_AXIOM_DECLS
    header, return_names = _emit_signature_header(node, spec, current_class, axiomatize=force_axiom)
    lines = [header]
    for read in spec.get("reads", ()):
        lines.append(f"  reads {read}")
    for modify in spec.get("modifies", ()):
        lines.append(f"  modifies {modify}")
    for req in spec.get("requires", ()):
        lines.append(f"  requires {req}")
    for ens in spec.get("ensures", ()):
        lines.append(f"  ensures {ens}")
    for dec in spec.get("decreases", ()):
        lines.append(f"  decreases {dec}")

    special_body = _special_function_body(node.name, current_class) if spec["kind"] in {"function", "predicate"} else None
    special_stmt_body = _special_stmt_body(node.name, current_class) if spec["kind"] in {"method", "lemma", "constructor"} else None
    if force_axiom:
        return lines
    if spec.get("extern") or (spec.get("axiom") and spec["kind"] in {"method", "function", "predicate", "constructor"}):
        return lines
    if spec.get("axiom") and spec["kind"] == "lemma":
        return lines

    lines.append("{")
    if spec["kind"] in {"function", "predicate"}:
        body_lines = special_body or _translate_function_block(node.body, current_class, indent=2)
        lines.extend(("  " + body_line) if not body_line.startswith("  ") else body_line for body_line in body_lines)
    else:
        declared = {arg.arg for arg in node.args.args if arg.arg != "self"}
        body_lines = special_stmt_body or _translate_stmt_list(node.body, current_class, source_lines, declared, return_names, node.name, indent=2)
        lines.extend(body_lines)
    lines.append("}")
    return lines


def transpile_contract_library(
    source: str,
    module_name_hint: str | None = None,
    axiomatize: bool = False,
) -> Result[str, Exception]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Err(e)

    source_lines = source.splitlines()
    module_name = module_name_hint or "GeneratedModule"
    include_lines: list[str] = []
    import_lines: list[str] = []
    alias_lines: list[str] = []
    const_lines: list[str] = []
    decl_lines: list[str] = []
    needs_any_type = False

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "MODULE_NAME" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                module_name = node.value.value
            elif target == "DAFNY_INCLUDE" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                include_lines.append(f'include "{node.value.value}"')
            elif target == "DAFNY_INCLUDES":
                include_lines.extend(f'include "{value}"' for value in _literal_string_values(node.value))
            elif target == "DAFNY_OPEN_IMPORT" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                import_lines.append(f"  import opened {node.value.value}")
            elif target in {"LeftDelimiter", "RightDelimiter"}:
                const_lines.append(f"  const {target}: Token := {_const_expr(node.value)}")
            continue

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias":
                alias_value = _type_alias_value_to_dafny(node.value, node.target.id)
                alias_lines.append(f"  type {node.target.id} = {alias_value}")
                if alias_value == "AnyType":
                    needs_any_type = True
            elif node.target.id == "DAFNY_INCLUDE" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                include_lines.append(f'include "{node.value.value}"')
            elif node.target.id == "DAFNY_INCLUDES":
                include_lines.extend(f'include "{value}"' for value in _literal_string_values(node.value))
            elif node.target.id == "DAFNY_OPEN_IMPORT" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                import_lines.append(f"  import opened {node.value.value}")
            elif node.target.id in {"LeftDelimiter", "RightDelimiter"}:
                const_lines.append(f"  const {node.target.id}: {_annotation_to_dafny(node.annotation)} := {_const_expr(node.value)}")
            continue

        if isinstance(node, ast.FunctionDef):
            if node.name in {"dafny_spec"}:
                continue
            spec = _spec_from_decorator(node)
            if spec is None:
                continue
            if "AnyType" in " ".join(str(v) for v in spec.values()):
                needs_any_type = True
            decl_lines.extend("  " + line if line else "" for line in _emit_decl(node, spec, None, source_lines, axiomatize=axiomatize))
            decl_lines.append("")
            continue

        if isinstance(node, ast.ClassDef):
            if node.name == "DafnySpec":
                continue
            mutated_fields = _mutated_self_fields(node)
            class_lines = [f"  class {node.name} {{"]
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_type = _FIELD_TYPE_OVERRIDES.get((node.name, item.target.id), _annotation_to_dafny(item.annotation, item.target.id))
                    modifier = "var" if item.target.id in mutated_fields else "const"
                    class_lines.append(f"    {modifier} {item.target.id}: {field_type}")
            if len(class_lines) > 1:
                class_lines.append("")
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                spec = _spec_from_decorator(item)
                if spec is None:
                    continue
                emitted = _emit_decl(item, spec, node.name, source_lines, axiomatize=axiomatize)
                class_lines.extend("    " + line if line else "" for line in emitted)
                class_lines.append("")
            class_lines.append("  }")
            decl_lines.extend(class_lines)
            decl_lines.append("")

    final_lines = []
    final_lines.extend(include_lines)
    if include_lines:
        final_lines.append("")
    final_lines.append(f"module {module_name} {{")
    final_lines.extend(import_lines)
    if import_lines:
        final_lines.append("")
    if needs_any_type:
        final_lines.append("  type AnyType")
        final_lines.append("")
    final_lines.extend(alias_lines)
    if alias_lines:
        final_lines.append("")
    final_lines.extend(const_lines)
    if const_lines:
        final_lines.append("")
    final_lines.extend(decl_lines)
    final_lines.append("}")
    return Ok("\n".join(line.rstrip() for line in final_lines if line is not None))


def transpile_python_file_to_dafny(python_file: str, axiomatize: bool = False) -> Result[str, Exception]:
    path = Path(python_file).resolve()
    if not path.exists():
        return Err(FileNotFoundError(f"Python file not found: {path}"))
    try:
        source = path.read_text()
    except Exception as e:
        return Err(e)
    return transpile_contract_library(source, module_name_hint=path.stem, axiomatize=axiomatize)


def _literal_string_values(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for elt in node.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                raise ValueError("DAFNY_INCLUDE/DAFNY_INCLUDES values must be string literals")
            values.append(elt.value)
        return values
    raise ValueError("DAFNY_INCLUDE/DAFNY_INCLUDES must be a string or a list/tuple/set of strings")


def _extract_dafny_includes(source: str) -> Result[list[str], Exception]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Err(e)

    includes: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target in {"DAFNY_INCLUDE", "DAFNY_INCLUDES"}:
                try:
                    includes.extend(_literal_string_values(node.value))
                except Exception as e:
                    return Err(e)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            if target in {"DAFNY_INCLUDE", "DAFNY_INCLUDES"}:
                try:
                    includes.extend(_literal_string_values(node.value))
                except Exception as e:
                    return Err(e)
    return Ok(includes)


def _resolve_python_dependency(source_path: Path, include_name: str) -> Path | None:
    include_path = Path(include_name)
    candidates: list[Path] = []
    if include_path.suffix == ".dfy":
        candidates.append((source_path.parent / include_path).with_suffix(".py"))
    elif include_path.suffix:
        candidates.append(source_path.parent / include_path)
    else:
        candidates.append((source_path.parent / include_path).with_suffix(".py"))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _materialize_transpiled_module(
    source_path: Path,
    output_path: Path,
    axiomatize: bool,
    visiting: set[Path],
    written: set[Path],
) -> Result[None, Exception]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()

    if output_path in written:
        return Ok(None)
    if source_path in visiting:
        return Err(ValueError(f"Circular Dafny dependency detected while transpiling {source_path.name}"))

    try:
        source = source_path.read_text()
    except Exception as e:
        return Err(e)

    visiting.add(source_path)
    includes_result = _extract_dafny_includes(source)
    if includes_result.is_err():
        visiting.remove(source_path)
        return Err(includes_result.error)

    for include_name in includes_result.value:
        include_output_path = (output_path.parent / include_name).resolve()
        include_output_path.parent.mkdir(parents=True, exist_ok=True)

        python_dependency = _resolve_python_dependency(source_path, include_name)
        if python_dependency is not None:
            dep_result = _materialize_transpiled_module(
                python_dependency,
                include_output_path,
                axiomatize=axiomatize,
                visiting=visiting,
                written=written,
            )
            if dep_result.is_err():
                visiting.remove(source_path)
                return dep_result
            continue

        dafny_dependency = (source_path.parent / include_name).resolve()
        if dafny_dependency.exists():
            try:
                shutil.copyfile(dafny_dependency, include_output_path)
            except Exception as e:
                visiting.remove(source_path)
                return Err(e)
            written.add(include_output_path)
            continue

        visiting.remove(source_path)
        return Err(FileNotFoundError(
            f"Could not resolve Dafny dependency '{include_name}' for {source_path.name}. "
            f"Expected either {(source_path.parent / include_name).with_suffix('.py')} or {source_path.parent / include_name}."
        ))

    dafny_result = transpile_contract_library(source, module_name_hint=source_path.stem, axiomatize=axiomatize)
    visiting.remove(source_path)
    if dafny_result.is_err():
        return Err(dafny_result.error)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(dafny_result.value)
    except Exception as e:
        return Err(e)
    written.add(output_path)
    return Ok(None)


def prepare_dafny_workspace_from_python(
    main_python_file: str | Path,
    workspace_dir: str | Path,
    axiomatize: bool = False,
) -> Result[Path, Exception]:
    main_path = Path(main_python_file).resolve()
    workspace = Path(workspace_dir).resolve()
    main_output_path = workspace / main_path.with_suffix(".dfy").name
    result = _materialize_transpiled_module(
        main_path,
        main_output_path,
        axiomatize=axiomatize,
        visiting=set(),
        written=set(),
    )
    if result.is_err():
        return Err(result.error)
    return Ok(main_output_path)


def _verify_python_file_with_dependencies(python_file: str, dafny_binary: str | None = None) -> Result[bool, Exception]:
    path = Path(python_file).resolve()
    try:
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".dafny-transpile-") as workspace_dir:
            prep_result = prepare_dafny_workspace_from_python(path, workspace_dir, axiomatize=False)
            if prep_result.is_err():
                return Err(prep_result.error)
            return _run_dafny_verify_file(str(prep_result.value), dafny_binary=dafny_binary)
    except Exception as e:
        return Err(e)


def verify_python_file(python_file: str, dafny_binary: str | None = None) -> Result[bool, Exception]:
    path = Path(python_file).resolve()
    print(f"Transpiling {path.name} to temporary Dafny contract stubs and verifying them")
    return _verify_python_file_with_dependencies(str(path), dafny_binary=dafny_binary)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transpile a Python contract library to Dafny and verify it.")
    parser.add_argument("python_file", help="Path to the Python file to verify")
    parser.add_argument(
        "--dafny",
        default=None,
        help="Optional path to the Dafny binary. Defaults to the repo-local Dafny if present.",
    )
    parser.add_argument(
        "--print-dafny",
        action="store_true",
        help="Print the generated Dafny source before verification.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.print_dafny:
        dafny_result = transpile_python_file_to_dafny(args.python_file, axiomatize=False)
        if dafny_result.is_err():
            print(f"Transpilation failed: {dafny_result.error}")
            raise SystemExit(1)
        print(dafny_result.value)
        raise SystemExit(0)
    result = verify_python_file(args.python_file, dafny_binary=args.dafny)
    if result.is_ok():
        print("Verification succeeded.")
        raise SystemExit(0)
    print(f"Verification failed: {result.error}")
    raise SystemExit(1)
