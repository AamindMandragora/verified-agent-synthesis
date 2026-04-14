import ast
import tempfile
import os
from typing import Dict, Any, List, Protocol
import uuid

import re
from xml.dom import NotFoundErr
from mypy import api
import mypy.main as MAIN
import mypy.build as BUILD

try:
    from .type_resolver import Result, Ok, Err
except ImportError:
    from type_resolver import Result, Ok, Err

# Helper from a previous answer to unparse AST annotations
def _unparse_annotation(node: ast.AST | None) -> str:
    if node is None: return "Any"
    # ast.unparse is the most robust, available in Python 3.9+
    try: return ast.unparse(node)
    except AttributeError: # Fallback for older Python
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return f"{_unparse_annotation(node.value)}.{node.attr}"
        return "Any"

def clean_ansi_output(raw_text: str) -> str:
    """
    Removes ANSI escape sequences (colors, bolding) from a string.
    """
    # Regex explanation:
    # \x1b  -> matches the ESC character (hex 1B)
    # \[    -> matches the literal bracket
    # [0-9;]* -> matches any number of digits or semicolons (the color codes)
    # m     -> matches the literal 'm' which terminates the sequence
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    
    cleaned_text = ansi_escape.sub('', raw_text)
    
    return cleaned_text
        

class SupportsTypeContext(Protocol):
    global_context_signature_name_only: dict[str, Any]
    custom_types_list: list[str]


class TypeCheck:
    def __init__(self, 
                 context_builder: SupportsTypeContext, 
                 agent_context: SupportsTypeContext) -> None:
        self.context_builder = context_builder
        # self.function_signature = task_config.get('function_signature')
        self.global_context_signature = context_builder.global_context_signature_name_only
        self.custom_type_names = context_builder.custom_types_list
        # Agent signature plus any custom type constructor
        self.agent_context = agent_context
        self.agent_signature = self.agent_context.global_context_signature_name_only["agent"]
        self.agent_custom_types = agent_context.custom_types_list
        # Update the info
        self._update_custom_types()

    def _update_custom_types(self):
        for custom_type in self.agent_custom_types:
            if custom_type not in self.custom_type_names:
                self.custom_type_names.append(custom_type)
            if custom_type not in self.global_context_signature:
                self.global_context_signature[custom_type] = self.agent_context.global_context_signature_name_only[custom_type]


    def _check_main_function_signature(self, code: str) -> Result[bool, Exception]:
        """
        Performs a fast, preliminary check of the main function's signature using AST.
        This gives clearer errors for signature mismatches before running full mypy.
        """
        try:
            function_node = ast.parse(code).body[0]
            if not isinstance(function_node, ast.FunctionDef):
                 return Err(ValueError("Code is not a single function definition."))

            expected_inputs = self.agent_signature.get('input_signature', {})
            # for k, v in expected_inputs.items():
            #     expected_inputs[k] = self.context_builder.internal_map.get(v, v)
            print([arg for arg in function_node.args.args])
            actual_inputs_map = {
                arg.arg: _unparse_annotation(arg.annotation) for arg in function_node.args.args
            }
            if actual_inputs_map != expected_inputs:
                 return Err(TypeError(f"Input signature mismatch. Expected: {expected_inputs}, Got: {actual_inputs_map}"))

            expected_output = self.agent_signature.get('output_signature', 'Any')
            # expected_output = self.context_builder.internal_map.get(expected_output, expected_output)
            actual_output = _unparse_annotation(function_node.returns) 
            if actual_output != expected_output:
                 return Err(TypeError(f"Output signature mismatch. Expected: '{expected_output}', Got: '{actual_output}'"))

            return Ok(True)
        except (SyntaxError, IndexError) as e:
            return Err(e)
    
    def _process_type_str(self, type_str: str) -> str:
        if type_str == 'List':
            return 'List[Any]'
        elif type_str == 'Dict':
            return 'Dict[Any, Any]'
        else:
            return type_str

    def _generate_custom_type_stubs(self) -> List[str]:
        stubs = []
        for custom_type in self.custom_type_names:
            signature = self.global_context_signature[custom_type]
            inputs = signature.get('input_signature')
            params = ''
            for name, type_str in inputs.items():
                if name == 'self':
                    params += 'self, '
                else:
                    params += f"{name}: {self._process_type_str(type_str)}"
            stubs.append(f'class {custom_type}:\n\tdef __init__({params}) -> None: ...')            
        return stubs

    def _generate_context_stubs(self) -> str:
        """
        Generates a string of Python code containing type stubs for mypy.
        """
        stubs = ["from typing import Any, List, Dict, Set, Type"] + self._generate_custom_type_stubs()
        
        # 1. Create stubs for all operators
        for op_name, sig in self.global_context_signature.items():
            if op_name in self.custom_type_names:
                continue
            inputs = sig.get('input_signature', {})
            output = sig.get('output_signature', 'Any')
            params = ", ".join([f"{name}: {self._process_type_str(type_str)}" for name, type_str in inputs.items()])
            stubs.append(f"def {op_name}({params}) -> {output}: ...")
        
        return "\n".join(stubs)

    def get_stub(self) -> str:
        return self._generate_context_stubs()

    def _forward(self, code: str) -> Result[bool, Exception]:
        """
        Validates the user's code by running mypy with a generated type context.
        """
        # 1. Fast check for the main function's signature
        preliminary_check = self._check_main_function_signature(code)
        if preliminary_check.is_err():
            return preliminary_check

        # 2. Generate the context stubs and combine with user code
        context_stubs = self._generate_context_stubs()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Generate a unique identifier to avoid module name clashes.
                unique_id = uuid.uuid4().hex[:12]
                unique_stub_module_name = f"stubs_{unique_id}"
                unique_program_name = f"program_{unique_id}.py"


                # 2. Define unique file paths inside the temporary directory.
                stub_path = os.path.join(temp_dir, f"{unique_stub_module_name}.pyi")
                program_path = os.path.join(temp_dir, unique_program_name)

                # 3. CRITICAL: Modify the user's code to import from the unique stub module.
                # This is the key to preventing parallel execution clashes.
                modified_code = f"from {unique_stub_module_name} import *\n"+code

                # 4. Write the stub and the modified program files.
                try:
                    with open(stub_path, "w") as f:
                        f.write(context_stubs)
                        # print(context_stubs)
                        # print(f'---------End of Stub--------')
                    with open(program_path, "w") as f:
                        f.write(modified_code)
                        # print(modified_code)

                except IOError as e:
                    # print(f"Error writing files: {e}")
                    return Err(e)

                # 5. Run MyPy on the uniquely named files.
                mypy_args = [
                    program_path,
                    '--strict',
                    '--check-untyped-defs',
                    '--allow-redefinition',
                    '--no-color',  
                    '--no-error-summary' 
                ]
                
                report, errors, exit_status = api.run(mypy_args)
                if exit_status == 0:
                    return Ok(True)
                else:
                    print(f'errors {errors}')
                    print(f'report {report}')
                    return Err(TypeError(f'Generated code has type error. See report \n\n{clean_ansi_output(report)}'))
        except Exception as e:
            return Err(e)
    
    def get_typed_tree(self, code: str) -> Result[Any, Exception]:
        # 1. Fast check for the main function's signature
        preliminary_check = self._check_main_function_signature(code)
        if preliminary_check.is_err():
            return preliminary_check

        # 2. Generate the context stubs and combine with user code
        context_stubs = self._generate_context_stubs()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Generate a unique identifier to avoid module name clashes.
                unique_id = uuid.uuid4().hex[:12]
                unique_stub_module_name = f"stubs_{unique_id}"
                unique_program_name = f"program_{unique_id}.py"


                # 2. Define unique file paths inside the temporary directory.
                stub_path = os.path.join(temp_dir, f"{unique_stub_module_name}.pyi")
                program_path = os.path.join(temp_dir, unique_program_name)

                # 3. CRITICAL: Modify the user's code to import from the unique stub module.
                # This is the key to preventing parallel execution clashes.
                modified_code = f"from {unique_stub_module_name} import *\n"+code

                # 4. Write the stub and the modified program files.
                try:
                    with open(stub_path, "w") as f:
                        # Add the builtin functions 
                        f.write('import builtins\n')
                        f.write(context_stubs)
                        # print(context_stubs)
                        # print(f'---------End of Stub--------')
                    with open(program_path, "w") as f:
                        f.write(modified_code)
                        # print(modified_code)
                except IOError as e:
                    # print(f"Error writing files: {e}")
                    return Err(e)
                # 5. Run MyPy on the uniquely named files.
                mypy_args = [
                    program_path,
                    '--strict',
                    '--check-untyped-defs',
                    '--allow-redefinition',
                    '--no-color',  
                    '--no-error-summary' 
                ]
                files, opt = MAIN.process_options(mypy_args)
                # opt.preserve_asts = True
                # opt.fine_grained_incremental = True
                # result = BUILD.build(files, options=opt)
                # print(f'graph dict keys {result.graph.keys()}')
                # tree = result.graph[unique_program_name[:-3]].tree
                # return Ok(tree)
                if not files:
                    return Err(NotFoundErr(f"MyPy could not find source file: {program_path}"))

                # Get the module ID that mypy assigned.
                # This is the *only* correct key to use.
                module_id = files[0].module
                # --- END FIX ---
                
                opt.preserve_asts = True
                opt.fine_grained_incremental = True
                result = BUILD.build(files, options=opt)
                
                # print(f'graph dict keys {result.graph.keys()}')

                # --- USE THE CORRECT KEY (with __main__ fallback) ---
                if module_id not in result.graph:
                    # For simple scripts, the ID is almost always '__main__'
                    if '__main__' in result.graph:
                        module_id = '__main__'
                    else:
                        # Handle error: module not found in graph
                        return Err(NotFoundErr(f"Module '{module_id}' not in graph. Found: {result.graph.keys()}"))

                # This tree will be fully analyzed and have built-ins resolved
                tree = result.graph[module_id].tree 
                
                if tree is None:
                    return Err(TypeError(f"MyPy did not preserve AST for module: {module_id}"))

                return Ok(tree)
        except Exception as e:
            print(f'Error happened while accessing the typed ast {str(e)}')
            return Err(e)
