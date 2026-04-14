from __future__ import annotations

import ast
import os
import re
import tempfile
import uuid
from typing import Any, Dict, List, Protocol

from mypy import api
import mypy.build as mypy_build
import mypy.main as mypy_main

from .result import Err, Ok, Result


def _unparse_annotation(node: ast.AST | None) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except AttributeError:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_unparse_annotation(node.value)}.{node.attr}"
        return "Any"


def clean_ansi_output(raw_text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", raw_text)


class SupportsTypeContext(Protocol):
    global_context_signature_name_only: dict[str, Any]
    custom_types_list: list[str]


class MypyTypeChecker:
    """
    Build mypy stubs for the synthesis context and return typed ASTs.
    """

    def __init__(
        self,
        context_builder: SupportsTypeContext,
        agent_context: SupportsTypeContext,
    ) -> None:
        self.context_builder = context_builder
        self.global_context_signature = context_builder.global_context_signature_name_only
        self.custom_type_names = context_builder.custom_types_list
        self.agent_context = agent_context
        self.agent_signature = self.agent_context.global_context_signature_name_only["agent"]
        self.agent_custom_types = agent_context.custom_types_list
        self._update_custom_types()

    def _update_custom_types(self) -> None:
        for custom_type in self.agent_custom_types:
            if custom_type not in self.custom_type_names:
                self.custom_type_names.append(custom_type)
            if custom_type not in self.global_context_signature:
                self.global_context_signature[custom_type] = (
                    self.agent_context.global_context_signature_name_only[custom_type]
                )

    def _check_main_function_signature(self, code: str) -> Result[bool, Exception]:
        try:
            function_node = ast.parse(code).body[0]
            if not isinstance(function_node, ast.FunctionDef):
                return Err(ValueError("Code is not a single function definition."))

            expected_inputs = self.agent_signature.get("input_signature", {})
            actual_inputs_map = {
                arg.arg: _unparse_annotation(arg.annotation) for arg in function_node.args.args
            }
            if actual_inputs_map != expected_inputs:
                return Err(
                    TypeError(
                        f"Input signature mismatch. Expected: {expected_inputs}, Got: {actual_inputs_map}"
                    )
                )

            expected_output = self.agent_signature.get("output_signature", "Any")
            actual_output = _unparse_annotation(function_node.returns)
            if actual_output != expected_output:
                return Err(
                    TypeError(
                        f"Output signature mismatch. Expected: '{expected_output}', Got: '{actual_output}'"
                    )
                )

            return Ok(True)
        except (SyntaxError, IndexError) as exc:
            return Err(exc)

    def _process_type_str(self, type_str: str) -> str:
        if type_str == "List":
            return "List[Any]"
        if type_str == "Dict":
            return "Dict[Any, Any]"
        return type_str

    def _generate_custom_type_stubs(self) -> List[str]:
        stubs: List[str] = []
        for custom_type in self.custom_type_names:
            signature = self.global_context_signature[custom_type]
            inputs = signature.get("input_signature", {})
            params: list[str] = []
            for name, type_str in inputs.items():
                if name == "self":
                    params.append("self")
                else:
                    params.append(f"{name}: {self._process_type_str(type_str)}")
            stubs.append(
                f"class {custom_type}:\n\tdef __init__({', '.join(params)}) -> None: ..."
            )
        return stubs

    def _generate_context_stubs(self) -> str:
        stubs = ["from typing import Any, Dict, List, Set, Type"]
        stubs.extend(self._generate_custom_type_stubs())

        for op_name, sig in self.global_context_signature.items():
            if op_name in self.custom_type_names:
                continue
            inputs = sig.get("input_signature", {})
            output = sig.get("output_signature", "Any")
            params = ", ".join(
                f"{name}: {self._process_type_str(type_str)}"
                for name, type_str in inputs.items()
            )
            stubs.append(f"def {op_name}({params}) -> {output}: ...")

        return "\n".join(stubs)

    def get_stub(self) -> str:
        return self._generate_context_stubs()

    def _write_typecheck_files(
        self,
        *,
        temp_dir: str,
        code: str,
        context_stubs: str,
    ) -> tuple[str, str]:
        unique_id = uuid.uuid4().hex[:12]
        stub_module_name = f"stubs_{unique_id}"
        program_name = f"program_{unique_id}.py"
        stub_path = os.path.join(temp_dir, f"{stub_module_name}.pyi")
        program_path = os.path.join(temp_dir, program_name)
        modified_code = f"from {stub_module_name} import *\n{code}"

        with open(stub_path, "w", encoding="utf-8") as stub_file:
            stub_file.write("import builtins\n")
            stub_file.write(context_stubs)
        with open(program_path, "w", encoding="utf-8") as program_file:
            program_file.write(modified_code)

        return stub_path, program_path

    def _mypy_args(self, program_path: str) -> list[str]:
        return [
            program_path,
            "--strict",
            "--check-untyped-defs",
            "--allow-redefinition",
            "--no-color",
            "--no-error-summary",
        ]

    def _forward(self, code: str) -> Result[bool, Exception]:
        preliminary_check = self._check_main_function_signature(code)
        if preliminary_check.is_err():
            return preliminary_check

        context_stubs = self._generate_context_stubs()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _, program_path = self._write_typecheck_files(
                    temp_dir=temp_dir,
                    code=code,
                    context_stubs=context_stubs,
                )
                report, _errors, exit_status = api.run(self._mypy_args(program_path))
                if exit_status == 0:
                    return Ok(True)
                return Err(
                    TypeError(
                        f"Generated code has type error. See report \n\n{clean_ansi_output(report)}"
                    )
                )
        except Exception as exc:
            return Err(exc)

    def get_typed_tree(self, code: str) -> Result[Any, Exception]:
        preliminary_check = self._check_main_function_signature(code)
        if preliminary_check.is_err():
            return preliminary_check

        context_stubs = self._generate_context_stubs()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                _, program_path = self._write_typecheck_files(
                    temp_dir=temp_dir,
                    code=code,
                    context_stubs=context_stubs,
                )
                files, options = mypy_main.process_options(self._mypy_args(program_path))
                if not files:
                    return Err(FileNotFoundError(f"MyPy could not find source file: {program_path}"))

                module_id = files[0].module
                options.preserve_asts = True
                options.fine_grained_incremental = True
                result = mypy_build.build(files, options=options)

                if module_id not in result.graph:
                    if "__main__" in result.graph:
                        module_id = "__main__"
                    else:
                        return Err(
                            FileNotFoundError(
                                f"Module '{module_id}' not in graph. Found: {result.graph.keys()}"
                            )
                        )

                tree = result.graph[module_id].tree
                if tree is None:
                    return Err(TypeError(f"MyPy did not preserve AST for module: {module_id}"))
                return Ok(tree)
        except Exception as exc:
            return Err(exc)


TypeCheck = MypyTypeChecker
