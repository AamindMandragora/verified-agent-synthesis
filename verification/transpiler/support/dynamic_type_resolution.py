from __future__ import annotations

import builtins
import collections.abc
import importlib
import typing
from typing import Any


def is_builtin_type_string(type_string: str) -> bool:
    """Return whether ``type_string`` names a builtin type."""
    if not hasattr(builtins, type_string):
        return False
    return isinstance(getattr(builtins, type_string), type)


def resolve_type_from_string(type_string: str) -> Any:
    """
    Resolve a callable or builtin type object from its string path.
    """
    if "." not in type_string:
        if hasattr(builtins, type_string):
            return getattr(builtins, type_string)
        raise ValueError(f"Type '{type_string}' is not a valid builtin.")

    parts = type_string.split(".")
    for index in range(len(parts), 0, -1):
        module_path = ".".join(parts[:index])
        attr_path = parts[index:]

        try:
            module = importlib.import_module(module_path)
            resolved_obj = module
            for attr in attr_path:
                resolved_obj = getattr(resolved_obj, attr)

            origin = typing.get_origin(resolved_obj)
            is_callable_alias = origin in (collections.abc.Callable, typing.Callable)
            if is_callable_alias or callable(resolved_obj):
                return resolved_obj
        except (ImportError, AttributeError):
            continue

    raise ImportError(f"Could not resolve a callable type for '{type_string}'.")
