import importlib
import builtins
import typing
import collections.abc
from typing import Any, Type, TypeVar, Generic, Union, Callable
from dataclasses import dataclass

# Define generic type variables for our container
T = TypeVar('T') # For the success value type
E = TypeVar('E') # For the error value type

@dataclass(frozen=True)
class Ok(Generic[T]):
    """Represents a successful result, containing a value."""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

@dataclass(frozen=True)
class Err(Generic[E]):
    """Represents an error, containing an error value."""
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

# A Result is either an Ok or an Err. We use Union for type hinting.
Result = Union[Ok[T], Err[E]]

def is_builtin_type_string(type_string: str) -> bool:
    """
    Checks if a string corresponds to the name of a built-in type.

    Args:
        type_string: The string to check (e.g., "int", "list", "MyClass").

    Returns:
        True if the string is a name of a built-in type, False otherwise.
    """
    # 1. Check if the name exists in the builtins module
    if not hasattr(builtins, type_string):
        return False
    
    # 2. Get the attribute from the builtins module
    the_type = getattr(builtins, type_string)
    
    # 3. Check if the attribute is actually a type (class)
    return isinstance(the_type, type)



def resolve_type_from_string(type_string: str) -> Any:
    # 1. Handle built-ins
    if '.' not in type_string:
        if hasattr(builtins, type_string):
            return getattr(builtins, type_string)
        raise ValueError(f"Type '{type_string}' is not a valid builtin.")

    parts = type_string.split('.')
    
    # 2. Iterative resolution
    for i in range(len(parts), 0, -1):
        module_path = '.'.join(parts[:i])
        attr_path = parts[i:]
        
        try:
            module = importlib.import_module(module_path)
            resolved_obj = module
            
            for attr in attr_path:
                resolved_obj = getattr(resolved_obj, attr)
            
            # --- NEW CHECK LOGIC ---
            # We check if the object is a Callable Type Alias (e.g. Callable[[float], float])
            # OR if it is a standard callable (function/class).
            
            # Check 1: Is it a Typing Alias for Callable?
            origin = typing.get_origin(resolved_obj)
            is_callable_alias = (origin is collections.abc.Callable or 
                                 origin is typing.Callable)

            # Check 2: Is it an actual callable function/class? 
            # (Optional: remove this if you ONLY want Type Aliases)
            is_actual_callable = callable(resolved_obj)

            if is_callable_alias or is_actual_callable:
                return resolved_obj
            else:
                # If we found an object but it's not Callable, treat it as 
                # the wrong path and trigger the 'except' block to retry.
                raise AttributeError(f"Object found but is not Callable: {type(resolved_obj)}")
            # -----------------------

        except (ImportError, AttributeError):
            continue

    raise ImportError(f"Could not resolve a callable type for '{type_string}'.") # def resolve_type_from_string(type_string: str) -> Type:

#     """
#     Dynamically imports and returns a class type from its string path.
#     For example, 'builtins.str' will return the `str` type.
#     """
#     try:
#         module_path, class_name = type_string.rsplit('.', 1)
#         module = importlib.import_module(module_path)
#         return getattr(module, class_name)
#     except (ImportError, AttributeError, ValueError) as e:
#         # Handle built-in types like 'str', 'int', etc. which don't have a module path.
#         if '.' not in type_string:
#             try:
#                 return getattr(importlib.import_module('builtins'), type_string)
#             except AttributeError:
#                 pass # Fall through to the original error
#         raise ImportError(f"Could not resolve the type '{type_string}'. Error: {e}")