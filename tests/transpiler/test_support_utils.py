from verification.transpiler.support import (
    Err,
    Ok,
    is_builtin_type_string,
    remove_comments_and_docstrings,
    resolve_type_from_string,
)


def test_remove_comments_and_docstrings_strips_module_and_function_docs():
    source = '''"""module docs"""
# a comment
def helper():
    """function docs"""
    # another comment
    return 1
'''

    cleaned = remove_comments_and_docstrings(source)

    assert "module docs" not in cleaned
    assert "function docs" not in cleaned
    assert "return 1" in cleaned


def test_builtin_type_resolution_helpers_cover_common_cases():
    assert is_builtin_type_string("int")
    assert resolve_type_from_string("str") is str


def test_result_types_report_success_and_failure():
    assert Ok(1).is_ok()
    assert not Ok(1).is_err()
    assert Err(ValueError("boom")).is_err()
    assert not Err(ValueError("boom")).is_ok()
