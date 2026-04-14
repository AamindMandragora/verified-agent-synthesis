import ast

from verification.transpiler.transpiler import _translate_compare, _translate_expr, _translate_stmt_list


def test_translate_isinstance_str_to_true():
    expr = ast.parse("isinstance(next_token, str)", mode="eval").body
    assert _translate_expr(expr) == "true"


def test_translate_str_identity():
    expr = ast.parse("str(next_token)", mode="eval").body
    assert _translate_expr(expr) == "next_token"


def test_translate_is_not_none_to_true():
    expr = ast.parse("next_token is not None", mode="eval").body
    assert _translate_compare(expr, None) == "true"


def test_translate_is_none_to_false():
    expr = ast.parse("next_token is None", mode="eval").body
    assert _translate_compare(expr, None) == "false"


def test_translate_token_not_in_string_literal():
    expr = ast.parse('next_token not in "()"', mode="eval").body
    assert _translate_compare(expr, None) == '(next_token != "(" && next_token != ")")'


def test_translate_token_in_tuple_literal():
    expr = ast.parse("next_token in ('+', '-', '*', '/')", mode="eval").body
    translated = _translate_compare(expr, None)
    assert translated == '(next_token == "+" || next_token == "-" || next_token == "*" || next_token == "/")'


def test_translate_isalpha_predicate():
    expr = ast.parse("answer[-1].isalpha()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'a'" in translated
    assert "'Z'" in translated


def test_translate_isdigit_predicate():
    expr = ast.parse("answer[-1].isdigit()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'0'" in translated
    assert "'9'" in translated


def test_translate_isnumeric_predicate():
    expr = ast.parse("answer[-1].isnumeric()", mode="eval").body
    translated = _translate_expr(expr)
    assert "forall i ::" in translated
    assert "'0'" in translated
    assert "'9'" in translated


def test_translate_list_append_as_sequence_update():
    expr = ast.parse("free_form_buffer.append(next_token)", mode="eval").body
    translated = _translate_expr(expr)
    assert translated == "free_form_buffer := (free_form_buffer + [next_token])"


def test_translate_mycsd_none_initializers_to_defaults():
    source = "next_token = None\nnew_steps = None\n"
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated[0] == "var next_token := eosToken;"
    assert translated[1] == "var new_steps := stepsLeft;"


def test_translate_break_statement():
    source = "while keep_going:\n    break\n"
    while_stmt = ast.parse(source).body[0]
    translated = _translate_stmt_list(
        while_stmt.body,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated == ["break;"]


def test_translate_while_with_specs_before_setup_lines():
    source = """# invariant lm.ValidTokensIdsLogits()
# invariant parser.IsValidPrefix(answer)
# decreases stepsLeft
# Initialize loop state
phase = 0
answer_tokens = 0
while stepsLeft > 0 and not parser.IsCompletePrefix(answer):
    answer_tokens = answer_tokens + 1
"""
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    while_line = "while ((stepsLeft > 0) && (!parser.IsCompletePrefix(answer)))"
    assert while_line in translated
    while_index = translated.index(while_line)
    assert translated[while_index + 1] == "  invariant lm.ValidTokensIdsLogits()"
    assert translated[while_index + 2] == "  invariant parser.IsValidPrefix(answer)"
    assert translated[while_index + 3] == "  decreases stepsLeft"


def test_translate_any_isdigit_over_token_uses_index_quantifier():
    expr = ast.parse("any(char.isdigit() for char in next_token)", mode="eval").body
    translated = _translate_expr(expr)

    assert "exists char_idx ::" in translated
    assert "next_token[char_idx]" in translated
    assert "char in next_token" not in translated


def test_translate_mixed_any_predicates_parenthesizes_quantifiers():
    expr = ast.parse(
        "isinstance(next_token, str) and (any(c.isalpha() for c in next_token) or any(c.isdigit() for c in next_token))",
        mode="eval",
    ).body
    translated = _translate_expr(expr)

    assert translated.startswith("(")
    assert "((exists c_idx ::" in translated
    assert ") || ((exists c_idx ::" in translated or ") || ((exists c_idx_2 ::" in translated
    assert "c in next_token" not in translated


def test_translate_startswith_tuple_literal():
    expr = ast.parse("next_token.startswith(('n', 'x'))", mode="eval").body
    translated = _translate_expr(expr)

    assert '|next_token| >= |"n"|' in translated
    assert 'next_token[..|"n"|] == "n"' in translated
    assert '||' in translated


def test_translate_tuple_literal_assignment():
    source = "next_token, new_steps = None, 0\n"
    stmts = ast.parse(source).body
    translated = _translate_stmt_list(
        stmts,
        current_class=None,
        source_lines=source.splitlines(),
        declared=set(),
        return_names=[],
        method_name="MyCSDStrategy",
        indent=0,
    )

    assert translated == ["var next_token := eosToken;", "var new_steps := 0;"]
