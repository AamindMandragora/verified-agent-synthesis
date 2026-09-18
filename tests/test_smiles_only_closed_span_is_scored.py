from synthesis.evaluate.benchmarks.smiles import eval_logic


def test_only_a_closed_span_is_an_answer():
    f = eval_logic._span_content_for_scoring
    assert f("<<CCO>> trailing") == "CCO"
    assert f("<<CCO>> <<CCN>>") == "CCN"
    assert f("<<CCO") == ""
    assert f("<<CCO>> <<CCN") == "CCO"
    assert f("CCO") == "CCO"
    assert f("") == ""
