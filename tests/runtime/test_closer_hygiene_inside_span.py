"""Inside a span the content never contains `>>`: the span helper writes the closer, the grammar
never does, and the grammar mask leaks look-alikes (` >>`, `>>\\n`, `)>>`) that must be banned."""
import pytest
import torch

from synthesis.evaluate.benchmarks.common.delimiter_hygiene import apply_closer_rule, closer_allowed


@pytest.mark.parametrize("content,token", [
    ("1+2", ">>"), ("1+2", " >>"), ("(1+2", ")>>"), ("1+2", ">>\n"), ("1+2", " >>\n\n"), ("1+2", ">>>"),
    ("1+2>", ">"), ("1+2>", ">\n"), ("x = '", ">>'"), ("a >", "> b"), ("1+2>>", "3"),
])
def test_any_token_that_puts_a_closer_in_the_content_is_banned(content, token):
    assert not closer_allowed(content, token)


@pytest.mark.parametrize("content,token", [("1", "+2"), ("a ", "> 1"), ("a >", "= 1"), ("a", ">")])
def test_ordinary_tokens_and_a_single_greater_than_still_work(content, token):
    assert closer_allowed(content, token)


def test_mask_application_bans_only_offenders_and_does_not_touch_the_input():
    texts = ["1", ">>", ">>\n", " >>", ">"]
    mask = torch.ones(len(texts), dtype=torch.bool)
    gt = [(i, t) for i, t in enumerate(texts) if ">" in t]
    assert apply_closer_rule(mask, gt, "1+2").tolist() == [True, False, False, False, True]
    assert apply_closer_rule(mask, gt, "1+2>").tolist() == [True, False, False, False, False]
    assert mask.all()
