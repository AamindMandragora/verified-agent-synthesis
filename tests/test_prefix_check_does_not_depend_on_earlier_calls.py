"""The prefix check must give the same answer however the text was built up.

Found 2026-09-19: syncode's incremental parser keeps parser state between calls. When
characters that were separate words (``O``, ``=``, ``C`` ...) later merge into one long
word (the acrylate group ``O=C(O)C(=C)``), the remembered state is wrong, and the
unbalanced ``O=C(O)C(=C))`` was accepted - but only after a character-by-character
history, which is how decoding really calls it.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser(grammar_name):
    transformers = pytest.importorskip("transformers")
    from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser

    class _VDA:
        class Parser:
            pass

    class _Dafny:
        Seq = staticmethod(lambda s: s)
        SeqWithoutIsStrInference = staticmethod(lambda items: items)

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3.5-4B tokenizer is not cached on this machine")
    vocab = [tok.decode([i]) for i in range(len(tok))]
    grammar = (REPO_ROOT / f"synthesis/evaluate/grammars/{grammar_name}.lark").read_text()
    return create_lark_dafny_parser(grammar, _VDA(), _Dafny(), tokenizer=tok, start="start")(vocab)


@pytest.mark.parametrize("group", ["O=C(O)C(=C)", "C=CC(=O)O", "OC(=O)C(=C)"])
def test_an_extra_closing_bracket_after_a_long_group_is_rejected_after_any_history(group):
    parser = _parser("smiles_acrylates")
    for end in range(1, len(group) + 1):
        assert parser._is_valid_prefix(group[:end])
    assert not parser._is_valid_prefix(group + ")")
    assert not parser._is_valid_prefix(group + "C)")


def test_a_long_group_can_still_be_followed_by_more_molecule():
    parser = _parser("smiles_acrylates")
    for text in ["O=C(O)C(=C)C", "C=CC(=O)OC(C)", "CC(=C)C(=O)OCCO", "N=C=O"]:
        for end in range(1, len(text) + 1):
            assert parser._is_valid_prefix(text[:end]), text[:end]
