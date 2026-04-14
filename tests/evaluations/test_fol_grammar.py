from pathlib import Path

from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_parser():
    grammar = (PROJECT_ROOT / "utils" / "grammars" / "folio.lark").read_text(encoding="utf-8")
    parser = Lark(grammar, start="start", parser="lalr")
    parser._UnexpectedCharacters = UnexpectedCharacters
    parser._UnexpectedToken = UnexpectedToken
    parser._UnexpectedEOF = UnexpectedEOF
    return parser


def _is_valid_prefix(parser, text: str) -> bool:
    if not text:
        return True
    try:
        parser.parse(text)
        return True
    except parser._UnexpectedEOF:
        return True
    except parser._UnexpectedToken as exc:
        return exc.token.type == "$END"
    except parser._UnexpectedCharacters:
        return False
    except Exception:
        return False


def _is_complete(parser, text: str) -> bool:
    if not text:
        return False
    try:
        parser.parse(text)
        return True
    except Exception:
        return False


def _valid_next_tokens(parser, text: str, vocabulary: list[str]) -> list[str]:
    if text and not _is_valid_prefix(parser, text):
        return []
    valid = []
    for token in vocabulary:
        if token and _is_valid_prefix(parser, text + token):
            valid.append(token)
    return valid


def test_single_folio_atom_is_valid_and_complete():
    parser = _build_parser()

    assert _is_valid_prefix(parser, "P(x)")
    assert _is_complete(parser, "P(x)")


def test_incomplete_folio_formula_stays_valid_prefix():
    parser = _build_parser()

    assert _is_valid_prefix(parser, "P(x) {and}")
    assert not _is_complete(parser, "P(x) {and}")


def test_invalid_folio_formula_is_rejected():
    parser = _build_parser()

    assert not _is_valid_prefix(parser, "({and})")
    assert not _is_complete(parser, "({and})")


def test_complete_atom_can_extend_with_connective_tokens():
    parser = _build_parser()
    vocabulary = ["{and}", "{or}", "{not}", " ", "%", "P", "Q", "x", "(", ")"]

    valid_next = _valid_next_tokens(parser, "P(x)", vocabulary)

    assert "{and}" in valid_next or " " in valid_next
