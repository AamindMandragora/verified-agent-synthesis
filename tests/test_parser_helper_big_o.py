import sys
import types

from synthesis.evaluate.benchmarks.common import parser_utils


class _FakeBaseParser:
    pass


class _FakeCompleteLark:
    def __init__(self):
        self.parse_calls = 0

    def parse(self, text):
        self.parse_calls += 1
        return object()


class _FakeParseResult:
    def __init__(self, function_end=True, remainder_state_name="COMPLETE"):
        self.function_end = function_end
        self.remainder_state = types.SimpleNamespace(name=remainder_state_name)


class _FakeIncrementalParser:
    instances = []

    def __init__(self, base_parser, ignore_whitespace=False):
        self.calls = []
        type(self).instances.append(self)

    def get_acceptable_next_terminals(self, text):
        self.calls.append(text)
        return _FakeParseResult(function_end=True, remainder_state_name="COMPLETE")


class _FakeVerifiedDecoderAgent:
    class Parser:
        def IsDeadPrefix(self, prefix):
            return (not self.IsCompletePrefix(prefix)) and self.ValidNextTokenCountUpTo(prefix, 1) == 0


class _FakeDafny:
    @staticmethod
    def SeqWithoutIsStrInference(values):
        return list(values)


def _install_fake_syncode(monkeypatch):
    fake_lark = types.ModuleType("lark")
    fake_lark_exceptions = types.ModuleType("lark.exceptions")

    class UnexpectedCharacters(Exception):
        pass

    class UnexpectedToken(Exception):
        def __init__(self, token=None):
            self.token = token

    class UnexpectedEOF(Exception):
        pass

    fake_lark_exceptions.UnexpectedCharacters = UnexpectedCharacters
    fake_lark_exceptions.UnexpectedToken = UnexpectedToken
    fake_lark_exceptions.UnexpectedEOF = UnexpectedEOF
    fake_lark.exceptions = fake_lark_exceptions

    fake_syncode = types.ModuleType("syncode")
    fake_parsers = types.ModuleType("syncode.parsers")
    fake_incremental_parser = types.ModuleType("syncode.parsers.incremental_parser")
    fake_incremental_parser.IncrementalParser = _FakeIncrementalParser
    fake_parsers.incremental_parser = fake_incremental_parser
    fake_syncode.parsers = fake_parsers

    monkeypatch.setitem(sys.modules, "lark", fake_lark)
    monkeypatch.setitem(sys.modules, "lark.exceptions", fake_lark_exceptions)
    monkeypatch.setitem(sys.modules, "syncode", fake_syncode)
    monkeypatch.setitem(sys.modules, "syncode.parsers", fake_parsers)
    monkeypatch.setitem(sys.modules, "syncode.parsers.incremental_parser", fake_incremental_parser)


def test_complete_prefix_uses_incremental_end_state_before_full_parse(monkeypatch):
    _install_fake_syncode(monkeypatch)
    _FakeIncrementalParser.instances.clear()
    complete_lark = _FakeCompleteLark()

    monkeypatch.setattr(
        parser_utils,
        "_get_parser_components",
        lambda grammar_text, start: (object(), _FakeBaseParser(), object(), complete_lark),
    )
    monkeypatch.setattr(parser_utils, "_get_cached_dfa_mask_store", lambda *args: None)

    parser_cls = parser_utils.create_lark_dafny_parser(
        "start: \"C\"",
        _FakeVerifiedDecoderAgent,
        _FakeDafny,
    )

    class CountingParser(parser_cls):
        def __init__(self, lm_tokens):
            super().__init__(lm_tokens)
            self.tokens_to_text_calls = 0

        def _tokens_to_text(self, tokens):
            self.tokens_to_text_calls += 1
            return super()._tokens_to_text(tokens)

    parser = CountingParser(["C"])
    prefix = ["C"]

    assert parser.IsCompletePrefix(prefix) is True
    assert parser.IsCompletePrefix(prefix) is True
    assert _FakeIncrementalParser.instances[0].calls == ["C"]
    assert complete_lark.parse_calls == 0
    assert parser.tokens_to_text_calls == 1
