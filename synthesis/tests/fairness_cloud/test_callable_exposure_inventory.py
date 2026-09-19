"""Every verified callable must be public or deliberately classified as internal."""

import importlib.util
import pathlib
import re


_REPO = pathlib.Path(__file__).resolve().parents[3]
_LIBRARY_PATH = _REPO / "synthesis" / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
_PROMPTS_PATH = _REPO / "synthesis" / "generate" / "prompts.py"
_CALLABLE_RE = re.compile(
    r"^\s*(?:static\s+)?(?:method|function|predicate)\s+"
    r"(?:\{:[^}]+\}\s*)*([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


PUBLIC_LM = {
    "IdToToken", "TokenToId", "IdToLogit", "TokenToLogit", "TokensToLogits",
    "IdsToLogits", "MaskToken", "MaskTokens", "MaskTokensExcept", "IsMasked",
    "HasUnmaskedToken", "GenerateLogits", "ChooseNextToken",
    "ChooseNextTokenUnconstrained", "GenerateUnconstrainedChunk",
    "MaskValidNextAndEos", "BoostValidNextAndEos",
}
INTERNAL_LM = {
    "ValidTokensIdsLogits",  # proof invariant, not an operation
    "TokenToIdRecursive",  # implementation behind TokenToId
    "AppendTaskGuidance",  # raw host method, wrapped by helpers.AppendTaskGuidance
    "PenalizeTriedTokenAt",  # host-state mutation with a weak public contract
    "SpanGrounded",  # raw grounding check used by the bounded repair helper
    "SpanAppearsInPrompt",  # raw string check, wrapped with rendered-prefix input
    "SpanResemblanceToPromptExamples",  # raw string check, wrapped with rendered-prefix input
    "FirstUngroundedIdentifierTokenIdx",  # internal signal used by bounded grounding repair
}
PUBLIC_PARSER = {
    "IsValidPrefix", "IsCompletePrefix", "ValidNextTokenCountUpTo", "IsDeadPrefix",
    "ValidNextToken", "ValidNextTokens", "ParseG",
}
INTERNAL_PARSER = {
    "CompletedSchemaSymbolCount",  # internal boundary signal for grounding repair
}
PUBLIC_RENDERED_TEXT = {"Contains", "RenderPrefix", "RenderedEndsWith"}
INTERNAL_HELPERS = {
    "RollbackToValidPrefix",  # implementation behind safe rollback helpers
    "RollbackConstrainedSpan",  # removed because it requires a caller-managed stable prefix
    "RollbackToCompletePrefix",  # implementation behind complete-prefix rollback helpers
    "RollbackAndRegenerate",  # duplicate repair path with no accepted-run use
    "FindSubstring",  # implementation behind delimiter extraction
    "RolloutConstrainedWithPenalties",  # redundant with safer bounded step helpers
    "RegenerateUnitOnCheckFailure",  # caller-provided answer-key-like units are unsafe
}


def _load_prompts():
    spec = importlib.util.spec_from_file_location("_prompts_callable_inventory", _PROMPTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_block(library: str, name: str, next_name: str | None) -> str:
    start = library.index(f"class {name} {{")
    end = library.index(f"class {next_name} {{", start) if next_name else library.rindex("\n  }")
    return library[start:end]


def test_every_lm_and_parser_callable_has_an_explicit_exposure_decision():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")
    lm = set(_CALLABLE_RE.findall(_class_block(library, "LM", "Parser")))
    parser = set(_CALLABLE_RE.findall(_class_block(library, "Parser", "CSDHelpers")))
    parser -= PUBLIC_RENDERED_TEXT

    assert lm == PUBLIC_LM | INTERNAL_LM
    assert parser == PUBLIC_PARSER | INTERNAL_PARSER


def test_every_helper_callable_is_public_or_deliberately_internal():
    library = _LIBRARY_PATH.read_text(encoding="utf-8")
    prompts = _load_prompts()
    helpers = set(_CALLABLE_RE.findall(_class_block(library, "CSDHelpers", None)))

    assert helpers == prompts._ALL_HELPER_NAMES | INTERNAL_HELPERS
    assert not (prompts._ALL_HELPER_NAMES & INTERNAL_HELPERS)


def test_global_rendered_text_predicates_are_all_documented():
    reference = _load_prompts()._build_tool_reference_block(None)
    for name in PUBLIC_RENDERED_TEXT:
        assert f"{name}(" in reference


def test_every_public_lm_and_parser_callable_is_named_in_its_menu_section():
    reference = _load_prompts()._build_tool_reference_block(None)
    start = reference.index("## LM and `Parser` surface")
    end = reference.index("### Verified rendered-text functions", start)
    block = reference[start:end]

    for name in PUBLIC_LM | PUBLIC_PARSER:
        assert name in block, f"{name} is public but missing from the LM/Parser menu section"
