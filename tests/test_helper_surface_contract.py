from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _prompt_helper_refs() -> set[str]:
    prompt_source = (REPO_ROOT / "synthesis/generate/prompts.py").read_text()
    return set(
        re.findall(
            r"\b(?:helpers|CSDHelpers)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            prompt_source,
        )
    )


def _all_helper_names() -> set[str]:
    from synthesis.generate import prompts

    helper_names = getattr(prompts, "_ALL_HELPER_NAMES")
    assert isinstance(helper_names, set)
    return set(helper_names)


def _dafny_helper_defs() -> set[str]:
    dafny_source = (
        REPO_ROOT / "synthesis/verify/library/VerifiedAgentSynthesis.dfy"
    ).read_text()
    return set(
        re.findall(
            r"(?:^|\n)\s*(?:static\s+)?(?:method|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            dafny_source,
        )
    )


def _feedback_helper_classifications() -> set[str]:
    feedback_source = (REPO_ROOT / "synthesis/evaluate/feedback_loop.py").read_text()
    module = ast.parse(feedback_source)
    classified: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or node.name != "SynthesisPipeline":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in {"NON_PRUNABLE_HELPERS", "PRUNABLE_HELPERS"}
                ):
                    classified.update(ast.literal_eval(stmt.value))
    return classified


def _core_lm_helpers() -> set[str]:
    return {
        "Contains",
        "RenderPrefix",
        "GenerateLogits",
        "ChooseNextToken",
        "ChooseNextTokenUnconstrained",
        "GenerateUnconstrainedChunk",
        "MaskValidNextAndEos",
        "BoostValidNextAndEos",
        "IdToToken",
        "TokenToId",
        "TokenToIdRecursive",
        "IdToLogit",
        "TokenToLogit",
        "TokensToLogits",
        "IdsToLogits",
        "MaskToken",
        "MaskTokens",
        "MaskTokensExcept",
        "IsMasked",
        "HasUnmaskedToken",
        "IsValidPrefix",
        "IsCompletePrefix",
        "IsDeadPrefix",
        "ValidNextTokenCountUpTo",
        "ValidNextToken",
        "ValidNextTokens",
        "ParseG",
    }


def test_prompt_exposed_helpers_exist_in_dafny_library():
    missing = _prompt_helper_refs() - _dafny_helper_defs()
    assert not missing


def test_prompt_exposed_helpers_are_classified_for_feedback_policy():
    missing = _prompt_helper_refs() - _feedback_helper_classifications()
    assert not missing


def test_core_lm_helpers_are_classified_for_feedback_policy():
    missing = _core_lm_helpers() - _feedback_helper_classifications()
    assert not missing


def test_all_prompt_universe_helpers_have_docs_and_feedback_classification():
    """Name-only helpers are too weak for cold discovery.

    Every helper in the generated helper universe must have prompt-visible docs
    and a feedback-loop classification. Otherwise the allowed-helper block can
    name a helper without giving the model the call shape or letting the helper
    policy reason about it.
    """
    helper_names = _all_helper_names()

    missing_docs = helper_names - _prompt_helper_refs()
    assert not missing_docs

    missing_classification = helper_names - _feedback_helper_classifications()
    assert not missing_classification


def test_append_task_guidance_preserves_the_earlier_task_contract():
    prompt_source = (REPO_ROOT / "synthesis/generate/prompts.py").read_text()
    start = prompt_source.index("### Task prompt guidance")
    end = prompt_source.index("### Outside-span generation", start)
    contract = " ".join(prompt_source[start:end].split())

    assert "must not contradict, weaken, or replace earlier task instructions" in contract
    assert "examples, schema, or output-format requirements" in contract
