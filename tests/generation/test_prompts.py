from generation import prompts


def test_build_initial_prompt_excludes_helper_reference_by_default(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "CSD Helper Library — Function Reference" not in system_prompt
    assert "[BEGIN CURATED_HELPER_REFERENCE]" not in system_prompt
    assert "demo task" in user_prompt


def test_build_initial_prompt_includes_curated_helper_reference_when_enabled(monkeypatch):
    monkeypatch.setenv("CSD_HELPER_REFERENCE_MODE", "curated")
    monkeypatch.delenv("CSD_INCLUDE_HELPER_REFERENCE_MD", raising=False)

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "Curated Helper Mini-Reference" in system_prompt
    assert "[BEGIN CURATED_HELPER_REFERENCE]" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" not in system_prompt
    assert "demo task" in user_prompt


def test_build_initial_prompt_includes_full_helper_reference_when_enabled(monkeypatch):
    monkeypatch.delenv("CSD_HELPER_REFERENCE_MODE", raising=False)
    monkeypatch.setenv("CSD_INCLUDE_HELPER_REFERENCE_MD", "1")

    system_prompt, user_prompt = prompts.build_initial_prompt("demo task")

    assert "CSD Helper Library — Function Reference" in system_prompt
    assert "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]" in system_prompt
    assert "demo task" in user_prompt
