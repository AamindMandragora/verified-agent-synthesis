from types import SimpleNamespace
import hashlib
import os

from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.evaluate.feedback_loop import FailureStage, SynthesisAttempt, SynthesisPipeline
from synthesis.generate.generator import StrategyGenerator


class _FakeGenerator:
    def __init__(self):
        self.rationales = []

    def summarize_rationale_claim(self, rationale: str) -> str:
        self.rationales.append(rationale)
        if "Recent branch" in rationale:
            return "summary: allowed int() and / while only penalizing braces"
        if "Best branch" in rationale:
            return "summary: dropped braces and kept strict operator guidance"
        return "summary: " + rationale


def _pipeline() -> SynthesisPipeline:
    generator = _FakeGenerator()
    return SynthesisPipeline(
        evaluator=object(),
        generator=generator,
        verifier=object(),
        compiler=object(),
    )


def _attempt(
    number: int,
    strategy: str,
    accuracy: float,
    syntax_rate: float,
    failure_location: str | list[str],
) -> SynthesisAttempt:
    num_examples = 10
    num_correct = int(accuracy * num_examples)
    syntax_valid = int(syntax_rate * num_examples)
    samples = []
    for index in range(num_examples):
        is_correct = index < num_correct
        if is_correct:
            location = "correct"
        elif isinstance(failure_location, list):
            location = failure_location[(index - num_correct) % len(failure_location)]
        else:
            location = failure_location
        samples.append(
            {
                "is_correct": is_correct,
                "is_syntax_valid": index < syntax_valid,
                "failure_location": location,
                "time_seconds": 1.0,
            }
        )
    return SynthesisAttempt(
        attempt_number=number,
        strategy_code=strategy,
        full_dafny_code=strategy,
        timestamp="2026-05-23T00:00:00",
        failed_at=FailureStage.EVALUATION,
        eval_result=EvaluationResult(
            success=False,
            accuracy=accuracy,
            contains_delimiters=True,
            syntax_rate=syntax_rate,
            num_examples=num_examples,
            num_correct=num_correct,
            total_time_seconds=10.0,
            max_sample_time_seconds=1.0,
            sample_outputs=samples,
        ),
    )


def test_attempt_outcome_ledger_summarizes_best_and_recent_branches():
    pipeline = _pipeline()
    best = _attempt(
        2,
        """// CSD_RATIONALE_BEGIN
// Best branch: drop braces and keep strict operator guidance.
// CSD_RATIONALE_END
generated := generatedPrefix;""",
        0.40,
        0.70,
        "no_valid_visible_span",
    )
    recent = _attempt(
        3,
        """// CSD_RATIONALE_BEGIN
// Recent branch: allow int() and / while only penalizing braces.
// CSD_RATIONALE_END
generated := prompt;""",
        0.20,
        0.70,
        ["syntax_valid_semantic_mismatch", "no_valid_visible_span"],
    )

    ledger = pipeline._build_attempt_outcome_ledger([best, recent], best_attempt_number=2)

    assert "Use this as empirical search context" in ledger
    assert "Best result:" in ledger
    assert "Attempt 2: accuracy 40.0%" in ledger
    assert "Recent evaluated branches:" in ledger
    assert "Attempt 3: accuracy 20.0%" in ledger
    assert "rationale claim: summary: allowed int() and / while only penalizing braces" in ledger
    assert "measured effect vs best: accuracy -20.0pp, syntax +0.0pp" in ledger
    assert "failure locations: no_valid_visible_span=4, syntax_valid_semantic_mismatch=4" in ledger
    assert isinstance(pipeline.generator, _FakeGenerator)
    assert any("Recent branch" in rationale for rationale in pipeline.generator.rationales)


def test_rationale_summary_disabled_returns_full_rationale(monkeypatch):
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "off")
    generator = StrategyGenerator()
    rationale = (
        "Attempt changed the branch by allowing int() and / because prior "
        "feedback indicated gold answers used those constructs. It narrowed "
        "penalties to braces and span reopening while preserving the rest of "
        "the best-scoring runtime structure."
    )

    assert generator.summarize_rationale_claim(rationale) == rationale


class _FakeSummaryCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="short causal summary"))]
        )


class _FakeSummaryChat:
    def __init__(self):
        self.completions = _FakeSummaryCompletions()


class _FakeSummaryClient:
    def __init__(self):
        self.chat = _FakeSummaryChat()


def test_rationale_summary_openai_request_omits_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    generator = StrategyGenerator()
    generator._summary_client = _FakeSummaryClient()

    summary = generator._summarize_rationale_claim_openai(
        "Attempt changed the branch by keeping chunked outside generation."
    )

    assert summary == "short causal summary"
    kwargs = generator._summary_client.chat.completions.kwargs
    assert kwargs["model"] == "chat-latest"
    assert "temperature" not in kwargs
    assert "reasoning_effort" not in kwargs


class _FailingSummaryCompletions:
    def create(self, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: Gemini credits depleted")


class _FailingSummaryChat:
    def __init__(self):
        self.completions = _FailingSummaryCompletions()


class _FailingSummaryClient:
    def __init__(self):
        self.chat = _FailingSummaryChat()


class _FakeAnthropicMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="haiku fallback summary")]
        )


class _FakeAnthropicSummaryClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


class _FakeBedrockClient:
    def __init__(self):
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        return {
            "output": {
                "message": {
                    "content": [{"text": "bedrock strategy"}],
                },
            },
        }


def test_rationale_summary_falls_back_to_anthropic_haiku_when_primary_fails():
    old_env = {key: os.environ.get(key) for key in (
        "CSD_RATIONALE_SUMMARY_BACKEND",
        "CSD_RATIONALE_SUMMARY_MODEL",
        "CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND",
        "CSD_RATIONALE_SUMMARY_FALLBACK_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    )}
    try:
        os.environ["CSD_RATIONALE_SUMMARY_BACKEND"] = "openai"
        os.environ["CSD_RATIONALE_SUMMARY_MODEL"] = "gemini-3.5-flash"
        os.environ.pop("CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND", None)
        os.environ.pop("CSD_RATIONALE_SUMMARY_FALLBACK_MODEL", None)
        os.environ["OPENAI_API_KEY"] = "openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"

        generator = StrategyGenerator()
        generator._summary_client = _FailingSummaryClient()
        generator._summary_anthropic_client = _FakeAnthropicSummaryClient()

        summary = generator.summarize_rationale_claim(
            "Attempt changed the branch by keeping chunked outside generation."
        )

        assert summary == "haiku fallback summary"
        kwargs = generator._summary_anthropic_client.messages.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == 96
        assert "thinking" not in kwargs
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_rationale_summary_defaults_to_anthropic_haiku_without_openai():
    old_env = {key: os.environ.get(key) for key in (
        "CSD_RATIONALE_SUMMARY_BACKEND",
        "CSD_RATIONALE_SUMMARY_MODEL",
        "CSD_RATIONALE_SUMMARY_FALLBACK_BACKEND",
        "CSD_RATIONALE_SUMMARY_FALLBACK_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    )}
    try:
        for key in old_env:
            os.environ.pop(key, None)
        os.environ["ANTHROPIC_API_KEY"] = "anthropic-key"

        generator = StrategyGenerator()
        generator._summary_anthropic_client = _FakeAnthropicSummaryClient()

        summary = generator.summarize_rationale_claim(
            "Attempt changed the branch by keeping chunked outside generation."
        )

        assert summary == "haiku fallback summary"
        kwargs = generator._summary_anthropic_client.messages.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert "OPENAI_API_KEY" not in os.environ
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_bedrock_generation_uses_aws_converse_without_anthropic_client(monkeypatch):
    monkeypatch.delenv("BEDROCK_BASE_URL", raising=False)
    generator = StrategyGenerator(
        model_name="us.anthropic.claude-sonnet-4-6",
        backend="bedrock",
        api_key="bedrock-key",
        max_new_tokens=8192,
    )
    generator._client = _FakeBedrockClient()

    output = generator._generate_bedrock("system prompt", "user prompt")

    assert output == "bedrock strategy"
    assert generator.api_base_url is None
    kwargs = generator._client.kwargs
    assert kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert kwargs["system"] == [{"text": "system prompt"}]
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"text": "user prompt"}]},
    ]
    assert kwargs["inferenceConfig"] == {"maxTokens": 8192}
    assert "model" not in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_gemini_generation_uses_google_ai_studio_generate_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    generator = StrategyGenerator(
        model_name="gemini-3.1-flash-lite",
        backend="gemini",
        max_new_tokens=2048,
    )
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "// CSD_RATIONALE_BEGIN\n// ok\n// CSD_RATIONALE_END\n"}
                        ]
                    }
                }
            ]
        }

    generator._post_json = fake_post_json

    output = generator._generate_gemini("system prompt", "user prompt")

    assert output.startswith("// CSD_RATIONALE_BEGIN")
    assert "models/gemini-3.1-flash-lite:generateContent" in captured["url"]
    assert "gemini-key" not in captured["url"]
    assert captured["headers"] == {"x-goog-api-key": "gemini-key"}
    assert captured["payload"]["systemInstruction"] == {
        "parts": [{"text": "system prompt"}]
    }
    assert captured["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "user prompt"}]}
    ]
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 2048


def test_gemini_generation_rotates_to_backup_key_on_quota_exhaustion(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP_1", "backup-1-key")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP_2", "backup-2-key")
    generator = StrategyGenerator(
        model_name="gemini-3.1-flash-lite",
        backend="gemini",
        max_new_tokens=2048,
    )
    calls = []

    def fake_post_json(url, headers, payload, max_retries=None, retryable_statuses=None):
        calls.append((url, headers, max_retries))
        if headers.get("x-goog-api-key") == "primary-key":
            error = RuntimeError("HTTP 429: RESOURCE_EXHAUSTED: Gemini credits depleted")
            error.status_code = 429
            error.response_body = "RESOURCE_EXHAUSTED: Gemini credits depleted"
            raise error
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "// CSD_RATIONALE_BEGIN\n// backup key ok\n// CSD_RATIONALE_END\n"}
                        ]
                    }
                }
            ]
        }

    generator._post_json = fake_post_json

    output = generator._generate_gemini("system prompt", "user prompt")

    assert output.startswith("// CSD_RATIONALE_BEGIN")
    assert any(headers.get("x-goog-api-key") == "primary-key" for _, headers, _ in calls)
    assert any(headers.get("x-goog-api-key") == "backup-1-key" for _, headers, _ in calls)
    assert all(max_retries == 0 for _, _, max_retries in calls)
    assert all("key=" not in url for url, _, _ in calls)
    assert generator.author_route_identity() == {
        "auth_mode": "gemini_api_key",
        "api_key_sha256": hashlib.sha256(b"backup-1-key").hexdigest(),
    }


def test_gemini37_generation_omits_deprecated_sampling_parameters(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    generator = StrategyGenerator(
        model_name="gemini-3.7-flash",
        backend="gemini",
        max_new_tokens=32768,
    )
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["payload"] = payload
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    generator._post_json = fake_post_json
    assert generator._generate_gemini("system", "user") == "ok"
    assert captured["payload"]["generationConfig"] == {
        "maxOutputTokens": 32768,
    }
    assert generator.author_route_identity() == {
        "auth_mode": "gemini_api_key",
        "api_key_sha256": hashlib.sha256(b"gemini-key").hexdigest(),
    }


def test_rationale_summary_uses_gemini_flash_lite_backend(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "gemini")
    monkeypatch.delenv("CSD_RATIONALE_SUMMARY_MODEL", raising=False)
    generator = StrategyGenerator()
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "candidates": [
                {"content": {"parts": [{"text": "short gemini summary"}]}}
            ]
        }

    generator._post_json = fake_post_json

    summary = generator.summarize_rationale_claim(
        "Attempt changed the branch by keeping chunked outside generation."
    )

    assert summary == "short gemini summary"
    assert "models/gemini-2.5-flash-lite:generateContent" in captured["url"]
    assert "gemini-key" not in captured["url"]
    assert captured["headers"] == {"x-goog-api-key": "gemini-key"}
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 96


def test_rationale_summary_rotates_to_backup_key_on_quota_exhaustion(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP_1", "backup-1-key")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP_2", "backup-2-key")
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "gemini")
    monkeypatch.delenv("CSD_RATIONALE_SUMMARY_MODEL", raising=False)
    generator = StrategyGenerator()
    calls = []

    def fake_post_json(url, headers, payload, max_retries=None, retryable_statuses=None):
        calls.append((url, headers, max_retries))
        if headers.get("x-goog-api-key") == "primary-key":
            error = RuntimeError("HTTP 429: RESOURCE_EXHAUSTED: Gemini credits depleted")
            error.status_code = 429
            error.response_body = "RESOURCE_EXHAUSTED: Gemini credits depleted"
            raise error
        return {
            "candidates": [
                {"content": {"parts": [{"text": "short backup summary"}]}}
            ]
        }

    generator._post_json = fake_post_json

    summary = generator.summarize_rationale_claim(
        "Attempt changed the branch by keeping chunked outside generation."
    )

    assert summary == "short backup summary"
    assert any(headers.get("x-goog-api-key") == "primary-key" for _, headers, _ in calls)
    assert any(headers.get("x-goog-api-key") == "backup-1-key" for _, headers, _ in calls)
    assert all(max_retries == 0 for _, _, max_retries in calls)
    assert all("key=" not in url for url, _, _ in calls)


def test_vertex_generation_uses_aiplatform_generate_content(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_PROJECT", "paper-project")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    monkeypatch.setenv("VERTEX_AI_ACCESS_TOKEN", "vertex-token")
    generator = StrategyGenerator(
        model_name="gemini-3-pro-preview",
        backend="vertex",
        max_new_tokens=4096,
    )
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "// CSD_RATIONALE_BEGIN\n// vertex ok\n// CSD_RATIONALE_END\n"}
                        ]
                    }
                }
            ]
        }

    generator._post_json = fake_post_json

    output = generator._generate_vertex("system prompt", "user prompt")

    assert output.startswith("// CSD_RATIONALE_BEGIN")
    assert captured["url"] == (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/paper-project/locations/us-central1/publishers/google/"
        "models/gemini-3-pro-preview:generateContent"
    )
    assert captured["headers"] == {"Authorization": "Bearer vertex-token"}
    assert captured["payload"]["systemInstruction"] == {
        "parts": [{"text": "system prompt"}]
    }
    assert captured["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "user prompt"}]}
    ]
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 4096


def test_vertex_generation_rotates_to_backup_gemini_key_on_quota_exhaustion(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_PROJECT", "paper-project")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "us-central1")
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP_1", "backup-1-key")
    generator = StrategyGenerator(
        model_name="gemini-3-pro-preview",
        backend="vertex",
        max_new_tokens=4096,
    )
    captured = []

    def fake_post_json(url, headers, payload, max_retries=None, retryable_statuses=None):
        captured.append((url, headers, max_retries))
        if headers.get("x-goog-api-key") == "primary-key":
            error = RuntimeError("HTTP 429: RESOURCE_EXHAUSTED: Gemini credits depleted")
            error.status_code = 429
            error.response_body = "RESOURCE_EXHAUSTED: Gemini credits depleted"
            raise error
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "// CSD_RATIONALE_BEGIN\n// vertex backup ok\n// CSD_RATIONALE_END\n"}
                        ]
                    }
                }
            ]
        }

    generator._post_json = fake_post_json

    output = generator._generate_vertex("system prompt", "user prompt")

    assert output.startswith("// CSD_RATIONALE_BEGIN")
    assert any(headers.get("x-goog-api-key") == "primary-key" for _, headers, _ in captured)
    assert any(headers.get("x-goog-api-key") == "backup-1-key" for _, headers, _ in captured)
    assert all(max_retries == 0 for _, _, max_retries in captured if max_retries is not None)


def test_rationale_summary_uses_vertex_flash_lite_backend(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_PROJECT", "paper-project")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "global")
    monkeypatch.setenv("VERTEX_AI_ACCESS_TOKEN", "vertex-token")
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "vertex")
    monkeypatch.delenv("CSD_RATIONALE_SUMMARY_MODEL", raising=False)
    generator = StrategyGenerator()
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "candidates": [
                {"content": {"parts": [{"text": "short vertex summary"}]}}
            ]
        }

    generator._post_json = fake_post_json

    summary = generator.summarize_rationale_claim(
        "Attempt changed the branch by keeping chunked outside generation."
    )

    assert summary == "short vertex summary"
    assert captured["url"] == (
        "https://aiplatform.googleapis.com/v1/"
        "projects/paper-project/locations/global/publishers/google/"
        "models/gemini-2.5-flash-lite:generateContent"
    )
    assert captured["headers"] == {"Authorization": "Bearer vertex-token"}
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 96


def test_vertex_summary_prefers_google_api_key_over_access_token(monkeypatch):
    monkeypatch.setenv("VERTEX_AI_PROJECT", "paper-project")
    monkeypatch.setenv("VERTEX_AI_LOCATION", "global")
    monkeypatch.setenv("VERTEX_AI_ACCESS_TOKEN", "unsupported-token")
    monkeypatch.setenv("GEMINI_API_KEY", "working-api-key")
    monkeypatch.setenv("CSD_RATIONALE_SUMMARY_BACKEND", "vertex")
    generator = StrategyGenerator()
    captured = {}

    def fake_post_json(url, headers, payload):
        captured["headers"] = headers
        return {
            "candidates": [
                {"content": {"parts": [{"text": "api key vertex summary"}]}}
            ]
        }

    generator._post_json = fake_post_json

    summary = generator.summarize_rationale_claim(
        "Attempt changed the branch by keeping chunked outside generation."
    )

    assert summary == "api key vertex summary"
    assert captured["headers"] == {"x-goog-api-key": "working-api-key"}
