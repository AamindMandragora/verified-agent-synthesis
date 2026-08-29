import io
import json
import urllib.error

import pytest

from synthesis.evaluate.evaluator import _print_realtime_completion
from synthesis.generate.generator import StrategyGenerator
from synthesis.safe_logging import display_text


def test_campaign_safe_logging_never_writes_raw_prompt_response_or_completion(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("CSD_REDACT_SENSITIVE_LOGS", "1")
    monkeypatch.setenv("CSD_PROMPT_LOG_DIR", str(tmp_path))
    generator = object.__new__(StrategyGenerator)
    generator._prompt_log_counter = 0
    generator.backend = "claude"
    generator.model_name = "claude-opus-5"
    private_system = "PRIVATE SYSTEM SCHEMA"
    private_user = "PRIVATE USER SQL"
    private_output = "PRIVATE PROVIDER RESPONSE"

    generator._log_prompt_io(private_system, private_user, private_output)
    _print_realtime_completion(1, 1, private_output)
    print(display_text("Strategy", "PRIVATE STRATEGY"))

    record_text = (tmp_path / "prompt_io.jsonl").read_text(encoding="utf-8")
    record = json.loads(record_text)
    captured = capsys.readouterr().out
    for private in (
        private_system,
        private_user,
        private_output,
        "PRIVATE STRATEGY",
    ):
        assert private not in record_text
        assert private not in captured
    assert record["redacted"] is True
    assert set(record["system_prompt_metadata"]) == {"chars", "bytes", "sha256"}
    assert "redacted" in captured


def test_default_debug_logging_contract_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("CSD_REDACT_SENSITIVE_LOGS", raising=False)
    monkeypatch.setenv("CSD_PROMPT_LOG_DIR", str(tmp_path))
    generator = object.__new__(StrategyGenerator)
    generator._prompt_log_counter = 0
    generator.backend = "vllm"
    generator.model_name = "local-model"

    generator._log_prompt_io("system", "user", "output")

    record = json.loads((tmp_path / "prompt_io.jsonl").read_text(encoding="utf-8"))
    assert record["system_prompt"] == "system"
    assert record["user_prompt"] == "user"
    assert record["output"] == "output"


def test_campaign_safe_logging_redacts_vertex_http_error_body(monkeypatch, capsys):
    monkeypatch.setenv("CSD_REDACT_SENSITIVE_LOGS", "1")
    private_body = b"PRIVATE PROVIDER ERROR BODY WITH SQL"

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://aiplatform.googleapis.com/v1/test",
            400,
            "bad request",
            {},
            io.BytesIO(private_body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    generator = object.__new__(StrategyGenerator)
    generator.backend = "vertex"

    with pytest.raises(RuntimeError) as raised:
        generator._post_json("https://example.invalid", {}, {}, max_retries=1)

    combined = str(raised.value) + capsys.readouterr().out
    assert private_body.decode() not in combined
    assert "redacted" in combined.lower()
