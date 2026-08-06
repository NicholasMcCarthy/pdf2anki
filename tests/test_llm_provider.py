"""Tests for LLMProvider provider selection (openai/anthropic)."""

from unittest.mock import Mock, patch

import pytest

from pdf2anki.config import LLMConfig
from pdf2anki.llm import LLMProvider, ModelRegistry, _extract_text_content, _rejects_temperature


def _make_config(provider: str, model: str = "gpt-4-1106-preview") -> LLMConfig:
    return LLMConfig(provider=provider, model=model, api_key="dummy-key")


def test_openai_provider_initializes_chat_openai(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("pdf2anki.llm.ChatOpenAI") as mock_chat_openai:
        LLMProvider(_make_config("openai"))
        mock_chat_openai.assert_called_once()


def test_anthropic_provider_initializes_chat_anthropic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic:
        LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        mock_chat_anthropic.assert_called_once()
        _, kwargs = mock_chat_anthropic.call_args
        # Anthropic requires an explicit max_tokens even if unset in config.
        assert kwargs["max_tokens"]


def test_anthropic_default_max_tokens_is_generous(tmp_path, monkeypatch):
    """Defaults to a larger budget than a typical non-reasoning model needs -
    models that reject temperature (reasoning-first models) can consume part
    of max_tokens on internal reasoning before emitting the visible answer."""
    monkeypatch.chdir(tmp_path)
    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic:
        LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        _, kwargs = mock_chat_anthropic.call_args
        assert kwargs["max_tokens"] >= 8192


def test_anthropic_explicit_max_tokens_still_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key="dummy", max_tokens=2048)
    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic:
        LLMProvider(config)
        _, kwargs = mock_chat_anthropic.call_args
        assert kwargs["max_tokens"] == 2048


def test_unsupported_provider_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMProvider(_make_config("not-a-real-provider"))


def test_llm_cache_directory_is_created(tmp_path, monkeypatch):
    """LLMProvider.__init__ must create .llm_cache/ before pointing SQLite at it."""
    monkeypatch.chdir(tmp_path)
    with patch("pdf2anki.llm.ChatOpenAI"):
        LLMProvider(_make_config("openai"))
    assert (tmp_path / ".llm_cache").is_dir()


def test_model_registry_has_anthropic_models():
    info = ModelRegistry.get_model_info("claude-sonnet-5")
    assert info.get("provider") == "anthropic"


def test_extract_text_content_passes_through_plain_string():
    assert _extract_text_content("hello") == "hello"


def test_extract_text_content_concatenates_text_blocks():
    content = [
        {"type": "thinking", "thinking": "reasoning about the answer..."},
        {"type": "text", "text": '{"cards": []}'},
    ]
    assert _extract_text_content(content) == '{"cards": []}'


def test_extract_text_content_joins_multiple_text_blocks():
    content = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    assert _extract_text_content(content) == "part onepart two"


def test_extract_text_content_ignores_non_text_blocks_entirely():
    content = [{"type": "redacted_thinking", "data": "..."}, {"type": "tool_use", "input": {}}]
    assert _extract_text_content(content) == ""


def test_extract_text_content_warns_when_result_is_empty_but_content_wasnt(caplog):
    """Reproduces a real failure: a successful (200 OK) call whose content was
    entirely thinking/reasoning blocks with no text block - previously this
    silently produced an empty string and the caller only ever saw a
    downstream 'Expecting value: line 1 column 1' JSON error with no clue why."""
    content = [{"type": "thinking", "thinking": "reasoning with no final answer emitted..."}]
    with caplog.at_level("WARNING", logger="pdf2anki.llm"):
        result = _extract_text_content(content)

    assert result == ""
    assert any("no text content blocks" in record.message for record in caplog.records)
    assert any("thinking" in record.message for record in caplog.records)


def test_extract_text_content_does_not_warn_for_empty_list():
    # Genuinely empty content (not "content with no text blocks") shouldn't
    # trigger the diagnostic warning - there's nothing to explain.
    assert _extract_text_content([]) == ""


def test_generate_handles_list_content_from_structured_response(tmp_path, monkeypatch):
    """Reproduces a real failure: some models return AIMessage.content as a
    list of content blocks (e.g. a thinking block plus a text block) instead
    of a plain string, which previously broke json.loads() with 'the JSON
    object must be str, bytes or bytearray, not list'."""
    monkeypatch.chdir(tmp_path)

    list_content_response = Mock(content=[
        {"type": "thinking", "thinking": "let me work through this..."},
        {"type": "text", "text": '{"cards": [{"front": "Q", "back": "A"}]}'},
    ])

    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic_cls:
        client = Mock()
        client.invoke.return_value = list_content_response
        mock_chat_anthropic_cls.return_value = client

        provider = LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        response = provider.generate(prompt="test", json_mode=True, max_retries=0)

    assert response.content == '{"cards": [{"front": "Q", "back": "A"}]}'


def test_rejects_temperature_detects_anthropic_error_message():
    error = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': '`temperature` is deprecated for this model.'}}"
    )
    assert _rejects_temperature(error) is True


def test_rejects_temperature_ignores_unrelated_errors():
    assert _rejects_temperature(Exception("rate limit exceeded")) is False
    assert _rejects_temperature(Exception("temperature must be between 0 and 1")) is False


def test_first_call_includes_temperature_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic:
        LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        _, kwargs = mock_chat_anthropic.call_args
        assert "temperature" in kwargs


def test_generate_recovers_by_omitting_temperature_and_retries_immediately(tmp_path, monkeypatch):
    """Simulates the real failure mode: the model rejects `temperature` on the
    first call. The provider should rebuild its client without it and retry
    immediately (no sleep - this isn't a transient error)."""
    monkeypatch.chdir(tmp_path)

    temp_rejected_error = Exception(
        "Error code: 400 - {'error': {'message': '`temperature` is deprecated for this model.'}}"
    )
    success_response = Mock(content='{"cards": []}')

    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic_cls, \
         patch("pdf2anki.llm.time.sleep") as mock_sleep:
        first_client = Mock()
        first_client.invoke.side_effect = temp_rejected_error
        second_client = Mock()
        second_client.invoke.return_value = success_response
        mock_chat_anthropic_cls.side_effect = [first_client, second_client]

        provider = LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        response = provider.generate(prompt="test prompt", max_retries=3)

    assert response.content == '{"cards": []}'
    assert provider._omit_temperature is True
    # Rebuilt once (initial + after the rejection), and the second client had
    # temperature omitted from its construction kwargs.
    assert mock_chat_anthropic_cls.call_count == 2
    _, second_call_kwargs = mock_chat_anthropic_cls.call_args
    assert "temperature" not in second_call_kwargs
    # No backoff sleep - the recovery path retries immediately.
    mock_sleep.assert_not_called()


def test_generate_does_not_loop_forever_if_temperature_rejection_persists(tmp_path, monkeypatch):
    """Defensive: if the same 'temperature rejected' error somehow recurs after
    already rebuilding without it, the provider must not retry the recovery
    path a second time (which would loop forever) - it should fall through to
    the normal bounded retry/backoff and eventually raise."""
    monkeypatch.chdir(tmp_path)

    temp_rejected_error = Exception(
        "Error code: 400 - {'error': {'message': '`temperature` is deprecated for this model.'}}"
    )

    with patch("pdf2anki.llm.ChatAnthropic") as mock_chat_anthropic_cls, \
         patch("pdf2anki.llm.time.sleep"):
        always_failing_client = Mock()
        always_failing_client.invoke.side_effect = temp_rejected_error
        mock_chat_anthropic_cls.return_value = always_failing_client

        provider = LLMProvider(_make_config("anthropic", model="claude-sonnet-5"))
        with pytest.raises(Exception):
            provider.generate(prompt="test prompt", max_retries=1)

    # One rebuild for the recovery attempt, but not runaway/infinite rebuilding.
    assert mock_chat_anthropic_cls.call_count == 2
