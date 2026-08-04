"""Tests for LLMProvider provider selection (openai/anthropic)."""

from unittest.mock import patch

import pytest

from pdf2anki.config import LLMConfig
from pdf2anki.llm import LLMProvider, ModelRegistry


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
