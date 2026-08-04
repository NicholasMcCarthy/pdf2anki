"""Tests for HighlightPriorityStrategy: prompt grounding, media attachment,
and should_apply_to_chunk gating."""

import json
from unittest.mock import Mock

from pdf2anki.chunking import TextChunk
from pdf2anki.prompts import create_prompt_manager
from pdf2anki.strategies.highlight_priority import HighlightPriorityStrategy


def _make_highlight_chunk() -> TextChunk:
    highlights = [{
        "page_num": 1,
        "type": "highlight",
        "text": "Plants convert light energy into chemical energy via photosynthesis.",
        "rect": (0, 0, 10, 10),
        "color": [1.0, 1.0, 0.0],
        "author": "nick",
        "content": "key mechanism",
        "created": "",
        "screenshot": "highlight_p1_0_abc123.png",
    }]
    chunk = TextChunk(
        text="[HIGHLIGHT - nick]: Plants convert light energy into chemical energy via "
             "photosynthesis.\n[NOTE]: key mechanism\n\n[PAGE CONTEXT]:\nIntroduction",
        start_page=1,
        end_page=1,
        chunk_type="highlight",
        highlights=highlights,
    )
    return chunk


class _FakeStrategyConfig:
    enabled = True
    params = {}
    template_version = "1.0"


def _make_strategy() -> HighlightPriorityStrategy:
    llm_provider = Mock()
    llm_provider.config.model = "gpt-4"
    return HighlightPriorityStrategy(
        llm_provider=llm_provider,
        prompt_manager=create_prompt_manager(),
        strategy_config=_FakeStrategyConfig(),
        strategy_name="highlight_priority",
    )


def test_should_apply_to_chunk_requires_highlights():
    strategy = _make_strategy()
    highlight_chunk = _make_highlight_chunk()
    plain_chunk = TextChunk(text="no highlights here", start_page=1, end_page=1)

    assert strategy.should_apply_to_chunk(highlight_chunk, {}) is True
    assert strategy.should_apply_to_chunk(plain_chunk, {}) is False


def test_prompt_includes_highlight_markers():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    strategy.llm_provider.generate.return_value = Mock(
        content=json.dumps({"cards": []})
    )
    strategy.generate_cards(chunk, pdf_metadata={"title": "Test", "path": "test.pdf"}, max_cards=5)

    prompt_arg = strategy.llm_provider.generate.call_args.kwargs["prompt"]
    assert "[HIGHLIGHT" in prompt_arg
    assert "key mechanism" in prompt_arg


def test_parse_cards_attaches_highlight_screenshots():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    response_data = {
        "cards": [
            {"front": "Q1", "back": "A1", "page_citation": "p. 1", "core_concept": "X"},
        ]
    }
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata={"title": "Test"})

    assert len(cards) == 1
    assert cards[0].media == ["highlight_p1_0_abc123.png"]
    assert "highlight-priority" in cards[0].tags


def test_validate_response_rejects_missing_fields():
    strategy = _make_strategy()
    assert strategy.validate_response({"cards": [{"front": "Q1"}]}) is False
    assert strategy.validate_response({"cards": [{"front": "Q1", "back": "A1"}]}) is True
