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


def test_prompt_includes_abstract_when_present():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    strategy.llm_provider.generate.return_value = Mock(content=json.dumps({"cards": []}))
    strategy.generate_cards(
        chunk,
        pdf_metadata={"title": "Test", "path": "test.pdf", "abstract": "This paper studies photosynthesis."},
        max_cards=5,
    )

    prompt_arg = strategy.llm_provider.generate.call_args.kwargs["prompt"]
    assert "This paper studies photosynthesis." in prompt_arg
    assert "Paper Abstract" in prompt_arg


def test_prompt_omits_abstract_section_when_absent():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    strategy.llm_provider.generate.return_value = Mock(content=json.dumps({"cards": []}))
    strategy.generate_cards(chunk, pdf_metadata={"title": "Test", "path": "test.pdf"}, max_cards=5)

    prompt_arg = strategy.llm_provider.generate.call_args.kwargs["prompt"]
    assert "Paper Abstract" not in prompt_arg


def test_validate_response_accepts_cloze_card_with_card_type():
    strategy = _make_strategy()
    response = {"cards": [{"card_type": "cloze", "cloze_text": "The {{c1::mitochondria}} is the powerhouse."}]}
    assert strategy.validate_response(response) is True


def test_validate_response_rejects_malformed_cloze():
    strategy = _make_strategy()
    response = {"cards": [{"card_type": "cloze", "cloze_text": "no cloze markers here"}]}
    assert strategy.validate_response(response) is False


def test_parse_cards_handles_mixed_basic_and_cloze():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    response_data = {
        "cards": [
            {"card_type": "basic", "front": "Q1", "back": "A1", "core_concept": "X"},
            {"card_type": "cloze", "cloze_text": "Plants use {{c1::chlorophyll}} to capture light.",
             "extra": "context", "core_concept": "Y"},
        ]
    }
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata={"title": "Test"})

    assert len(cards) == 2
    basic, cloze = cards
    assert basic.note_type == "Basic"
    assert basic.front == "Q1" and basic.back == "A1"
    assert cloze.note_type == "Cloze"
    assert cloze.cloze_text == "Plants use {{c1::chlorophyll}} to capture light."
    assert cloze.extra == "context"
    # Both card types still get highlight screenshots attached.
    assert cloze.media == ["highlight_p1_0_abc123.png"]


def test_parse_cards_defaults_to_basic_without_card_type():
    """Backward compatibility: a response with no card_type field (the old
    prompt shape) should still parse as a Basic card."""
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    response_data = {"cards": [{"front": "Q1", "back": "A1", "core_concept": "X"}]}
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata={"title": "Test"})

    assert len(cards) == 1
    assert cards[0].note_type == "Basic"


def test_parse_cards_skips_invalid_cloze_format():
    strategy = _make_strategy()
    chunk = _make_highlight_chunk()

    response_data = {"cards": [{"card_type": "cloze", "cloze_text": "no markers at all"}]}
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata={"title": "Test"})

    assert cards == []
