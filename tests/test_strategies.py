"""Tests for generation strategies with mock LLM."""

import json
from unittest.mock import Mock, patch

import pytest

from pdf2anki.chunking import TextChunk
from pdf2anki.config import StrategyConfig
from pdf2anki.llm import LLMResponse
from pdf2anki.prompts import PromptManager
from pdf2anki.strategies import KeyPointsStrategy, ClozeDefinitionsStrategy


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    mock_provider = Mock()
    mock_provider.config.model = "gpt-4"
    return mock_provider


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    mock_manager = Mock()
    return mock_manager


@pytest.fixture
def sample_chunk():
    """Create a sample text chunk for testing."""
    return TextChunk(
        text="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Neural networks are a type of machine learning model inspired by the human brain.",
        start_page=1,
        end_page=1,
        section="Introduction to ML",
        chunk_index=0,
        total_chunks=1
    )


@pytest.fixture
def sample_pdf_metadata():
    """Create sample PDF metadata."""
    return {
        "title": "Introduction to Machine Learning",
        "author": "Dr. Test Author",
        "path": "test.pdf"
    }


def test_key_points_strategy_valid_response(mock_llm_provider, mock_prompt_manager, sample_chunk, sample_pdf_metadata):
    """Test key points strategy with valid LLM response."""
    # Setup strategy
    config = StrategyConfig(enabled=True, template_version="1.0")
    strategy = KeyPointsStrategy(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        strategy_config=config,
        strategy_name="key_points"
    )
    
    # Mock prompt rendering
    mock_prompt_manager.render_template.return_value = "Mock prompt for key points"
    
    # Mock LLM response
    valid_response = {
        "cards": [
            {
                "front": "What is machine learning?",
                "back": "A subset of artificial intelligence that focuses on algorithms that can learn from data",
                "page_citation": "p. 1",
                "core_concept": "Machine Learning Definition",
                "difficulty": "medium",
                "tags": ["ml", "definition"]
            },
            {
                "front": "What are neural networks?",
                "back": "A type of machine learning model inspired by the human brain",
                "page_citation": "p. 1",
                "core_concept": "Neural Networks",
                "difficulty": "medium",
                "tags": ["neural-networks", "ml"]
            }
        ]
    }
    
    mock_llm_response = LLMResponse(
        content=json.dumps(valid_response),
        model="gpt-4",
        tokens_used=150,
        cost_estimate=0.01,
        cached=False,
        response_time=1.5
    )
    
    mock_llm_provider.generate.return_value = mock_llm_response
    
    # Generate cards
    cards = strategy.generate_cards(sample_chunk, sample_pdf_metadata, max_cards=5)
    
    # Verify results
    assert len(cards) == 2
    
    card1 = cards[0]
    assert card1.note_type == "Basic"
    assert card1.front == "What is machine learning?"
    assert "subset of artificial intelligence" in card1.back
    assert card1.page_start == 1
    assert card1.page_end == 1
    assert card1.strategy == "key_points"
    assert "key-points" in card1.tags
    
    # Verify prompt was called correctly
    mock_prompt_manager.render_template.assert_called_once()
    mock_llm_provider.generate.assert_called_once()


def test_key_points_strategy_invalid_response(mock_llm_provider, mock_prompt_manager, sample_chunk, sample_pdf_metadata):
    """Test key points strategy with invalid LLM response."""
    config = StrategyConfig(enabled=True)
    strategy = KeyPointsStrategy(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        strategy_config=config,
        strategy_name="key_points"
    )
    
    mock_prompt_manager.render_template.return_value = "Mock prompt"
    
    # Mock invalid JSON response
    invalid_response = LLMResponse(
        content="This is not valid JSON",
        model="gpt-4",
        tokens_used=50,
        cost_estimate=0.005,
        cached=False,
        response_time=1.0
    )
    
    mock_llm_provider.generate.return_value = invalid_response
    
    # Should handle gracefully and return empty list
    cards = strategy.generate_cards(sample_chunk, sample_pdf_metadata)
    assert cards == []


def test_cloze_strategy_valid_response(mock_llm_provider, mock_prompt_manager, sample_chunk, sample_pdf_metadata):
    """Test cloze definitions strategy with valid response."""
    config = StrategyConfig(enabled=True)
    strategy = ClozeDefinitionsStrategy(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        strategy_config=config,
        strategy_name="cloze_definitions"
    )
    
    mock_prompt_manager.render_template.return_value = "Mock cloze prompt"
    
    valid_response = {
        "cards": [
            {
                "cloze_text": "{{c1::Machine learning}} is a subset of artificial intelligence that focuses on {{c2::algorithms that can learn from data}}.",
                "extra": "This is a fundamental definition in AI",
                "page_citation": "p. 1",
                "core_concept": "Machine Learning",
                "difficulty": "medium",
                "tags": ["definition", "ml"]
            }
        ]
    }
    
    mock_llm_response = LLMResponse(
        content=json.dumps(valid_response),
        model="gpt-4",
        tokens_used=100,
        cost_estimate=0.008,
        cached=False,
        response_time=1.2
    )
    
    mock_llm_provider.generate.return_value = mock_llm_response
    
    cards = strategy.generate_cards(sample_chunk, sample_pdf_metadata)
    
    assert len(cards) == 1
    card = cards[0]
    assert card.note_type == "Cloze"
    assert "{{c1::Machine learning}}" in card.cloze_text
    assert "{{c2::algorithms that can learn from data}}" in card.cloze_text
    assert card.extra == "This is a fundamental definition in AI"
    assert "cloze-definitions" in card.tags


def test_cloze_strategy_invalid_cloze_format(mock_llm_provider, mock_prompt_manager, sample_chunk, sample_pdf_metadata):
    """Test cloze strategy with invalid cloze format."""
    config = StrategyConfig(enabled=True)
    strategy = ClozeDefinitionsStrategy(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        strategy_config=config,
        strategy_name="cloze_definitions"
    )
    
    mock_prompt_manager.render_template.return_value = "Mock prompt"
    
    # Response without proper cloze markers
    invalid_response = {
        "cards": [
            {
                "cloze_text": "Machine learning is a subset of artificial intelligence.",  # No cloze markers
                "extra": "No cloze deletions here",
                "page_citation": "p. 1",
                "core_concept": "ML",
                "difficulty": "medium",
                "tags": ["ml"]
            }
        ]
    }
    
    mock_llm_response = LLMResponse(
        content=json.dumps(invalid_response),
        model="gpt-4",
        tokens_used=80,
        cost_estimate=0.006,
        cached=False,
        response_time=1.0
    )
    
    mock_llm_provider.generate.return_value = mock_llm_response
    
    cards = strategy.generate_cards(sample_chunk, sample_pdf_metadata)
    
    # Should filter out invalid cloze cards
    assert len(cards) == 0


def test_strategy_disabled(mock_llm_provider, mock_prompt_manager, sample_chunk, sample_pdf_metadata):
    """Test that disabled strategies don't generate cards."""
    config = StrategyConfig(enabled=False)  # Disabled
    strategy = KeyPointsStrategy(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        strategy_config=config,
        strategy_name="key_points"
    )
    
    cards = strategy.generate_cards(sample_chunk, sample_pdf_metadata)
    
    # Should return empty list without calling LLM
    assert cards == []
    mock_llm_provider.generate.assert_not_called()


def test_strategy_deduplication():
    """Test strategy-level deduplication."""
    config = StrategyConfig(enabled=True)
    strategy = KeyPointsStrategy(
        llm_provider=Mock(),
        prompt_manager=Mock(),
        strategy_config=config,
        strategy_name="key_points"
    )
    
    from pdf2anki.strategies.base import FlashcardData
    
    # Create duplicate cards
    cards = [
        FlashcardData(
            note_type="Basic",
            front="What is Python?",
            back="A programming language",
            page_citation="p. 1",
            core_concept="Python"
        ),
        FlashcardData(
            note_type="Basic",
            front="What is Python?",  # Duplicate
            back="A programming language",
            page_citation="p. 1",
            core_concept="Python"
        ),
        FlashcardData(
            note_type="Basic",
            front="What is Java?",  # Different
            back="Another programming language",
            page_citation="p. 1",
            core_concept="Java"
        )
    ]
    
    deduplicated = strategy.deduplicate_cards(cards)
    
    # Should remove one duplicate
    assert len(deduplicated) == 2
    
    # Should keep both unique questions
    fronts = [card.front for card in deduplicated]
    assert "What is Python?" in fronts
    assert "What is Java?" in fronts