"""Tests for the reviewer step functionality."""

import json
import pytest
from unittest.mock import Mock, patch

from pdf2anki.config import Config, ReviewConfig
from pdf2anki.llm import LLMProvider
from pdf2anki.preprocess import apply_review_step
from pdf2anki.prompts import PromptManager
from pdf2anki.strategies.base import FlashcardData
from pdf2anki.telemetry import TelemetryCollector


@pytest.fixture
def sample_cards():
    """Create sample flashcards for testing."""
    cards = []
    
    # Basic card
    card1 = FlashcardData(
        id="card1",
        note_type="basic",
        page_citation="Page 1-2",
        core_concept="machine learning definition",
        front="What is machine learning?",
        back="A subset of AI that enables computers to learn without explicit programming",
        tags=["AI", "ML"],
        source_pdf="test.pdf",
        page_start=1,
        page_end=2,
        strategy="key_points",
        section="Introduction",
        metadata={}
    )
    
    # Cloze card
    card2 = FlashcardData(
        id="card2",
        note_type="cloze", 
        page_citation="Page 3",
        core_concept="neural networks definition",
        cloze_text="{{c1::Neural networks}} are inspired by biological neurons",
        front="{{c1::Neural networks}} are inspired by biological neurons",
        back="",
        tags=["AI", "neural"],
        source_pdf="test.pdf",
        page_start=3,
        page_end=3,
        strategy="cloze_definitions",
        section="Architecture",
        metadata={}
    )
    
    cards.extend([card1, card2])
    return cards


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LLMProvider)
    
    # Mock successful review response
    def mock_review_card(card_data, context, prompt_manager, template_name):
        # Return different scores based on card ID
        if card_data.get("id") == "card1":
            return {
                "id": "card1",
                "score": 8.5,
                "issues": [],
                "edited": {
                    "front": "What is machine learning and how does it work?",
                    "back": "A subset of AI that enables computers to learn from data without explicit programming"
                }
            }
        elif card_data.get("id") == "card2":
            return {
                "id": "card2", 
                "score": 6.5,
                "issues": ["Could be more specific"],
                "edited": None
            }
        else:
            return {
                "id": card_data.get("id", "unknown"),
                "score": 5.0,
                "issues": ["Generic issues"],
                "edited": None
            }
    
    provider.review_card.side_effect = mock_review_card
    provider.get_stats.return_value = {
        "total_tokens": 150,
        "total_cost": 0.002,
        "cache_hits": 0,
        "api_calls": 2
    }
    
    return provider


@pytest.fixture
def mock_prompt_manager():
    """Create a mock prompt manager."""
    manager = Mock(spec=PromptManager)
    manager.list_templates.return_value = ["reviewer_special.j2", "key_points.j2"]
    manager.render_template.return_value = "Mock rendered template"
    return manager


@pytest.fixture
def mock_telemetry():
    """Create a mock telemetry collector."""
    telemetry = Mock(spec=TelemetryCollector)
    return telemetry


class TestReviewerDisabled:
    """Test reviewer when disabled."""
    
    def test_reviewer_disabled_returns_original_cards(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test that when reviewer is disabled, original cards are returned unchanged."""
        config = ReviewConfig(enabled=False)
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider, 
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should return all original cards unchanged
        assert len(result) == len(sample_cards)
        assert result == sample_cards
        
        # Should not call LLM
        mock_llm_provider.review_card.assert_not_called()
        
        # Should not record telemetry
        mock_telemetry.record_reviewer_metrics.assert_not_called()


class TestReviewerEnabled:
    """Test reviewer when enabled."""
    
    def test_reviewer_enabled_processes_cards(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test that when reviewer is enabled, cards are processed correctly."""
        config = ReviewConfig(enabled=True, min_score=7.0, allow_edits=True)
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager, 
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should only keep cards with score >= 7.0 (card1 with 8.5)
        assert len(result) == 1
        assert result[0].id == "card1"
        
        # Should apply edits to card1
        assert result[0].front == "What is machine learning and how does it work?"
        assert result[0].back == "A subset of AI that enables computers to learn from data without explicit programming"
        
        # Should have review metadata
        assert "review_score" in result[0].metadata
        assert result[0].metadata["review_score"] == 8.5
        assert result[0].metadata["review_edited"] is True
        
        # Should call LLM for each card
        assert mock_llm_provider.review_card.call_count == 2
        
        # Should record telemetry
        mock_telemetry.record_reviewer_metrics.assert_called_once()
        call_args = mock_telemetry.record_reviewer_metrics.call_args[1]
        assert call_args["cards_reviewed"] == 2
        assert call_args["cards_dropped"] == 1
        assert call_args["cards_edited"] == 1
    
    def test_reviewer_filters_low_scores(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test that reviewer filters out cards below minimum score."""
        config = ReviewConfig(enabled=True, min_score=9.0, allow_edits=False)
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should filter out all cards (highest score is 8.5)
        assert len(result) == 0
        
        # Should record metrics showing all cards dropped
        mock_telemetry.record_reviewer_metrics.assert_called_once()
        call_args = mock_telemetry.record_reviewer_metrics.call_args[1]
        assert call_args["cards_dropped"] == 2
        assert call_args["cards_edited"] == 0  # edits disabled
    
    def test_reviewer_without_edits(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test reviewer with edits disabled."""
        config = ReviewConfig(enabled=True, min_score=6.0, allow_edits=False)
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should keep both cards (scores 8.5 and 6.5)
        assert len(result) == 2
        
        # Should not apply edits
        for card in result:
            if card.id == "card1":
                # Original content should be preserved
                assert card.front == "What is machine learning?"
                assert card.metadata["review_edited"] is False
    
    def test_reviewer_missing_template(self, sample_cards, mock_llm_provider, mock_telemetry):
        """Test reviewer behavior when template is missing."""
        config = ReviewConfig(enabled=True)
        
        # Mock prompt manager without reviewer template
        mock_prompt_manager = Mock(spec=PromptManager)
        mock_prompt_manager.list_templates.return_value = ["key_points.j2"]
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should return original cards unchanged
        assert len(result) == len(sample_cards)
        assert result == sample_cards
        
        # Should not call LLM
        mock_llm_provider.review_card.assert_not_called()
    
    def test_reviewer_handles_errors_gracefully(self, sample_cards, mock_prompt_manager, mock_telemetry):
        """Test that reviewer handles errors gracefully and keeps cards."""
        config = ReviewConfig(enabled=True, min_score=7.0)
        
        # Mock LLM provider that raises exceptions
        mock_llm_provider = Mock(spec=LLMProvider)
        mock_llm_provider.review_card.side_effect = Exception("LLM error")
        mock_llm_provider.get_stats.return_value = {"total_tokens": 0, "total_cost": 0.0, "cache_hits": 0, "api_calls": 0}
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Should keep all cards when errors occur
        assert len(result) == len(sample_cards)
        
        # Should still record telemetry
        mock_telemetry.record_reviewer_metrics.assert_called_once()


class TestReviewerProvenance:
    """Test reviewer provenance and metadata tracking."""
    
    def test_reviewer_metadata_recorded(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test that reviewer metadata is properly recorded."""
        config = ReviewConfig(enabled=True, min_score=6.0, allow_edits=True, template_version="2.0")
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Check that telemetry was called with correct config
        mock_telemetry.record_reviewer_metrics.assert_called_once()
        call_args = mock_telemetry.record_reviewer_metrics.call_args[1]
        
        config_arg = call_args["config"]
        assert config_arg["min_score"] == 6.0
        assert config_arg["template_version"] == "2.0"  
        assert config_arg["allow_edits"] is True
        
        # Check LLM stats are recorded
        llm_stats = call_args["llm_stats"]
        assert "total_tokens" in llm_stats
        assert "total_cost" in llm_stats
    
    def test_card_metadata_populated(self, sample_cards, mock_llm_provider, mock_prompt_manager, mock_telemetry):
        """Test that individual card metadata is populated correctly."""
        config = ReviewConfig(enabled=True, min_score=6.0, allow_edits=True)
        
        result = apply_review_step(
            cards=sample_cards,
            llm_provider=mock_llm_provider,
            prompt_manager=mock_prompt_manager,
            review_config=config,
            telemetry=mock_telemetry
        )
        
        # Check card metadata for kept cards
        for card in result:
            assert "review_score" in card.metadata
            assert "review_issues" in card.metadata  
            assert "review_edited" in card.metadata
            
            if card.id == "card1":
                assert card.metadata["review_score"] == 8.5
                assert card.metadata["review_issues"] == []
                assert card.metadata["review_edited"] is True