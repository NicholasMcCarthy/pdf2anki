"""Tests for strategy registry and runners."""

import pytest
from unittest.mock import Mock, patch

from src.pdf2anki.strategy_registry import (
    StrategyRegistry, GenericStrategyRunner, get_strategy_registry,
    list_available_strategies
)
from src.pdf2anki.config import Config
from src.pdf2anki.chunking import TextChunk


class TestStrategyRegistry:
    """Test strategy registry functionality."""
    
    def test_registry_initialization(self):
        """Test registry creates with default strategies."""
        registry = StrategyRegistry()
        strategies = registry.list_strategies()
        
        assert "key_points" in strategies
        assert "cloze_definitions" in strategies
        assert len(strategies) >= 2
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        registry = StrategyRegistry()
        
        def custom_factory(config):
            return Mock()
        
        registry.register_strategy("custom_test", custom_factory)
        
        assert "custom_test" in registry.list_strategies()
        
        runner = registry.create_runner("custom_test", Config())
        assert runner is not None
    
    def test_create_nonexistent_strategy(self):
        """Test creating runner for non-existent strategy."""
        registry = StrategyRegistry()
        
        runner = registry.create_runner("nonexistent", Config())
        assert runner is None
    
    def test_global_registry(self):
        """Test global registry functions."""
        strategies = list_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) >= 2
        
        registry = get_strategy_registry()
        assert isinstance(registry, StrategyRegistry)


class TestGenericStrategyRunner:
    """Test generic strategy runner."""
    
    def test_runner_initialization(self):
        """Test runner initialization with valid prompt."""
        config = Config()
        
        # This should work if our key_points prompt exists
        runner = GenericStrategyRunner(config, "key_points")
        
        assert runner.config == config
        assert runner.prompt_name == "key_points"
        assert runner.get_note_type() == "basic"  # key_points uses basic note type
    
    def test_runner_invalid_prompt(self):
        """Test runner with invalid prompt name."""
        config = Config()
        
        with pytest.raises(ValueError, match="Prompt template 'invalid' not found"):
            GenericStrategyRunner(config, "invalid")
    
    def test_generate_cards_mock(self):
        """Test card generation with mocked components."""
        config = Config()
        
        # Create a test chunk
        chunk = TextChunk(
            text="This is test content for card generation.",
            start_page=1,
            end_page=1,
            section="Test Section"
        )
        
        with patch('src.pdf2anki.strategy_registry.get_prompt_manager') as mock_prompt_mgr:
            # Mock prompt manager and template
            mock_template = Mock()
            mock_template.note_type = "basic"
            mock_template.version = "1.0"
            mock_template.token_limits = {}
            
            mock_mgr = Mock()
            mock_mgr.get_prompt.return_value = mock_template
            mock_mgr.render_prompt.return_value = "Rendered prompt"
            mock_prompt_mgr.return_value = mock_mgr
            
            runner = GenericStrategyRunner(config, "test_prompt")
            cards = runner.generate_cards(chunk)
            
            # Should return mock cards
            assert isinstance(cards, list)
            if cards:  # If not empty
                assert "strategy" in cards[0]
                assert "note_type" in cards[0]
    
    def test_mock_response_generation(self):
        """Test mock response generation for different note types."""
        config = Config()
        
        chunk = TextChunk(
            text="Test content",
            start_page=1,
            end_page=2,
            section="Test"
        )
        
        with patch('src.pdf2anki.strategy_registry.get_prompt_manager') as mock_prompt_mgr:
            # Test basic note type
            mock_template = Mock()
            mock_template.note_type = "basic"
            mock_template.version = "1.0"
            mock_template.token_limits = {}
            
            mock_mgr = Mock()
            mock_mgr.get_prompt.return_value = mock_template
            mock_prompt_mgr.return_value = mock_mgr
            
            runner = GenericStrategyRunner(config, "test")
            response = runner._generate_mock_response(chunk)
            
            assert '"front"' in response
            assert '"back"' in response
            assert f"pages {chunk.start_page}-{chunk.end_page}" in response
            
            # Test cloze note type
            mock_template.note_type = "cloze"
            response = runner._generate_mock_response(chunk)
            
            assert '"cloze_text"' in response
            assert "{{c1::" in response


def test_integration_with_actual_templates():
    """Test integration with actual prompt templates."""
    config = Config()
    
    # Test creating runners for actual templates
    registry = get_strategy_registry()
    
    key_points_runner = registry.create_runner("key_points", config)
    assert key_points_runner is not None
    assert key_points_runner.get_note_type() == "basic"
    
    cloze_runner = registry.create_runner("cloze_definitions", config)
    assert cloze_runner is not None
    assert cloze_runner.get_note_type() == "cloze"