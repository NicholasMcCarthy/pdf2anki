"""Strategy registry and runners for flashcard generation."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
import json

from .templates import get_prompt_manager, get_note_type_manager
from .chunking import TextChunk
from .config import Config

logger = logging.getLogger(__name__)


class StrategyRunner(ABC):
    """Abstract base class for strategy runners."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def generate_cards(self, chunk: TextChunk, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate flashcards from a text chunk.
        
        Args:
            chunk: Text chunk to process
            **kwargs: Additional parameters
            
        Returns:
            List of card dictionaries
        """
        pass
    
    @abstractmethod
    def get_note_type(self) -> str:
        """Get the note type this strategy produces."""
        pass


class GenericStrategyRunner(StrategyRunner):
    """
    Generic strategy runner that uses prompt templates and note types.
    
    This runner can work with any prompt template that follows the standard format.
    """
    
    def __init__(self, config: Config, prompt_name: str):
        super().__init__(config)
        self.prompt_name = prompt_name
        self.prompt_manager = get_prompt_manager()
        self.note_type_manager = get_note_type_manager()
        
        # Load the prompt template
        self.prompt_template = self.prompt_manager.get_prompt(prompt_name)
        if not self.prompt_template:
            raise ValueError(f"Prompt template '{prompt_name}' not found")
    
    def generate_cards(self, chunk: TextChunk, **kwargs) -> List[Dict[str, Any]]:
        """Generate cards using the configured prompt template."""
        try:
            # Prepare context for template rendering
            context = {
                'chunk_text': chunk.text,
                'page_start': chunk.start_page,
                'page_end': chunk.end_page,
                'section_title': chunk.section,
                **kwargs  # Allow override of template parameters
            }
            
            # Render the prompt
            rendered_prompt = self.prompt_manager.render_prompt(
                self.prompt_name,
                context,
                truncate_tokens=self.prompt_template.token_limits.get('max_chunk_tokens')
            )
            
            # TODO: Call LLM with rendered prompt
            # For now, return mock data
            logger.info(f"Generated prompt for {self.prompt_name} (LLM call stubbed)")
            
            mock_response = self._generate_mock_response(chunk)
            
            # Parse and validate response
            cards = self._parse_llm_response(mock_response)
            
            # Add provenance metadata
            for card in cards:
                card.update({
                    'strategy': self.prompt_name,
                    'note_type': self.get_note_type(),
                    'template_version': self.prompt_template.version,
                    'source_chunk_id': chunk.chunk_id if hasattr(chunk, 'chunk_id') else None,
                    'page_start': chunk.start_page,
                    'page_end': chunk.end_page,
                    'section': chunk.section,
                })
            
            return cards
            
        except Exception as e:
            logger.error(f"Error generating cards with {self.prompt_name}: {e}")
            return []
    
    def get_note_type(self) -> str:
        """Get the note type this strategy produces."""
        return self.prompt_template.note_type
    
    def _generate_mock_response(self, chunk: TextChunk) -> str:
        """Generate mock LLM response for testing."""
        note_type = self.get_note_type()
        
        if note_type == "basic":
            return json.dumps([
                {
                    "front": f"What is the main concept discussed in this section?",
                    "back": f"The main concept is related to the content from pages {chunk.start_page}-{chunk.end_page}."
                }
            ])
        elif note_type == "cloze":
            return json.dumps([
                {
                    "cloze_text": f"The key concept from this section is {{{{c1::placeholder concept}}}}.",
                    "extra": f"This appears on pages {chunk.start_page}-{chunk.end_page}."
                }
            ])
        else:
            return "[]"
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse and validate LLM response."""
        try:
            cards = json.loads(response)
            if not isinstance(cards, list):
                logger.warning(f"LLM response is not a list: {type(cards)}")
                return []
            
            # TODO: Add validation based on note type definition
            return cards
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []


class StrategyRegistry:
    """Registry for mapping strategy names to runner classes."""
    
    def __init__(self):
        self._strategies: Dict[str, Callable[[Config], StrategyRunner]] = {}
        self._register_default_strategies()
    
    def register_strategy(
        self,
        name: str,
        runner_factory: Callable[[Config], StrategyRunner]
    ) -> None:
        """
        Register a strategy runner factory.
        
        Args:
            name: Strategy name
            runner_factory: Function that takes Config and returns StrategyRunner
        """
        self._strategies[name] = runner_factory
        logger.info(f"Registered strategy: {name}")
    
    def create_runner(self, name: str, config: Config) -> Optional[StrategyRunner]:
        """
        Create a strategy runner instance.
        
        Args:
            name: Strategy name
            config: Configuration object
            
        Returns:
            StrategyRunner instance or None if not found
        """
        factory = self._strategies.get(name)
        if not factory:
            logger.error(f"Strategy '{name}' not found in registry")
            return None
        
        try:
            return factory(config)
        except Exception as e:
            logger.error(f"Failed to create strategy runner '{name}': {e}")
            return None
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def _register_default_strategies(self) -> None:
        """Register default strategies that use prompt templates."""
        
        # Key points strategy using new prompt template
        self.register_strategy(
            "key_points",
            lambda config: GenericStrategyRunner(config, "key_points")
        )
        
        # Cloze definitions strategy using new prompt template  
        self.register_strategy(
            "cloze_definitions",
            lambda config: GenericStrategyRunner(config, "cloze_definitions")
        )
        
        # TODO: Add more strategies as prompt templates are created
        # self.register_strategy(
        #     "figure_based",
        #     lambda config: GenericStrategyRunner(config, "figure_based")
        # )


# Global registry instance
_global_registry = StrategyRegistry()


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    return _global_registry


def create_strategy_runner(strategy_name: str, config: Config) -> Optional[StrategyRunner]:
    """Convenience function to create a strategy runner."""
    return _global_registry.create_runner(strategy_name, config)


def list_available_strategies() -> List[str]:
    """List all available strategies."""
    return _global_registry.list_strategies()