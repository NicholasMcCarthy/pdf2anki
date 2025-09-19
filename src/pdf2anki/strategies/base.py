"""Base strategy class for flashcard generation."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator

from ..chunking import TextChunk
from ..config import StrategyConfig
from ..llm import LLMProvider
from ..prompts import PromptManager

logger = logging.getLogger(__name__)


class FlashcardData(BaseModel):
    """Base flashcard data structure."""
    
    # Core fields (all cards have these)
    note_type: str
    page_citation: str
    core_concept: str
    difficulty: str = "medium"
    tags: List[str] = []
    
    # Strategy-specific fields (will be added by subclasses)
    front: Optional[str] = None
    back: Optional[str] = None
    cloze_text: Optional[str] = None
    extra: Optional[str] = None
    
    # Metadata fields
    source_pdf: str = ""
    page_start: int = 0
    page_end: int = 0
    section: Optional[str] = None
    ref_citation: str = ""
    llm_model: str = ""
    llm_version: str = ""
    strategy: str = ""
    template_version: str = "1.0"
    original_text: Optional[str] = None
    
    @validator("difficulty")
    def validate_difficulty(cls, v):
        valid_levels = ["easy", "medium", "hard"]
        if v not in valid_levels:
            return "medium"
        return v
    
    @validator("tags", pre=True)
    def validate_tags(cls, v):
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(";") if tag.strip()]
        return v or []


class BaseStrategy(ABC):
    """Base class for all flashcard generation strategies."""
    
    def __init__(
        self, 
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        strategy_config: StrategyConfig,
        strategy_name: str
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.config = strategy_config
        self.name = strategy_name
        
    @abstractmethod
    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        pass
    
    @abstractmethod
    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        pass
    
    @abstractmethod
    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for this strategy."""
        pass
    
    @abstractmethod
    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into flashcard data objects."""
        pass
    
    def generate_cards(
        self, 
        chunk: TextChunk, 
        pdf_metadata: Dict[str, Any],
        max_cards: int = 5
    ) -> List[FlashcardData]:
        """Generate flashcards for a text chunk."""
        if not self.config.enabled:
            logger.debug(f"Strategy {self.name} is disabled, skipping")
            return []
        
        try:
            # Prepare template variables
            template_vars = {
                "chunk": chunk.text,
                "section": chunk.section,
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "pdf_title": pdf_metadata.get("title", "Unknown"),
                "author": pdf_metadata.get("author", ""),
                "strategy": self.name,
                "max_cards": max_cards,
                **self.config.params
            }
            
            # Render prompt
            prompt = self.prompt_manager.render_template(
                self.get_template_name(),
                **template_vars
            )
            
            # Get system prompt
            system_prompt = self._get_system_prompt()
            
            # Generate response
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True,
                max_retries=3
            )
            
            # Parse response
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from {self.name}: {e}")
                logger.debug(f"Raw response: {response.content}")
                return []
            
            # Validate response format
            if not self.validate_response(response_data):
                logger.warning(f"Invalid response format from {self.name}")
                return []
            
            # Parse into flashcard objects
            cards = self.parse_cards(response_data, chunk, pdf_metadata)
            
            # Add common metadata
            for card in cards:
                card.source_pdf = str(pdf_metadata.get("path", ""))
                card.page_start = chunk.start_page
                card.page_end = chunk.end_page
                card.section = chunk.section
                card.ref_citation = f"p. {chunk.start_page}"
                card.llm_model = self.llm_provider.config.model
                card.strategy = self.name
                card.template_version = self.config.template_version
                card.original_text = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
            
            logger.debug(f"Generated {len(cards)} cards using {self.name} strategy")
            return cards
            
        except Exception as e:
            logger.error(f"Error generating cards with {self.name} strategy: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for this strategy."""
        from ..prompts import get_default_system_prompt
        return get_default_system_prompt(self.name)
    
    def apply_quality_filter(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Apply quality filtering to generated cards."""
        if not self.config.min_score:
            return cards
        
        # TODO: Implement quality scoring
        # For now, just return all cards
        logger.debug(f"Quality filtering not implemented yet for {self.name}")
        return cards
    
    def deduplicate_cards(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Remove duplicate cards within the same generation."""
        if not cards:
            return cards
        
        seen_fronts = set()
        unique_cards = []
        
        for card in cards:
            # Create a key based on the question/cloze content
            if hasattr(card, 'front') and card.front:
                key = card.front.lower().strip()
            elif hasattr(card, 'cloze_text') and card.cloze_text:
                # Remove cloze markers for comparison
                import re
                key = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', card.cloze_text).lower().strip()
            else:
                key = str(card.dict())
            
            if key not in seen_fronts:
                seen_fronts.add(key)
                unique_cards.append(card)
            else:
                logger.debug(f"Filtered duplicate card: {key[:50]}...")
        
        if len(unique_cards) < len(cards):
            logger.info(f"Filtered {len(cards) - len(unique_cards)} duplicate cards")
        
        return unique_cards