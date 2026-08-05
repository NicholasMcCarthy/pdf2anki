"""Base strategy class for flashcard generation."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..chunking import TextChunk
from ..config import StrategyConfig
from ..llm import LLMProvider
from ..prompts import PromptManager

logger = logging.getLogger(__name__)

_CLOZE_PATTERN = re.compile(r"\{\{c\d+::[^}]+\}\}")


def validate_cloze_format(cloze_text: Any, max_deletions: int = 3) -> bool:
    """Validate that cloze_text contains 1-max_deletions well-formed
    {{cN::...}} deletion markers. Shared by every strategy that can emit
    Cloze cards (ClozeDefinitionsStrategy, and any strategy that lets the LLM
    choose per-card between Basic and Cloze - see infer_card_type())."""
    if not isinstance(cloze_text, str):
        return False
    matches = _CLOZE_PATTERN.findall(cloze_text)
    if not matches:
        return False
    if len(matches) > max_deletions:
        logger.warning(f"Too many cloze deletions ({len(matches)}) in: {cloze_text[:100]}...")
        return False
    return True


def infer_card_type(card_data: Dict[str, Any]) -> str:
    """Determine whether a card dict from an LLM response represents a Basic
    or Cloze note. Prefers an explicit "card_type" field (used by strategies
    that let the LLM choose per card, e.g. highlight_priority and
    readwise_highlight), falling back to field-presence inference if it's
    missing or invalid."""
    declared = str(card_data.get("card_type", "")).strip().lower()
    if declared in ("basic", "cloze"):
        return declared
    return "cloze" if card_data.get("cloze_text") else "basic"


@dataclass
class FlashcardData:
    """Base flashcard data structure."""
    
    # Core fields (all cards have these)
    note_type: str
    page_citation: str
    core_concept: str
    id: Optional[str] = None  # Added for reviewer functionality
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    
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
    media: List[str] = field(default_factory=list)  # e.g. highlight/figure screenshot filenames
    metadata: Dict[str, Any] = field(default_factory=dict)  # Added for reviewer functionality
    
    def __post_init__(self):
        """Validate and clean up fields after initialization."""
        # Validate difficulty
        valid_levels = ["easy", "medium", "hard"]
        if self.difficulty not in valid_levels:
            self.difficulty = "medium"
        
        # Validate and clean tags
        if isinstance(self.tags, str):
            self.tags = [tag.strip() for tag in self.tags.split(";") if tag.strip()]
        elif not self.tags:
            self.tags = []


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
                "abstract": pdf_metadata.get("abstract", ""),
                "other_highlights": pdf_metadata.get("other_highlights", ""),
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
                # Page-based citation is the sensible default for PDF-derived chunks;
                # a strategy may pre-set ref_citation in parse_cards() instead (e.g.
                # ReadwiseHighlightStrategy citing a source URL), which wins here.
                card.ref_citation = card.ref_citation or f"p. {chunk.start_page}"
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
                key = str(asdict(card))
            
            if key not in seen_fronts:
                seen_fronts.add(key)
                unique_cards.append(card)
            else:
                logger.debug(f"Filtered duplicate card: {key[:50]}...")
        
        if len(unique_cards) < len(cards):
            logger.info(f"Filtered {len(cards) - len(unique_cards)} duplicate cards")
        
        return unique_cards