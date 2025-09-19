"""Cloze definitions strategy for generating cloze deletion flashcards."""

import logging
import re
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData

logger = logging.getLogger(__name__)


class ClozeDefinitionsStrategy(BaseStrategy):
    """Strategy for generating cloze deletion flashcards from definitions and key terms."""
    
    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        return "Cloze"
    
    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        return "cloze_definitions.j2"
    
    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for cloze definitions strategy."""
        if not isinstance(response_data, dict):
            return False
        
        cards = response_data.get("cards", [])
        if not isinstance(cards, list):
            return False
        
        for card in cards:
            if not isinstance(card, dict):
                return False
            
            # Check required fields
            cloze_text = card.get("cloze_text")
            if not isinstance(cloze_text, str) or not cloze_text.strip():
                logger.warning(f"Missing or invalid 'cloze_text' in card: {card}")
                return False
            
            # Validate cloze format
            if not self._validate_cloze_format(cloze_text):
                logger.warning(f"Invalid cloze format in: {cloze_text}")
                return False
        
        return True
    
    def _validate_cloze_format(self, cloze_text: str) -> bool:
        """Validate that cloze text contains proper cloze deletion markers."""
        # Check for cloze deletion patterns like {{c1::word}} or {{c2::phrase}}
        cloze_pattern = r'\{\{c\d+::[^}]+\}\}'
        matches = re.findall(cloze_pattern, cloze_text)
        
        if not matches:
            return False
        
        # Check that we don't have too many cloze deletions (max 3 per card)
        if len(matches) > 3:
            logger.warning(f"Too many cloze deletions ({len(matches)}) in: {cloze_text[:100]}...")
            return False
        
        return True
    
    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into cloze deletion flashcard data objects."""
        cards = []
        
        for card_data in response_data.get("cards", []):
            try:
                # Validate and clean cloze text
                cloze_text = card_data["cloze_text"].strip()
                if not self._validate_cloze_format(cloze_text):
                    continue
                
                # Create flashcard with required fields
                flashcard = FlashcardData(
                    note_type=self.get_note_type(),
                    cloze_text=cloze_text,
                    extra=card_data.get("extra", "").strip(),
                    page_citation=card_data.get("page_citation", f"p. {chunk.start_page}"),
                    core_concept=card_data.get("core_concept", "Definition"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", [])),
                )
                
                cards.append(flashcard)
                
            except Exception as e:
                logger.warning(f"Failed to parse cloze card data: {card_data}, error: {e}")
                continue
        
        return cards
    
    def _process_tags(self, tags: List[str]) -> List[str]:
        """Process and clean tags."""
        processed_tags = []
        
        # Add strategy-specific tag
        processed_tags.append("cloze-definitions")
        
        # Add user-provided tags
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    processed_tags.append(tag.strip().lower())
        
        return list(set(processed_tags))  # Remove duplicates
    
    def deduplicate_cards(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Remove duplicate cloze cards with special handling for cloze text."""
        if not cards:
            return cards
        
        seen_clozes = set()
        unique_cards = []
        
        for card in cards:
            if not hasattr(card, 'cloze_text') or not card.cloze_text:
                continue
            
            # Normalize cloze text for comparison
            normalized_cloze = self._normalize_cloze_text(card.cloze_text)
            
            if normalized_cloze not in seen_clozes:
                seen_clozes.add(normalized_cloze)
                unique_cards.append(card)
            else:
                logger.debug(f"Filtered duplicate cloze card: {normalized_cloze[:50]}...")
        
        if len(unique_cards) < len(cards):
            logger.info(f"Filtered {len(cards) - len(unique_cards)} duplicate cloze cards")
        
        return unique_cards
    
    def _normalize_cloze_text(self, cloze_text: str) -> str:
        """Normalize cloze text for duplicate detection."""
        # Remove cloze markers and normalize whitespace
        normalized = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', cloze_text)
        normalized = re.sub(r'\s+', ' ', normalized.strip().lower())
        return normalized