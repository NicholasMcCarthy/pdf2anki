"""Key points strategy for generating basic Q&A flashcards."""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData

logger = logging.getLogger(__name__)


class KeyPointsStrategy(BaseStrategy):
    """Strategy for generating question-answer flashcards from key points."""
    
    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        return "Basic"
    
    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        return "key_points.j2"
    
    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for key points strategy."""
        if not isinstance(response_data, dict):
            return False
        
        cards = response_data.get("cards", [])
        if not isinstance(cards, list):
            return False
        
        for card in cards:
            if not isinstance(card, dict):
                return False
            
            # Check required fields
            required_fields = ["front", "back"]
            for field in required_fields:
                if field not in card or not isinstance(card[field], str):
                    logger.warning(f"Missing or invalid field '{field}' in card: {card}")
                    return False
                
                if not card[field].strip():
                    logger.warning(f"Empty field '{field}' in card: {card}")
                    return False
        
        return True
    
    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into key points flashcard data objects."""
        cards = []
        
        for card_data in response_data.get("cards", []):
            try:
                # Create flashcard with required fields
                flashcard = FlashcardData(
                    note_type=self.get_note_type(),
                    front=card_data["front"].strip(),
                    back=card_data["back"].strip(),
                    page_citation=card_data.get("page_citation", f"p. {chunk.start_page}"),
                    core_concept=card_data.get("core_concept", "Key Point"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", [])),
                )
                
                cards.append(flashcard)
                
            except Exception as e:
                logger.warning(f"Failed to parse card data: {card_data}, error: {e}")
                continue
        
        return cards
    
    def _process_tags(self, tags: List[str]) -> List[str]:
        """Process and clean tags."""
        processed_tags = []
        
        # Add strategy-specific tag
        processed_tags.append("key-points")
        
        # Add user-provided tags
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    processed_tags.append(tag.strip().lower())
        
        return list(set(processed_tags))  # Remove duplicates