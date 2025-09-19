"""Figure-based strategy for generating flashcards from visual content."""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData

logger = logging.getLogger(__name__)


class FigureBasedStrategy(BaseStrategy):
    """Strategy for generating flashcards based on figures, tables, and visual content."""
    
    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        return "Basic"  # Could be extended to support image occlusion
    
    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        return "figure_based.j2"  # Will be created later
    
    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for figure-based strategy."""
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
                    logger.warning(f"Missing or invalid field '{field}' in figure card: {card}")
                    return False
                
                if not card[field].strip():
                    logger.warning(f"Empty field '{field}' in figure card: {card}")
                    return False
        
        return True
    
    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into figure-based flashcard data objects."""
        cards = []
        
        for card_data in response_data.get("cards", []):
            try:
                # Create flashcard with required fields
                flashcard = FlashcardData(
                    note_type=self.get_note_type(),
                    front=card_data["front"].strip(),
                    back=card_data["back"].strip(),
                    page_citation=card_data.get("page_citation", f"p. {chunk.start_page}"),
                    core_concept=card_data.get("core_concept", "Visual Content"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", [])),
                )
                
                cards.append(flashcard)
                
            except Exception as e:
                logger.warning(f"Failed to parse figure card data: {card_data}, error: {e}")
                continue
        
        return cards
    
    def _process_tags(self, tags: List[str]) -> List[str]:
        """Process and clean tags."""
        processed_tags = []
        
        # Add strategy-specific tag
        processed_tags.append("figure-based")
        
        # Add user-provided tags
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    processed_tags.append(tag.strip().lower())
        
        return list(set(processed_tags))  # Remove duplicates
    
    def should_apply_to_chunk(self, chunk: TextChunk, pdf_content: Dict) -> bool:
        """Check if this strategy should be applied to a chunk based on visual content."""
        # Look for indicators of visual content in the text
        visual_indicators = [
            "figure", "fig.", "table", "chart", "graph", "diagram", 
            "image", "picture", "plot", "illustration", "exhibit"
        ]
        
        chunk_text_lower = chunk.text.lower()
        
        # Check if chunk references visual elements
        has_visual_references = any(indicator in chunk_text_lower for indicator in visual_indicators)
        
        if not has_visual_references:
            return False
        
        # Check if there are actual images on these pages
        chunk_pages = set(range(chunk.start_page, chunk.end_page + 1))
        pdf_images = pdf_content.get("images", [])
        
        images_in_chunk = [img for img in pdf_images if img["page_num"] in chunk_pages]
        
        if images_in_chunk:
            logger.debug(f"Found {len(images_in_chunk)} images in chunk pages {chunk.start_page}-{chunk.end_page}")
            return True
        
        # Even without images, if there are strong visual references, we might want to process
        strong_indicators = ["figure", "table", "chart", "graph"]
        has_strong_references = any(indicator in chunk_text_lower for indicator in strong_indicators)
        
        return has_strong_references