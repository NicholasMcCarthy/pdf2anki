"""Highlight-priority strategy: generates flashcards anchored on PDF
highlight/annotation text, with a screenshot of the highlighted region attached
as supporting media. General (non-highlighted) coverage of the document is left
to the sibling key_points/cloze_definitions strategies, which still run over the
document's non-highlight chunks."""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData

logger = logging.getLogger(__name__)


class HighlightPriorityStrategy(BaseStrategy):
    """Strategy for generating flashcards grounded in highlighted/annotated PDF text."""

    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        return "Basic"

    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        return "highlight_priority.j2"

    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for the highlight-priority strategy."""
        if not isinstance(response_data, dict):
            return False

        cards = response_data.get("cards", [])
        if not isinstance(cards, list):
            return False

        for card in cards:
            if not isinstance(card, dict):
                return False

            required_fields = ["front", "back"]
            for req_field in required_fields:
                if req_field not in card or not isinstance(card[req_field], str):
                    logger.warning(f"Missing or invalid field '{req_field}' in highlight card: {card}")
                    return False

                if not card[req_field].strip():
                    logger.warning(f"Empty field '{req_field}' in highlight card: {card}")
                    return False

        return True

    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into highlight-grounded flashcard data objects."""
        cards = []

        # All highlight screenshots captured on this page-level chunk are attached to
        # every card generated from it. The LLM isn't asked to map individual cards to
        # individual highlights, so this is page-granularity association, not a
        # precise per-highlight one.
        screenshots = [
            h["screenshot"] for h in (chunk.highlights or []) if h.get("screenshot")
        ]

        for card_data in response_data.get("cards", []):
            try:
                flashcard = FlashcardData(
                    note_type=self.get_note_type(),
                    front=card_data["front"].strip(),
                    back=card_data["back"].strip(),
                    page_citation=card_data.get("page_citation", f"p. {chunk.start_page}"),
                    core_concept=card_data.get("core_concept", "Highlighted Content"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", [])),
                    media=list(screenshots),
                )

                cards.append(flashcard)

            except Exception as e:
                logger.warning(f"Failed to parse highlight card data: {card_data}, error: {e}")
                continue

        return cards

    def _process_tags(self, tags: List[str]) -> List[str]:
        """Process and clean tags."""
        processed_tags = ["highlight-priority"]

        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    processed_tags.append(tag.strip().lower())

        return list(set(processed_tags))

    def should_apply_to_chunk(self, chunk: TextChunk, pdf_content: Dict) -> bool:
        """Only apply to chunks that actually carry highlight annotations."""
        return bool(getattr(chunk, "highlights", None))
