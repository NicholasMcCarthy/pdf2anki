"""Readwise-highlight strategy: generates flashcards from a single saved web
highlight (Readwise/Obsidian markdown export), grounding each card in the
highlighted text (and the reader's own note on it, if present) and citing the
source URL instead of a PDF page number."""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData

logger = logging.getLogger(__name__)


class ReadwiseHighlightStrategy(BaseStrategy):
    """Strategy for generating flashcards from Readwise web highlights."""

    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy."""
        return "Basic"

    def get_template_name(self) -> str:
        """Get the prompt template name for this strategy."""
        return "readwise_highlight.j2"

    def validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response format for the readwise-highlight strategy."""
        if not isinstance(response_data, dict):
            return False

        cards = response_data.get("cards", [])
        if not isinstance(cards, list):
            return False

        for card in cards:
            if not isinstance(card, dict):
                return False

            for req_field in ("front", "back"):
                if req_field not in card or not isinstance(card[req_field], str):
                    logger.warning(f"Missing or invalid field '{req_field}' in readwise card: {card}")
                    return False

                if not card[req_field].strip():
                    logger.warning(f"Empty field '{req_field}' in readwise card: {card}")
                    return False

        return True

    def parse_cards(self, response_data: Dict[str, Any], chunk: TextChunk, pdf_metadata: Dict) -> List[FlashcardData]:
        """Parse LLM response into flashcard data objects citing the source URL."""
        cards = []
        source_url = pdf_metadata.get("source_url", "")
        source_title = pdf_metadata.get("title", "source")

        for card_data in response_data.get("cards", []):
            try:
                flashcard = FlashcardData(
                    note_type=self.get_note_type(),
                    front=card_data["front"].strip(),
                    back=card_data["back"].strip(),
                    page_citation=source_url or source_title,
                    ref_citation=source_url or source_title,
                    core_concept=card_data.get("core_concept", "Highlight"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", []), pdf_metadata),
                    extra=f'<a href="{source_url}">{source_title}</a>' if source_url else "",
                )

                cards.append(flashcard)

            except Exception as e:
                logger.warning(f"Failed to parse readwise card data: {card_data}, error: {e}")
                continue

        return cards

    def _process_tags(self, tags: List[str], pdf_metadata: Dict) -> List[str]:
        """Process and clean tags, folding in the document's own category/tags."""
        processed_tags = ["readwise"]

        category = pdf_metadata.get("category")
        if category:
            processed_tags.append(str(category).strip().lower())

        for extra_tag in pdf_metadata.get("extra_tags", []) or []:
            if isinstance(extra_tag, str) and extra_tag.strip():
                processed_tags.append(extra_tag.strip().lower())

        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    processed_tags.append(tag.strip().lower())

        return list(set(processed_tags))
