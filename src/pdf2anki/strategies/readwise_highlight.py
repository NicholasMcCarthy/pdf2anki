"""Readwise-highlight strategy: generates flashcards from a single saved web
highlight (Readwise/Obsidian markdown export), grounding each card in the
highlighted text (and the reader's own note on it, if present) and citing the
source URL instead of a PDF page number.

The LLM is given the article's other highlights (if any) as background context
and may draw on limited general background knowledge for clarification, but
every card must still be grounded in its own highlight - see
prompts/readwise_highlight.j2. It also chooses per card whether Basic or
Cloze format tests the material better (see infer_card_type())."""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData, infer_card_type, validate_cloze_format

logger = logging.getLogger(__name__)


class ReadwiseHighlightStrategy(BaseStrategy):
    """Strategy for generating flashcards from Readwise web highlights."""

    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy. Cards can individually be
        Basic or Cloze (the LLM decides per card - see parse_cards()); this is
        just the representative default for callers that need a single value."""
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

            if infer_card_type(card) == "cloze":
                if not validate_cloze_format(card.get("cloze_text")):
                    logger.warning(f"Missing or invalid 'cloze_text' in readwise card: {card}")
                    return False
                continue

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
                card_type = infer_card_type(card_data)
                common = dict(
                    page_citation=source_url or source_title,
                    ref_citation=source_url or source_title,
                    core_concept=card_data.get("core_concept", "Highlight"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", []), pdf_metadata),
                    extra=f'<a href="{source_url}">{source_title}</a>' if source_url else "",
                )

                if card_type == "cloze":
                    cloze_text = card_data["cloze_text"].strip()
                    if not validate_cloze_format(cloze_text):
                        continue
                    # Cloze notes' `extra` field is for card-specific supplementary
                    # context, not the source link - fold the source-link markup
                    # (built into `common["extra"]` above) after any LLM-provided extra.
                    llm_extra = card_data.get("extra", "").strip()
                    common["extra"] = f"{llm_extra}<br>{common['extra']}" if llm_extra and common["extra"] else (llm_extra or common["extra"])
                    flashcard = FlashcardData(
                        note_type="Cloze",
                        cloze_text=cloze_text,
                        **common,
                    )
                else:
                    flashcard = FlashcardData(
                        note_type="Basic",
                        front=card_data["front"].strip(),
                        back=card_data["back"].strip(),
                        **common,
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
