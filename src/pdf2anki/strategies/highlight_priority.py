"""Highlight-priority strategy: generates flashcards anchored on PDF
highlight/annotation text, with a screenshot of the highlighted region attached
as supporting media. General (non-highlighted) coverage of the document is left
to the sibling key_points/cloze_definitions strategies, which still run over the
document's non-highlight chunks.

The LLM chooses per card whether Basic or Cloze format tests the material
better (see infer_card_type()), and is given the paper's abstract (if
extracted) as framing context for identifying key concepts - see
prompts/highlight_priority.j2.
"""

import logging
from typing import Any, Dict, List

from ..chunking import TextChunk
from .base import BaseStrategy, FlashcardData, default_page_citation, infer_card_type, validate_cloze_format

logger = logging.getLogger(__name__)


class HighlightPriorityStrategy(BaseStrategy):
    """Strategy for generating flashcards grounded in highlighted/annotated PDF text."""

    def get_note_type(self) -> str:
        """Get the Anki note type for this strategy. Cards can individually be
        Basic or Cloze (the LLM decides per card - see parse_cards()); this is
        just the representative default for callers that need a single value."""
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

            if infer_card_type(card) == "cloze":
                cloze_text = card.get("cloze_text")
                if not validate_cloze_format(cloze_text):
                    logger.warning(f"Missing or invalid 'cloze_text' in highlight card: {card}")
                    return False
                continue

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

        # All highlight screenshots captured on this chunk are attached to every
        # card generated from it. The LLM isn't asked to map individual cards to
        # individual highlights, so this is chunk-granularity association, not a
        # precise per-highlight one - and since a chunk can now span an entire
        # short paper (see chunking.py:_chunk_by_highlights), that granularity can
        # be coarse (e.g. a 10-highlight paper's every card carries all 10
        # screenshots). Accepted as a known limitation: a substring-match
        # heuristic (attach only the highlight whose text overlaps a given card)
        # would be fragile against paraphrased cards and risks silently dropping
        # a legitimate screenshot, which is worse than Anki storing a few unused
        # media references.
        screenshots = [
            h["screenshot"] for h in (chunk.highlights or []) if h.get("screenshot")
        ]

        for card_data in response_data.get("cards", []):
            try:
                card_type = infer_card_type(card_data)
                common = dict(
                    page_citation=card_data.get("page_citation", default_page_citation(chunk)),
                    core_concept=card_data.get("core_concept", "Highlighted Content"),
                    difficulty=card_data.get("difficulty", "medium"),
                    tags=self._process_tags(card_data.get("tags", [])),
                    media=list(screenshots),
                )

                if card_type == "cloze":
                    cloze_text = card_data["cloze_text"].strip()
                    if not validate_cloze_format(cloze_text):
                        continue
                    flashcard = FlashcardData(
                        note_type="Cloze",
                        cloze_text=cloze_text,
                        extra=card_data.get("extra", "").strip(),
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
