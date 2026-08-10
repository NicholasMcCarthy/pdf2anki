"""Textbook workflow support: outline-driven chaptering for full-coverage
generation, and a per-book instructions.yml profile that overrides a shipped
default template - mirroring the "default.yml -> per-book <slug>.yml, with the
book's own outline folded back in" pattern from a reference textbook pipeline
the chunking/config granularity choices here were modeled on.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import yaml

from .config import Chunking, _as_plain, _dataclass_from_dict, _deep_merge
from .pdf import PDFDocument

logger = logging.getLogger(__name__)


@dataclass
class OutlineEntry:
    """One chapter/section entry from a PDF's table of contents."""
    title: str
    level: int
    start_page: int
    end_page: int


@dataclass
class TextbookProfile:
    """Per-textbook processing profile: chunking granularity, card budget, and
    deck/tag naming. Loaded via load_textbook_profile(), which deep-merges a
    hand-authored instructions.yml (if present next to the book's PDF) over a
    shipped default profile - unset fields fall back to the default.
    """
    chunking: Chunking = field(default_factory=lambda: Chunking(
        mode="outline", tokens_per_chunk=2500, max_chunk_tokens=4000
    ))
    strategies: List[str] = field(default_factory=lambda: [
        "key_points", "figure_based", "cloze_definitions"
    ])
    min_cards_per_section: int = 1
    max_cards_per_section: int = 8
    deck_name: Optional[str] = None
    extra_tags: List[str] = field(default_factory=list)


def load_textbook_profile(
    book_dir: Path,
    default_profile_path: Optional[Path] = None,
    instructions_filename: str = "instructions.yml",
) -> TextbookProfile:
    """Load a per-book instructions.yml if present in book_dir, deep-merged over
    the shipped default profile. Falls back to the default (or dataclass
    defaults, if no default file is given/found either) when the book has no
    instructions.yml of its own - the whole point is that a book needs no
    per-book file to get sensible full-coverage behavior.
    """
    default_data: Dict[str, Any] = {}
    if default_profile_path and Path(default_profile_path).exists():
        with open(default_profile_path, "r", encoding="utf-8") as f:
            default_data = yaml.safe_load(f) or {}

    instructions_path = Path(book_dir) / instructions_filename
    override_data: Dict[str, Any] = {}
    if instructions_path.exists():
        with open(instructions_path, "r", encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}
        logger.info(f"Loaded per-book profile from {instructions_path}")
    else:
        logger.info(f"No {instructions_filename} found in {book_dir}, using default textbook profile")

    merged = _deep_merge(_deep_merge(_as_plain(TextbookProfile()), default_data), override_data)
    return _dataclass_from_dict(TextbookProfile, merged)


def extract_outline(pdf_path: Path) -> List[OutlineEntry]:
    """Extract a chapter/section outline from the PDF's embedded TOC/bookmarks.

    Falls back to heading-based structure detection (pdf.PDFDocument.detect_structure)
    when the PDF has no embedded TOC, so textbooks without bookmarks still get an
    (approximate) outline rather than none at all.
    """
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()
    page_count = len(doc)
    doc.close()

    if toc:
        entries = [
            OutlineEntry(title=title, level=level, start_page=max(1, page), end_page=max(1, page))
            for level, title, page in toc
        ]
    else:
        logger.info(f"{pdf_path} has no embedded TOC, falling back to heading detection for outline")
        with PDFDocument(Path(pdf_path)) as pdf_doc:
            structure = pdf_doc.detect_structure()
        entries = [
            OutlineEntry(title=h["text"], level=h["level"], start_page=h["page"], end_page=h["page"])
            for h in structure.get("headings", [])
            if h["level"] <= 2
        ]

    # Fill in end_page: up to (but not including) the next entry at the same or
    # shallower level, or the end of the document if this is the last such entry.
    for i, entry in enumerate(entries):
        next_start = page_count + 1
        for other in entries[i + 1:]:
            if other.level <= entry.level:
                next_start = other.start_page
                break
        entry.end_page = max(entry.start_page, next_start - 1)

    logger.info(f"Extracted {len(entries)} outline entries from {pdf_path}")
    return entries


def build_coverage_report(outline: List[OutlineEntry], cards: List[Any]) -> Dict[str, Any]:
    """Cross-check which outline sections got at least one generated card.

    A card is considered to cover an outline entry when the card's page range
    overlaps the entry's page range. Returns a summary plus the list of
    zero-card sections so gaps are visible rather than silently dropped.
    """
    covered_counts = {i: 0 for i in range(len(outline))}

    for card in cards:
        card_start = getattr(card, "page_start", 0) or 0
        card_end = getattr(card, "page_end", card_start) or card_start
        for i, entry in enumerate(outline):
            if card_start <= entry.end_page and card_end >= entry.start_page:
                covered_counts[i] += 1

    sections = []
    uncovered = []
    for i, entry in enumerate(outline):
        count = covered_counts[i]
        sections.append({
            "title": entry.title,
            "level": entry.level,
            "start_page": entry.start_page,
            "end_page": entry.end_page,
            "card_count": count,
        })
        if count == 0:
            uncovered.append(entry.title)

    total = len(outline)
    covered = total - len(uncovered)

    return {
        "total_sections": total,
        "covered_sections": covered,
        "coverage_ratio": (covered / total) if total else 1.0,
        "uncovered_sections": uncovered,
        "sections": sections,
    }
