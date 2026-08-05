"""Readwise/Obsidian-plugin markdown export ingestion.

Handles the format produced by the Readwise Obsidian plugin (and similarly by
Readwise Reader's "Export to Markdown"): YAML frontmatter (title/author/url/
category/tags), an H1 title line, and a "## Highlights" section containing one
Obsidian callout block per highlight - `> [!info]  [icon](readwise-deep-link)`
followed by the highlighted text, which may itself contain markdown links or
footnote-style citations.

Only a single-highlight sample was available while building this, so the
block-splitting in _parse_highlights() is deliberately tolerant of 0, 1, or
many highlight blocks and doesn't hard-fail on unrecognized callout types -
a non-"info" callout (e.g. a user's own annotation on a highlight) is folded
into the preceding highlight's `note` rather than treated as a parse error.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .chunking import TextChunk
from .config import Strategy as StrategyConfig
from .strategies.base import FlashcardData
from .strategies.readwise_highlight import ReadwiseHighlightStrategy

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_HEADING_LINK_RE = re.compile(r"\s*\[[^\]]*\]\([^)]*\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CALLOUT_RE = re.compile(r"^\[!(\w+)\]\s*(.*)$")
_LINK_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)]+)\)")


@dataclass
class ReadwiseHighlight:
    """A single highlighted passage, with any user note folded in."""
    text: str
    note: Optional[str] = None
    source_url: Optional[str] = None
    callout_type: str = "info"


@dataclass
class ReadwiseDocument:
    """A parsed Readwise/Obsidian markdown export: one source document (article,
    webpage, book, etc.) and the highlights saved from it."""
    title: str
    highlights: List[ReadwiseHighlight] = field(default_factory=list)
    author: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created: Optional[str] = None
    source_path: Optional[str] = None


def parse_readwise_markdown(path: Path) -> ReadwiseDocument:
    """Parse a Readwise/Obsidian-export markdown file into a ReadwiseDocument."""
    raw = Path(path).read_text(encoding="utf-8")

    frontmatter: Dict[str, Any] = {}
    body = raw
    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter in {path}: {e}")
        body = fm_match.group(2)

    title = frontmatter.get("title")
    if not title:
        h1_match = _H1_RE.search(body)
        if h1_match:
            title = _HEADING_LINK_RE.sub("", h1_match.group(1)).strip()
    if not title:
        title = Path(path).stem

    highlights = _parse_highlights(body)

    tags = frontmatter.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    created = frontmatter.get("created")

    doc = ReadwiseDocument(
        title=str(title),
        highlights=highlights,
        author=frontmatter.get("author"),
        url=frontmatter.get("url"),
        category=frontmatter.get("category"),
        tags=[str(t) for t in tags],
        created=str(created) if created else None,
        source_path=str(path),
    )

    logger.info(f"Parsed {len(highlights)} highlights from {path}")
    return doc


def _parse_highlights(body: str) -> List[ReadwiseHighlight]:
    """Extract highlight callout blocks from the (post-frontmatter) document body."""
    lines = body.splitlines()

    # Narrow to a "## Highlights"-ish section if one exists; otherwise scan the
    # whole body, since not every export necessarily uses that exact heading.
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if heading and "highlight" in heading.group(2).lower():
            start = i + 1
            level = len(heading.group(1))
            for j in range(start, len(lines)):
                next_heading = _HEADING_RE.match(lines[j])
                if next_heading and len(next_heading.group(1)) <= level:
                    end = j
                    break
            break

    highlights: List[ReadwiseHighlight] = []
    current_lines: List[str] = []
    current_type = "info"
    current_source_url: Optional[str] = None

    def flush() -> None:
        nonlocal current_lines, current_type, current_source_url
        text = "\n".join(l for l in current_lines if l.strip()).strip()
        current_lines = []
        if not text:
            return
        if current_type == "info" or not highlights:
            highlights.append(ReadwiseHighlight(
                text=text, source_url=current_source_url, callout_type=current_type
            ))
        else:
            # A non-"info" callout right after a highlight is treated as the
            # reader's own note on it, not a second highlight.
            prev = highlights[-1]
            prev.note = f"{prev.note}\n{text}" if prev.note else text

    for raw_line in lines[start:end]:
        line = raw_line.rstrip()

        if not line.strip():
            if current_lines:
                flush()
            continue

        if not line.lstrip().startswith(">"):
            if current_lines:
                flush()
            continue

        content = line.lstrip()[1:].strip()  # drop the leading "> "
        callout = _CALLOUT_RE.match(content)
        if callout:
            if current_lines:
                flush()
            current_type = callout.group(1).lower()
            remainder = callout.group(2).strip()
            url_match = _LINK_URL_RE.search(remainder)
            current_source_url = url_match.group(1) if url_match else None
            continue

        current_lines.append(content)

    if current_lines:
        flush()

    return highlights


def _build_other_highlights_context(doc: ReadwiseDocument, exclude_index: int, max_chars: int = 1500) -> str:
    """Join the OTHER highlights in this document (excluding the one currently
    being turned into cards) into a short bulleted list, for use as background
    context in the per-highlight prompt - see process_readwise_document(). This
    lets the LLM understand the surrounding argument/narrative of a multi-highlight
    article without grounding cards in anything that wasn't itself highlighted.
    """
    parts: List[str] = []
    total_len = 0
    for i, highlight in enumerate(doc.highlights):
        if i == exclude_index:
            continue
        snippet = highlight.text.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "..."
        line = f"- {snippet}"
        if total_len + len(line) > max_chars:
            break
        parts.append(line)
        total_len += len(line)
    return "\n".join(parts)


def readwise_document_to_chunks(doc: ReadwiseDocument) -> List[TextChunk]:
    """Turn each highlight into its own chunk so per-highlight metadata (note)
    survives into card generation - each highlight is treated as its own
    atomic unit rather than merged with others in the same document.
    """
    chunks = []
    for i, highlight in enumerate(doc.highlights):
        text = highlight.text
        if highlight.note:
            text += f"\n\n[NOTE]: {highlight.note}"

        chunks.append(TextChunk(
            text=text,
            start_page=0,
            end_page=0,
            section=doc.title,
            chunk_index=i,
            total_chunks=len(doc.highlights),
        ))
    return chunks


def process_readwise_document(
    path: Path,
    llm_provider,
    prompt_manager,
    strategy_config: Optional[StrategyConfig] = None,
    max_cards_per_highlight: int = 2,
) -> List[FlashcardData]:
    """Parse a Readwise markdown export and generate one flashcard-call per
    highlight via ReadwiseHighlightStrategy, returning the combined cards.
    """
    if strategy_config is None:
        strategy_config = StrategyConfig(
            name="readwise_highlight", prompt="readwise_highlight", note="basic", enabled=True
        )

    strategy = ReadwiseHighlightStrategy(
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        strategy_config=strategy_config,
        strategy_name="readwise_highlight",
    )

    doc = parse_readwise_markdown(path)
    if not doc.highlights:
        logger.info(f"No highlights found in {path}, skipping")
        return []

    chunks = readwise_document_to_chunks(doc)

    pdf_metadata = {
        "title": doc.title,
        "author": doc.author or "",
        "path": doc.source_path,
        "source_url": doc.url or "",
        "category": doc.category,
        "extra_tags": doc.tags,
    }

    all_cards: List[FlashcardData] = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = dict(pdf_metadata)
        if len(doc.highlights) > 1:
            chunk_metadata["other_highlights"] = _build_other_highlights_context(doc, exclude_index=i)
        cards = strategy.generate_cards(chunk, chunk_metadata, max_cards=max_cards_per_highlight)
        all_cards.extend(strategy.deduplicate_cards(cards))

    logger.info(f"Generated {len(all_cards)} cards from {len(chunks)} highlights in {path}")
    return all_cards
