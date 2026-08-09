"""Text chunking with heading-aware strategies."""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import tiktoken

from .config import ChunkingConfig, ChunkingMode

logger = logging.getLogger(__name__)

# Headings that mark the start of a terminal, never-flashcard-worthy section of
# an academic paper - References/Bibliography/Works Cited, Appendix,
# Supplementary Material, Acknowledgments/Funding. Matched as a whole line
# (MULTILINE, so `^`/`$` anchor to line boundaries, including the very start
# of a page's text - a heading is very commonly the first line of a fresh
# page, which a "must be preceded by \n" pattern would miss). Shared by
# _trim_terminal_sections() (entire-mode, operates on one concatenated string)
# and _strip_terminal_sections_from_pages() (highlights-mode, operates at page
# granularity so a paper's page_count for the single-call-vs-chunk decision
# stays meaningful) - keep both in sync by editing only here.
TERMINAL_SECTION_HEADING_RE = re.compile(
    r"^[ \t]*("
    r"References|Bibliography|Works Cited|"
    r"Appendix(?:\s+[A-Z0-9]+)?|"
    r"Supplementary Material(?:s)?|"
    r"Acknowledge?ments?|Funding"
    r")[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class TextChunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(
        self,
        text: str,
        start_page: int,
        end_page: int,
        section: Optional[str] = None,
        subsection: Optional[str] = None,
        chunk_index: int = 0,
        total_chunks: int = 1,
        chunk_type: str = "text",
        highlights: Optional[List[Dict]] = None,
    ):
        self.text = text
        self.start_page = start_page
        self.end_page = end_page
        self.section = section
        self.subsection = subsection
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.token_count = 0
        self.word_count = len(text.split())
        self.char_count = len(text)
        # chunk_type/highlights: populated for highlight-priority chunks (see
        # TextChunker._chunk_by_highlights) so strategies can attach highlight
        # screenshots to generated cards. Unused (defaults) for every other mode.
        self.chunk_type = chunk_type
        self.highlights = highlights
    
    def __repr__(self) -> str:
        return f"TextChunk(pages={self.start_page}-{self.end_page}, tokens={self.token_count}, section='{self.section}')"


class TextChunker:
    """Handles various text chunking strategies."""
    
    def __init__(self, config: ChunkingConfig, model: str = "gpt-4"):
        self.config = config
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding for {model}: {e}")
            # Fallback to a simple token counter for testing
            self.encoding = None
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the specified encoding."""
        if self.encoding is None:
            # Fallback: rough approximation of 1 token per 4 characters
            return len(text) // 4
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, start_page: int = 1) -> List[TextChunk]:
        """Convenience method to chunk plain text."""
        # Create a minimal pdf_content structure for compatibility
        pdf_content = {
            "pages": [{"page_num": start_page, "raw_text": text}],
            "page_count": 1,
            "structure": {"headings": [], "sections": [], "chapters": []},
        }
        return self.chunk_document(pdf_content)
    
    def chunk_document(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk document based on the configured strategy."""
        if self.config.mode == ChunkingMode.PAGES:
            return self._chunk_by_pages(pdf_content)
        elif self.config.mode == ChunkingMode.SECTIONS:
            return self._chunk_by_sections(pdf_content)
        elif self.config.mode == ChunkingMode.PARAGRAPHS:
            return self._chunk_by_paragraphs(pdf_content)
        elif self.config.mode == ChunkingMode.SMART:
            return self._chunk_smart(pdf_content)
        elif self.config.mode == ChunkingMode.FIGURES:
            return self._chunk_by_figures(pdf_content)
        elif self.config.mode == ChunkingMode.HIGHLIGHTS:
            return self._chunk_by_highlights(pdf_content)
        elif self.config.mode == ChunkingMode.ENTIRE:
            return self._chunk_entire(pdf_content)
        elif self.config.mode == ChunkingMode.OUTLINE:
            return self._chunk_by_outline(pdf_content)
        else:
            raise ValueError(f"Unknown chunking mode: {self.config.mode}")
    
    def _chunk_by_pages(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk text by pages with token limits."""
        chunks = []
        current_text = ""
        current_pages = []
        
        for page_data in pdf_content["pages"]:
            page_text = page_data["raw_text"].strip()
            page_num = page_data["page_num"]
            
            if not page_text:
                continue
            
            # Check if this single page exceeds token limits and needs splitting
            page_token_count = self.count_tokens(page_text)
            if page_token_count > self.config.tokens_per_chunk:
                # Page exceeds preferred chunk size, split it
                page_chunks = self._split_large_text(page_text, page_num, page_num, target_tokens=self.config.tokens_per_chunk)
                chunks.extend(page_chunks)
                continue
            
            # Check if adding this page exceeds token limit
            potential_text = current_text + "\n\n" + page_text if current_text else page_text
            token_count = self.count_tokens(potential_text)
            
            if token_count > self.config.tokens_per_chunk and current_text:
                # Create chunk from accumulated pages
                chunk = TextChunk(
                    text=current_text.strip(),
                    start_page=current_pages[0],
                    end_page=current_pages[-1],
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new chunk
                current_text = page_text
                current_pages = [page_num]
            else:
                # Add page to current chunk
                current_text = potential_text
                current_pages.append(page_num)
        
        # Add final chunk
        if current_text:
            chunk = TextChunk(
                text=current_text.strip(),
                start_page=current_pages[0],
                end_page=current_pages[-1],
                chunk_index=len(chunks),
            )
            chunk.token_count = self.count_tokens(chunk.text)
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks using page-based strategy")
        return chunks
    
    def _chunk_by_sections(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk text by detected sections."""
        chunks = []
        structure = pdf_content.get("structure", {})
        sections = structure.get("sections", [])
        
        if not sections:
            logger.warning("No sections detected, falling back to page-based chunking")
            return self._chunk_by_pages(pdf_content)
        
        for section in sections:
            section_text = self._extract_section_text(pdf_content, section)
            
            if not section_text.strip():
                continue
            
            # If section is too large, split it further
            section_chunks = self._split_large_text(
                section_text,
                start_page=section["start_page"],
                end_page=section["end_page"],
                section_title=section["title"]
            )
            
            chunks.extend(section_chunks)
        
        # Update chunk indices and totals
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks using section-based strategy")
        return chunks
    
    def _chunk_by_paragraphs(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk text by paragraphs with smart grouping."""
        chunks = []
        
        for page_data in pdf_content["pages"]:
            page_text = page_data["raw_text"]
            page_num = page_data["page_num"]
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(page_text)
            
            current_chunk_text = ""
            chunk_start_page = page_num
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                
                # Check if adding this paragraph exceeds token limit
                potential_text = current_chunk_text + "\n\n" + paragraph if current_chunk_text else paragraph
                token_count = self.count_tokens(potential_text)
                
                if token_count > self.config.tokens_per_chunk and current_chunk_text:
                    # Create chunk from accumulated paragraphs
                    chunk = TextChunk(
                        text=current_chunk_text.strip(),
                        start_page=chunk_start_page,
                        end_page=page_num,
                        chunk_index=len(chunks),
                    )
                    chunk.token_count = self.count_tokens(chunk.text)
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk_text = paragraph
                    chunk_start_page = page_num
                else:
                    current_chunk_text = potential_text
            
            # Handle remaining text at end of page
            if current_chunk_text and (
                page_num == pdf_content["pages"][-1]["page_num"] or
                self.count_tokens(current_chunk_text) >= self.config.min_chunk_tokens
            ):
                chunk = TextChunk(
                    text=current_chunk_text.strip(),
                    start_page=chunk_start_page,
                    end_page=page_num,
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                current_chunk_text = ""
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks using paragraph-based strategy")
        return chunks
    
    def _chunk_smart(self, pdf_content: Dict) -> List[TextChunk]:
        """Smart chunking that combines multiple strategies."""
        structure = pdf_content.get("structure", {})
        
        # If we have good section detection, use it
        if structure.get("sections") and len(structure["sections"]) > 2:
            return self._chunk_by_sections(pdf_content)
        
        # Otherwise, use paragraph-based chunking with heading awareness
        chunks = []
        headings = structure.get("headings", [])
        heading_pages = {h["page"] for h in headings}
        
        current_text = ""
        current_pages = []
        current_section = None
        
        for page_data in pdf_content["pages"]:
            page_text = page_data["raw_text"].strip()
            page_num = page_data["page_num"]
            
            if not page_text:
                continue
            
            # Check if this page has a major heading
            page_headings = [h for h in headings if h["page"] == page_num and h["level"] <= 2]
            
            # If we hit a major heading and have accumulated text, create a chunk
            if page_headings and current_text and len(current_pages) > 0:
                chunk = TextChunk(
                    text=current_text.strip(),
                    start_page=current_pages[0],
                    end_page=current_pages[-1],
                    section=current_section,
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new section
                current_text = page_text
                current_pages = [page_num]
                current_section = page_headings[0]["text"] if page_headings else None
                continue
            
            # Add page to current chunk, checking token limits
            potential_text = current_text + "\n\n" + page_text if current_text else page_text
            token_count = self.count_tokens(potential_text)
            
            if token_count > self.config.tokens_per_chunk and current_text:
                # Create chunk from accumulated text
                chunk = TextChunk(
                    text=current_text.strip(),
                    start_page=current_pages[0],
                    end_page=current_pages[-1],
                    section=current_section,
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text, page_text)
                current_text = overlap_text
                current_pages = [page_num]
            else:
                current_text = potential_text
                current_pages.append(page_num)
                
                # Update current section if we encounter a heading
                if page_headings:
                    current_section = page_headings[0]["text"]
        
        # Add final chunk
        if current_text:
            chunk = TextChunk(
                text=current_text.strip(),
                start_page=current_pages[0],
                end_page=current_pages[-1],
                section=current_section,
                chunk_index=len(chunks),
            )
            chunk.token_count = self.count_tokens(chunk.text)
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks using smart strategy")
        return chunks
    
    def _extract_section_text(self, pdf_content: Dict, section: Dict) -> str:
        """Extract text for a specific section."""
        section_text = ""
        start_page = section["start_page"]
        end_page = section["end_page"]
        
        for page_data in pdf_content["pages"]:
            page_num = page_data["page_num"]
            if start_page <= page_num <= end_page:
                section_text += page_data["raw_text"] + "\n\n"
        
        return section_text.strip()
    
    def _split_large_text(self, text: str, start_page: int, end_page: int, section_title: str = None, target_tokens: Optional[int] = None) -> List[TextChunk]:
        """Split large text into smaller chunks with overlap."""
        chunks = []
        
        # Use max_chunk_tokens as the target size for splitting large text (unless specified)
        target_tokens = target_tokens or self.config.max_chunk_tokens
        
        if self.count_tokens(text) <= target_tokens:
            # Text fits in one chunk
            chunk = TextChunk(
                text=text,
                start_page=start_page,
                end_page=end_page,
                section=section_title,
                chunk_index=0,
                total_chunks=1,
            )
            chunk.token_count = self.count_tokens(chunk.text)
            return [chunk]
        
        # Split into paragraphs for better boundaries
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        chunk_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self.count_tokens(potential_chunk)
            
            if token_count > target_tokens and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    start_page=start_page,
                    end_page=end_page,
                    section=section_title,
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new chunk with overlap, but ensure we don't exceed limits
                overlap_paras = chunk_paragraphs[-2:] if len(chunk_paragraphs) >= 2 else chunk_paragraphs
                overlap_text = "\n\n".join(overlap_paras)
                
                # Check if overlap plus new paragraph exceeds target
                potential_new_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                if self.count_tokens(potential_new_chunk) > target_tokens:
                    # Skip overlap if it would cause the new chunk to exceed limits
                    current_chunk = paragraph
                    chunk_paragraphs = [paragraph]
                else:
                    current_chunk = potential_new_chunk
                    chunk_paragraphs = overlap_paras + [paragraph]
            else:
                current_chunk = potential_chunk
                chunk_paragraphs.append(paragraph)
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                text=current_chunk.strip(),
                start_page=start_page,
                end_page=end_page,
                section=section_title,
                chunk_index=len(chunks),
            )
            chunk.token_count = self.count_tokens(chunk.text)
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines, but also handle various paragraph markers
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split very long paragraphs
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is very long, split on sentence boundaries
            if len(para) > 1000 or self.count_tokens(para) > self.config.tokens_per_chunk:
                sentences = re.split(r'[.!?]+\s+', para)
                current_group = ""
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    potential_group = current_group + ". " + sentence if current_group else sentence
                    
                    # Use token count for more accurate splitting
                    if self.count_tokens(potential_group) > self.config.tokens_per_chunk and current_group:
                        result.append(current_group.strip())
                        current_group = sentence
                    else:
                        current_group = potential_group
                
                if current_group:
                    result.append(current_group.strip())
            else:
                result.append(para)
        
        return result
    
    def _get_overlap_text(self, current_text: str, new_text: str) -> str:
        """Get overlap text for chunk boundaries."""
        if not self.config.overlap_tokens:
            return new_text
        
        # Get last portion of current text for overlap
        words = current_text.split()
        overlap_words = words[-min(len(words), self.config.overlap_tokens // 4):]  # Rough approximation
        overlap_text = " ".join(overlap_words)
        
        return overlap_text + "\n\n" + new_text
    
    def _chunk_by_figures(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk text by figures and their associated content."""
        # TODO: Implement figure-based chunking
        # This should:
        # 1. Identify figures/images in the PDF
        # 2. Extract surrounding text context for each figure
        # 3. Create chunks that combine figure references with relevant text
        # 4. Include figure metadata (captions, page numbers, etc.)
        
        logger.info("Figure-based chunking not yet implemented, falling back to smart chunking")
        return self._chunk_smart(pdf_content)
    
    def _strip_terminal_sections_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """Remove References/Bibliography/Works Cited/Appendix/Supplementary
        Material/Acknowledgments (and everything after) from a page list, at
        page granularity - so academic-paper text never sends citation lists
        or appendices to the LLM, and a paper's real page_count (used for the
        single-call-vs-chunk size decision) isn't inflated by trailing matter
        nobody wants cards from.

        Scans pages in order; the first page whose raw_text contains a
        terminal-section heading is truncated at that heading (keeping any
        real content before it on that page) and every later page is dropped
        entirely. Returns a new list of page dicts (shallow copies) - the
        input list/dicts are never mutated.
        """
        result = []
        for page in pages:
            text = page.get("raw_text", "") or ""
            match = TERMINAL_SECTION_HEADING_RE.search(text)

            if match:
                truncated_text = text[:match.start()].strip()
                if truncated_text:
                    truncated_page = dict(page)
                    truncated_page["raw_text"] = truncated_text
                    result.append(truncated_page)
                logger.info(
                    f"Stopped academic-paper text at page {page.get('page_num')} "
                    f"(found '{match.group(1)}') - excluding it onward"
                )
                return result

            result.append(page)

        return result

    def _build_highlight_markers(self, annotations: List[Dict], start_index: int = 1) -> List[str]:
        """Render numbered [HIGHLIGHT N]/[NOTE] marker blocks for a list of
        annotation records, in the shared format both the single-call and
        per-chunk highlight paths use. Numbering starts at `start_index` and
        follows `annotations`' order - callers must keep that order
        consistent with `chunk.highlights` so a strategy's LLM response can
        reference "highlight N" (e.g. to pick that highlight's screenshot
        for a specific card - see strategies/highlight_priority.py) and have
        it map unambiguously back to `chunk.highlights[N - 1]`."""
        blocks = []
        for i, ann in enumerate(annotations, start=start_index):
            marker = f"[HIGHLIGHT {i}"
            if ann.get("author"):
                marker += f" - {ann['author']}"
            marker += f"]: {ann['text']}"
            if ann.get("content"):
                marker += "\n[NOTE]: " + ann["content"]
            blocks.append(marker)
        return blocks

    def _build_single_paper_chunk(self, pages: List[Dict], annotations: List[Dict]) -> TextChunk:
        """Build one chunk covering an entire (already reference-stripped)
        short paper, for a single LLM call instead of one call per page/
        highlight. Highlight markers are inserted inline at each page's
        position (not extracted into a separate block) so a reader-highlighted
        passage still appears exactly where it occurs in the paper's flow."""
        # Global 1-based index matching `annotations`' order (== the order
        # chunk.highlights ends up in below) - kept even though markers get
        # interleaved page-by-page, so "[HIGHLIGHT N]" numbering in the
        # rendered text is unambiguous regardless of which page N appears on.
        numbered = list(enumerate(annotations, start=1))
        by_page: Dict[int, List[tuple]] = defaultdict(list)
        for i, ann in numbered:
            by_page[ann["page_num"]].append((i, ann))

        parts = []
        for page in pages:
            page_num = page.get("page_num")
            for i, ann in by_page.get(page_num, []):
                parts.extend(self._build_highlight_markers([ann], start_index=i))

            page_text = (page.get("raw_text") or "").strip()
            if page_text:
                parts.append(f"[PAGE {page_num}]:\n{page_text}")

        chunk = TextChunk(
            text="\n\n".join(parts),
            start_page=pages[0]["page_num"] if pages else 1,
            end_page=pages[-1]["page_num"] if pages else 1,
            chunk_index=0,
            total_chunks=1,
            chunk_type="highlight" if annotations else "text",
            highlights=annotations or None,
        )
        chunk.token_count = self.count_tokens(chunk.text)
        return chunk

    def _chunk_smart_with_highlights(self, pages: List[Dict], annotations: List[Dict], pdf_content: Dict) -> List[TextChunk]:
        """Smart-chunk an (already reference-stripped) long paper, then attach
        whatever highlight annotations fall within each chunk's page range -
        so highlight_priority (which gates on `bool(chunk.highlights)`) still
        applies to exactly the chunks that actually contain a highlight,
        at smart-chunk granularity instead of one call per highlighted page."""
        stripped_content = dict(pdf_content)
        stripped_content["pages"] = pages
        chunks = self._chunk_smart(stripped_content)

        for chunk in chunks:
            chunk_annotations = [
                ann for ann in annotations if chunk.start_page <= ann["page_num"] <= chunk.end_page
            ]
            if not chunk_annotations:
                continue

            chunk.highlights = chunk_annotations
            chunk.chunk_type = "highlight"
            marker_blocks = self._build_highlight_markers(chunk_annotations)
            chunk.text = "\n\n".join(marker_blocks) + f"\n\n[PAGE CONTEXT]:\n{chunk.text}"
            chunk.token_count = self.count_tokens(chunk.text)

        return chunks

    def _chunk_by_highlights(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk an academic paper for the highlight_priority/key_points/
        cloze_definitions strategies, minimizing LLM call count:

        - References/Bibliography/Works Cited/Appendix/Supplementary Material/
          Acknowledgments are stripped first, at page granularity, regardless
          of paper length - never sent to the LLM.
        - Papers at or under `config.single_call_max_pages` (default 12) get
          exactly ONE chunk covering the whole (stripped) paper, with any
          highlight/note markers inserted inline at their real page position
          and every highlight's screenshot attached - one call total instead
          of one call per highlighted page.
        - Longer papers are smart-chunked instead; each resulting chunk picks
          up whatever highlights fall within its own page range.
        - A single-call chunk that turns out to be unexpectedly token-dense
          (still possible even under the page threshold) falls back to the
          smart-chunk path rather than risking one oversized, over-budget call.
        """
        pages = self._strip_terminal_sections_from_pages(pdf_content.get("pages", []))
        retained_page_nums = {p.get("page_num") for p in pages}
        # A highlight physically made inside a stripped section (e.g. someone
        # highlighted a citation in the References) must not surface at all -
        # neither as inline marker text nor as a chunk.highlights entry (which
        # would otherwise still get its screenshot attached to generated cards
        # even though the LLM never saw that highlight's text).
        annotations = [
            a for a in pdf_content.get("annotations", []) if a.get("page_num") in retained_page_nums
        ]
        page_count = pdf_content.get("page_count") or len(pdf_content.get("pages", []))
        max_single_call_pages = getattr(self.config, "single_call_max_pages", 12)

        if page_count <= max_single_call_pages:
            chunk = self._build_single_paper_chunk(pages, annotations)
            if chunk.token_count <= self.config.token_budget:
                logger.info(
                    f"Paper is {page_count} pages (<= {max_single_call_pages}) - "
                    f"sending as a single {chunk.token_count}-token chunk"
                )
                return [chunk]
            logger.info(
                f"Paper is {page_count} pages but the single chunk would be "
                f"{chunk.token_count} tokens (over budget {self.config.token_budget}) - "
                f"falling back to smart chunking instead"
            )

        chunks = self._chunk_smart_with_highlights(pages, annotations, pdf_content)
        logger.info(
            f"Created {len(chunks)} chunks for a {page_count}-page paper "
            f"({len(annotations)} highlight annotations, references/appendix stripped)"
        )
        return chunks
    
    def _chunk_by_outline(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk a textbook using its authoritative PDF outline/TOC (see
        textbook.extract_outline(), which populates pdf_content["outline"])
        instead of heuristic heading detection, so every outline section
        produces at least one chunk - the basis for full-coverage generation.

        Reuses the same section-text extraction and token-bounded splitting as
        _chunk_by_sections(); the only difference is the section list's source
        (authoritative TOC vs. detected headings).
        """
        outline = pdf_content.get("outline", [])
        if not outline:
            logger.info("No outline available for outline-driven chunking, falling back to smart chunking")
            return self._chunk_smart(pdf_content)

        chunks = []
        for entry in outline:
            section = {
                "title": entry["title"],
                "start_page": entry["start_page"],
                "end_page": entry["end_page"],
            }
            section_text = self._extract_section_text(pdf_content, section)

            if not section_text.strip():
                continue

            section_chunks = self._split_large_text(
                section_text,
                start_page=section["start_page"],
                end_page=section["end_page"],
                section_title=section["title"],
            )
            chunks.extend(section_chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using outline-driven strategy ({len(outline)} outline entries)")
        return chunks

    def _chunk_entire(self, pdf_content: Dict) -> List[TextChunk]:
        """Chunk text as entire document with optional trimming and auto-splitting."""
        logger.info("Starting entire document chunking")
        
        # Extract all text from the document
        full_text = self._extract_full_text(pdf_content)
        
        if not full_text.strip():
            logger.warning("No text found in document")
            return []
        
        original_tokens = self.count_tokens(full_text)
        logger.info(f"Original document has {original_tokens} tokens ({len(full_text)} characters)")
        
        # Apply trimming if enabled
        if self.config.enable_trimming:
            logger.info("Trimming enabled - scanning for terminal sections")
            full_text = self._trim_terminal_sections(full_text, pdf_content)
        else:
            logger.info("Trimming disabled - preserving full document content")
        
        # Check if the entire document fits within token budget
        total_tokens = self.count_tokens(full_text)
        logger.info(f"Final document has {total_tokens} tokens, budget is {self.config.token_budget}")
        
        if total_tokens <= self.config.token_budget:
            # Create single chunk
            logger.info("Document fits in single chunk - entire mode successful")
            chunk = TextChunk(
                text=full_text.strip(),
                start_page=1,
                end_page=pdf_content["page_count"],
                section="Entire Document",
                chunk_index=0,
                total_chunks=1,
            )
            chunk.token_count = total_tokens
            return [chunk]
        else:
            # Auto-split into large chunks
            logger.info(f"Document exceeds token budget ({total_tokens} > {self.config.token_budget}), auto-splitting")
            return self._auto_split_large_document(full_text, pdf_content)
    
    def _extract_full_text(self, pdf_content: Dict) -> str:
        """Extract all text from the PDF document."""
        full_text_parts = []
        
        for page_data in pdf_content["pages"]:
            page_text = page_data["raw_text"].strip()
            if page_text:
                full_text_parts.append(page_text)
        
        return "\n\n".join(full_text_parts)
    
    def _trim_terminal_sections(self, text: str, pdf_content: Dict) -> str:
        """Trim terminal sections like References, Bibliography, etc."""
        original_length = len(text)
        original_tokens = self.count_tokens(text)

        trimmed_text = text
        sections_removed = []

        # Only the first (earliest) terminal heading matters - once one is found,
        # everything from there on is dropped in a single cut.
        match = TERMINAL_SECTION_HEADING_RE.search(trimmed_text)
        if match:
            section_start = match.group(1)
            sections_removed.append(section_start)
            trim_boundary_pos = match.start()
            trimmed_text = trimmed_text[:match.start()] + "\n"

            # Log the trim decision with detailed boundary information
            chars_removed = len(text) - len(trimmed_text)
            logger.info(f"TRIM DECISION: Removed terminal section '{section_start}'")
            logger.info(f"TRIM BOUNDARY: Position {trim_boundary_pos} in document")
            logger.info(f"TRIM IMPACT: Removed {chars_removed} characters")
        
        # Ensure we don't remove conclusions
        conclusion_patterns = [
            r'\n\s*(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*\n',
            r'\n\s*(Summary|SUMMARY)\s*\n',
            r'\n\s*(Final Thoughts|FINAL THOUGHTS)\s*\n',
        ]
        
        # If we accidentally trimmed conclusions, warn about it
        for pattern in conclusion_patterns:
            if re.search(pattern, text, re.IGNORECASE) and not re.search(pattern, trimmed_text, re.IGNORECASE):
                logger.warning("WARNING: Conclusion section may have been removed during trimming - consider adjusting patterns")
        
        final_tokens = self.count_tokens(trimmed_text)
        tokens_saved = original_tokens - final_tokens
        
        if sections_removed:
            logger.info(f"TRIM SUMMARY: Removed sections: {', '.join(sections_removed)}")
            logger.info(f"TRIM SUMMARY: Text reduced from {original_length} to {len(trimmed_text)} characters")
            logger.info(f"TRIM SUMMARY: Tokens reduced from {original_tokens} to {final_tokens} (saved {tokens_saved} tokens)")
        else:
            logger.info("TRIM SUMMARY: No terminal sections found to trim")
        
        return trimmed_text.strip()
    
    def _auto_split_large_document(self, text: str, pdf_content: Dict) -> List[TextChunk]:
        """Auto-split large document into sequential chunks within token budget."""
        chunks = []
        
        # Try to split at section boundaries first
        sections = self._detect_section_boundaries(text)
        
        if len(sections) > 1:
            logger.info(f"Found {len(sections)} sections, splitting at section boundaries")
            return self._split_by_detected_sections(sections, pdf_content)
        
        # Fall back to paragraph-based splitting for large chunks
        logger.info("No clear sections found, splitting by paragraphs into large chunks")
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk_text = ""
        chunk_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            potential_text = current_chunk_text + "\n\n" + paragraph if current_chunk_text else paragraph
            token_count = self.count_tokens(potential_text)
            
            if token_count > self.config.token_budget and current_chunk_text:
                # Create chunk
                chunk = TextChunk(
                    text=current_chunk_text.strip(),
                    start_page=1,  # We don't have precise page mapping in entire mode
                    end_page=pdf_content["page_count"],
                    section=f"Document Part {len(chunks) + 1}",
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new chunk with minimal overlap for large chunks
                overlap_paras = chunk_paragraphs[-1:] if chunk_paragraphs else []
                overlap_text = "\n\n".join(overlap_paras)
                current_chunk_text = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                chunk_paragraphs = overlap_paras + [paragraph]
            else:
                current_chunk_text = potential_text
                chunk_paragraphs.append(paragraph)
        
        # Add final chunk
        if current_chunk_text:
            chunk = TextChunk(
                text=current_chunk_text.strip(),
                start_page=1,
                end_page=pdf_content["page_count"],
                section=f"Document Part {len(chunks) + 1}",
                chunk_index=len(chunks),
            )
            chunk.token_count = self.count_tokens(chunk.text)
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.info(f"Created {len(chunks)} large chunks using auto-split in entire mode")
        return chunks
    
    def _detect_section_boundaries(self, text: str) -> List[Dict]:
        """Detect major section boundaries in the text."""
        sections = []
        
        # Common academic paper section patterns
        section_patterns = [
            r'^(Abstract|ABSTRACT)\s*$',
            r'^(Introduction|INTRODUCTION)\s*$',
            r'^(Methods|METHODS|Methodology|METHODOLOGY)\s*$',
            r'^(Results|RESULTS)\s*$',
            r'^(Discussion|DISCUSSION)\s*$',
            r'^(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*$',
            r'^(\d+\.?\s+[A-Z][^.\n]*)\s*$',  # Numbered sections like "1. Introduction"
            r'^([A-Z][A-Z\s]{3,})\s*$',  # ALL CAPS headers
        ]
        
        lines = text.split('\n')
        current_section = {"start": 0, "title": "Beginning", "text": ""}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            for pattern in section_patterns:
                if re.match(pattern, line_stripped, re.MULTILINE):
                    # End current section
                    if current_section["text"].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "start": i,
                        "title": line_stripped,
                        "text": ""
                    }
                    break
            else:
                # Add line to current section
                current_section["text"] += line + "\n"
        
        # Add final section
        if current_section["text"].strip():
            sections.append(current_section)
        
        return sections
    
    def _split_by_detected_sections(self, sections: List[Dict], pdf_content: Dict) -> List[TextChunk]:
        """Split document by detected sections, grouping small sections together."""
        chunks = []
        current_chunk_text = ""
        current_sections = []
        
        for section in sections:
            section_text = section["text"].strip()
            if not section_text:
                continue
            
            potential_text = current_chunk_text + "\n\n" + section_text if current_chunk_text else section_text
            token_count = self.count_tokens(potential_text)
            
            if token_count > self.config.token_budget and current_chunk_text:
                # Create chunk from accumulated sections
                section_titles = [s["title"] for s in current_sections]
                chunk = TextChunk(
                    text=current_chunk_text.strip(),
                    start_page=1,
                    end_page=pdf_content["page_count"],
                    section=" + ".join(section_titles),
                    chunk_index=len(chunks),
                )
                chunk.token_count = self.count_tokens(chunk.text)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_text = section_text
                current_sections = [section]
            else:
                current_chunk_text = potential_text
                current_sections.append(section)
        
        # Add final chunk
        if current_chunk_text:
            section_titles = [s["title"] for s in current_sections]
            chunk = TextChunk(
                text=current_chunk_text.strip(),
                start_page=1,
                end_page=pdf_content["page_count"],
                section=" + ".join(section_titles),
                chunk_index=len(chunks),
            )
            chunk.token_count = self.count_tokens(chunk.text)
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks