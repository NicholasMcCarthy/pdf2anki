"""Text chunking with heading-aware strategies."""

import logging
import re
from typing import Dict, List, Optional, Tuple

import tiktoken

from .config import ChunkingConfig, ChunkingMode

logger = logging.getLogger(__name__)


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
    
    def __repr__(self) -> str:
        return f"TextChunk(pages={self.start_page}-{self.end_page}, tokens={self.token_count}, section='{self.section}')"


class TextChunker:
    """Handles various text chunking strategies."""
    
    def __init__(self, config: ChunkingConfig, model: str = "gpt-4"):
        self.config = config
        self.encoding = tiktoken.encoding_for_model(model)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the specified encoding."""
        return len(self.encoding.encode(text))
    
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
    
    def _split_large_text(self, text: str, start_page: int, end_page: int, section_title: str = None) -> List[TextChunk]:
        """Split large text into smaller chunks with overlap."""
        chunks = []
        
        if self.count_tokens(text) <= self.config.tokens_per_chunk:
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
            
            if token_count > self.config.tokens_per_chunk and current_chunk:
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
                
                # Start new chunk with overlap
                overlap_paras = chunk_paragraphs[-2:] if len(chunk_paragraphs) >= 2 else chunk_paragraphs
                overlap_text = "\n\n".join(overlap_paras)
                
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
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
            if len(para) > 1000:
                sentences = re.split(r'[.!?]+\s+', para)
                current_group = ""
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    potential_group = current_group + ". " + sentence if current_group else sentence
                    
                    if len(potential_group) > 800 and current_group:
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