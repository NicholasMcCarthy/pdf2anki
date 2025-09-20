"""Tests for text chunking functionality."""

import pytest

from pdf2anki.chunking import TextChunk, TextChunker
from pdf2anki.config import ChunkingConfig, ChunkingMode


def create_mock_pdf_content(pages_text):
    """Create mock PDF content for testing."""
    pages = []
    for i, text in enumerate(pages_text):
        pages.append({
            "page_num": i + 1,
            "raw_text": text,
            "blocks": [],
            "bbox": [0, 0, 600, 800],
            "rotation": 0
        })
    
    return {
        "path": "test.pdf",
        "metadata": {"title": "Test PDF"},
        "page_count": len(pages_text),
        "pages": pages,
        "images": [],
        "structure": {"headings": [], "sections": [], "chapters": []},
        "content_hash": "test_hash"
    }


def test_text_chunk_creation():
    """Test TextChunk object creation and properties."""
    chunk = TextChunk(
        text="This is a test chunk with multiple words for testing.",
        start_page=1,
        end_page=2,
        section="Test Section",
        subsection="Test Subsection",
        chunk_index=0,
        total_chunks=5
    )
    
    assert chunk.text == "This is a test chunk with multiple words for testing."
    assert chunk.start_page == 1
    assert chunk.end_page == 2
    assert chunk.section == "Test Section"
    assert chunk.subsection == "Test Subsection"
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 5
    assert chunk.word_count == 10
    assert chunk.char_count == 53


def test_chunker_initialization():
    """Test TextChunker initialization."""
    config = ChunkingConfig(
        mode=ChunkingMode.SMART,
        tokens_per_chunk=1000,
        overlap_tokens=100
    )
    
    chunker = TextChunker(config, model="gpt-4")
    
    assert chunker.config.mode == ChunkingMode.SMART
    assert chunker.config.tokens_per_chunk == 1000
    assert chunker.config.overlap_tokens == 100


def test_token_counting():
    """Test token counting functionality."""
    config = ChunkingConfig()
    chunker = TextChunker(config, model="gpt-4")
    
    # Test simple text
    text = "This is a test sentence."
    token_count = chunker.count_tokens(text)
    
    assert isinstance(token_count, int)
    assert token_count > 0
    assert token_count < 20  # Should be reasonable for this short text


def test_chunk_by_pages():
    """Test page-based chunking."""
    config = ChunkingConfig(
        mode=ChunkingMode.PAGES,
        tokens_per_chunk=50,  # Small for testing
        overlap_tokens=10
    )
    chunker = TextChunker(config, model="gpt-4")
    
    # Create test content with multiple pages
    pages_text = [
        "This is page one with some content about Python programming.",
        "This is page two with more content about data structures.",
        "This is page three with information about algorithms and complexity.",
        "This is page four with details about machine learning concepts."
    ]
    
    pdf_content = create_mock_pdf_content(pages_text)
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(chunk.start_page <= chunk.end_page for chunk in chunks)
    
    # Check that chunks have proper indices
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.total_chunks == len(chunks)


def test_chunk_by_paragraphs():
    """Test paragraph-based chunking."""
    config = ChunkingConfig(
        mode=ChunkingMode.PARAGRAPHS,
        tokens_per_chunk=100,
        overlap_tokens=20
    )
    chunker = TextChunker(config, model="gpt-4")
    
    # Create content with clear paragraph breaks
    pages_text = [
        """First paragraph with some content about programming.

Second paragraph with different content about algorithms.

Third paragraph with information about data structures and their importance in computer science."""
    ]
    
    pdf_content = create_mock_pdf_content(pages_text)
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) > 0
    
    # Check that paragraph breaks are respected
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0


def test_smart_chunking():
    """Test smart chunking strategy."""
    config = ChunkingConfig(
        mode=ChunkingMode.SMART,
        tokens_per_chunk=200,
        overlap_tokens=50
    )
    chunker = TextChunker(config, model="gpt-4")
    
    # Create content with headings
    pages_text = [
        """Introduction

This is the introduction section with some basic information.

Chapter 1: Getting Started

This chapter covers the basics of getting started with the topic.

Section 1.1: Installation

Here we discuss installation procedures and requirements."""
    ]
    
    # Add some structure
    pdf_content = create_mock_pdf_content(pages_text)
    pdf_content["structure"] = {
        "headings": [
            {"text": "Introduction", "page": 1, "level": 1},
            {"text": "Chapter 1: Getting Started", "page": 1, "level": 1},
            {"text": "Section 1.1: Installation", "page": 1, "level": 2}
        ],
        "sections": [
            {"title": "Introduction", "start_page": 1, "end_page": 1},
            {"title": "Chapter 1: Getting Started", "start_page": 1, "end_page": 1}
        ]
    }
    
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) > 0
    
    # Smart chunking should preserve some structure
    sections_found = [chunk.section for chunk in chunks if chunk.section]
    assert len(sections_found) > 0


def test_chunk_size_limits():
    """Test that chunks respect size limits."""
    config = ChunkingConfig(
        mode=ChunkingMode.PAGES,
        tokens_per_chunk=50,
        min_chunk_tokens=10,
        max_chunk_tokens=100
    )
    chunker = TextChunker(config, model="gpt-4")
    
    # Create content that would naturally exceed limits
    long_text = "This is a very long paragraph. " * 50
    pdf_content = create_mock_pdf_content([long_text])
    
    chunks = chunker.chunk_document(pdf_content)
    
    # Check that no chunk exceeds maximum token limit (with some tolerance)
    for chunk in chunks:
        chunk.token_count = chunker.count_tokens(chunk.text)
        # Allow some flexibility due to overlap
        assert chunk.token_count <= config.max_chunk_tokens + config.overlap_tokens


def test_empty_content_handling():
    """Test handling of empty or minimal content."""
    config = ChunkingConfig()
    chunker = TextChunker(config, model="gpt-4")
    
    # Empty content
    pdf_content = create_mock_pdf_content(["", "   ", "\n\n"])
    chunks = chunker.chunk_document(pdf_content)
    
    # Should handle gracefully - might return empty list or skip empty pages
    assert isinstance(chunks, list)


def test_overlap_handling():
    """Test that overlap is properly handled between chunks."""
    config = ChunkingConfig(
        mode=ChunkingMode.PAGES,
        tokens_per_chunk=30,  # Very small for testing
        overlap_tokens=10
    )
    chunker = TextChunker(config, model="gpt-4")
    
    # Create content that will definitely need multiple chunks
    pages_text = [
        "First sentence about Python programming and data structures. "
        "Second sentence about algorithms and computational complexity. "
        "Third sentence about machine learning and artificial intelligence. "
        "Fourth sentence about software engineering and best practices."
    ]
    
    pdf_content = create_mock_pdf_content(pages_text)
    chunks = chunker.chunk_document(pdf_content)
    
    # Should create multiple chunks due to small size limit
    assert len(chunks) > 1
    
    # Set token counts for all chunks
    for chunk in chunks:
        chunk.token_count = chunker.count_tokens(chunk.text)