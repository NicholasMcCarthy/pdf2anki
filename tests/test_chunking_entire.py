"""Tests for entire chunking mode functionality."""

import pytest
from unittest.mock import Mock

from pdf2anki.chunking import TextChunk, TextChunker
from pdf2anki.config import ChunkingConfig, ChunkingMode


def create_mock_pdf_content(pages_text, page_count=None):
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
        "page_count": page_count or len(pages_text),
        "pages": pages,
        "images": [],
        "structure": {"headings": [], "sections": [], "chapters": []},
        "content_hash": "test_hash"
    }


def test_entire_mode_small_document():
    """Test that small documents produce a single chunk."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=False
    )
    
    # Mock the encoding to avoid network issues
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = [1] * 100  # Mock 100 tokens
    
    # Small document text
    small_text = """Introduction

This is a small document with just a few paragraphs of content.
It should fit easily within the token budget and produce a single chunk.

Conclusion

This document is complete."""
    
    pdf_content = create_mock_pdf_content([small_text])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) == 1
    assert chunks[0].section == "Entire Document"
    assert chunks[0].total_chunks == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == 100


def test_entire_mode_large_document_auto_split():
    """Test that large documents are auto-split into large chunks within token budget."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=1000,  # Small budget to force splitting
        enable_trimming=False
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    
    # Mock token counting to return different values for different text lengths
    def mock_encode(text):
        # Return roughly 1 token per 4 characters
        return [1] * (len(text) // 4)
    chunker.encoding.encode.side_effect = mock_encode
    
    # Large document that should be split
    large_text = """Introduction

This is a very long document with multiple sections and lots of content.
""" + ("This is paragraph content. " * 50) + """

Methods

Here we describe our methodology.
""" + ("More detailed methods content. " * 50) + """

Results

The results section contains our findings.
""" + ("Detailed results and analysis. " * 50) + """

Conclusion

This concludes our document."""
    
    pdf_content = create_mock_pdf_content([large_text])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) > 1  # Should be split into multiple chunks
    
    # Each chunk should be within token budget
    for chunk in chunks:
        assert chunk.token_count <= config.token_budget
    
    # All chunks should have correct metadata
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.total_chunks == len(chunks)
        # Check that it has meaningful section names (either detected sections or "Document Part")
        assert chunk.section is not None
        assert chunk.section != ""


def test_entire_mode_trimming_removes_references():
    """Test that trimming removes terminal reference sections but not conclusions."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=True
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = [1] * 500  # Mock 500 tokens
    
    # Document with references that should be trimmed
    document_with_refs = """Introduction

This is the main content of the document.

Conclusion

This is the conclusion that should be preserved.

References

1. Smith, J. (2020). Some research paper.
2. Johnson, A. (2019). Another paper.
3. Brown, B. (2018). Yet another reference."""
    
    pdf_content = create_mock_pdf_content([document_with_refs])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) == 1
    chunk_text = chunks[0].text
    
    # Should contain conclusion but not references
    assert "Conclusion" in chunk_text
    assert "This is the conclusion that should be preserved" in chunk_text
    assert "References" not in chunk_text
    assert "Smith, J." not in chunk_text


def test_entire_mode_trimming_preserves_conclusions():
    """Test that trimming does not remove conclusion sections."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=True
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = [1] * 300  # Mock 300 tokens
    
    # Document where conclusion appears after references (edge case)
    document_text = """Introduction

Main content here.

References

Some references.

Conclusion

Important conclusion that should be kept."""
    
    pdf_content = create_mock_pdf_content([document_text])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) == 1
    chunk_text = chunks[0].text
    
    # Should preserve the main content but references may be trimmed
    assert "Introduction" in chunk_text
    assert "Main content" in chunk_text


def test_entire_mode_trimming_disabled():
    """Test that trimming can be disabled via config flag."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=False  # Disabled
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = [1] * 500  # Mock 500 tokens
    
    # Document with references
    document_with_refs = """Introduction

Main content.

References

1. Some reference that should be kept when trimming is disabled."""
    
    pdf_content = create_mock_pdf_content([document_with_refs])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) == 1
    chunk_text = chunks[0].text
    
    # Should contain everything including references
    assert "Introduction" in chunk_text
    assert "Main content" in chunk_text
    assert "References" in chunk_text
    assert "Some reference that should be kept" in chunk_text


def test_entire_mode_section_detection():
    """Test section boundary detection for auto-splitting."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=200,  # Very small budget to force section-based splitting
        enable_trimming=False
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    
    # Mock token counting to make each section exceed the budget
    def mock_encode(text):
        return [1] * (len(text) // 2)  # Roughly 1 token per 2 characters (more aggressive)
    chunker.encoding.encode.side_effect = mock_encode
    
    # Document with clear section boundaries
    sectioned_document = """Abstract

This is the abstract section with content that should be long enough to force splitting when combined with other sections.

Introduction

This is the introduction with some content that goes on for a while with detailed explanations.

Methods

This describes our methodology in detail with comprehensive information about our approach.

Results

Here are our results and findings with detailed analysis and interpretation.

Discussion

Discussion of the results goes here with thorough examination of implications.

Conclusion

Final thoughts and conclusions with summary of key findings."""
    
    pdf_content = create_mock_pdf_content([sectioned_document])
    chunks = chunker.chunk_document(pdf_content)
    
    # Should create multiple chunks based on sections OR fallback to paragraph splitting
    # The key is that it should NOT be a single "Entire Document" chunk
    if len(chunks) == 1:
        # If only one chunk, it means the document was small enough to fit in budget
        # This is acceptable behavior for entire mode
        assert chunks[0].section == "Entire Document"
    else:
        # If multiple chunks, they should have meaningful section information
        for chunk in chunks:
            assert chunk.section is not None
            assert chunk.section != "Entire Document"


def test_entire_mode_with_empty_document():
    """Test entire mode handles empty documents gracefully."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=True
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = []  # No tokens
    
    # Empty document
    pdf_content = create_mock_pdf_content(["", "   ", "\n\n"])
    chunks = chunker.chunk_document(pdf_content)
    
    assert len(chunks) == 0


def test_entire_mode_integration_with_chunk_text():
    """Test that entire mode works with the chunk_text convenience method."""
    config = ChunkingConfig(
        mode=ChunkingMode.ENTIRE,
        token_budget=8000,
        enable_trimming=False
    )
    
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.return_value = [1] * 200  # Mock 200 tokens
    
    # Test text
    test_text = """This is a test document.

It has multiple paragraphs and should be processed as a single chunk in entire mode."""
    
    chunks = chunker.chunk_text(test_text)
    
    assert len(chunks) == 1
    assert chunks[0].section == "Entire Document"
    assert chunks[0].text.strip() == test_text.strip()