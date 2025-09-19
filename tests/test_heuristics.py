"""Tests for document heuristics and analysis."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.pdf2anki.heuristics import DocumentAnalyzer, get_heuristic_defaults
from src.pdf2anki.config import DocumentMetadata, DocumentType


class TestDocumentAnalyzer:
    """Test document analysis and heuristics."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = DocumentAnalyzer()
        assert analyzer is not None
    
    @patch('fitz.open')
    def test_analyze_document_mock(self, mock_fitz_open):
        """Test document analysis with mocked PyMuPDF."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=10)  # 10 pages
        mock_doc.get_toc = Mock(return_value=[])  # No TOC
        mock_doc.close = Mock()
        
        # Mock page for text extraction
        mock_page = Mock()
        mock_page.get_text = Mock(return_value="Abstract: This is a test paper with DOI: 10.1234/test")
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        
        mock_fitz_open.return_value = mock_doc
        
        analyzer = DocumentAnalyzer()
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            metadata = analyzer.analyze_document("test.pdf")
        
        assert metadata.page_count == 10
        assert metadata.abstract_present is True
        assert metadata.has_doi is True
        assert metadata.toc_present is False
        assert metadata.file_size == 1024
    
    def test_classify_research_paper(self):
        """Test research paper classification."""
        analyzer = DocumentAnalyzer()
        
        doc_type = analyzer._classify_document_type(
            toc_present=False,
            chapters_detected=False,
            abstract_present=True,
            references_present=True,
            has_doi=True,
            page_count=12
        )
        
        assert doc_type == DocumentType.RESEARCH_PAPER
    
    def test_classify_textbook(self):
        """Test textbook classification."""
        analyzer = DocumentAnalyzer()
        
        doc_type = analyzer._classify_document_type(
            toc_present=True,
            chapters_detected=True,
            abstract_present=False,
            references_present=True,
            has_doi=False,
            page_count=150
        )
        
        assert doc_type == DocumentType.TEXTBOOK
    
    def test_classify_unknown(self):
        """Test unknown document classification."""
        analyzer = DocumentAnalyzer()
        
        doc_type = analyzer._classify_document_type(
            toc_present=False,
            chapters_detected=False,
            abstract_present=False,
            references_present=False,
            has_doi=False,
            page_count=5
        )
        
        assert doc_type == DocumentType.UNKNOWN
    
    def test_detect_abstract(self):
        """Test abstract detection."""
        analyzer = DocumentAnalyzer()
        
        text_with_abstract = "Abstract: This paper presents..."
        assert analyzer._detect_abstract(text_with_abstract) is True
        
        text_without_abstract = "Introduction: This document..."
        assert analyzer._detect_abstract(text_without_abstract) is False
    
    def test_detect_doi(self):
        """Test DOI detection."""
        analyzer = DocumentAnalyzer()
        
        text_with_doi = "DOI: 10.1234/example.paper.2024"
        assert analyzer._detect_doi(text_with_doi) is True
        
        text_without_doi = "No DOI in this text"
        assert analyzer._detect_doi(text_without_doi) is False
    
    def test_detect_chapters(self):
        """Test chapter detection."""
        analyzer = DocumentAnalyzer()
        
        # Mock document for testing
        mock_doc = Mock()
        
        text_with_chapters = "Chapter 1: Introduction\nChapter 2: Methods"
        assert analyzer._detect_chapters(mock_doc, text_with_chapters) is True
        
        text_without_chapters = "This text has no chapter structure"
        assert analyzer._detect_chapters(mock_doc, text_without_chapters) is False


def test_get_heuristic_defaults():
    """Test heuristic defaults generation."""
    # Test research paper defaults
    research_metadata = DocumentMetadata(
        page_count=15,
        doc_type=DocumentType.RESEARCH_PAPER,
        abstract_present=True,
        has_doi=True
    )
    
    defaults = get_heuristic_defaults(research_metadata)
    
    assert defaults["chunking_mode"] == "sections"
    assert defaults["tokens_per_chunk"] == 1500  # Smaller for research papers
    assert "key_points" in defaults["strategies"]
    
    # Test textbook defaults
    textbook_metadata = DocumentMetadata(
        page_count=100,
        doc_type=DocumentType.TEXTBOOK,
        chapters_detected=True,
        toc_present=True
    )
    
    defaults = get_heuristic_defaults(textbook_metadata)
    
    assert defaults["chunking_mode"] == "smart"
    assert defaults["tokens_per_chunk"] == 2500  # Larger for textbooks
    assert "figure_based" in defaults["strategies"]
    
    # Test adjustment for large documents
    large_doc_metadata = DocumentMetadata(
        page_count=200,
        doc_type=DocumentType.UNKNOWN
    )
    
    defaults = get_heuristic_defaults(large_doc_metadata)
    assert defaults["tokens_per_chunk"] >= 3000  # Should be increased for large docs