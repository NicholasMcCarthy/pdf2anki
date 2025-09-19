"""Heuristics module for PDF document analysis and metadata extraction."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from .config import DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Analyzes PDF documents to extract metadata and heuristics."""
    
    def __init__(self):
        self.logger = logger
    
    def analyze_document(self, pdf_path: str) -> DocumentMetadata:
        """
        Perform fast analysis of PDF document to extract metadata and heuristics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentMetadata object with extracted information
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Basic file information
            page_count = len(doc)
            file_size = Path(pdf_path).stat().st_size
            
            # Extract text from first few pages for analysis
            sample_text = self._extract_sample_text(doc, max_pages=5)
            
            # Detect document structure and features
            toc_present = self._detect_toc(doc)
            chapters_detected = self._detect_chapters(doc, sample_text)
            abstract_present = self._detect_abstract(sample_text)
            references_present = self._detect_references(doc)
            two_column_layout = self._detect_two_column_layout(doc)
            has_doi = self._detect_doi(sample_text)
            
            # Determine document type based on features
            doc_type = self._classify_document_type(
                toc_present=toc_present,
                chapters_detected=chapters_detected,
                abstract_present=abstract_present,
                references_present=references_present,
                has_doi=has_doi,
                page_count=page_count
            )
            
            doc.close()
            
            return DocumentMetadata(
                page_count=page_count,
                toc_present=toc_present,
                chapters_detected=chapters_detected,
                abstract_present=abstract_present,
                references_present=references_present,
                two_column_layout=two_column_layout,
                has_doi=has_doi,
                doc_type=doc_type,
                file_size=file_size
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {pdf_path}: {e}")
            # Return minimal metadata for failed analysis
            return DocumentMetadata(
                page_count=0,
                doc_type=DocumentType.UNKNOWN
            )
    
    def _extract_sample_text(self, doc: fitz.Document, max_pages: int = 5) -> str:
        """Extract text from first few pages for analysis."""
        # TODO: Implement efficient text extraction for heuristics
        # For now, extract from first page only
        try:
            if len(doc) > 0:
                page = doc[0]
                return page.get_text()
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to extract sample text: {e}")
            return ""
    
    def _detect_toc(self, doc: fitz.Document) -> bool:
        """Detect if document has a table of contents."""
        # TODO: Implement TOC detection using PyMuPDF's TOC extraction
        try:
            toc = doc.get_toc()
            return len(toc) > 0
        except Exception:
            return False
    
    def _detect_chapters(self, doc: fitz.Document, sample_text: str) -> bool:
        """Detect if document has chapter structure."""
        # TODO: Implement chapter detection heuristics
        # Look for patterns like "Chapter 1", "Chapter I", etc.
        chapter_patterns = [
            r'\bchapter\s+\d+\b',
            r'\bchapter\s+[ivxlcdm]+\b',
            r'^\s*\d+\.\s+[A-Z]',  # Numbered sections
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, sample_text, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False
    
    def _detect_abstract(self, sample_text: str) -> bool:
        """Detect if document has an abstract."""
        # TODO: Implement abstract detection
        abstract_patterns = [
            r'\babstract\b',
            r'\bsummary\b',
        ]
        
        for pattern in abstract_patterns:
            if re.search(pattern, sample_text, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_references(self, doc: fitz.Document) -> bool:
        """Detect if document has a references section."""
        # TODO: Implement references detection
        # Look for "References", "Bibliography", etc. in later pages
        try:
            # Check last few pages for references
            for page_num in range(max(0, len(doc) - 5), len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if re.search(r'\b(references|bibliography|works cited)\b', text, re.IGNORECASE):
                    return True
            return False
        except Exception:
            return False
    
    def _detect_two_column_layout(self, doc: fitz.Document) -> bool:
        """Detect if document uses two-column layout."""
        # TODO: Implement layout detection using text block analysis
        # For now, return False as a stub
        return False
    
    def _detect_doi(self, sample_text: str) -> bool:
        """Detect if document has a DOI."""
        # TODO: Implement DOI detection
        doi_pattern = r'\bdoi:\s*10\.\d{4,}/[^\s]+'
        return bool(re.search(doi_pattern, sample_text, re.IGNORECASE))
    
    def _classify_document_type(
        self,
        toc_present: bool,
        chapters_detected: bool,
        abstract_present: bool,
        references_present: bool,
        has_doi: bool,
        page_count: int
    ) -> DocumentType:
        """Classify document type based on detected features."""
        # TODO: Refine classification heuristics
        
        # Research paper indicators
        research_score = 0
        if abstract_present:
            research_score += 2
        if references_present:
            research_score += 2
        if has_doi:
            research_score += 3
        if 5 <= page_count <= 30:  # Typical research paper length
            research_score += 1
        
        # Textbook indicators
        textbook_score = 0
        if chapters_detected:
            textbook_score += 3
        if toc_present:
            textbook_score += 2
        if page_count > 50:  # Textbooks are usually longer
            textbook_score += 2
        
        # Determine classification
        if research_score >= 4:
            return DocumentType.RESEARCH_PAPER
        elif textbook_score >= 4:
            return DocumentType.TEXTBOOK
        else:
            return DocumentType.UNKNOWN


def get_heuristic_defaults(metadata: DocumentMetadata) -> Dict[str, any]:
    """
    Generate heuristic defaults for configuration based on document metadata.
    
    Args:
        metadata: Document metadata from analysis
        
    Returns:
        Dictionary with suggested configuration defaults
    """
    # TODO: Implement heuristic-based configuration suggestions
    defaults = {}
    
    # Suggest chunking strategy based on document type
    if metadata.doc_type == DocumentType.RESEARCH_PAPER:
        defaults["chunking_mode"] = "sections"
        defaults["tokens_per_chunk"] = 1500  # Smaller chunks for papers
        defaults["strategies"] = ["key_points", "cloze_definitions"]
    elif metadata.doc_type == DocumentType.TEXTBOOK:
        defaults["chunking_mode"] = "smart"
        defaults["tokens_per_chunk"] = 2500  # Larger chunks for textbooks
        defaults["strategies"] = ["key_points", "figure_based"]
    else:
        defaults["chunking_mode"] = "pages"
        defaults["tokens_per_chunk"] = 2000
        defaults["strategies"] = ["key_points"]
    
    # Adjust for document features
    if metadata.two_column_layout:
        defaults["respect_page_bounds"] = False  # Better chunking across columns
    
    if metadata.page_count > 100:
        defaults["tokens_per_chunk"] = max(defaults.get("tokens_per_chunk", 2000), 3000)
    
    return defaults