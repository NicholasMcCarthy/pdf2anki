"""PDF processing and text extraction."""

import hashlib
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class PDFDocument:
    """Represents a PDF document with extracted content."""
    
    def __init__(self, path: Path):
        """Initialize PDF document."""
        self.path = path
        self.doc = fitz.open(str(path))
        self.metadata = self._extract_metadata()
        self.page_count = len(self.doc)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'doc'):
            self.doc.close()
    
    def _extract_metadata(self) -> Dict[str, str]:
        """Extract PDF metadata."""
        meta = self.doc.metadata
        return {
            "title": meta.get("title", self.path.stem),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
        }
    
    def extract_text(self, start_page: int = 0, end_page: Optional[int] = None) -> List[Dict]:
        """Extract text from PDF pages with structure information."""
        if end_page is None:
            end_page = self.page_count
        
        pages_data = []
        
        for page_num in range(start_page, min(end_page, self.page_count)):
            page = self.doc[page_num]
            
            # Extract text with formatting
            text_dict = page.get_text("dict")
            blocks = self._process_text_blocks(text_dict["blocks"])
            
            # Extract raw text for fallback
            raw_text = page.get_text()
            
            page_data = {
                "page_num": page_num + 1,  # 1-indexed
                "blocks": blocks,
                "raw_text": raw_text,
                "bbox": page.bound(),
                "rotation": page.rotation,
            }
            
            pages_data.append(page_data)
            logger.debug(f"Extracted text from page {page_num + 1}: {len(raw_text)} chars")
        
        return pages_data
    
    def _process_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Process text blocks to identify structure."""
        processed_blocks = []
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                block_data = {
                    "type": "text",
                    "bbox": block["bbox"],
                    "lines": [],
                    "font_info": [],
                }
                
                for line in block.get("lines", []):
                    line_text = ""
                    line_fonts = []
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        font_info = {
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0),
                        }
                        
                        line_text += text
                        line_fonts.append(font_info)
                    
                    if line_text.strip():
                        block_data["lines"].append(line_text)
                        block_data["font_info"].extend(line_fonts)
                
                if block_data["lines"]:
                    processed_blocks.append(block_data)
            
            elif block.get("type") == 1:  # Image block
                block_data = {
                    "type": "image",
                    "bbox": block["bbox"],
                    "width": block.get("width", 0),
                    "height": block.get("height", 0),
                }
                processed_blocks.append(block_data)
        
        return processed_blocks
    
    def extract_images(
        self, 
        start_page: int = 0, 
        end_page: Optional[int] = None,
        min_width: int = 100,
        min_height: int = 100
    ) -> List[Dict]:
        """Extract images from PDF pages."""
        if end_page is None:
            end_page = self.page_count
        
        images = []
        
        for page_num in range(start_page, min(end_page, self.page_count)):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(self.doc, xref)
                    
                    # Skip small images
                    if pix.width < min_width or pix.height < min_height:
                        pix = None
                        continue
                    
                    # Convert to PIL Image for processing
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("ppm")
                        pil_img = Image.open(io.BytesIO(img_data))
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("ppm")
                        pil_img = Image.open(io.BytesIO(img_data))
                        pix1 = None
                    
                    # Generate deterministic filename
                    content_hash = hashlib.md5(img_data).hexdigest()[:12]
                    filename = f"img_p{page_num+1}_{img_index}_{content_hash}.png"
                    
                    image_info = {
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "filename": filename,
                        "width": pix.width,
                        "height": pix.height,
                        "xref": xref,
                        "pil_image": pil_img,
                        "bbox": self._get_image_bbox(page, xref),
                    }
                    
                    images.append(image_info)
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                    continue
        
        logger.info(f"Extracted {len(images)} images from PDF")
        return images
    
    def _get_image_bbox(self, page, xref: int) -> Tuple[float, float, float, float]:
        """Get bounding box for an image on a page."""
        try:
            # Find image rectangles on the page
            img_rects = page.get_image_rects(xref)
            if img_rects:
                return tuple(img_rects[0])
        except Exception:
            pass
        
        # Fallback: return page bounds
        return tuple(page.bound())
    
    def detect_structure(self) -> Dict[str, List]:
        """Detect document structure (headings, sections, etc.)."""
        structure = {
            "headings": [],
            "sections": [],
            "chapters": [],
        }
        
        # Simple heuristic-based structure detection
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            font_size = span.get("size", 0)
                            flags = span.get("flags", 0)
                            
                            line_text += text + " "
                            max_font_size = max(max_font_size, font_size)
                            
                            # Check if bold (flag 16) or other emphasis
                            if flags & 16:
                                is_bold = True
                        
                        line_text = line_text.strip()
                        
                        # Heuristics for headings
                        if (line_text and 
                            len(line_text) < 200 and  # Not too long
                            (max_font_size > 14 or is_bold) and  # Larger or bold text
                            not line_text.endswith('.') and  # Doesn't end with period
                            len(line_text.split()) < 20):  # Not too many words
                            
                            heading_level = self._estimate_heading_level(max_font_size, is_bold)
                            
                            heading = {
                                "text": line_text,
                                "page": page_num + 1,
                                "level": heading_level,
                                "font_size": max_font_size,
                                "is_bold": is_bold,
                                "bbox": block["bbox"]
                            }
                            
                            structure["headings"].append(heading)
        
        # Group headings into sections and chapters
        structure["sections"] = self._group_into_sections(structure["headings"])
        structure["chapters"] = self._identify_chapters(structure["headings"])
        
        return structure
    
    def _estimate_heading_level(self, font_size: float, is_bold: bool) -> int:
        """Estimate heading level based on font properties."""
        if font_size >= 18:
            return 1
        elif font_size >= 16:
            return 2
        elif font_size >= 14 or is_bold:
            return 3
        else:
            return 4
    
    def _group_into_sections(self, headings: List[Dict]) -> List[Dict]:
        """Group headings into logical sections."""
        sections = []
        current_section = None
        
        for heading in headings:
            if heading["level"] <= 2:  # Major heading
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "title": heading["text"],
                    "start_page": heading["page"],
                    "end_page": heading["page"],
                    "headings": [heading],
                }
            elif current_section:
                current_section["headings"].append(heading)
                current_section["end_page"] = heading["page"]
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _identify_chapters(self, headings: List[Dict]) -> List[Dict]:
        """Identify chapter boundaries."""
        chapters = []
        
        chapter_keywords = ["chapter", "part", "section", "unit"]
        
        for heading in headings:
            text_lower = heading["text"].lower()
            
            # Check if this looks like a chapter heading
            if (heading["level"] == 1 or 
                any(keyword in text_lower for keyword in chapter_keywords) or
                any(char.isdigit() for char in heading["text"][:10])):
                
                chapter = {
                    "title": heading["text"],
                    "page": heading["page"],
                    "level": heading["level"],
                }
                chapters.append(chapter)
        
        return chapters
    
    def get_text_hash(self) -> str:
        """Get a hash of the document's text content for caching."""
        text_content = ""
        for page_num in range(min(5, self.page_count)):  # Use first few pages
            page = self.doc[page_num]
            text_content += page.get_text()
        
        return hashlib.sha256(text_content.encode()).hexdigest()[:16]


def extract_pdf_content(
    pdf_path: Path,
    extract_images: bool = True,
    extract_structure: bool = True,
    ocr_fallback: bool = False
) -> Dict:
    """Extract comprehensive content from a PDF file."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    with PDFDocument(pdf_path) as pdf_doc:
        # Extract text from all pages
        pages_data = pdf_doc.extract_text()
        
        # Extract images if requested
        images = []
        if extract_images:
            images = pdf_doc.extract_images()
        
        # Detect structure if requested
        structure = {}
        if extract_structure:
            structure = pdf_doc.detect_structure()
        
        # TODO: Implement OCR fallback if needed
        if ocr_fallback and not any(page["raw_text"].strip() for page in pages_data):
            logger.warning("OCR fallback requested but not implemented yet")
        
        content = {
            "path": pdf_path,
            "metadata": pdf_doc.metadata,
            "page_count": pdf_doc.page_count,
            "pages": pages_data,
            "images": images,
            "structure": structure,
            "content_hash": pdf_doc.get_text_hash(),
        }
    
    logger.info(f"Extracted content from {pdf_doc.page_count} pages with {len(images)} images")
    return content