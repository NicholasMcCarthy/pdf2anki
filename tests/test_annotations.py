"""Tests for PDF highlight-annotation extraction, region screenshots, and the
highlights chunking mode/strategy they feed."""

from pathlib import Path

import fitz
import pytest

from pdf2anki.chunking import TextChunker
from pdf2anki.config import Chunking, ChunkingMode
from pdf2anki.pdf import PDFDocument, extract_pdf_content


def _make_highlighted_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Introduction", fontsize=18)
    page.insert_text(
        (72, 100),
        "Plants convert light energy into chemical energy via photosynthesis.",
        fontsize=11,
    )
    page.insert_text(
        (72, 120), "This process occurs mainly in the chloroplasts of plant cells.", fontsize=11
    )

    quads = page.search_for(
        "Plants convert light energy into chemical energy via photosynthesis."
    )
    assert quads, "setup: search_for found nothing to highlight"
    annot = page.add_highlight_annot(quads)
    annot.set_info(info={"title": "nick", "content": "key mechanism"})
    annot.update()

    doc.save(str(path))
    doc.close()


@pytest.fixture
def highlighted_pdf(tmp_path) -> Path:
    path = tmp_path / "highlighted.pdf"
    _make_highlighted_pdf(path)
    return path


def test_extract_annotations_returns_highlight_text_and_metadata(highlighted_pdf):
    with PDFDocument(highlighted_pdf) as pdf_doc:
        annotations = pdf_doc.extract_annotations()

    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["type"] == "highlight"
    assert "photosynthesis" in ann["text"].lower()
    assert ann["author"] == "nick"
    assert ann["content"] == "key mechanism"
    assert ann["page_num"] == 1


def test_render_region_returns_nonempty_image(highlighted_pdf):
    with PDFDocument(highlighted_pdf) as pdf_doc:
        annotations = pdf_doc.extract_annotations()
        img = pdf_doc.render_region(page_num=annotations[0]["page_num"], rect=annotations[0]["rect"])

    assert img.size[0] > 0
    assert img.size[1] > 0


def test_extract_pdf_content_generates_screenshot_and_links_it_to_images(highlighted_pdf):
    content = extract_pdf_content(highlighted_pdf, extract_images=False, extract_annotations=True)

    assert len(content["annotations"]) == 1
    screenshot_name = content["annotations"][0]["screenshot"]
    assert screenshot_name is not None
    assert any(img["filename"] == screenshot_name for img in content["images"])


def test_extract_pdf_content_without_annotations_flag_skips_annotations(highlighted_pdf):
    content = extract_pdf_content(highlighted_pdf, extract_images=False, extract_annotations=False)
    assert content["annotations"] == []


def test_chunk_by_highlights_groups_by_page_with_context(highlighted_pdf):
    content = extract_pdf_content(highlighted_pdf, extract_images=False, extract_annotations=True)
    chunker = TextChunker(Chunking(mode=ChunkingMode.HIGHLIGHTS))
    chunks = chunker.chunk_document(content)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "highlight"
    assert chunk.highlights and len(chunk.highlights) == 1
    assert "[HIGHLIGHT" in chunk.text
    assert "[NOTE]: key mechanism" in chunk.text
    assert "[PAGE CONTEXT]" in chunk.text


def test_chunk_by_highlights_falls_back_to_smart_when_no_annotations(tmp_path):
    path = tmp_path / "plain.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Just some plain text with no highlights at all.", fontsize=11)
    doc.save(str(path))
    doc.close()

    content = extract_pdf_content(path, extract_images=False, extract_annotations=True)
    assert content["annotations"] == []

    chunker = TextChunker(Chunking(mode=ChunkingMode.HIGHLIGHTS))
    chunks = chunker.chunk_document(content)

    # Falls back to smart chunking rather than producing nothing.
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "text"
