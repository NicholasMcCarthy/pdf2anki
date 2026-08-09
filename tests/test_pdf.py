"""Tests for pdf.py's extract_abstract_text() - best-effort abstract extraction
used to give the highlight_priority strategy paper-level context (see
prompts/highlight_priority.j2)."""

import fitz

from pdf2anki.pdf import PDFDocument, extract_abstract_text, extract_pdf_content


def _pages(*texts):
    return [{"raw_text": t} for t in texts]


def test_extracts_abstract_after_standalone_heading():
    pages = _pages(
        "Title of Paper\n\nAbstract\nThis paper studies photosynthesis in detail.\n"
        "It shows several key findings.\n\nIntroduction\nPhotosynthesis is important because..."
    )
    abstract = extract_abstract_text(pages)
    assert abstract is not None
    assert "photosynthesis in detail" in abstract
    assert "Introduction" not in abstract
    assert "important because" not in abstract


def test_extracts_abstract_with_inline_lead_in():
    pages = _pages(
        "Title\n\nAbstract: This paper studies X and finds Y.\n\nKeywords\nX, Y, Z"
    )
    abstract = extract_abstract_text(pages)
    assert abstract is not None
    assert "This paper studies X and finds Y." in abstract
    assert "Keywords" not in abstract


def test_returns_none_when_no_abstract_heading():
    pages = _pages("Just a random document\nwith no abstract section at all.\n\nIntroduction\nSome text.")
    assert extract_abstract_text(pages) is None


def test_searches_across_first_two_pages():
    pages = _pages(
        "Title page with no abstract yet.",
        "Abstract\nThe abstract is on the second page.\n\nIntroduction\nBody text.",
    )
    abstract = extract_abstract_text(pages)
    assert abstract is not None
    assert "second page" in abstract


def test_ignores_pages_beyond_the_first_two():
    pages = _pages(
        "Title page.",
        "Some other front matter, still no abstract.",
        "Abstract\nThis abstract is too far in to be found.",
    )
    assert extract_abstract_text(pages) is None


def test_caps_abstract_length():
    long_text = "word " * 1000
    pages = _pages(f"Abstract\n{long_text}\n\nIntroduction\nBody.")
    abstract = extract_abstract_text(pages, max_chars=100)
    assert abstract is not None
    assert len(abstract) < 300  # generous margin over max_chars given line-based accumulation


def test_extract_pdf_content_includes_abstract_in_metadata(tmp_path):
    """End-to-end: extract_pdf_content() should surface the abstract text
    into content["metadata"]["abstract"] for a real PDF."""
    path = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Study of Photosynthesis", fontsize=16)
    page.insert_text((72, 110), "Abstract", fontsize=12)
    page.insert_text((72, 130), "This paper investigates photosynthesis mechanisms.", fontsize=10)
    page.insert_text((72, 150), "Introduction", fontsize=12)
    page.insert_text((72, 170), "Photosynthesis has been studied for decades.", fontsize=10)
    doc.save(str(path))
    doc.close()

    content = extract_pdf_content(path, extract_images=False, extract_structure=False)

    assert "abstract" in content["metadata"]
    assert "photosynthesis mechanisms" in content["metadata"]["abstract"]
    assert "Introduction" not in content["metadata"]["abstract"]


def test_extract_pdf_content_omits_abstract_key_when_not_found(tmp_path):
    path = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Just some plain document text.")
    doc.save(str(path))
    doc.close()

    content = extract_pdf_content(path, extract_images=False, extract_structure=False)
    assert "abstract" not in content["metadata"]


def test_extract_pdf_content_includes_source_path_in_metadata(tmp_path):
    """Reproduces a real bug: card.source_pdf (strategies/base.py) reads
    pdf_metadata["path"], but extract_pdf_content() only ever put the path at
    the top-level content["path"], never inside content["metadata"] - so
    every academic-PDF/textbook card's source_pdf was always "", and
    build.py's _build_source_info() rendered the literal string "Unknown"
    on every card."""
    path = tmp_path / "paper.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content.")
    doc.save(str(path))
    doc.close()

    content = extract_pdf_content(path, extract_images=False, extract_structure=False)

    assert content["metadata"]["path"] == str(path)


def test_pdf_document_title_falls_back_to_filename_when_metadata_title_empty(tmp_path):
    """Reproduces a real bug: PyMuPDF's metadata dict always has a "title"
    key present (empty string, not absent, when unset), so a
    meta.get("title", self.path.stem) default never actually fires - a PDF
    with no title metadata got title == "", not the filename stem."""
    path = tmp_path / "untitled_paper.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content.")
    doc.set_metadata({})  # explicit empty metadata - title key present but ""
    doc.save(str(path))
    doc.close()

    with PDFDocument(path) as pdf_doc:
        assert pdf_doc.metadata["title"] == "untitled_paper"


def test_pdf_document_title_uses_real_metadata_when_present(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content.")
    doc.set_metadata({"title": "A Real Paper Title"})
    doc.save(str(path))
    doc.close()

    with PDFDocument(path) as pdf_doc:
        assert pdf_doc.metadata["title"] == "A Real Paper Title"
