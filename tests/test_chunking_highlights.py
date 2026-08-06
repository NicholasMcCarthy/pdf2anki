"""Tests for the highlights-mode chunking rework (_chunk_by_highlights et al.):
reference/appendix/supplementary/acknowledgments stripping at page granularity,
single-call whole-paper chunking for short papers, and smart-chunk-with-
highlights for longer ones. See chunking.py:_chunk_by_highlights.
"""

from unittest.mock import Mock

from pdf2anki.chunking import TextChunker
from pdf2anki.config import ChunkingConfig, ChunkingMode


def make_page(page_num, text):
    return {
        "page_num": page_num,
        "raw_text": text,
        "blocks": [],
        "bbox": [0, 0, 600, 800],
        "rotation": 0,
    }


def make_annotation(page_num, text, author="", content="", screenshot=None):
    ann = {"page_num": page_num, "text": text, "author": author, "content": content}
    if screenshot is not None:
        ann["screenshot"] = screenshot
    return ann


def make_pdf_content(pages, annotations=None, page_count=None):
    return {
        "path": "test.pdf",
        "metadata": {"title": "Test Paper"},
        "page_count": page_count or len(pages),
        "pages": pages,
        "annotations": annotations or [],
        "images": [],
        "structure": {"headings": [], "sections": [], "chapters": []},
        "content_hash": "test_hash",
    }


def make_chunker(single_call_max_pages=12, token_budget=8000):
    config = ChunkingConfig(
        mode=ChunkingMode.HIGHLIGHTS,
        single_call_max_pages=single_call_max_pages,
        token_budget=token_budget,
    )
    chunker = TextChunker(config, model="gpt-4")
    chunker.encoding = Mock()
    chunker.encoding.encode.side_effect = lambda text: [1] * max(len(text) // 4, 0)
    return chunker


# --- _strip_terminal_sections_from_pages ---------------------------------

def test_strip_terminal_sections_truncates_mid_page():
    chunker = make_chunker()
    pages = [
        make_page(1, "Introduction\n\nSome real content here."),
        make_page(2, "More content.\n\nReferences\n\n1. Smith, J. (2020)."),
        make_page(3, "3. Another citation that should never be seen."),
    ]
    result = chunker._strip_terminal_sections_from_pages(pages)

    assert len(result) == 2
    assert "More content." in result[1]["raw_text"]
    assert "References" not in result[1]["raw_text"]
    assert "Smith, J." not in result[1]["raw_text"]


def test_strip_terminal_sections_drops_heading_at_start_of_page():
    chunker = make_chunker()
    pages = [
        make_page(1, "Body text."),
        make_page(2, "Appendix A\n\nSupplementary tables go here."),
    ]
    result = chunker._strip_terminal_sections_from_pages(pages)

    assert len(result) == 1
    assert result[0]["page_num"] == 1


def test_strip_terminal_sections_noop_when_nothing_found():
    chunker = make_chunker()
    pages = [make_page(1, "Intro"), make_page(2, "Conclusion. All done.")]
    result = chunker._strip_terminal_sections_from_pages(pages)

    assert len(result) == 2
    assert [p["raw_text"] for p in result] == [p["raw_text"] for p in pages]


def test_strip_terminal_sections_does_not_mutate_input():
    chunker = make_chunker()
    pages = [make_page(1, "Intro\n\nReferences\n\nSmith 2020.")]
    original_text = pages[0]["raw_text"]
    chunker._strip_terminal_sections_from_pages(pages)

    assert pages[0]["raw_text"] == original_text


def test_strip_terminal_sections_matches_acknowledgments_and_funding():
    chunker = make_chunker()
    pages = [
        make_page(1, "Body text."),
        make_page(2, "Acknowledgments\n\nThanks to our funders."),
    ]
    result = chunker._strip_terminal_sections_from_pages(pages)
    assert len(result) == 1


# --- _chunk_by_highlights: small paper -> single call ---------------------

def test_small_paper_produces_single_chunk_with_inline_highlights():
    chunker = make_chunker(single_call_max_pages=12)
    pages = [
        make_page(1, "Abstract\n\nThis paper is about X."),
        make_page(2, "Introduction\n\nX matters because Y."),
    ]
    annotations = [make_annotation(2, "X matters because Y", author="reader")]
    pdf_content = make_pdf_content(pages, annotations)

    chunks = chunker._chunk_by_highlights(pdf_content)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_page == 1
    assert chunk.end_page == 2
    assert chunk.highlights == annotations
    assert "[HIGHLIGHT - reader]: X matters because Y" in chunk.text
    assert "[PAGE 1]:" in chunk.text
    assert "[PAGE 2]:" in chunk.text


def test_small_paper_with_no_annotations_still_produces_single_chunk():
    chunker = make_chunker(single_call_max_pages=12)
    pages = [make_page(1, "Just some text, no highlights.")]
    pdf_content = make_pdf_content(pages, annotations=[])

    chunks = chunker._chunk_by_highlights(pdf_content)

    assert len(chunks) == 1
    assert chunks[0].highlights is None
    assert chunks[0].chunk_type == "text"


def test_small_paper_strips_references_before_building_single_chunk():
    chunker = make_chunker(single_call_max_pages=12)
    pages = [
        make_page(1, "Intro content."),
        make_page(2, "References\n\n1. Some citation."),
    ]
    pdf_content = make_pdf_content(pages, annotations=[], page_count=2)

    chunks = chunker._chunk_by_highlights(pdf_content)

    assert len(chunks) == 1
    assert "References" not in chunks[0].text
    assert "Some citation" not in chunks[0].text


def test_annotation_on_stripped_page_is_excluded_entirely():
    """A highlight physically made inside a stripped References section must
    not surface as marker text nor as a chunk.highlights entry."""
    chunker = make_chunker(single_call_max_pages=12)
    pages = [
        make_page(1, "Real content."),
        make_page(2, "References\n\n1. Smith 2020."),
    ]
    annotations = [make_annotation(2, "Smith 2020", author="reader")]
    pdf_content = make_pdf_content(pages, annotations, page_count=2)

    chunks = chunker._chunk_by_highlights(pdf_content)

    assert len(chunks) == 1
    assert chunks[0].highlights is None
    assert "Smith 2020" not in chunks[0].text


def test_small_but_token_dense_paper_falls_back_to_smart_chunking():
    """Even a paper under the page threshold can still be too token-dense for
    one call - the safety fallback should kick in rather than risk an
    oversized single call."""
    chunker = make_chunker(single_call_max_pages=12, token_budget=10)
    pages = [make_page(1, "word " * 200)]
    pdf_content = make_pdf_content(pages, annotations=[], page_count=1)

    original = chunker._chunk_smart_with_highlights
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    chunker._chunk_smart_with_highlights = spy
    chunks = chunker._chunk_by_highlights(pdf_content)

    assert calls, "expected fallback to _chunk_smart_with_highlights when the single chunk exceeds token_budget"
    assert isinstance(chunks, list)


# --- _chunk_by_highlights: large paper -> smart chunk with highlights -----

def test_large_paper_uses_smart_chunking_with_per_chunk_highlights():
    chunker = make_chunker(single_call_max_pages=2)
    pages = [make_page(i, f"Page {i} content. " * 20) for i in range(1, 5)]
    annotations = [make_annotation(3, "an important claim", author="reader")]
    pdf_content = make_pdf_content(pages, annotations, page_count=4)

    chunks = chunker._chunk_by_highlights(pdf_content)

    assert len(chunks) >= 1
    highlighted_chunks = [c for c in chunks if c.highlights]
    assert len(highlighted_chunks) == 1
    hc = highlighted_chunks[0]
    assert hc.start_page <= 3 <= hc.end_page
    assert "[HIGHLIGHT - reader]: an important claim" in hc.text
    assert "[PAGE CONTEXT]:" in hc.text


def test_large_paper_strips_references_before_smart_chunking():
    chunker = make_chunker(single_call_max_pages=1)
    pages = [
        make_page(1, "Intro content here. " * 10),
        make_page(2, "More body content. " * 10),
        make_page(3, "References\n\n1. Some citation that must not appear."),
    ]
    pdf_content = make_pdf_content(pages, annotations=[], page_count=3)

    chunks = chunker._chunk_by_highlights(pdf_content)

    combined = "\n".join(c.text for c in chunks)
    assert "Some citation" not in combined
    assert "References" not in combined


def test_zero_content_after_stripping_does_not_crash():
    chunker = make_chunker(single_call_max_pages=12)
    pages = [make_page(1, "References\n\n1. Only a citation list here.")]
    pdf_content = make_pdf_content(pages, annotations=[], page_count=1)

    chunks = chunker._chunk_by_highlights(pdf_content)

    # Nothing left to send - should degrade gracefully, not raise.
    assert isinstance(chunks, list)
