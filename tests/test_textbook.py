"""Tests for the textbook workflow: outline extraction, per-book instructions.yml
profile loading (with default-template fallback), outline-driven chunking, and
the full-coverage report."""

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from typer.testing import CliRunner

from pdf2anki.chunking import TextChunker
from pdf2anki.cli import app
from pdf2anki.config import Chunking, ChunkingMode
from pdf2anki.llm import LLMResponse
from pdf2anki.pdf import extract_pdf_content
from pdf2anki.strategies.base import FlashcardData
from pdf2anki.textbook import (
    OutlineEntry,
    TextbookProfile,
    build_coverage_report,
    extract_outline,
    load_textbook_profile,
)

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "examples" / "textbook_default.yml"


def _make_textbook_pdf(path: Path) -> None:
    doc = fitz.open()
    for title in ["Chapter 1: Basics", "Chapter 2: Advanced"]:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=16)
        page.insert_text((72, 100), f"This is the content of {title}. It covers important material.", fontsize=11)
    doc.set_toc([[1, "Chapter 1: Basics", 1], [1, "Chapter 2: Advanced", 2]])
    doc.save(str(path))
    doc.close()


@pytest.fixture
def textbook_pdf(tmp_path) -> Path:
    path = tmp_path / "textbook.pdf"
    _make_textbook_pdf(path)
    return path


def test_extract_outline_from_toc(textbook_pdf):
    outline = extract_outline(textbook_pdf)

    assert len(outline) == 2
    assert outline[0] == OutlineEntry(title="Chapter 1: Basics", level=1, start_page=1, end_page=1)
    assert outline[1] == OutlineEntry(title="Chapter 2: Advanced", level=1, start_page=2, end_page=2)


def test_extract_outline_falls_back_to_headings_without_toc(tmp_path):
    path = tmp_path / "no_toc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Heading", fontsize=20)
    page.insert_text((72, 100), "Some regular body text that is not a heading.", fontsize=10)
    doc.save(str(path))
    doc.close()

    outline = extract_outline(path)
    # No embedded TOC, but the large-font line should be detected as a heading.
    assert any(e.title == "A Heading" for e in outline)


def test_load_textbook_profile_defaults_without_instructions_file(tmp_path):
    profile = load_textbook_profile(tmp_path)
    assert isinstance(profile, TextbookProfile)
    assert profile.deck_name is None


def test_load_textbook_profile_uses_shipped_default_template(tmp_path):
    profile = load_textbook_profile(tmp_path, default_profile_path=DEFAULT_PROFILE_PATH)
    assert profile.chunking.mode == "outline"
    assert profile.strategies == ["key_points", "figure_based", "cloze_definitions"]
    assert profile.max_cards_per_section == 8


def test_instructions_yml_overrides_only_specified_fields(tmp_path):
    (tmp_path / "instructions.yml").write_text("max_cards_per_section: 20\ndeck_name: My Custom Deck\n")

    profile = load_textbook_profile(tmp_path, default_profile_path=DEFAULT_PROFILE_PATH)

    assert profile.max_cards_per_section == 20
    assert profile.deck_name == "My Custom Deck"
    # Untouched fields still inherit from the default template.
    assert profile.chunking.mode == "outline"
    assert profile.strategies == ["key_points", "figure_based", "cloze_definitions"]


def test_chunk_by_outline_produces_one_chunk_group_per_chapter(textbook_pdf):
    content = extract_pdf_content(textbook_pdf, extract_images=False)
    outline = extract_outline(textbook_pdf)
    content["outline"] = [
        {"title": e.title, "level": e.level, "start_page": e.start_page, "end_page": e.end_page}
        for e in outline
    ]

    chunker = TextChunker(Chunking(mode=ChunkingMode.OUTLINE))
    chunks = chunker.chunk_document(content)

    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"Chapter 1: Basics", "Chapter 2: Advanced"}


def test_chunk_by_outline_falls_back_to_smart_without_outline(textbook_pdf):
    content = extract_pdf_content(textbook_pdf, extract_images=False)
    # No "outline" key populated.
    chunker = TextChunker(Chunking(mode=ChunkingMode.OUTLINE))
    chunks = chunker.chunk_document(content)
    assert len(chunks) >= 1


def _card(page_start, page_end):
    return FlashcardData(
        note_type="Basic",
        page_citation=f"p. {page_start}",
        core_concept="X",
        page_start=page_start,
        page_end=page_end,
    )


def test_coverage_report_flags_zero_card_sections():
    outline = [
        OutlineEntry(title="Chapter 1", level=1, start_page=1, end_page=5),
        OutlineEntry(title="Chapter 2", level=1, start_page=6, end_page=10),
    ]
    cards = [_card(2, 2)]  # only covers Chapter 1

    report = build_coverage_report(outline, cards)

    assert report["total_sections"] == 2
    assert report["covered_sections"] == 1
    assert report["uncovered_sections"] == ["Chapter 2"]
    assert report["coverage_ratio"] == 0.5


def test_coverage_report_full_coverage():
    outline = [
        OutlineEntry(title="Chapter 1", level=1, start_page=1, end_page=5),
        OutlineEntry(title="Chapter 2", level=1, start_page=6, end_page=10),
    ]
    cards = [_card(2, 2), _card(7, 8)]

    report = build_coverage_report(outline, cards)

    assert report["covered_sections"] == 2
    assert report["uncovered_sections"] == []
    assert report["coverage_ratio"] == 1.0


@pytest.mark.integration
def test_outline_chunking_end_to_end_writes_coverage_report(tmp_path):
    """scan-docs -> generate with chunking.mode=outline should produce one card
    group per chapter and a coverage_report.json with no uncovered sections."""
    pdf_path = tmp_path / "textbook.pdf"
    _make_textbook_pdf(pdf_path)

    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "workspace"
    config_path.write_text(f"""
inputs:
  paths: ["{pdf_path}"]
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
  ingestion:
    chunking:
      mode: outline
  hallucination:
    require_citations: false
    verify_quotes: false
generate:
  strategies:
    key_points:
      enabled: true
    cloze_definitions:
      enabled: false
    figure_based:
      enabled: false
  output:
    workspace: {workspace}
    csv_path: {workspace}/cards.csv
    media_path: {workspace}/media
    apkg_path: {workspace}/deck.apkg
    manifest_path: {workspace}/manifest.json
""")
    documents_file = tmp_path / "documents.yaml"

    def fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
        payload = {"cards": [{
            "front": f"Q about {abs(hash(prompt)) % 1000}",
            "back": "A",
            "page_citation": "p. 1",
            "core_concept": "X",
        }]}
        return LLMResponse(
            content=json.dumps(payload), model=self.config.model, tokens_used=1,
            cost_estimate=0.0, cached=False, response_time=0.0,
        )

    runner = CliRunner()
    with patch("pdf2anki.llm.LLMProvider.generate", new=fake_generate):
        result = runner.invoke(app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)])
        assert result.exit_code == 0, result.output

        result = runner.invoke(app, ["generate", "--config", str(config_path), "--documents", str(documents_file)])
        assert result.exit_code == 0, result.output

    coverage_path = workspace / "textbook_coverage.json"
    assert coverage_path.exists()
    coverage = json.loads(coverage_path.read_text())
    assert coverage["total_sections"] == 2
    assert coverage["uncovered_sections"] == []
