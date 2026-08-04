"""End-to-end tests for the watcher service's per-file runner: classify,
generate, merge into the shared CSV, and rebuild the .apkg - the same path
exercised by `pdf2anki serve`."""

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from pdf2anki.config import Config
from pdf2anki.llm import LLMResponse
from pdf2anki.service.runner import process_new_file
from pdf2anki.workflow_router import Workflow


def _fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
    payload = {"cards": [{
        "front": f"Q-{abs(hash(prompt)) % 10000}", "back": "A",
        "page_citation": "p. 1", "core_concept": "X",
    }]}
    return LLMResponse(
        content=json.dumps(payload), model=self.config.model, tokens_used=1,
        cost_estimate=0.0, cached=False, response_time=0.0,
    )


def _make_config(tmp_path: Path) -> Config:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
generate:
  strategies:
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
    return Config.from_yaml(config_path)


@pytest.mark.integration
def test_process_new_pdf_generates_and_builds_deck(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Intro")
    page.insert_text((72, 100), "Plants convert light energy into chemical energy.")
    doc.save(str(pdf_path))
    doc.close()

    config = _make_config(tmp_path)

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = process_new_file(pdf_path, config)

    assert result["workflow"] == "generic"
    assert result["cards_generated"] >= 1
    assert result["cards_added"] == result["cards_generated"]
    assert result["apkg_path"] is not None
    assert Path(result["apkg_path"]).exists()


@pytest.mark.integration
def test_reprocessing_unchanged_file_does_not_duplicate_cards(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content here.")
    doc.save(str(pdf_path))
    doc.close()

    config = _make_config(tmp_path)

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result1 = process_new_file(pdf_path, config)
        result2 = process_new_file(pdf_path, config)

    assert result1["cards_added"] >= 1
    assert result2["cards_added"] == 0
    assert result2["cards_total"] == result1["cards_total"]
    # No new cards -> no need to rebuild the .apkg again.
    assert result2["apkg_path"] is None


@pytest.mark.integration
def test_process_new_readwise_file_generates_cards(tmp_path):
    md_path = tmp_path / "article.md"
    md_path.write_text("""---
author: example.com
url: https://example.com/article
category: articles
tags: []
---
# Test Article

## Highlights
> [!info]
> Plants convert light energy into chemical energy.
""")

    config = _make_config(tmp_path)

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = process_new_file(md_path, config)

    assert result["workflow"] == "readwise"
    assert result["cards_generated"] == 1
    assert result["cards_added"] == 1


@pytest.mark.integration
def test_ankiconnect_push_is_skipped_when_disabled(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content.")
    doc.save(str(pdf_path))
    doc.close()

    config = _make_config(tmp_path)
    assert config.service.ankiconnect.enabled is False

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = process_new_file(pdf_path, config)

    assert result["ankiconnect"] is None


@pytest.mark.integration
def test_textbook_workflow_honors_per_book_instructions_yml(tmp_path):
    """A hand-placed instructions.yml next to a textbook PDF (in what would be
    the mounted textbooks/ directory) should override deck name and strategies,
    per the requested "custom instructions.yml per textbook" design."""
    book_dir = tmp_path / "my-textbook"
    book_dir.mkdir()
    pdf_path = book_dir / "book.pdf"

    doc = fitz.open()
    for title in ["Chapter 1: Basics", "Chapter 2: Advanced"]:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=16)
        page.insert_text((72, 100), f"Content of {title}.", fontsize=11)
    doc.set_toc([[1, "Chapter 1: Basics", 1], [1, "Chapter 2: Advanced", 2]])
    doc.save(str(pdf_path))
    doc.close()

    (book_dir / "instructions.yml").write_text(
        "deck_name: My Custom Textbook Deck\nstrategies: [key_points]\n"
    )

    config = _make_config(tmp_path)

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        with patch("pdf2anki.service.classifier.select_workflow", return_value=Workflow.TEXTBOOK):
            result = process_new_file(pdf_path, config)

    assert result["workflow"] == "textbook"
    assert result["cards_added"] >= 1

    from pdf2anki.io import load_csv
    df = load_csv(config.output.csv_path)
    assert (df["deck"] == "My Custom Textbook Deck").all()


@pytest.mark.integration
def test_ankiconnect_push_reports_unreachable_when_enabled_but_down(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Some content.")
    doc.save(str(pdf_path))
    doc.close()

    config = _make_config(tmp_path)
    config.service.ankiconnect.enabled = True
    config.service.ankiconnect.url = "http://localhost:1"  # nothing listening

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = process_new_file(pdf_path, config)

    assert result["ankiconnect"] is not None
    assert result["ankiconnect"]["added"] == 0
    assert result["ankiconnect"]["failed"] == result["cards_added"]
