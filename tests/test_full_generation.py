"""End-to-end test for the generate CLI command wired to the real preprocess
pipeline (scan-docs -> generate -> cards.csv), with the LLM call mocked out so
the test needs no network access or API key.
"""

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from typer.testing import CliRunner

from pdf2anki.cli import app
from pdf2anki.llm import LLMResponse

runner = CliRunner()


def _make_sample_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Introduction", fontsize=18)
    page.insert_text((72, 100), "This is a short sample paragraph about photosynthesis.", fontsize=11)
    page.insert_text((72, 120), "Plants convert light energy into chemical energy.", fontsize=11)
    doc.save(str(path))
    doc.close()


def _fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
    payload = {
        "cards": [
            {
                "front": "What do plants convert light energy into?",
                "back": "Chemical energy (via photosynthesis).",
                "page_citation": "p. 1",
                "core_concept": "Photosynthesis",
                "tags": ["biology"],
            }
        ]
    }
    return LLMResponse(
        content=json.dumps(payload),
        model=self.config.model,
        tokens_used=42,
        cost_estimate=0.001,
        cached=False,
        response_time=0.01,
    )


def _make_pdf_with_abstract(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Paper About Photosynthesis", fontsize=16)
    page.insert_text((72, 110), "Abstract", fontsize=12)
    page.insert_text(
        (72, 130),
        "UNIQUE_ABSTRACT_MARKER_TEXT this paper studies photosynthesis mechanisms.",
        fontsize=10,
    )
    page.insert_text((72, 150), "Introduction", fontsize=12)
    page.insert_text(
        (72, 170),
        "Photosynthesis has been studied for decades in a wide range of plant species.",
        fontsize=10,
    )
    doc.save(str(path))
    doc.close()


@pytest.mark.integration
def test_abstract_generates_its_own_key_points_cards_alongside_body_cards(tmp_path):
    """The abstract, when detected, is sent as its own single chunk through
    the key_points ("high-level, key message") strategy - separate from
    whatever the main chunking mode produces for the body text - so a paper's
    core contribution gets covered even without a highlight on it.

    A single-page paper's main "whole document" chunk necessarily contains
    the abstract text too (same page), so this can't tell the two calls
    apart by response content - it captures every prompt actually sent and
    checks their shape instead: one call's chunk text is the abstract alone
    (no "Introduction" heading), and a separate call's chunk text is the
    full page (abstract heading through the Introduction/body)."""
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf_with_abstract(pdf_path)

    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "workspace"
    config_path.write_text(f"""
project:
  name: test-project
inputs:
  paths: ["{pdf_path}"]
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
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

    captured_prompts = []

    # Deliberately distinct front/back text per call (not just a counter
    # suffix) so the pipeline's text-similarity dedup doesn't collapse the
    # two cards into one before they reach the CSV.
    _card_texts = [
        ("What is the paper's central topic?", "Photosynthesis mechanisms, per the abstract."),
        ("What does the introduction say has been studied for decades?", "Photosynthesis in many plant species."),
    ]

    def _fake_generate_capturing(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
        front, back = _card_texts[len(captured_prompts) % len(_card_texts)]
        captured_prompts.append(prompt)
        payload = {"cards": [{
            "front": front,
            "back": back,
            "page_citation": "p. 1",
            "core_concept": "X",
            "tags": [],
        }]}
        return LLMResponse(
            content=json.dumps(payload), model=self.config.model, tokens_used=10,
            cost_estimate=0.0, cached=False, response_time=0.01,
        )

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate_capturing):
        result = runner.invoke(
            app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(
            app, ["generate", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

    assert len(captured_prompts) >= 2, "expected a separate call for the abstract chunk and the main chunk(s)"

    abstract_only_prompts = [
        p for p in captured_prompts if "UNIQUE_ABSTRACT_MARKER_TEXT" in p and "Introduction" not in p
    ]
    full_page_prompts = [
        p for p in captured_prompts if "UNIQUE_ABSTRACT_MARKER_TEXT" in p and "Introduction" in p
    ]
    assert abstract_only_prompts, "no call was made with the abstract text alone as the chunk"
    assert full_page_prompts, "no call was made with the full page (including the Introduction) as the chunk"

    # The abstract-chunk-derived card actually reaches the CSV, tagged with
    # its own "Abstract" section - not just requested from the LLM.
    content = (workspace / "cards.csv").read_text()
    assert "What is the paper's central topic?" in content
    assert ",Abstract," in content


@pytest.mark.integration
def test_scan_and_generate_produce_cards_csv(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_sample_pdf(pdf_path)

    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "workspace"
    config_path.write_text(f"""
project:
  name: test-project
inputs:
  paths: ["{pdf_path}"]
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
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

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = runner.invoke(
            app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

        result = runner.invoke(
            app, ["generate", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

    csv_path = workspace / "cards.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "Photosynthesis" in content
    assert "What do plants convert light energy into?" in content


@pytest.mark.integration
def test_cards_land_in_workflow_subdeck_honoring_manual_override(tmp_path):
    """A manual `workflow: academic_paper` override in documents.yaml (the
    same mechanism used for heuristic chunking/strategy defaults) should
    also drive the "workflow" deck_structure's subdeck placement - cards
    should land in "<deck_name>::Articles", not the base deck."""
    from pdf2anki.config import DocumentsConfig

    pdf_path = tmp_path / "sample.pdf"
    _make_sample_pdf(pdf_path)

    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "workspace"
    config_path.write_text(f"""
project:
  name: test-project
inputs:
  paths: ["{pdf_path}"]
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
  hallucination:
    require_citations: false
    verify_quotes: false
generate:
  anki:
    deck_name: My Deck
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

    with patch("pdf2anki.llm.LLMProvider.generate", new=_fake_generate):
        result = runner.invoke(
            app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

        # Force the academic-paper workflow, same as a user hand-editing
        # documents.yaml.
        documents_config = DocumentsConfig.from_yaml(documents_file)
        documents_config.documents["sample.pdf"].workflow = "academic_paper"
        documents_config.to_yaml(documents_file)

        result = runner.invoke(
            app, ["generate", "--config", str(config_path), "--documents", str(documents_file)]
        )
        assert result.exit_code == 0, result.output

    content = (workspace / "cards.csv").read_text()
    assert "My Deck::Articles" in content
