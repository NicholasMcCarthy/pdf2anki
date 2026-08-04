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
