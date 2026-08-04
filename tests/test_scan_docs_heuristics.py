"""Regression test: scan-docs must not write heuristic_* suggestions for
documents classified as UNKNOWN, since get_effective_config() treats
heuristic_* as taking precedence over the user's own config.yaml - writing a
bland generic suggestion for every document (including unclassifiable ones)
would silently stomp explicit user config for anything not confidently
recognized as a paper or textbook."""

from pathlib import Path

import fitz
from typer.testing import CliRunner

from pdf2anki.cli import app
from pdf2anki.config import DocumentsConfig

runner = CliRunner()


def test_unknown_doc_gets_no_heuristic_overrides(tmp_path):
    pdf_path = tmp_path / "plain.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Just an ordinary short document with no strong signals.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
inputs:
  paths: ["{pdf_path}"]
""")
    documents_file = tmp_path / "documents.yaml"

    result = runner.invoke(
        app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)]
    )
    assert result.exit_code == 0, result.output

    documents_config = DocumentsConfig.from_yaml(documents_file)
    doc_config = documents_config.documents["plain.pdf"]

    assert doc_config.metadata.doc_type == "unknown"
    assert doc_config.heuristic_chunking is None
    assert doc_config.heuristic_strategies is None
    assert doc_config.heuristic_extract_annotations is None
