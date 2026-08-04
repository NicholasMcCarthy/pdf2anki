"""Tests for workflow routing: file-type/doc-type based auto-selection, manual
override precedence, and scan-docs honoring a manual workflow override even
when the document itself is classified as unknown."""

from pathlib import Path

import fitz
from typer.testing import CliRunner

from pdf2anki.cli import app
from pdf2anki.config import DocumentMetadata, DocumentType, DocumentsConfig
from pdf2anki.workflow_router import Workflow, select_workflow

runner = CliRunner()


def test_markdown_files_route_to_readwise():
    assert select_workflow(Path("notes/article.md")) == Workflow.READWISE
    assert select_workflow(Path("notes/article.markdown")) == Workflow.READWISE


def test_research_paper_routes_to_academic_paper():
    metadata = DocumentMetadata(doc_type=DocumentType.RESEARCH_PAPER)
    assert select_workflow(Path("paper.pdf"), metadata) == Workflow.ACADEMIC_PAPER


def test_textbook_routes_to_textbook_workflow():
    metadata = DocumentMetadata(doc_type=DocumentType.TEXTBOOK)
    assert select_workflow(Path("book.pdf"), metadata) == Workflow.TEXTBOOK


def test_unknown_pdf_routes_to_generic():
    metadata = DocumentMetadata(doc_type=DocumentType.UNKNOWN)
    assert select_workflow(Path("mystery.pdf"), metadata) == Workflow.GENERIC


def test_pdf_without_metadata_routes_to_generic():
    assert select_workflow(Path("unscanned.pdf")) == Workflow.GENERIC


def test_override_wins_over_detected_type():
    metadata = DocumentMetadata(doc_type=DocumentType.UNKNOWN)
    assert select_workflow(Path("book.pdf"), metadata, override="textbook") == Workflow.TEXTBOOK


def test_invalid_override_falls_back_to_detection():
    metadata = DocumentMetadata(doc_type=DocumentType.RESEARCH_PAPER)
    assert select_workflow(Path("paper.pdf"), metadata, override="not-a-real-workflow") == Workflow.ACADEMIC_PAPER


def test_scan_docs_honors_manual_workflow_override_for_unknown_doc(tmp_path):
    """A user hand-editing documents.yaml to force workflow: textbook on a
    document the classifier scored as unknown should get textbook heuristic
    defaults on the next scan-docs run, not the no-heuristics unknown path."""
    pdf_path = tmp_path / "plain.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "An ordinary short document.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'inputs:\n  paths: ["{pdf_path}"]\n')
    documents_file = tmp_path / "documents.yaml"

    # First scan: classifier can't confidently classify it -> generic, no heuristics.
    result = runner.invoke(app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)])
    assert result.exit_code == 0, result.output

    documents_config = DocumentsConfig.from_yaml(documents_file)
    doc_config = documents_config.documents["plain.pdf"]
    assert doc_config.workflow == "generic"
    assert doc_config.heuristic_chunking is None

    # User hand-edits documents.yaml to force the textbook workflow.
    doc_config.workflow = "textbook"
    documents_config.to_yaml(documents_file)

    # Re-scanning should now apply textbook heuristics despite doc_type staying unknown.
    result = runner.invoke(app, ["scan-docs", "--config", str(config_path), "--documents", str(documents_file)])
    assert result.exit_code == 0, result.output

    documents_config = DocumentsConfig.from_yaml(documents_file)
    doc_config = documents_config.documents["plain.pdf"]
    assert doc_config.workflow == "textbook"
    assert doc_config.metadata.doc_type == "unknown"  # detection itself is unchanged
    assert doc_config.heuristic_chunking is not None
    assert doc_config.heuristic_chunking.mode == "outline"
    assert "figure_based" in doc_config.heuristic_strategies
