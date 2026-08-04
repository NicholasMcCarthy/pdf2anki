"""Tests for the watcher service's file classifier."""

from pathlib import Path

import fitz

from pdf2anki.service.classifier import classify_file
from pdf2anki.workflow_router import Workflow


def test_classifies_markdown_as_readwise(tmp_path):
    path = tmp_path / "article.md"
    path.write_text("# Title\n\n## Highlights\n> [!info]\n> some highlight\n")
    assert classify_file(path) == Workflow.READWISE


def test_classifies_unrecognized_extension_as_generic(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text")
    assert classify_file(path) == Workflow.GENERIC


def test_classifies_plain_pdf_as_generic(tmp_path):
    path = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Just a short document.")
    doc.save(str(path))
    doc.close()

    assert classify_file(path) == Workflow.GENERIC


def test_classifies_pdf_with_toc_and_chapters_as_textbook(tmp_path):
    path = tmp_path / "book.pdf"
    doc = fitz.open()
    for i in range(60):
        page = doc.new_page()
        if i == 0:
            page.insert_text((72, 72), "Chapter 1: Introduction")
    doc.set_toc([[1, f"Chapter {i}", i + 1] for i in range(1, 10)])
    doc.save(str(path))
    doc.close()

    assert classify_file(path) == Workflow.TEXTBOOK
