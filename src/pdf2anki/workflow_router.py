"""Workflow routing: decides which processing path applies to a given input
file - textbook (full coverage), academic paper (highlight-priority), Readwise
markdown, or the generic fallback - combining file-type detection,
DocumentAnalyzer's PDF classification, and any manual override.

This is the single source of truth both the CLI (scan-docs, to resolve a
manual `workflow:` override in documents.yaml into the right heuristic
defaults) and the Docker watcher service (to auto-classify newly dropped-in
files before deciding which pipeline to invoke) are meant to use, so routing
logic isn't duplicated between the two.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


class Workflow(str, Enum):
    TEXTBOOK = "textbook"
    ACADEMIC_PAPER = "academic_paper"
    READWISE = "readwise"
    GENERIC = "generic"


# Maps a resolved Workflow back to the DocumentType vocabulary get_heuristic_defaults()
# understands, so a manual workflow override can drive the same heuristic chunking/
# strategy defaults an auto-detected document of that type would get.
WORKFLOW_TO_DOCUMENT_TYPE = {
    Workflow.TEXTBOOK: DocumentType.TEXTBOOK,
    Workflow.ACADEMIC_PAPER: DocumentType.RESEARCH_PAPER,
    Workflow.GENERIC: DocumentType.UNKNOWN,
}

_DOCUMENT_TYPE_TO_WORKFLOW = {
    DocumentType.TEXTBOOK: Workflow.TEXTBOOK,
    DocumentType.RESEARCH_PAPER: Workflow.ACADEMIC_PAPER,
    DocumentType.UNKNOWN: Workflow.GENERIC,
}

MARKDOWN_SUFFIXES = (".md", ".markdown")

# Subdeck label per workflow (joined onto Anki.deck_name with "::" - Anki
# and genanki's own subdeck separator) for the "workflow" deck_structure
# mode - see build.py::AnkiDeckBuilder._create_decks(). GENERIC has no
# label: an unclassified document's cards stay in the base deck rather than
# a "Generic" subdeck nobody asked for.
_WORKFLOW_DECK_LABELS = {
    Workflow.READWISE: "Readwise",
    Workflow.ACADEMIC_PAPER: "Articles",
    Workflow.TEXTBOOK: "Textbooks",
    Workflow.GENERIC: None,
}


def deck_subdeck_for_workflow(workflow: "Workflow", book_name: Optional[str] = None) -> Optional[str]:
    """Subdeck path segment (relative to the base deck_name, not yet joined
    onto it) for a workflow, e.g. "Readwise", "Articles", or
    "Textbooks::My Book" - or None for GENERIC/unclassified documents,
    which stay in the base deck with no subdeck at all.

    `book_name` only affects TEXTBOOK - each book gets its own subdeck
    nested under "Textbooks" rather than sharing one flat textbook subdeck.
    """
    label = _WORKFLOW_DECK_LABELS.get(workflow)
    if label is None:
        return None
    if workflow == Workflow.TEXTBOOK and book_name:
        return f"{label}::{book_name}"
    return label


def select_workflow(
    path: Path,
    metadata: Optional[DocumentMetadata] = None,
    override: Optional[str] = None,
) -> Workflow:
    """Select the processing workflow for a file.

    Precedence: explicit override > file-type (markdown -> readwise) >
    DocumentAnalyzer's PDF classification (research_paper/textbook -> the
    matching workflow) > generic fallback.

    `metadata` is optional (e.g. the caller hasn't run DocumentAnalyzer yet)
    - without it, a PDF resolves to GENERIC unless overridden.
    """
    if override:
        try:
            return Workflow(override)
        except ValueError:
            logger.warning(f"Unknown workflow override '{override}' on {path}, ignoring")

    suffix = Path(path).suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return Workflow.READWISE

    if metadata is not None:
        return _DOCUMENT_TYPE_TO_WORKFLOW.get(metadata.doc_type, Workflow.GENERIC)

    return Workflow.GENERIC
