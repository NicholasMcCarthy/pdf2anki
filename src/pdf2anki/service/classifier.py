"""Thin wrapper around workflow_router.select_workflow for the watcher
service: classifies a newly-seen file, running DocumentAnalyzer first for
PDFs so the classification has real signal to work with (a bare file path
alone can't distinguish a textbook from a paper)."""

import logging
from pathlib import Path

from ..heuristics import DocumentAnalyzer
from ..workflow_router import Workflow, select_workflow

logger = logging.getLogger(__name__)

_analyzer = DocumentAnalyzer()


def classify_file(path: Path) -> Workflow:
    """Classify a newly-detected file into a Workflow. Markdown files route to
    Readwise without needing analysis; PDFs are run through DocumentAnalyzer
    first so research-paper/textbook detection has something to work with.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".markdown"):
        return select_workflow(path)

    if suffix == ".pdf":
        metadata = _analyzer.analyze_document(str(path))
        workflow = select_workflow(path, metadata)
        logger.info(f"Classified {path.name} as {workflow.value} (doc_type={metadata.doc_type})")
        return workflow

    logger.warning(f"Unrecognized file type for {path}, treating as generic")
    return Workflow.GENERIC
