"""Runs the actual generation pipeline for one newly-detected file, dispatching
by workflow, merging results into the shared CSV, rebuilding the .apkg, and
(if configured) pushing new notes into a live Anki instance via AnkiConnect.
This is what watcher.py calls for each file event."""

import copy
import logging
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..build import build_anki_deck
from ..chunking import TextChunker
from ..config import Config, DocumentType
from ..dedup import create_deduplication_manager
from ..heuristics import DocumentAnalyzer, get_heuristic_defaults
from ..ids import create_id_manager
from ..io import merge_cards_into_csv, save_images
from ..llm import create_llm_provider
from ..preprocess import process_single_pdf
from ..prompts import create_prompt_manager
from ..rag import create_rag_manager
from ..readwise import process_readwise_document
from ..telemetry import create_telemetry_collector
from ..textbook import load_textbook_profile
from ..workflow_router import WORKFLOW_TO_DOCUMENT_TYPE, Workflow
from .ankiconnect import AnkiConnectClient, push_notes
from .classifier import classify_file

logger = logging.getLogger(__name__)

# examples/textbook_default.yml, resolved relative to this file so it works
# both in the dev repo and in the Docker image (where examples/ is copied to
# a sibling of src/, matching prompts.py's PromptManager path resolution).
_DEFAULT_TEXTBOOK_PROFILE_PATH = Path(__file__).resolve().parents[3] / "examples" / "textbook_default.yml"


def _effective_config_for_pdf(path: Path, config: Config, workflow: Workflow) -> tuple[Config, Optional[str]]:
    """Build a one-off effective config for a single PDF, applying the same
    heuristic chunking/strategy defaults `scan-docs` would - without needing a
    persistent documents.yaml scan of a whole directory first, since the
    service discovers files one at a time as they arrive.

    Returns `(effective_config, book_name)` - `book_name` is only meaningful
    for TEXTBOOK (a per-book instructions.yml's `deck_name`, if set) and is
    the label process_single_pdf nests this book's cards under
    ("<deck_name>::Textbooks::<book_name>") for the default "workflow"
    deck_structure - see workflow_router.deck_subdeck_for_workflow(). Falls
    back to the book's own directory name (process_single_pdf's default)
    when instructions.yml doesn't set one.
    """
    doc_type = WORKFLOW_TO_DOCUMENT_TYPE.get(workflow, DocumentType.UNKNOWN)
    if doc_type == DocumentType.UNKNOWN:
        return config, None

    analyzer = DocumentAnalyzer()
    metadata = analyzer.analyze_document(str(path))
    metadata = replace(metadata, doc_type=doc_type)
    defaults = get_heuristic_defaults(metadata)

    effective = copy.deepcopy(config)

    if "chunking_mode" in defaults:
        effective.pipeline.ingestion.chunking.mode = defaults["chunking_mode"]
        effective.pipeline.ingestion.chunking.tokens_per_chunk = defaults.get(
            "tokens_per_chunk", effective.pipeline.ingestion.chunking.tokens_per_chunk
        )
    if "extract_annotations" in defaults:
        effective.pipeline.ingestion.extract_annotations = defaults["extract_annotations"]
    if "strategies" in defaults:
        # This is a one-off per-file config (no documents.yaml compose-with-base
        # layering to preserve), so just enable exactly the suggested strategies.
        suggested = set(defaults["strategies"])
        for name in ("key_points", "cloze_definitions", "figure_based", "highlight_priority"):
            getattr(effective.generate.strategies, name).enabled = name in suggested

    book_name = None
    if workflow == Workflow.TEXTBOOK:
        # Per-book instructions.yml (hand-placed next to the PDF in the mounted
        # textbooks/ directory) takes full precedence over the generic
        # heuristic defaults above - this is the "custom instructions.yml per
        # textbook, default template if none provided" mount layout requested.
        default_path = _DEFAULT_TEXTBOOK_PROFILE_PATH if _DEFAULT_TEXTBOOK_PROFILE_PATH.exists() else None
        profile = load_textbook_profile(path.parent, default_profile_path=default_path)

        effective.pipeline.ingestion.chunking = profile.chunking
        for name in ("key_points", "cloze_definitions", "figure_based", "highlight_priority"):
            getattr(effective.generate.strategies, name).enabled = name in profile.strategies
        book_name = profile.deck_name
        if profile.extra_tags:
            effective.generate.tags.default_tags = list(
                dict.fromkeys(effective.generate.tags.default_tags + profile.extra_tags)
            )

    return effective, book_name


def process_new_file(path: Path, config: Config) -> Dict[str, Any]:
    """Process one newly-detected file end to end: classify, generate cards,
    merge into the shared CSV, rebuild the .apkg, and optionally push to
    AnkiConnect. Returns a summary dict (used for Slack notifications and
    logging) - never raises for generation/push failures beyond the point
    where cards exist, so one bad file doesn't take down the watcher loop.
    """
    path = Path(path)
    config.create_workspace()

    workflow = classify_file(path)
    logger.info(f"Processing {path.name} as workflow={workflow.value}")

    prompt_manager = create_prompt_manager()
    llm_provider = create_llm_provider(config.llm)
    id_manager = create_id_manager(config.ids)

    if workflow == Workflow.READWISE:
        cards = process_readwise_document(path, llm_provider, prompt_manager, deck_name=config.anki.deck_name)
        images = []
    else:
        effective_config, book_name = _effective_config_for_pdf(path, config, workflow)
        dedup_manager = create_deduplication_manager(effective_config.deduplication)
        rag_manager = create_rag_manager(effective_config.rag)
        telemetry = create_telemetry_collector(effective_config.telemetry)
        text_chunker = TextChunker(effective_config.ingestion.chunking, effective_config.llm.model)

        cards, images = process_single_pdf(
            pdf_path=path,
            config=effective_config,
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            text_chunker=text_chunker,
            id_manager=id_manager,
            dedup_manager=dedup_manager,
            rag_manager=rag_manager,
            telemetry=telemetry,
            book_name=book_name,
        )

    saved_images = save_images(images, config.output.media_path) if images else []

    now = datetime.now().isoformat()
    new_rows = []
    for card in cards:
        card_dict = asdict(card)
        card_dict["id"] = id_manager.generate_id(card)
        card_dict["created_at"] = now
        card_dict["updated_at"] = now
        # generate_cards() already set this per-card from the document's
        # workflow classification (see process_single_pdf/
        # process_readwise_document) - config.anki.deck_name is only a
        # fallback for cards that somehow didn't get one set.
        card_dict["deck"] = card_dict.get("deck") or config.anki.deck_name
        card_dict.setdefault("longtext", "")
        card_dict.setdefault("my_notes", "")
        new_rows.append(card_dict)

    merge_result = merge_cards_into_csv(new_rows, config.output.csv_path)

    apkg_path = None
    if merge_result["added"] > 0:
        build_result = build_anki_deck(config)
        apkg_path = str(build_result["apkg_path"])

    result: Dict[str, Any] = {
        "file": str(path),
        "workflow": workflow.value,
        "cards_generated": len(cards),
        "cards_added": merge_result["added"],
        "cards_total": merge_result["total"],
        "images_saved": len(saved_images),
        "apkg_path": apkg_path,
        "ankiconnect": None,
    }

    if config.service.ankiconnect.enabled and merge_result["added"] > 0:
        result["ankiconnect"] = _push_to_ankiconnect(config, new_rows)

    logger.info(
        f"Finished {path.name}: {result['cards_generated']} cards generated "
        f"(+{result['cards_added']} new, {result['cards_total']} total)"
    )
    return result


def _push_to_ankiconnect(config: Config, new_rows: list) -> Dict[str, Any]:
    """Push only the newly-added rows to AnkiConnect (the full CSV may contain
    cards from earlier runs already present in the Anki collection)."""
    client = AnkiConnectClient(url=config.service.ankiconnect.url, timeout=config.service.ankiconnect.timeout)
    if not client.is_available():
        logger.warning(f"AnkiConnect enabled but not reachable at {config.service.ankiconnect.url}")
        return {"attempted": len(new_rows), "added": 0, "failed": len(new_rows), "synced": False}

    # push_notes() applies one deck to the whole batch - fine here since
    # process_new_file() handles exactly one file per call, and one file has
    # exactly one resolved workflow deck (every row's own "deck" - see
    # generate_cards() - agrees). Previously this used config.anki.deck_name
    # directly, which silently ignored per-book/per-workflow deck overrides
    # for the live push (they only ever reached the CSV) - reading it off
    # the rows themselves fixes that.
    deck_name = new_rows[0].get("deck") or config.anki.deck_name if new_rows else config.anki.deck_name

    return push_notes(
        client,
        deck_name=deck_name,
        rows=new_rows,
        sync_after=config.service.ankiconnect.sync_after_update,
        media_path=config.output.media_path,
    )
