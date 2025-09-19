"""Main preprocessing pipeline for converting PDFs to flashcards."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .chunking import TextChunker
from .config import Config
from .dedup import create_deduplication_manager
from .ids import create_id_manager
from .io import find_pdf_files, save_csv, save_images, save_manifest
from .llm import create_llm_provider
from .pdf import extract_pdf_content
from .prompts import create_prompt_manager
from .rag import create_rag_manager
from .strategies import ClozeDefinitionsStrategy, FigureBasedStrategy, KeyPointsStrategy
from .strategies.base import FlashcardData
from .telemetry import create_telemetry_collector

logger = logging.getLogger(__name__)


def preprocess_pdf(config: Config, verbose: bool = False) -> Dict[str, Any]:
    """Main preprocessing function that converts PDFs to flashcard CSV."""
    
    # Set up logging level
    if verbose:
        logging.getLogger("pdf2anki").setLevel(logging.DEBUG)
    
    logger.info("Starting PDF preprocessing pipeline")
    
    # Initialize telemetry
    telemetry = create_telemetry_collector(config.telemetry)
    telemetry.start_phase("initialization")
    
    # Create workspace directories
    config.create_workspace()
    
    # Initialize components
    llm_provider = create_llm_provider(config.llm)
    prompt_manager = create_prompt_manager()
    text_chunker = TextChunker(config.ingestion.chunking, config.llm.model)
    id_manager = create_id_manager(config.ids)
    dedup_manager = create_deduplication_manager(config.deduplication)
    rag_manager = create_rag_manager(config.rag)
    
    # Load existing persistent IDs if using persistent strategy
    if config.ids.strategy.value == "persistent":
        id_manager.load_persistent_ids(config.output.csv_path)
    
    telemetry.end_phase()
    telemetry.start_phase("pdf_discovery")
    
    # Find PDF files
    pdf_files = find_pdf_files(
        config.inputs.paths,
        config.inputs.patterns,
        config.inputs.recursive
    )
    
    if not pdf_files:
        raise ValueError("No PDF files found with the specified patterns")
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    telemetry.end_phase()
    
    # Process each PDF
    all_cards = []
    all_images = []
    
    for pdf_path in pdf_files:
        try:
            cards, images = process_single_pdf(
                pdf_path=pdf_path,
                config=config,
                llm_provider=llm_provider,
                prompt_manager=prompt_manager,
                text_chunker=text_chunker,
                id_manager=id_manager,
                dedup_manager=dedup_manager,
                rag_manager=rag_manager,
                telemetry=telemetry
            )
            
            all_cards.extend(cards)
            all_images.extend(images)
            telemetry.record_pdf_processed()
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            telemetry.record_error("pdf_processing_error")
            continue
    
    telemetry.start_phase("finalization")
    
    # Apply global deduplication
    logger.info(f"Applying global deduplication to {len(all_cards)} cards")
    all_cards = dedup_manager.deduplicate_cards(all_cards)
    
    # Save images
    saved_images = []
    if all_images:
        saved_images = save_images(all_images, config.output.media_path)
    
    # Convert cards to dictionaries for CSV
    cards_data = []
    for card in all_cards:
        card_dict = card.dict()
        
        # Generate ID
        card_dict["id"] = id_manager.generate_id(card)
        
        # Add timestamps
        now = datetime.now().isoformat()
        card_dict["created_at"] = now
        card_dict["updated_at"] = now
        
        # Set deck name
        card_dict["deck"] = config.anki.deck_name
        
        # Add media references if relevant
        card_media = []
        # TODO: Link images to cards based on page ranges
        card_dict["media"] = card_media
        
        # Add additional metadata
        card_dict["longtext"] = ""  # For future use
        card_dict["my_notes"] = ""  # For user annotations
        
        cards_data.append(card_dict)
    
    # Save CSV
    save_csv(cards_data, config.output.csv_path)
    
    # Update persistent index
    dedup_manager.add_to_persistent_index(all_cards)
    if config.deduplication.index_path:
        dedup_manager.save_persistent_index(Path(config.deduplication.index_path))
    
    # Save RAG index
    if config.rag.index_path:
        rag_manager.save_index(Path(config.rag.index_path))
    
    telemetry.end_phase()
    
    # Create manifest
    manifest_data = {
        "project": config.project.dict(),
        "processing": {
            "pdf_files": [str(p) for p in pdf_files],
            "total_pdfs": len(pdf_files),
            "total_cards": len(all_cards),
            "total_images": len(saved_images),
            "strategies_used": list(set(card.strategy for card in all_cards)),
        },
        "output": {
            "csv_path": str(config.output.csv_path),
            "media_path": str(config.output.media_path),
            "apkg_path": str(config.output.apkg_path),
        },
        **telemetry.get_manifest_data(config.llm.model)
    }
    
    save_manifest(manifest_data, config.output.manifest_path)
    
    # Log summary
    telemetry.log_summary()
    
    result = {
        "total_cards": len(all_cards),
        "processed_pdfs": len(pdf_files),
        "csv_path": config.output.csv_path,
        "media_path": config.output.media_path,
        "manifest_path": config.output.manifest_path,
        "images_saved": len(saved_images),
    }
    
    logger.info(f"Preprocessing complete: {len(all_cards)} cards generated from {len(pdf_files)} PDFs")
    return result


def process_single_pdf(
    pdf_path: Path,
    config: Config,
    llm_provider,
    prompt_manager,
    text_chunker,
    id_manager,
    dedup_manager,
    rag_manager,
    telemetry
) -> tuple[List[FlashcardData], List[Dict[str, Any]]]:
    """Process a single PDF file."""
    
    logger.info(f"Processing PDF: {pdf_path}")
    telemetry.start_phase(f"pdf_extraction_{pdf_path.name}")
    
    # Extract PDF content
    pdf_content = extract_pdf_content(
        pdf_path=pdf_path,
        extract_images=config.ingestion.extract_images,
        extract_structure=True,
        ocr_fallback=config.ingestion.ocr_fallback
    )
    
    telemetry.end_phase()
    telemetry.start_phase(f"chunking_{pdf_path.name}")
    
    # Chunk the text
    chunks = text_chunker.chunk_document(pdf_content)
    logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
    
    telemetry.end_phase()
    
    # Build RAG index if enabled
    if config.rag.enabled:
        telemetry.start_phase(f"rag_indexing_{pdf_path.name}")
        rag_manager.build_index(chunks)
        telemetry.end_phase()
    
    # Initialize strategies
    strategies = []
    
    if config.strategies.key_points.enabled:
        strategies.append(KeyPointsStrategy(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            strategy_config=config.strategies.key_points,
            strategy_name="key_points"
        ))
    
    if config.strategies.cloze_definitions.enabled:
        strategies.append(ClozeDefinitionsStrategy(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            strategy_config=config.strategies.cloze_definitions,
            strategy_name="cloze_definitions"
        ))
    
    if config.strategies.figure_based.enabled:
        strategies.append(FigureBasedStrategy(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            strategy_config=config.strategies.figure_based,
            strategy_name="figure_based"
        ))
    
    # Generate cards from chunks
    all_cards = []
    
    for chunk_idx, chunk in enumerate(chunks):
        telemetry.start_phase(f"generation_chunk_{chunk_idx}")
        telemetry.record_chunk_processed()
        
        logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                    f"(pages {chunk.start_page}-{chunk.end_page})")
        
        chunk_cards = []
        
        for strategy in strategies:
            try:
                # Enhance chunk with RAG context if enabled
                enhanced_chunk = chunk
                if config.rag.enabled:
                    enhanced_text = rag_manager.enhance_chunk_context(chunk)
                    if enhanced_text != chunk.text:
                        enhanced_chunk = chunk
                        enhanced_chunk.text = enhanced_text
                
                # Check if strategy should apply to this chunk
                should_apply = True
                if hasattr(strategy, 'should_apply_to_chunk'):
                    should_apply = strategy.should_apply_to_chunk(chunk, pdf_content)
                
                if not should_apply:
                    logger.debug(f"Skipping {strategy.name} for chunk {chunk_idx} (not applicable)")
                    continue
                
                # Generate cards
                strategy_cards = strategy.generate_cards(
                    chunk=enhanced_chunk,
                    pdf_metadata=pdf_content["metadata"],
                    max_cards=strategy.config.params.get("max_cards", 5)
                )
                
                # Apply strategy-level deduplication
                strategy_cards = strategy.deduplicate_cards(strategy_cards)
                
                chunk_cards.extend(strategy_cards)
                
                # Record telemetry
                telemetry.record_cards_generated(len(strategy_cards), strategy.name)
                telemetry.record_strategy_metrics(strategy.name, llm_provider.get_stats())
                
                logger.debug(f"Generated {len(strategy_cards)} cards using {strategy.name}")
                
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed on chunk {chunk_idx}: {e}")
                telemetry.record_error("strategy_error", strategy.name)
                continue
        
        # Apply hallucination checks
        if config.hallucination.require_citations or config.hallucination.verify_quotes:
            chunk_cards = apply_hallucination_checks(chunk_cards, chunk, config.hallucination)
        
        all_cards.extend(chunk_cards)
        telemetry.end_phase()
    
    # Apply review step if enabled
    if config.review.enabled:
        telemetry.start_phase(f"review_{pdf_path.name}")
        all_cards = apply_review_step(all_cards, llm_provider, prompt_manager, config.review)
        telemetry.end_phase()
    
    logger.info(f"Generated {len(all_cards)} cards from {pdf_path}")
    
    return all_cards, pdf_content.get("images", [])


def apply_hallucination_checks(
    cards: List[FlashcardData], 
    chunk, 
    hallucination_config
) -> List[FlashcardData]:
    """Apply hallucination mitigation checks to generated cards."""
    
    if not cards:
        return cards
    
    valid_cards = []
    
    for card in cards:
        is_valid = True
        
        # Check citations
        if hallucination_config.require_citations:
            if not card.page_citation or not card.ref_citation:
                logger.debug(f"Card missing citation: {card.dict()}")
                is_valid = False
        
        # Verify quotes (basic implementation)
        if hallucination_config.verify_quotes and is_valid:
            # Extract quoted text from card content
            card_text = ""
            if card.front:
                card_text += card.front + " "
            if card.back:
                card_text += card.back + " "
            if card.cloze_text:
                card_text += card.cloze_text + " "
            if card.extra:
                card_text += card.extra + " "
            
            # Simple quote verification - check if quoted phrases exist in source
            import re
            quotes = re.findall(r'"([^"]*)"', card_text)
            for quote in quotes:
                if len(quote) > 10 and quote.lower() not in chunk.text.lower():
                    logger.debug(f"Quote not found in source: {quote}")
                    is_valid = False
                    break
        
        if is_valid or not hallucination_config.drop_on_failure:
            valid_cards.append(card)
        else:
            logger.debug(f"Dropped card due to hallucination check failure")
    
    if len(valid_cards) < len(cards):
        logger.info(f"Hallucination checks removed {len(cards) - len(valid_cards)} cards")
    
    return valid_cards


def apply_review_step(cards, llm_provider, prompt_manager, review_config) -> List[FlashcardData]:
    """Apply LLM review step to improve card quality."""
    
    logger.info(f"Applying review step to {len(cards)} cards")
    
    # TODO: Implement LLM-based review
    # For now, just return cards as-is
    logger.debug("Review step not fully implemented yet")
    
    return cards