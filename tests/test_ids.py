"""Tests for ID generation and management."""

import pytest

from pdf2anki.config import IdsConfig, IdStrategy
from pdf2anki.ids import IDManager
from pdf2anki.strategies.base import FlashcardData


def test_content_hash_id_generation():
    """Test content-hash based ID generation."""
    config = IdsConfig(strategy=IdStrategy.CONTENT_HASH, salt="test_salt")
    id_manager = IDManager(config)
    
    # Create test card
    card = FlashcardData(
        note_type="Basic",
        front="What is Python?",
        back="A programming language",
        page_citation="p. 1",
        core_concept="Programming",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1
    )
    
    # Generate ID
    id1 = id_manager.generate_id(card)
    
    # Should be deterministic
    id2 = id_manager.generate_id(card)
    assert id1 == id2
    
    # Should be reasonable length
    assert len(id1) == 32  # BLAKE2b with 16 bytes = 32 hex chars
    
    # Should change if content changes
    card.front = "What is Java?"
    id3 = id_manager.generate_id(card)
    assert id1 != id3


def test_persistent_id_generation():
    """Test persistent ID generation."""
    config = IdsConfig(strategy=IdStrategy.PERSISTENT, salt="test_salt")
    id_manager = IDManager(config)
    
    # Create test card
    card = FlashcardData(
        note_type="Basic",
        front="What is Python?",
        back="A programming language",
        page_citation="p. 1",
        core_concept="Programming",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1,
        strategy="key_points"
    )
    
    # Generate ID
    id1 = id_manager.generate_id(card)
    
    # Should be persistent for same key
    id2 = id_manager.generate_id(card)
    assert id1 == id2
    
    # Should follow expected format
    assert id1.startswith("test_salt_")
    assert len(id1) > len("test_salt_")
    
    # Should survive content changes
    card.front = "What is Java?"
    id3 = id_manager.generate_id(card)
    assert id1 == id3  # Same key, same ID
    
    # Should change for different core concept
    card.core_concept = "Different Concept"
    id4 = id_manager.generate_id(card)
    assert id1 != id4  # Different key, different ID


def test_id_consistency_across_note_types():
    """Test that different note types generate different IDs."""
    config = IdsConfig(strategy=IdStrategy.CONTENT_HASH, salt="test")
    id_manager = IDManager(config)
    
    # Basic card
    basic_card = FlashcardData(
        note_type="Basic",
        front="Question",
        back="Answer",
        page_citation="p. 1",
        core_concept="Test",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1
    )
    
    # Cloze card with same content
    cloze_card = FlashcardData(
        note_type="Cloze",
        cloze_text="{{c1::Question}} and Answer",
        page_citation="p. 1",
        core_concept="Test",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1
    )
    
    basic_id = id_manager.generate_id(basic_card)
    cloze_id = id_manager.generate_id(cloze_card)
    
    # Should be different due to different note types
    assert basic_id != cloze_id


def test_salt_affects_ids():
    """Test that different salts produce different IDs."""
    card = FlashcardData(
        note_type="Basic",
        front="Question",
        back="Answer",
        page_citation="p. 1",
        core_concept="Test",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1
    )
    
    # Different salts
    config1 = IdsConfig(strategy=IdStrategy.CONTENT_HASH, salt="salt1")
    config2 = IdsConfig(strategy=IdStrategy.CONTENT_HASH, salt="salt2")
    
    id_manager1 = IDManager(config1)
    id_manager2 = IDManager(config2)
    
    id1 = id_manager1.generate_id(card)
    id2 = id_manager2.generate_id(card)
    
    assert id1 != id2


def test_id_manager_stats():
    """Test ID manager statistics."""
    config = IdsConfig(strategy=IdStrategy.PERSISTENT, salt="test")
    id_manager = IDManager(config)
    
    # Initial stats
    stats = id_manager.get_stats()
    assert stats["persistent_ids_loaded"] == 0
    assert stats["next_id_counter"] == 1
    
    # Generate some IDs
    card = FlashcardData(
        note_type="Basic",
        front="Question",
        back="Answer",
        page_citation="p. 1",
        core_concept="Test",
        source_pdf="test.pdf",
        page_start=1,
        page_end=1,
        strategy="test"
    )
    
    id_manager.generate_id(card)
    id_manager.generate_id(card)  # Same card, should reuse ID
    
    # Change card to force new ID
    card.core_concept = "Different"
    id_manager.generate_id(card)
    
    # Check updated stats
    stats = id_manager.get_stats()
    assert stats["next_id_counter"] == 3  # Two unique cards generated