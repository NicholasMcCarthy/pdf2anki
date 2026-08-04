"""Tests for build.py's Anki deck building - specifically that model/deck ids
are now deterministic across rebuilds (required for the watcher service's
repeated-rebuild-and-reimport design to actually be idempotent in Anki).

AnkiDeckBuilder is always constructed with the Anki sub-config (config.anki),
not the top-level Config - see build_anki_deck()."""

from pathlib import Path

import pandas as pd

from pdf2anki.build import AnkiDeckBuilder, _stable_id, build_anki_deck
from pdf2anki.config import Config
from pdf2anki.io import save_csv


def test_stable_id_is_deterministic():
    assert _stable_id("same-seed") == _stable_id("same-seed")
    assert _stable_id("seed-a") != _stable_id("seed-b")


def test_stable_id_within_genanki_range():
    value = _stable_id("pdf2anki-basic-note-type-v1")
    assert (1 << 30) <= value < (1 << 31)


def test_note_type_model_ids_are_stable_across_builder_instances():
    config = Config()
    builder1 = AnkiDeckBuilder(config.anki)
    builder2 = AnkiDeckBuilder(config.anki)

    assert builder1.note_types["Basic"].model_id == builder2.note_types["Basic"].model_id
    assert builder1.note_types["Cloze"].model_id == builder2.note_types["Cloze"].model_id
    assert builder1.note_types["Basic"].model_id != builder1.note_types["Cloze"].model_id


def test_flat_deck_id_is_stable_across_builder_instances():
    config = Config()
    config.anki.deck_structure = "flat"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([{"section": None, "strategy": "key_points"}])

    decks1 = AnkiDeckBuilder(config.anki)._create_decks(df)
    decks2 = AnkiDeckBuilder(config.anki)._create_decks(df)

    assert decks1["main"].deck_id == decks2["main"].deck_id


def test_chapter_subdeck_ids_are_stable_and_keyed_by_name():
    config = Config()
    config.anki.deck_structure = "chapter"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([
        {"section": "Chapter 1", "strategy": "key_points"},
        {"section": "Chapter 2", "strategy": "key_points"},
    ])

    decks1 = AnkiDeckBuilder(config.anki)._create_decks(df)
    decks2 = AnkiDeckBuilder(config.anki)._create_decks(df)

    assert decks1["Chapter 1"].deck_id == decks2["Chapter 1"].deck_id
    assert decks1["Chapter 2"].deck_id == decks2["Chapter 2"].deck_id
    assert decks1["Chapter 1"].deck_id != decks1["Chapter 2"].deck_id


def test_explicit_deck_id_config_still_wins():
    config = Config()
    config.anki.deck_structure = "flat"
    config.anki.deck_id = 123456789

    df = pd.DataFrame([{"section": None, "strategy": "key_points"}])
    decks = AnkiDeckBuilder(config.anki)._create_decks(df)

    assert decks["main"].deck_id == 123456789


def test_build_anki_deck_end_to_end_writes_apkg_twice_with_same_ids(tmp_path):
    """build_anki_deck() is the actual public entrypoint (called from both the
    `pdf2anki build` CLI command and the watcher service's runner) - exercise
    it directly rather than only its internals."""
    config = Config()
    config.generate.output.workspace = tmp_path
    config.generate.output.csv_path = tmp_path / "cards.csv"
    config.generate.output.media_path = tmp_path / "media"
    config.generate.output.apkg_path = tmp_path / "deck.apkg"
    config.anki.deck_name = "Test Deck"

    save_csv([{
        "id": "card1",
        "note_type": "Basic",
        "front": "What is 2+2?",
        "back": "4",
        "tags": ["math"],
        "source_pdf": "test.pdf",
        "page_start": 1,
        "page_end": 1,
    }], config.output.csv_path)

    result1 = build_anki_deck(config)
    assert config.output.apkg_path.exists()
    assert result1["total_cards"] == 1

    # Rebuilding (as the watcher service does after every new file) should
    # succeed again and use the same deck/model ids both times.
    from pdf2anki.build import AnkiDeckBuilder
    deck_id_1 = AnkiDeckBuilder(config.anki)._create_decks(pd.DataFrame([{"section": None, "strategy": "key_points"}]))["main"].deck_id

    result2 = build_anki_deck(config)
    deck_id_2 = AnkiDeckBuilder(config.anki)._create_decks(pd.DataFrame([{"section": None, "strategy": "key_points"}]))["main"].deck_id

    assert result2["total_cards"] == 1
    assert deck_id_1 == deck_id_2
