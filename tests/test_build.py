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
from pdf2anki.note_types import BASIC_FIELDS, CLOZE_FIELDS


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


def test_workflow_deck_structure_is_the_default():
    assert Config().anki.deck_structure == "workflow"


def test_workflow_subdecks_keyed_by_full_resolved_deck_name():
    config = Config()
    config.anki.deck_structure = "workflow"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([
        {"deck": "My Deck::Readwise"},
        {"deck": "My Deck::Articles"},
        {"deck": "My Deck::Textbooks::Some Book"},
    ])

    builder = AnkiDeckBuilder(config.anki)
    decks = builder._create_decks(df)

    assert set(decks.keys()) >= {"My Deck::Readwise", "My Deck::Articles", "My Deck::Textbooks::Some Book"}
    assert decks["My Deck::Readwise"].name == "My Deck::Readwise"
    assert decks["My Deck::Textbooks::Some Book"].name == "My Deck::Textbooks::Some Book"


def test_workflow_subdeck_ids_are_stable_and_distinct():
    config = Config()
    config.anki.deck_structure = "workflow"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([{"deck": "My Deck::Readwise"}, {"deck": "My Deck::Articles"}])

    decks1 = AnkiDeckBuilder(config.anki)._create_decks(df)
    decks2 = AnkiDeckBuilder(config.anki)._create_decks(df)

    assert decks1["My Deck::Readwise"].deck_id == decks2["My Deck::Readwise"].deck_id
    assert decks1["My Deck::Readwise"].deck_id != decks1["My Deck::Articles"].deck_id


def test_workflow_deck_structure_falls_back_to_main_for_generic_cards():
    """A card with no resolved "deck" (e.g. GENERIC/unclassified) lands in
    the base deck, not a missing/empty-named subdeck."""
    config = Config()
    config.anki.deck_structure = "workflow"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([{"deck": ""}])
    builder = AnkiDeckBuilder(config.anki)
    decks = builder._create_decks(df)

    assert "main" in decks
    assert decks["main"].name == "My Deck"
    assert builder._get_deck_for_card(df.iloc[0], decks).name == "My Deck"


def test_workflow_deck_structure_handles_missing_deck_column():
    """_create_decks() is sometimes called directly with a hand-built
    DataFrame that has no "deck" column at all (not just empty values) -
    must not crash."""
    config = Config()
    config.anki.deck_structure = "workflow"
    df = pd.DataFrame([{"section": None, "strategy": "key_points"}])

    decks = AnkiDeckBuilder(config.anki)._create_decks(df)
    assert "main" in decks


def test_get_deck_for_card_routes_to_correct_workflow_subdeck():
    config = Config()
    config.anki.deck_structure = "workflow"
    config.anki.deck_name = "My Deck"

    df = pd.DataFrame([{"deck": "My Deck::Readwise"}, {"deck": "My Deck::Articles"}])
    builder = AnkiDeckBuilder(config.anki)
    decks = builder._create_decks(df)

    assert builder._get_deck_for_card(df.iloc[0], decks).name == "My Deck::Readwise"
    assert builder._get_deck_for_card(df.iloc[1], decks).name == "My Deck::Articles"


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


def test_basic_note_includes_image_field_from_media():
    """Reproduces a real bug: screenshot files were bundled into the .apkg
    (genanki.Package.media_files globs the whole media dir) but no note
    field ever referenced them via <img>, so they were orphaned and never
    displayed on any card."""
    config = Config()
    builder = AnkiDeckBuilder(config.anki)
    row = pd.Series({
        "front": "Q", "back": "A", "media": ["highlight_p1_0_abc.png"],
        "source_pdf": "test.pdf", "source_title": "", "page_start": 1, "page_end": 1,
        "section": "", "tags": [], "extra": "",
    })

    note = builder._create_basic_note(row, builder.note_types["Basic"])

    image_field_index = BASIC_FIELDS.index("Image")
    assert note.fields[image_field_index] == '<img src="highlight_p1_0_abc.png">'


def test_cloze_note_includes_image_field_from_media():
    config = Config()
    builder = AnkiDeckBuilder(config.anki)
    row = pd.Series({
        "cloze_text": "The {{c1::mitochondria}} is...", "extra": "",
        "media": ["fig_p2_0_def.png"],
        "source_pdf": "test.pdf", "source_title": "", "page_start": 2, "page_end": 2,
        "section": "", "tags": [],
    })

    note = builder._create_cloze_note(row, builder.note_types["Cloze"])

    image_field_index = CLOZE_FIELDS.index("Image")
    assert note.fields[image_field_index] == '<img src="fig_p2_0_def.png">'


def test_basic_note_image_field_empty_without_media():
    config = Config()
    builder = AnkiDeckBuilder(config.anki)
    row = pd.Series({
        "front": "Q", "back": "A", "media": [],
        "source_pdf": "test.pdf", "source_title": "", "page_start": 1, "page_end": 1,
        "section": "", "tags": [], "extra": "",
    })

    note = builder._create_basic_note(row, builder.note_types["Basic"])

    assert note.fields[BASIC_FIELDS.index("Image")] == ""
