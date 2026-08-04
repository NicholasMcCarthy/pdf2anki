"""Tests for the AnkiConnect client, using a mocked HTTP layer - there's no
real Anki instance available in CI/sandbox environments."""

from unittest.mock import Mock, patch

import pytest
import requests

from pdf2anki.service.ankiconnect import (
    AnkiConnectClient,
    AnkiConnectError,
    card_row_to_note,
    push_notes,
)


def _mock_response(json_data, status_ok=True):
    resp = Mock()
    resp.json.return_value = json_data
    resp.raise_for_status = Mock() if status_ok else Mock(side_effect=requests.HTTPError("boom"))
    return resp


def test_invoke_returns_result_on_success():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"result": 42, "error": None})
        result = client.invoke("version")
    assert result == 42


def test_invoke_raises_on_ankiconnect_error():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"result": None, "error": "deck missing"})
        with pytest.raises(AnkiConnectError):
            client.invoke("addNotes", notes=[])


def test_is_available_false_on_connection_error():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post", side_effect=requests.ConnectionError()):
        assert client.is_available() is False


def test_card_row_to_note_basic():
    row = {
        "note_type": "Basic",
        "front": "Q?",
        "back": "A.",
        "source_pdf": "doc.pdf",
        "page_start": 5,
        "section": "Intro",
        "extra": "extra info",
        "tags": "bio;cells",
    }
    note = card_row_to_note(row, deck_name="My Deck")

    assert note["deckName"] == "My Deck"
    assert note["modelName"] == "PDF2Anki Basic"
    assert note["fields"]["Front"] == "Q?"
    assert note["fields"]["Back"] == "A."
    assert note["fields"]["Page"] == "p. 5"
    assert note["tags"] == ["bio", "cells"]


def test_card_row_to_note_cloze():
    row = {
        "note_type": "Cloze",
        "cloze_text": "The {{c1::mitochondria}} is...",
        "extra": "context",
        "source_pdf": "doc.pdf",
        "page_start": 2,
        "tags": [],
    }
    note = card_row_to_note(row, deck_name="My Deck")

    assert note["modelName"] == "PDF2Anki Cloze"
    assert "mitochondria" in note["fields"]["Text"]


def test_push_notes_counts_partial_failures():
    client = Mock()
    client.add_notes.return_value = [111, None, 222]  # middle one failed (e.g. duplicate)

    rows = [
        {"note_type": "Basic", "front": "Q1", "back": "A1", "tags": []},
        {"note_type": "Basic", "front": "Q2", "back": "A2", "tags": []},
        {"note_type": "Basic", "front": "Q3", "back": "A3", "tags": []},
    ]
    result = push_notes(client, deck_name="Deck", rows=rows, sync_after=True)

    assert result["attempted"] == 3
    assert result["added"] == 2
    assert result["failed"] == 1
    assert result["synced"] is True
    client.create_deck.assert_called_once_with("Deck")
    client.sync.assert_called_once()


def test_push_notes_handles_total_failure_gracefully():
    client = Mock()
    client.create_deck.side_effect = Exception("connection refused")

    rows = [{"note_type": "Basic", "front": "Q", "back": "A", "tags": []}]
    result = push_notes(client, deck_name="Deck", rows=rows)

    assert result["failed"] == 1
    assert result["added"] == 0


def test_push_notes_empty_rows_is_a_noop():
    client = Mock()
    result = push_notes(client, deck_name="Deck", rows=[])
    assert result == {"attempted": 0, "added": 0, "failed": 0, "synced": False}
    client.create_deck.assert_not_called()
