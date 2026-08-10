"""Tests for the AnkiConnect client, using a mocked HTTP layer - there's no
real Anki instance available in CI/sandbox environments."""

import base64
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from pdf2anki.note_types import BASIC_MODEL_NAME, CLOZE_MODEL_NAME
from pdf2anki.service.ankiconnect import (
    AnkiConnectClient,
    AnkiConnectError,
    card_row_to_note,
    ensure_media_uploaded,
    ensure_note_types,
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


def test_model_names_invokes_modelNames_action():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"result": [BASIC_MODEL_NAME], "error": None})
        result = client.model_names()

    assert result == [BASIC_MODEL_NAME]
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["action"] == "modelNames"


def test_create_model_invokes_createModel_with_expected_params():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"result": None, "error": None})
        client.create_model(
            CLOZE_MODEL_NAME,
            ["Text", "Extra"],
            "body { color: red }",
            [{"Name": "Cloze", "Front": "{{cloze:Text}}", "Back": "{{cloze:Text}}"}],
            is_cloze=True,
        )

    sent_params = mock_post.call_args.kwargs["json"]["params"]
    assert sent_params["modelName"] == CLOZE_MODEL_NAME
    assert sent_params["inOrderFields"] == ["Text", "Extra"]
    assert sent_params["isCloze"] is True
    assert sent_params["cardTemplates"][0]["Name"] == "Cloze"


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
        "media": ["highlight_p5_0_abc.png"],
    }
    note = card_row_to_note(row, deck_name="My Deck")

    assert note["deckName"] == "My Deck"
    assert note["modelName"] == "PDF2Anki Basic"
    assert note["fields"]["Front"] == "Q?"
    assert note["fields"]["Back"] == "A."
    assert note["fields"]["Page"] == "p. 5"
    assert note["fields"]["Image"] == '<img src="highlight_p5_0_abc.png">'
    assert note["tags"] == ["bio", "cells"]


def test_card_row_to_note_basic_empty_image_field_without_media():
    row = {"note_type": "Basic", "front": "Q?", "back": "A.", "tags": []}
    note = card_row_to_note(row, deck_name="My Deck")
    assert note["fields"]["Image"] == ""


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


def test_ensure_note_types_creates_both_missing_models():
    """Reproduces the real failure: a collection that has only ever received
    notes via the live AnkiConnect push (never imported a .apkg) never gets
    the PDF2Anki note types created, so addNotes fails outright with 'model
    was not found'."""
    client = Mock()
    client.model_names.return_value = []

    ensure_note_types(client)

    assert client.create_model.call_count == 2
    created_names = {call.args[0] for call in client.create_model.call_args_list}
    assert created_names == {BASIC_MODEL_NAME, CLOZE_MODEL_NAME}


def test_ensure_note_types_skips_models_that_already_exist():
    client = Mock()
    client.model_names.return_value = [BASIC_MODEL_NAME, CLOZE_MODEL_NAME, "Some Other Model"]

    ensure_note_types(client)

    client.create_model.assert_not_called()


def test_ensure_note_types_creates_only_the_missing_one():
    client = Mock()
    client.model_names.return_value = [BASIC_MODEL_NAME]

    ensure_note_types(client)

    client.create_model.assert_called_once()
    assert client.create_model.call_args.args[0] == CLOZE_MODEL_NAME
    assert client.create_model.call_args.kwargs.get("is_cloze") is True


def test_ensure_note_types_does_not_raise_if_model_names_fails():
    client = Mock()
    client.model_names.side_effect = Exception("connection refused")

    ensure_note_types(client)  # must not raise

    client.create_model.assert_not_called()


def test_ensure_note_types_does_not_raise_if_create_model_fails():
    client = Mock()
    client.model_names.return_value = []
    client.create_model.side_effect = Exception("boom")

    ensure_note_types(client)  # must not raise despite both creations failing

    assert client.create_model.call_count == 2


def test_push_notes_ensures_note_types_before_adding():
    client = Mock()
    client.model_names.return_value = []
    client.add_notes.return_value = [111]

    rows = [{"note_type": "Basic", "front": "Q", "back": "A", "tags": []}]
    result = push_notes(client, deck_name="Deck", rows=rows, sync_after=False)

    assert client.create_model.call_count == 2
    assert result["added"] == 1


def test_store_media_file_base64_encodes_and_invokes_storeMediaFile():
    client = AnkiConnectClient()
    with patch("pdf2anki.service.ankiconnect.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"result": "a.png", "error": None})
        client.store_media_file("a.png", b"\x89PNG\r\n")

    sent_params = mock_post.call_args.kwargs["json"]["params"]
    assert sent_params["filename"] == "a.png"
    assert base64.b64decode(sent_params["data"]) == b"\x89PNG\r\n"


def test_ensure_media_uploaded_uploads_each_unique_file(tmp_path):
    (tmp_path / "a.png").write_bytes(b"aaa")
    (tmp_path / "b.png").write_bytes(b"bbb")

    client = Mock()
    rows = [
        {"media": ["a.png"]},
        {"media": ["a.png", "b.png"]},  # "a.png" repeated - should upload once
        {"media": []},
    ]
    ensure_media_uploaded(client, rows, tmp_path)

    assert client.store_media_file.call_count == 2
    uploaded_names = {call.args[0] for call in client.store_media_file.call_args_list}
    assert uploaded_names == {"a.png", "b.png"}


def test_ensure_media_uploaded_skips_missing_file_without_raising(tmp_path):
    client = Mock()
    rows = [{"media": ["does_not_exist.png"]}]

    ensure_media_uploaded(client, rows, tmp_path)  # must not raise

    client.store_media_file.assert_not_called()


def test_push_notes_uploads_media_when_media_path_given(tmp_path):
    client = Mock()
    client.model_names.return_value = ["PDF2Anki Basic", "PDF2Anki Cloze"]
    client.add_notes.return_value = [111]

    (tmp_path / "a.png").write_bytes(b"aaa")

    rows = [{"note_type": "Basic", "front": "Q", "back": "A", "tags": [], "media": ["a.png"]}]
    push_notes(client, deck_name="Deck", rows=rows, sync_after=False, media_path=tmp_path)

    client.store_media_file.assert_called_once_with("a.png", b"aaa")


def test_push_notes_skips_media_upload_when_media_path_omitted():
    client = Mock()
    client.model_names.return_value = ["PDF2Anki Basic", "PDF2Anki Cloze"]
    client.add_notes.return_value = [111]

    rows = [{"note_type": "Basic", "front": "Q", "back": "A", "tags": [], "media": ["a.png"]}]
    push_notes(client, deck_name="Deck", rows=rows, sync_after=False)

    client.store_media_file.assert_not_called()
