"""Tests for Slack webhook notifications, mocking the HTTP layer."""

from unittest.mock import Mock, patch

import requests

from pdf2anki.service.notify import (
    notify_error,
    notify_file_processed,
    notify_startup,
    send_slack_notification,
)


def test_no_webhook_configured_is_a_noop():
    assert send_slack_notification(None, "hello") is False


def test_send_notification_posts_text_payload():
    with patch("pdf2anki.service.notify.requests.post") as mock_post:
        mock_post.return_value = Mock(raise_for_status=Mock())
        result = send_slack_notification("https://hooks.slack.com/xyz", "hello world")

    assert result is True
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"text": "hello world"}


def test_send_notification_failure_does_not_raise():
    with patch("pdf2anki.service.notify.requests.post", side_effect=requests.ConnectionError("down")):
        result = send_slack_notification("https://hooks.slack.com/xyz", "hello")
    assert result is False


def test_notify_file_processed_includes_counts_and_apkg_path():
    result = {
        "file": "book.pdf",
        "workflow": "textbook",
        "cards_generated": 10,
        "cards_added": 4,
        "cards_total": 40,
        "images_saved": 2,
        "apkg_path": "/data/workspace/deck.apkg",
        "ankiconnect": None,
    }
    with patch("pdf2anki.service.notify.requests.post") as mock_post:
        mock_post.return_value = Mock(raise_for_status=Mock())
        notify_file_processed("https://hooks.slack.com/xyz", result)

    text = mock_post.call_args.kwargs["json"]["text"]
    assert "book.pdf" in text
    assert "textbook" in text
    assert "deck.apkg" in text


def test_notify_file_processed_includes_ankiconnect_summary():
    result = {
        "file": "book.pdf", "workflow": "textbook", "cards_generated": 1,
        "cards_added": 1, "cards_total": 1, "images_saved": 0, "apkg_path": None,
        "ankiconnect": {"attempted": 1, "added": 1, "failed": 0, "synced": True},
    }
    with patch("pdf2anki.service.notify.requests.post") as mock_post:
        mock_post.return_value = Mock(raise_for_status=Mock())
        notify_file_processed("https://hooks.slack.com/xyz", result)

    text = mock_post.call_args.kwargs["json"]["text"]
    assert "synced to AnkiWeb" in text


def test_notify_error_includes_error_text():
    with patch("pdf2anki.service.notify.requests.post") as mock_post:
        mock_post.return_value = Mock(raise_for_status=Mock())
        notify_error("https://hooks.slack.com/xyz", "book.pdf", "boom, it broke")

    text = mock_post.call_args.kwargs["json"]["text"]
    assert "book.pdf" in text
    assert "boom, it broke" in text


def test_notify_startup_lists_watch_dirs():
    with patch("pdf2anki.service.notify.requests.post") as mock_post:
        mock_post.return_value = Mock(raise_for_status=Mock())
        notify_startup("https://hooks.slack.com/xyz", {"pdfs": "/data/pdfs", "textbooks": None})

    text = mock_post.call_args.kwargs["json"]["text"]
    assert "/data/pdfs" in text
    assert "textbooks" not in text  # None-valued dirs are skipped entirely
