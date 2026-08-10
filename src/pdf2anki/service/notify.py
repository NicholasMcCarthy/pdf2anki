"""Slack webhook notifications for the watcher service. Every function here
swallows its own failures (logs and returns False) rather than raising, so a
Slack outage or bad webhook URL never blocks or crashes the generation
pipeline itself."""

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def send_slack_notification(webhook_url: Optional[str], text: str) -> bool:
    """POST a simple text message to a Slack incoming webhook."""
    if not webhook_url:
        logger.debug("No Slack webhook configured, skipping notification")
        return False

    try:
        response = requests.post(webhook_url, json={"text": text}, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")
        return False


def notify_file_processed(webhook_url: Optional[str], result: Dict[str, Any]) -> bool:
    """Notify that a file finished processing (see service.runner.process_new_file)."""
    text = (
        f":books: *pdf2anki* processed `{result['file']}` as *{result['workflow']}*\n"
        f"- Cards generated: {result['cards_generated']} "
        f"(+{result['cards_added']} new, {result['cards_total']} total in deck)\n"
    )
    if result.get("images_saved"):
        text += f"- Media saved: {result['images_saved']}\n"
    if result.get("apkg_path"):
        text += f"- Deck rebuilt: `{result['apkg_path']}`\n"
    if "ankiconnect" in result and result["ankiconnect"] is not None:
        ac = result["ankiconnect"]
        text += (
            f"- AnkiConnect: {ac['added']} added"
            + (f", {ac['failed']} failed" if ac.get("failed") else "")
            + (", synced to AnkiWeb" if ac.get("synced") else "")
            + "\n"
        )
    return send_slack_notification(webhook_url, text)


def notify_error(webhook_url: Optional[str], file: str, error: str) -> bool:
    """Notify that processing a file failed."""
    text = f":warning: *pdf2anki* failed to process `{file}`\n```{error}```"
    return send_slack_notification(webhook_url, text)


def notify_startup(webhook_url: Optional[str], watch_dirs: Dict[str, Optional[str]]) -> bool:
    """Notify that the watcher service has started."""
    dirs_text = "\n".join(f"- {name}: `{path}`" for name, path in watch_dirs.items() if path)
    text = f":rocket: *pdf2anki* watcher service started, watching:\n{dirs_text}"
    return send_slack_notification(webhook_url, text)
