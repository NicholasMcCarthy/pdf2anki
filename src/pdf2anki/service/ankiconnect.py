"""Minimal AnkiConnect HTTP client: adds notes to a running Anki desktop
instance (via the AnkiConnect add-on) and triggers its built-in AnkiWeb sync.

AnkiWeb has no public API for direct .apkg upload - talking to a real Anki
client that already has AnkiWeb sync configured is the only realistic way to
push new cards into AnkiWeb automatically, hence this rather than trying to
speak AnkiWeb's (private, undocumented) sync protocol directly.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AnkiConnectError(Exception):
    """Raised when AnkiConnect itself reports an error for a request."""


class AnkiConnectClient:
    """Thin wrapper over AnkiConnect's single-endpoint JSON-RPC-ish API."""

    def __init__(self, url: str = "http://localhost:8765", timeout: int = 30):
        self.url = url
        self.timeout = timeout

    def invoke(self, action: str, **params: Any) -> Any:
        payload: Dict[str, Any] = {"action": action, "version": 6}
        if params:
            payload["params"] = params

        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if data.get("error") is not None:
            raise AnkiConnectError(data["error"])

        return data.get("result")

    def is_available(self) -> bool:
        try:
            self.invoke("version")
            return True
        except Exception as e:
            logger.debug(f"AnkiConnect not reachable at {self.url}: {e}")
            return False

    def create_deck(self, deck_name: str) -> None:
        self.invoke("createDeck", deck=deck_name)

    def add_notes(self, notes: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Add notes, in AnkiConnect's own note-object shape. AnkiConnect
        returns null (not an error) per-note for ones it can't add - most
        commonly an exact duplicate - so a partial failure doesn't abort the
        whole batch."""
        return self.invoke("addNotes", notes=notes)

    def sync(self) -> None:
        """Trigger Anki's own AnkiWeb sync (equivalent to clicking the sync
        button) - requires the user to already be logged into AnkiWeb in that
        Anki desktop instance."""
        self.invoke("sync")


def card_row_to_note(row: Dict[str, Any], deck_name: str) -> Dict[str, Any]:
    """Convert one cards.csv row (dict form) into an AnkiConnect note object,
    matching the same field layout build.py's genanki models use (Basic:
    Front/Back/Source/Page/Section/Tags/Extra; Cloze: Text/Extra/Source/Page/
    Section/Tags) so notes pushed live via AnkiConnect look the same as ones
    in the local .apkg.
    """
    note_type = row.get("note_type", "Basic")
    tags = row.get("tags", "")
    if isinstance(tags, str):
        tags = [t for t in tags.split(";") if t]
    elif not isinstance(tags, list):
        tags = []

    source = row.get("source_pdf", "")
    page_start = row.get("page_start")
    page = f"p. {page_start}" if page_start not in (None, "", 0) else ""
    section = row.get("section", "") or ""
    extra = row.get("extra", "") or ""

    if note_type == "Cloze":
        fields = {
            "Text": str(row.get("cloze_text", "")),
            "Extra": str(extra),
            "Source": str(source),
            "Page": str(page),
            "Section": str(section),
        }
        model_name = "PDF2Anki Cloze"
    else:
        fields = {
            "Front": str(row.get("front", "")),
            "Back": str(row.get("back", "")),
            "Source": str(source),
            "Page": str(page),
            "Section": str(section),
            "Extra": str(extra),
        }
        model_name = "PDF2Anki Basic"

    return {
        "deckName": deck_name,
        "modelName": model_name,
        "fields": fields,
        "tags": tags,
    }


def push_notes(
    client: AnkiConnectClient,
    deck_name: str,
    rows: List[Dict[str, Any]],
    sync_after: bool = True,
) -> Dict[str, Any]:
    """Create the deck if needed, add the given card rows as notes, and
    optionally sync. Never raises - failures are logged and reflected in the
    returned counts so a partial or failed AnkiConnect push can be reported
    (e.g. via Slack) without crashing the pipeline; the local .apkg is always
    still written regardless of this outcome.
    """
    result: Dict[str, Any] = {"attempted": len(rows), "added": 0, "failed": 0, "synced": False}
    if not rows:
        return result

    try:
        client.create_deck(deck_name)
        notes = [card_row_to_note(row, deck_name) for row in rows]
        added_ids = client.add_notes(notes)
        result["added"] = sum(1 for i in added_ids if i is not None)
        result["failed"] = sum(1 for i in added_ids if i is None)

        if sync_after:
            try:
                client.sync()
                result["synced"] = True
            except Exception as e:
                logger.warning(f"AnkiConnect sync failed: {e}")
    except Exception as e:
        logger.warning(f"AnkiConnect push failed: {e}")
        result["failed"] = len(rows)

    return result
