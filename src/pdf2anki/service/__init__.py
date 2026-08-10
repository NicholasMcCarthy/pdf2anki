"""The `pdf2anki serve` watcher service: watches mounted directories for new
PDFs/textbooks/Readwise markdown exports, classifies and processes each one,
keeps a running Anki deck (.apkg) up to date, optionally pushes new notes into
a live Anki instance via AnkiConnect (for AnkiWeb sync), and posts Slack
notifications."""

from .classifier import classify_file
from .runner import process_new_file
from .watcher import DirectoryWatcher, WatcherState

__all__ = ["classify_file", "process_new_file", "DirectoryWatcher", "WatcherState"]
