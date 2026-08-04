"""Tests for the directory watcher: reconciliation scanning, dedup-by-mtime,
and live watchdog event detection (debounced)."""

import time
from pathlib import Path

from pdf2anki.service.watcher import DirectoryWatcher, scan_existing_files


def test_scan_existing_files_finds_pdfs_and_markdown(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "b.md").write_text("# B")
    (tmp_path / "c.txt").write_text("ignored")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.markdown").write_text("# D")

    found = scan_existing_files([tmp_path])
    names = {p.name for p in found}
    assert names == {"a.pdf", "b.md", "d.markdown"}


def test_scan_existing_files_skips_missing_directories(tmp_path):
    assert scan_existing_files([tmp_path / "does-not-exist"]) == []


def test_handle_processes_each_file_once_per_mtime(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"v1")

    processed = []
    watcher = DirectoryWatcher(directories=[tmp_path], on_file=processed.append)

    watcher.handle(path)
    watcher.handle(path)  # unchanged - should not be reprocessed
    assert processed == [path]

    time.sleep(0.01)
    path.write_bytes(b"v2 - changed")  # mtime changes
    watcher.handle(path)
    assert processed == [path, path]


def test_reconcile_picks_up_all_existing_files(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.md").write_text("# B")

    processed = []
    watcher = DirectoryWatcher(directories=[tmp_path], on_file=processed.append)
    watcher.reconcile()

    assert {p.name for p in processed} == {"a.pdf", "b.md"}

    # A second reconcile pass with nothing changed should not reprocess.
    watcher.reconcile()
    assert len(processed) == 2


def test_on_file_exception_does_not_propagate(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")

    def boom(p):
        raise RuntimeError("processing failed")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=boom)
    watcher.handle(path)  # must not raise


def test_none_directories_are_filtered_out():
    watcher = DirectoryWatcher(directories=[None, "/some/path", None], on_file=lambda p: None)
    assert watcher.directories == [Path("/some/path")]


def test_live_watchdog_event_triggers_on_file(tmp_path):
    processed = []
    watcher = DirectoryWatcher(
        directories=[tmp_path],
        on_file=processed.append,
        debounce_seconds=0.1,
        poll_interval_seconds=999999,
    )
    watcher.start()
    try:
        time.sleep(0.1)
        new_file = tmp_path / "new.pdf"
        new_file.write_bytes(b"content")

        deadline = time.time() + 3
        while time.time() < deadline and not processed:
            time.sleep(0.1)
    finally:
        watcher.stop()

    assert any(p.name == "new.pdf" for p in processed)
