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


def test_state_persists_across_simulated_restart(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"a")
    (tmp_path / "b.md").write_text("# B")
    state_path = tmp_path / "state.json"

    processed1 = []
    watcher1 = DirectoryWatcher(
        directories=[tmp_path], on_file=processed1.append, state_path=state_path
    )
    watcher1.reconcile()
    assert {p.name for p in processed1} == {"a.pdf", "b.md"}
    assert state_path.exists()

    # A brand-new DirectoryWatcher instance pointed at the same state_path
    # (simulating a container restart) should not reprocess unchanged files.
    processed2 = []
    watcher2 = DirectoryWatcher(
        directories=[tmp_path], on_file=processed2.append, state_path=state_path
    )
    watcher2.reconcile()
    assert processed2 == []

    # But a genuinely new file should still be picked up.
    time.sleep(0.01)
    (tmp_path / "c.pdf").write_bytes(b"c")
    watcher2.reconcile()
    assert {p.name for p in processed2} == {"c.pdf"}


def test_state_persists_touched_file_gets_reprocessed(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"v1")
    state_path = tmp_path / "state.json"

    processed1 = []
    DirectoryWatcher(
        directories=[tmp_path], on_file=processed1.append, state_path=state_path
    ).reconcile()
    assert processed1 == [path]

    time.sleep(0.01)
    path.write_bytes(b"v2 - touched")  # mtime changes

    processed2 = []
    watcher2 = DirectoryWatcher(
        directories=[tmp_path], on_file=processed2.append, state_path=state_path
    )
    watcher2.reconcile()
    assert processed2 == [path]


def test_load_state_tolerates_missing_or_corrupt_file(tmp_path):
    state_path = tmp_path / "state.json"

    # Missing file - fine, starts empty.
    watcher = DirectoryWatcher(directories=[tmp_path], on_file=lambda p: None, state_path=state_path)
    assert watcher._processed == {}

    # Corrupt file - tolerated, logged, starts empty rather than crashing.
    state_path.write_text("not valid json{{{")
    watcher2 = DirectoryWatcher(directories=[tmp_path], on_file=lambda p: None, state_path=state_path)
    assert watcher2._processed == {}


def test_reset_state_clears_memory_and_disk_and_triggers_reprocessing(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")
    state_path = tmp_path / "state.json"

    processed = []
    watcher = DirectoryWatcher(
        directories=[tmp_path], on_file=processed.append, state_path=state_path
    )
    watcher.reconcile()
    assert processed == [path]
    assert state_path.exists()

    watcher.reset_state()
    assert watcher._processed == {}
    assert not state_path.exists()

    # Next reconciliation reprocesses everything, since nothing is "seen" anymore.
    watcher.reconcile()
    assert processed == [path, path]


def test_failed_handle_is_not_marked_seen_and_is_retried(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")
    state_path = tmp_path / "state.json"

    calls = []
    fail = {"on": True}

    def flaky(p):
        calls.append(p)
        if fail["on"]:
            raise RuntimeError("boom")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=flaky, state_path=state_path)
    watcher.handle(path)  # fails - must not raise, must not persist as seen
    assert len(calls) == 1
    assert watcher._processed == {}
    assert not state_path.exists()

    # Next attempt succeeds and is now recorded.
    fail["on"] = False
    watcher.handle(path)
    assert len(calls) == 2
    assert watcher._processed.get(str(path)) is not None
    assert state_path.exists()

    # A further handle() with unchanged mtime is now a no-op.
    watcher.handle(path)
    assert len(calls) == 2


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
