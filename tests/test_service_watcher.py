"""Tests for the directory watcher: reconciliation scanning, dedup-by-mtime,
error tracking/retry, and live watchdog event detection (debounced)."""

import time
from pathlib import Path

from pdf2anki.service.watcher import DirectoryWatcher, WatcherState, scan_existing_files


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
    assert watcher.state.entries == {}

    # Corrupt file - tolerated, logged, starts empty rather than crashing.
    state_path.write_text("not valid json{{{")
    watcher2 = DirectoryWatcher(directories=[tmp_path], on_file=lambda p: None, state_path=state_path)
    assert watcher2.state.entries == {}


def test_load_state_upgrades_old_plain_mtime_format(tmp_path):
    """State files written before error-tracking existed are {path: mtime_ns}
    - loading must treat those as prior successes, not wipe/ignore them."""
    import json

    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")
    mtime_ns = path.stat().st_mtime_ns
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({str(path): mtime_ns}))

    processed = []
    watcher = DirectoryWatcher(directories=[tmp_path], on_file=processed.append, state_path=state_path)
    watcher.handle(path)
    assert processed == []  # already "seen" per the old-format entry
    assert watcher.state.entries[str(path)]["status"] == "success"


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
    assert watcher.state.entries == {}
    assert not state_path.exists()

    # Next reconciliation reprocesses everything, since nothing is "seen" anymore.
    watcher.reconcile()
    assert processed == [path, path]


def test_failed_handle_is_recorded_as_error_and_not_auto_retried(tmp_path):
    """A failure must not crash the watch loop, but - unlike a success -
    must NOT be silently retried on the next handle()/reconcile(): a
    persistent failure (e.g. exhausted API credits) would otherwise burn
    credits retrying the same doomed call on every reconciliation pass."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")
    state_path = tmp_path / "state.json"

    calls = []

    def boom(p):
        calls.append(p)
        raise RuntimeError("insufficient credits")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=boom, state_path=state_path)
    watcher.handle(path)  # fails - must not raise
    assert len(calls) == 1

    entry = watcher.state.entries[str(path)]
    assert entry["status"] == "error"
    assert entry["error"] == "insufficient credits"
    assert entry["attempts"] == 1
    assert state_path.exists()

    # Reconciliation must NOT retry it - that's the whole point.
    watcher.reconcile()
    assert len(calls) == 1
    watcher.handle(path)
    assert len(calls) == 1


def test_retry_errors_clears_only_errors_and_allows_reprocessing(tmp_path):
    ok_path = tmp_path / "ok.pdf"
    ok_path.write_bytes(b"ok")
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_bytes(b"bad")
    state_path = tmp_path / "state.json"

    calls = []

    def on_file(p):
        calls.append(p)
        if p == bad_path:
            raise RuntimeError("boom")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=on_file, state_path=state_path)
    watcher.reconcile()
    assert set(calls) == {ok_path, bad_path}
    assert len(watcher.state.errors()) == 1
    assert len(watcher.state.successes()) == 1

    cleared = watcher.retry_errors()
    assert cleared == 1
    # The prior success must be untouched.
    assert len(watcher.state.successes()) == 1
    assert bad_path.name not in {Path(k).name for k in watcher.state.errors()}

    calls.clear()
    watcher.reconcile()
    # Only the previously-errored file is retried; the successful one is skipped.
    assert calls == [bad_path]


def test_retry_errors_limited_to_specific_paths(tmp_path):
    bad1 = tmp_path / "bad1.pdf"
    bad1.write_bytes(b"1")
    bad2 = tmp_path / "bad2.pdf"
    bad2.write_bytes(b"2")
    state_path = tmp_path / "state.json"

    def boom(p):
        raise RuntimeError("boom")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=boom, state_path=state_path)
    watcher.reconcile()
    assert len(watcher.state.errors()) == 2

    cleared = watcher.retry_errors(paths=[bad1])
    assert cleared == 1
    remaining = {Path(k).name for k in watcher.state.errors()}
    assert remaining == {"bad2.pdf"}


def test_reset_state_also_clears_errors(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"a")
    state_path = tmp_path / "state.json"

    def boom(p):
        raise RuntimeError("boom")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=boom, state_path=state_path)
    watcher.handle(path)
    assert len(watcher.state.errors()) == 1

    watcher.reset_state()
    assert watcher.state.entries == {}
    assert not state_path.exists()


def test_touching_an_errored_file_makes_it_eligible_again_without_retry_errors(tmp_path):
    """mtime changing always overrides settled state (success or error),
    even without an explicit retry - editing the file is itself a signal
    the user wants it re-attempted."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"v1")
    state_path = tmp_path / "state.json"

    calls = []

    def boom(p):
        calls.append(p)
        raise RuntimeError("boom")

    watcher = DirectoryWatcher(directories=[tmp_path], on_file=boom, state_path=state_path)
    watcher.handle(path)
    assert len(calls) == 1
    watcher.handle(path)  # not retried - same mtime, still errored
    assert len(calls) == 1

    time.sleep(0.01)
    path.write_bytes(b"v2 - touched")
    watcher.handle(path)
    assert len(calls) == 2


def test_watcher_state_directly_records_success_and_error(tmp_path):
    state_path = tmp_path / "state.json"
    state = WatcherState(state_path)
    path = tmp_path / "a.pdf"

    state.record_success(path, 100)
    assert state.is_settled(path, 100) is True
    assert state.is_settled(path, 200) is False  # different mtime - not settled

    state.record_error(path, 200, "boom")
    assert state.is_settled(path, 200) is True
    assert state.errors()[str(path)]["error"] == "boom"
    assert str(path) not in state.successes()


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
