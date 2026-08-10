"""Directory watcher: observes configured watch directories for new PDF/
markdown files using watchdog, debounced so a file isn't processed mid-copy,
plus an initial startup catch-up scan and a periodic full reconciliation scan
(protects against missed filesystem events on some mounted/network volumes -
a known watchdog gotcha, especially relevant for Docker bind mounts)."""

import json
import logging
import time
from pathlib import Path
from threading import Timer
from typing import Callable, Dict, Iterable, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

WATCHED_SUFFIXES = (".pdf", ".md", ".markdown")


class WatcherState:
    """Persists which watched files have already been handled - and how
    (success or error) - to a JSON file keyed by absolute path, so a
    container restart/rebuild doesn't rerun the full (LLM-calling, so not
    free) pipeline on files it already dealt with.

    Each entry is tied to the file's mtime at the time of that attempt:
    editing/`touch`ing a file always makes it eligible again regardless of
    its prior status. Entries look like:
        {"status": "success"|"error", "mtime_ns": int, "attempts": int,
         "last_attempt": <unix ts>, "error": <str, only when status=="error">}

    Deliberately does NOT auto-retry errors on a later scan (see
    is_settled()) - a persistent failure (expired API credits, a bad model
    name, etc.) would otherwise get retried against every matching file on
    every reconciliation pass, burning time/API credits on calls doomed to
    fail the same way again. Use retry_errors() (`pdf2anki serve
    --retry-errors`, or `pdf2anki watch-status --retry-errors`) to
    explicitly clear error entries once the underlying problem is fixed.

    Tolerates the plain {path: mtime_ns} format from before error-tracking
    existed - loaded as status="success" entries so upgrading doesn't
    force a full reprocess of everything already recorded.
    """

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = Path(state_path) if state_path else None
        self.entries: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if not self.state_path or not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load watcher state from {self.state_path}, starting fresh: {e}")
            return {}
        if not isinstance(data, dict):
            return {}
        entries: Dict[str, dict] = {}
        for key, value in data.items():
            if isinstance(value, dict) and "mtime_ns" in value:
                entries[str(key)] = value
            else:
                # Pre-error-tracking format: {path: mtime_ns}.
                try:
                    entries[str(key)] = {"status": "success", "mtime_ns": int(value), "attempts": 1}
                except (TypeError, ValueError):
                    continue
        return entries

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f)
            tmp_path.replace(self.state_path)  # atomic on POSIX - avoids a truncated file on crash
        except Exception as e:
            logger.warning(f"Failed to persist watcher state to {self.state_path}: {e}")

    def is_settled(self, path: Path, mtime_ns: Optional[int]) -> bool:
        """True if this exact (path, mtime) has already been handled -
        successfully OR with a recorded error - and should be skipped
        rather than re-attempted."""
        if mtime_ns is None:
            return False
        entry = self.entries.get(str(path))
        return bool(entry) and entry.get("mtime_ns") == mtime_ns

    def record_success(self, path: Path, mtime_ns: Optional[int]) -> None:
        if mtime_ns is None:
            return
        prior = self.entries.get(str(path)) or {}
        attempts = prior.get("attempts", 0) + 1 if prior.get("mtime_ns") == mtime_ns else 1
        self.entries[str(path)] = {
            "status": "success",
            "mtime_ns": mtime_ns,
            "attempts": attempts,
            "last_attempt": time.time(),
        }
        self._save()

    def record_error(self, path: Path, mtime_ns: Optional[int], error: str) -> None:
        if mtime_ns is None:
            return
        prior = self.entries.get(str(path)) or {}
        attempts = prior.get("attempts", 0) + 1 if prior.get("mtime_ns") == mtime_ns else 1
        self.entries[str(path)] = {
            "status": "error",
            "mtime_ns": mtime_ns,
            "attempts": attempts,
            "last_attempt": time.time(),
            "error": error,
        }
        self._save()

    def errors(self) -> Dict[str, dict]:
        return {k: v for k, v in self.entries.items() if v.get("status") == "error"}

    def successes(self) -> Dict[str, dict]:
        return {k: v for k, v in self.entries.items() if v.get("status") == "success"}

    def retry_errors(self, paths: Optional[Iterable[Path]] = None) -> int:
        """Clear error entries (leaving successes untouched) so the next
        handle()/reconcile() treats those files as unseen again. Limits to
        `paths` when given, otherwise clears every errored file. Returns
        the number of entries cleared."""
        if paths is not None:
            targets = {str(p) for p in paths}
        else:
            targets = {k for k, v in self.entries.items() if v.get("status") == "error"}
        cleared = 0
        for key in list(targets):
            entry = self.entries.get(key)
            if entry and entry.get("status") == "error":
                del self.entries[key]
                cleared += 1
        if cleared:
            self._save()
        return cleared

    def reset(self) -> None:
        """Forget every previously-seen file - successes and errors alike,
        in memory and (if configured) on disk. See `pdf2anki serve
        --reset-state` for a full "reprocess everything" reset; prefer
        retry_errors() when only errored files need another attempt."""
        self.entries = {}
        if self.state_path and self.state_path.exists():
            try:
                self.state_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove watcher state file {self.state_path}: {e}")


class _DebouncedHandler(FileSystemEventHandler):
    """Collects created/modified file paths and calls on_ready(path) once no
    further events for that path have arrived within debounce_seconds."""

    def __init__(self, on_ready: Callable[[Path], None], debounce_seconds: float):
        self.on_ready = on_ready
        self.debounce_seconds = debounce_seconds
        self._timers: dict = {}

    def _schedule(self, path: Path) -> None:
        if path.suffix.lower() not in WATCHED_SUFFIXES:
            return
        existing = self._timers.get(path)
        if existing:
            existing.cancel()
        timer = Timer(self.debounce_seconds, self._fire, args=(path,))
        timer.daemon = True
        self._timers[path] = timer
        timer.start()

    def _fire(self, path: Path) -> None:
        self._timers.pop(path, None)
        if path.exists():
            self.on_ready(path)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path))

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._schedule(Path(event.src_path))


def scan_existing_files(directories: Iterable[Path]) -> List[Path]:
    """One-shot discovery of already-present watched files - used for the
    startup catch-up scan and the periodic reconciliation fallback."""
    found = []
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for suffix in WATCHED_SUFFIXES:
            found.extend(directory.rglob(f"*{suffix}"))
    return sorted(set(found))


class DirectoryWatcher:
    """Watches a set of directories, calling on_file(path) for each new/changed
    watched file - deduplicated by (path, mtime) so a file isn't reprocessed
    on every reconciliation pass unless it has actually changed since last seen.

    Without `state_path`, that dedup memory is in-process only - a container
    restart starts with nothing seen, so the startup reconciliation scan
    (see start()) treats every already-handled file as new again and reruns
    the full (LLM-calling) pipeline on all of them. Passing `state_path`
    persists a WatcherState (see above) to a JSON file after every handle(),
    success or error, so a restart resumes from where it left off: only
    genuinely new or modified (mtime-changed - e.g. explicitly `touch`ed)
    files get (re)attempted, and files that previously errored stay skipped
    until explicitly retried (retry_errors()) rather than being silently
    retried - and silently burning API credits - on every reconciliation
    pass. Use reset_state() (or delete the file) for an explicit "reprocess
    everything" - see `pdf2anki serve --reset-state` / `--retry-errors`.
    """

    def __init__(
        self,
        directories: Iterable[Optional[str]],
        on_file: Callable[[Path], None],
        debounce_seconds: float = 5.0,
        poll_interval_seconds: int = 300,
        state_path: Optional[Path] = None,
    ):
        self.directories = [Path(d) for d in directories if d]
        self.on_file = on_file
        self.debounce_seconds = debounce_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.state_path = Path(state_path) if state_path else None
        self.state = WatcherState(self.state_path)
        self._observer: Optional[Observer] = None

    def _current_mtime_ns(self, path: Path) -> Optional[int]:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def reset_state(self) -> None:
        """Forget every previously-seen file (successes AND errors), in
        memory and (if configured) on disk - the next reconciliation scan
        reprocesses everything found. See `pdf2anki serve --reset-state`."""
        self.state.reset()

    def retry_errors(self, paths: Optional[Iterable[Path]] = None) -> int:
        """Clear recorded errors (leaving successfully-processed files
        alone) so the next reconciliation retries them. Limits to `paths`
        when given. Returns how many were cleared. See `pdf2anki serve
        --retry-errors` / `pdf2anki watch-status --retry-errors`."""
        return self.state.retry_errors(paths)

    def handle(self, path: Path) -> None:
        """Process one file if it hasn't already been settled (successfully
        processed, or errored and not yet explicitly retried) at its
        current mtime. Public (not just used internally by the observer
        callback/reconciliation loop) so callers can feed it a specific
        path directly, e.g. for tests."""
        mtime_ns = self._current_mtime_ns(path)
        if self.state.is_settled(path, mtime_ns):
            return
        try:
            self.on_file(path)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            self.state.record_error(path, mtime_ns, str(e))
            return
        self.state.record_success(path, mtime_ns)

    def reconcile(self) -> None:
        """Run one full directory scan, handling anything not yet settled."""
        for path in scan_existing_files(self.directories):
            self.handle(path)

    def start(self) -> None:
        """Create watch directories, run the startup catch-up scan, and start
        the watchdog observer. Does not block - call run_forever() for that,
        or drive reconcile()/stop() yourself (e.g. in tests)."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.reconcile()

        handler = _DebouncedHandler(self.handle, self.debounce_seconds)
        self._observer = Observer()
        for directory in self.directories:
            self._observer.schedule(handler, str(directory), recursive=True)
        self._observer.start()

        logger.info(
            f"Watching {[str(d) for d in self.directories]} "
            f"(debounce={self.debounce_seconds}s, reconciliation every {self.poll_interval_seconds}s)"
        )

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def run_forever(self) -> None:
        """Start watching and block, periodically re-reconciling, until
        interrupted. This is what `pdf2anki serve` actually calls."""
        self.start()
        try:
            while True:
                time.sleep(self.poll_interval_seconds)
                self.reconcile()
        except KeyboardInterrupt:
            logger.info("Stopping watcher")
        finally:
            self.stop()
