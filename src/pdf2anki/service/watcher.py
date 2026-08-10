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
    (see start()) treats every already-processed file as new again and
    reruns the full (LLM-calling) pipeline on all of them. Passing
    `state_path` persists the seen-file map to a JSON file after every
    successful handle(), so a restart resumes from where it left off and
    only genuinely new or modified (mtime-changed - e.g. explicitly
    `touch`ed) files get reprocessed. Use reset_state() (or delete the file)
    for an explicit "reprocess everything" - see `pdf2anki serve --reset-state`.
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
        self._processed: Dict[str, int] = self._load_state()
        self._observer: Optional[Observer] = None

    def _seen_key(self, path: Path) -> str:
        return str(path)

    def _current_mtime_ns(self, path: Path) -> Optional[int]:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def _load_state(self) -> Dict[str, int]:
        if not self.state_path or not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load watcher state from {self.state_path}, starting fresh: {e}")
        return {}

    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._processed, f)
            tmp_path.replace(self.state_path)  # atomic on POSIX - avoids a truncated file on crash
        except Exception as e:
            logger.warning(f"Failed to persist watcher state to {self.state_path}: {e}")

    def reset_state(self) -> None:
        """Forget every previously-seen file, in memory and (if configured)
        on disk - the next reconciliation scan reprocesses everything found.
        See `pdf2anki serve --reset-state`."""
        self._processed = {}
        if self.state_path and self.state_path.exists():
            try:
                self.state_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove watcher state file {self.state_path}: {e}")

    def handle(self, path: Path) -> None:
        """Process one file if it hasn't been seen at its current mtime yet.
        Public (not just used internally by the observer callback/reconciliation
        loop) so callers can feed it a specific path directly, e.g. for tests.
        """
        key = self._seen_key(path)
        mtime_ns = self._current_mtime_ns(path)
        if mtime_ns is not None and self._processed.get(key) == mtime_ns:
            return
        try:
            self.on_file(path)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return
        # Only recorded as "seen" after on_file() succeeds, so a failed
        # attempt gets retried on the next reconciliation pass rather than
        # being silently skipped forever.
        if mtime_ns is not None:
            self._processed[key] = mtime_ns
            self._save_state()

    def reconcile(self) -> None:
        """Run one full directory scan, handling anything not yet processed."""
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
