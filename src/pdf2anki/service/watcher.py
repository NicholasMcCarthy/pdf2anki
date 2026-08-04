"""Directory watcher: observes configured watch directories for new PDF/
markdown files using watchdog, debounced so a file isn't processed mid-copy,
plus an initial startup catch-up scan and a periodic full reconciliation scan
(protects against missed filesystem events on some mounted/network volumes -
a known watchdog gotcha, especially relevant for Docker bind mounts)."""

import logging
import time
from pathlib import Path
from threading import Timer
from typing import Callable, Iterable, List, Optional, Set

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
    """

    def __init__(
        self,
        directories: Iterable[Optional[str]],
        on_file: Callable[[Path], None],
        debounce_seconds: float = 5.0,
        poll_interval_seconds: int = 300,
    ):
        self.directories = [Path(d) for d in directories if d]
        self.on_file = on_file
        self.debounce_seconds = debounce_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._processed: Set[str] = set()
        self._observer: Optional[Observer] = None

    def _seen_key(self, path: Path) -> str:
        try:
            return f"{path}:{path.stat().st_mtime_ns}"
        except OSError:
            return f"{path}:missing"

    def handle(self, path: Path) -> None:
        """Process one file if it hasn't been seen at its current mtime yet.
        Public (not just used internally by the observer callback/reconciliation
        loop) so callers can feed it a specific path directly, e.g. for tests.
        """
        key = self._seen_key(path)
        if key in self._processed:
            return
        self._processed.add(key)
        try:
            self.on_file(path)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")

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
