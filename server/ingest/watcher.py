"""Watched-folder capture (spec 4.1).

Waits for file size to stabilise across two polls before enqueueing, so a
half-written screenshot is never ingested.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from server.config import load
from server.ingest.events import hub
from server.ingest.stages.intake import (ingest_image, ingest_media, ingest_text_file,
                                         is_image, is_media, is_text)

log = logging.getLogger(__name__)
SETTLE_POLL_S = 0.5


class _Handler(FileSystemEventHandler):
    def __init__(self, submit) -> None:
        self._submit = submit

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        dest = getattr(event, "dest_path", None)
        if not event.is_directory and dest:
            self._submit(Path(str(dest)))


class Watcher:
    def __init__(self) -> None:
        self._observer: Observer | None = None
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        dirs = [d for d in load().paths("paths.watch_dirs")]
        live = []
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
                live.append(d)
            except OSError as exc:
                log.warning("watch dir unavailable: %s (%s)", d, exc)
        if not live:
            return
        self._observer = Observer()
        handler = _Handler(self._enqueue)
        for d in live:
            self._observer.schedule(handler, str(d), recursive=True)
        self._observer.start()
        self._worker = threading.Thread(target=self._drain, name="watch-drain", daemon=True)
        self._worker.start()
        log.info("watching %s", ", ".join(str(d) for d in live))
        # Pick up anything already sitting in the folder at boot.
        for d in live:
            for p in d.rglob("*"):
                if p.is_file():
                    self._enqueue(p)

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3)

    def _enqueue(self, path: Path) -> None:
        if is_image(path) or is_text(path) or is_media(path):
            with self._lock:
                self._pending.add(path)

    def _drain(self) -> None:
        sizes: dict[Path, int] = {}
        while not self._stop.wait(SETTLE_POLL_S):
            with self._lock:
                batch = list(self._pending)
            for p in batch:
                try:
                    size = p.stat().st_size
                except OSError:
                    with self._lock:
                        self._pending.discard(p)
                    sizes.pop(p, None)
                    continue
                if sizes.get(p) != size or size == 0:
                    sizes[p] = size
                    continue  # still growing; check again next tick
                with self._lock:
                    self._pending.discard(p)
                sizes.pop(p, None)
                try:
                    if is_text(p):
                        r = ingest_text_file(p, source="watch")
                    elif is_media(p):
                        r = ingest_media(p, source="watch")
                    else:
                        r = ingest_image(p, source="watch")
                    if r.created:
                        hub.emit("item.created", {"id": r.item_id, "near_dup_of": r.near_dup_of})
                        log.info("ingested %s -> item %d", p.name, r.item_id)
                except Exception:
                    log.exception("intake failed for %s", p)


watcher = Watcher()
