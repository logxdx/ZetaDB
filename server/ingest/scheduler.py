"""Stage-batched ingest scheduler.

The 6 GB budget forbids co-residency, and a multi-GB GGUF costs 10-20 s to
load, so the queue drains one STAGE at a time over the whole pending batch
rather than running each item end to end (spec 4.2). CPU stages run while a
GPU stage is blocked because they take no lease.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.engine.core import engine as model_engine
from server.ingest.events import hub

log = logging.getLogger(__name__)

# stage -> (next stage, pool or None for CPU)
PIPELINE: dict[str, tuple[str | None, str | None]] = {
    "ocr":        ("layout",     "A"),
    "layout":     ("media",      None),
    # Audio and video enter here: windows are cut and embedded with the EG2
    # vision and audio encoders, so this needs the pool A lease. Images pass
    # straight through (the handler ignores them).
    "media":      ("tag_visual", "A"),
    "tag_visual": ("caption",    "A"),
    "caption":    ("extract",    "B"),
    "extract":    ("embed",      None),
    # Pool A: an image's item vector needs a vision pass (its text - OCR and
    # caption - only exists by now), and the encoder must not sit in VRAM
    # outside a lease while pool B holds llama.cpp. [M8] measured it doing
    # exactly that: ~0.7 GB resident after eviction. On the CPU it costs
    # 2.15 s/image. Search is unaffected - queries use the CPU backbone and
    # take no lease.
    "embed":      ("finalize",   "A"),
    "finalize":   (None,         None),
}
STAGE_ORDER = list(PIPELINE)
MAX_ATTEMPTS = 3


class Scheduler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, Callable[[list[int]], None]] = {}
        self._current: str | None = None

    def register(self, stage: str, fn: Callable[[list[int]], None]) -> None:
        """Register a batch handler. Unregistered stages pass through, which
        is what lets M1 run before OCR exists and M2 run before the VLM."""
        self._handlers[stage] = fn

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="ingest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)

    def nudge(self) -> None:
        self._wake.set()

    def status(self) -> dict:
        conn = reader()
        depth = {r["stage"]: r["n"] for r in conn.execute(
            "SELECT stage, count(*) AS n FROM ingest_queue GROUP BY stage")}
        return {"depths": depth, "total": sum(depth.values()),
                "current_stage": self._current,
                "handlers": sorted(self._handlers)}

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = False
            for stage in STAGE_ORDER:
                if self._stop.is_set():
                    break
                ids = self._pending(stage)
                if ids:
                    worked = True
                    self._drain(stage, ids)
            model_engine.sweep_idle()
            if not worked:
                self._wake.wait(timeout=2.0)
                self._wake.clear()

    def _pending(self, stage: str) -> list[int]:
        rows = reader().execute(
            "SELECT item_id FROM ingest_queue WHERE stage = ? AND attempts < ? "
            "ORDER BY enqueued_at LIMIT 256", (stage, MAX_ATTEMPTS)).fetchall()
        return [int(r["item_id"]) for r in rows]

    def _drain(self, stage: str, ids: list[int]) -> None:
        nxt, pool = PIPELINE[stage]
        handler = self._handlers.get(stage)
        self._current = stage
        t0 = time.monotonic()
        lease = None
        try:
            if handler is not None:
                if pool:
                    lease = model_engine.lease(pool, model=stage)
                handler(ids)
            elif stage not in ("finalize",):
                log.debug("stage %s has no handler; passing %d item(s) through", stage, len(ids))
        except Exception as exc:
            log.exception("stage %s failed for %d item(s)", stage, len(ids))
            self._record_errors(stage, ids, exc)
        finally:
            if lease is not None:
                model_engine.release(lease)
            self._current = None

        self._advance(stage, ids, nxt)
        log.info("stage %-10s %3d item(s) in %.1fs -> %s",
                 stage, len(ids), time.monotonic() - t0, nxt or "done")
        hub.emit("queue.changed", self.status())

    def _record_errors(self, stage: str, ids: list[int],
                       exc: BaseException | None = None) -> None:
        # Store what actually went wrong. The literal "stage handler raised"
        # this used to write left 29 caption failures in the database with
        # no way to tell a missing binary from a dead connection, since the
        # traceback only ever reached a log that had long since rotated -
        # and spec 4.4 wants this field readable in the UI.
        detail = f"{type(exc).__name__}: {exc}"[:500] if exc else "stage handler raised"

        def work(conn: sqlite3.Connection) -> None:
            now = int(time.time())
            for i in ids:
                conn.execute(
                    "INSERT INTO item_errors(item_id, stage, error, ts) VALUES (?,?,?,?)",
                    (i, stage, detail, now))
                conn.execute(
                    "UPDATE ingest_queue SET attempts = attempts + 1 "
                    "WHERE item_id = ? AND stage = ?", (i, stage))
        writer.submit(work)

    def _advance(self, stage: str, ids: list[int], nxt: str | None) -> None:
        def work(conn: sqlite3.Connection) -> None:
            now = int(time.time())
            for i in ids:
                conn.execute("DELETE FROM ingest_queue WHERE item_id = ? AND stage = ?",
                             (i, stage))
                if nxt:
                    conn.execute(
                        "INSERT INTO ingest_queue(item_id, stage, enqueued_at) VALUES (?,?,?)",
                        (i, nxt, now))
                else:
                    conn.execute("UPDATE items SET status = 'ready', updated_at = ? WHERE id = ?",
                                 (now, i))
        writer.submit(work)
        for i in ids:
            hub.emit("item.stage", {"id": i, "stage": stage, "next": nxt})


scheduler = Scheduler()
