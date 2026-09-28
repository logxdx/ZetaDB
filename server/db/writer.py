"""Single-writer queue.

SQLite in WAL mode allows one writer. The ingest worker and the API both
need to write, so every write funnels through this one thread. Anything
else produces SQLITE_BUSY under concurrent enrichment (spec 9).
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from server.db.conn import connect

_SENTINEL = object()


class Writer:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()
        self._started.wait(timeout=10)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._q.put(_SENTINEL)
        self._thread.join(timeout=5)
        self._thread = None

    def _run(self) -> None:
        self._conn = connect()
        self._started.set()
        while True:
            job = self._q.get()
            if job is _SENTINEL:
                break
            fn, args, kwargs, result = job
            try:
                result.value = fn(self._conn, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
                result.error = exc
            finally:
                result.done.set()
        if self._conn is not None:
            self._conn.close()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run fn(conn, *args) on the writer thread and return its result.

        Blocks. Exceptions raised in fn are re-raised here.
        """
        if self._thread is None:
            raise RuntimeError("writer not started")
        slot = _Result()
        self._q.put((fn, args, kwargs, slot))
        slot.done.wait()
        if slot.error is not None:
            raise slot.error
        return slot.value

    @property
    def depth(self) -> int:
        return self._q.qsize()


class _Result:
    __slots__ = ("value", "error", "done")

    def __init__(self) -> None:
        self.value: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()


writer = Writer()


def tx(conn: sqlite3.Connection, statements: list[tuple[str, tuple]]) -> None:
    """Helper for submitting a batch of statements as one transaction."""
    conn.execute("BEGIN")
    try:
        for sql, params in statements:
            conn.execute(sql, params)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
