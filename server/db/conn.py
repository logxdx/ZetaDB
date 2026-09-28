"""SQLite connection factory.

Loads sqlite-vec, applies pragmas, and initialises the schema. Read
connections are cheap and per-thread; all writes go through db.writer.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from server.config import load

_SCHEMA = Path(__file__).parent / "schema.sql"
_local = threading.local()


def connect(db_path: Path | None = None, *, readonly: bool = False) -> sqlite3.Connection:
    cfg = load()
    path = db_path or cfg.path("paths.db")
    path.parent.mkdir(parents=True, exist_ok=True)

    if readonly and path.exists():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)

    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def reader() -> sqlite3.Connection:
    """Thread-local read connection. Safe to share within a request."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _local.conn = connect()
    return conn


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    conn = connect(db_path)
    conn.executescript(_SCHEMA.read_text())
    _migrate(conn)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('schema_version','2') "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    )
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations. SQLite has no ADD COLUMN IF NOT EXISTS,
    so check the table info first."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(vocab_terms)")}
    if "requires_text" not in have:
        conn.execute("ALTER TABLE vocab_terms ADD COLUMN "
                     "requires_text INTEGER NOT NULL DEFAULT 0")


def versions(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": conn.execute("SELECT vec_version()").fetchone()[0],
    }
