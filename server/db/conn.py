"""SQLite connection factory.

Loads sqlite-vec, applies pragmas, and initialises the schema. Read
connections are cheap and per-thread; all writes go through db.writer.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from server.config import load

log = logging.getLogger(__name__)

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
        "INSERT INTO meta(key, value) VALUES ('schema_version','3') "
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
    _migrate_v3_vectors(conn)


def _migrate_v3_vectors(conn: sqlite3.Connection) -> None:
    """v2 -> v3: three 1024-d spaces become one 768-d space.

    Nothing is salvageable. The dimension changed AND the model changed, so
    the stored floats are not vectors in the new space - they are noise with
    the right shape. A vec0 virtual table also cannot be ALTERed to a new
    dimension. So the old tables are dropped outright and every item is
    requeued for the embed stage, which recomputes from blobs and text that
    are all still on disk. Tag scores survive untouched in tag_scores, which
    is exactly what spec 7.x retains them for.

    Idempotent: once the old tables are gone this is a no-op, so it is safe
    on the init_db() path that runs at every startup.
    """
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    stale = [t for t in ("vec_text", "vec_jina") if t in tables]
    # vec_vocab survives by name but not by dimension; detect the old width
    # by asking vec0 for its declaration rather than guessing.
    if "vec_vocab" in tables:
        try:
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'vec_vocab'").fetchone()
            if ddl and "1024" in (ddl["sql"] or ""):
                stale.append("vec_vocab")
        except sqlite3.Error:
            pass
    if not stale:
        return

    log.warning("schema v3: dropping %s - 1024-d vectors cannot be converted "
                "to the 768-d space, every item will be re-embedded",
                ", ".join(stale))
    for t in stale:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(_SCHEMA.read_text())

    n = conn.execute(
        "INSERT INTO ingest_queue(item_id, stage, enqueued_at) "
        "SELECT id, 'embed', unixepoch() FROM items WHERE status != 'error'"
    ).rowcount
    # The visual rows come from tag_visual, not embed, so images need that
    # stage too or "looks similar" and zero-shot tagging stay empty.
    m = conn.execute(
        "INSERT INTO ingest_queue(item_id, stage, enqueued_at) "
        "SELECT id, 'tag_visual', unixepoch() FROM items "
        "WHERE kind = 'image' AND blob_path IS NOT NULL AND status != 'error'"
    ).rowcount
    # Calibration describes a DISTRIBUTION OF COSINES, so it belongs to the
    # space it was measured in and is actively harmful once that space is
    # gone. jina put a term's mean at ~0.18 with sigma ~0.04; EG2 puts the
    # same term at ~0.56. Left in place those numbers read every image as
    # 9 sigma above the mean, logistic() saturates, and every term is
    # accepted for every image - 1275 of 1278 scores on this corpus, with
    # no error anywhere. Null them so terms fall back until the tagging
    # stage recalibrates (spec 7.3).
    c = conn.execute(
        "UPDATE vocab_terms SET clip_mu = NULL, clip_sigma = NULL "
        "WHERE clip_mu IS NOT NULL OR clip_sigma IS NOT NULL").rowcount
    log.warning("schema v3: requeued %d item(s) for embed, %d for tag_visual, "
                "cleared stale calibration on %d term(s)", n, m, c)


def versions(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": conn.execute("SELECT vec_version()").fetchone()[0],
    }
