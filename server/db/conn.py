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
        "INSERT INTO meta(key, value) VALUES ('schema_version','4') "
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
    _migrate_v4_eye_and_media(conn)


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


def _items_ddl(name: str = "items") -> str:
    """The items CREATE TABLE statement from schema.sql, optionally renamed."""
    text = _SCHEMA.read_text()
    start = text.index("CREATE TABLE IF NOT EXISTS items (")
    end = text.index(");", start) + 2
    ddl = text[start:end]
    return ddl.replace("CREATE TABLE IF NOT EXISTS items (", f"CREATE TABLE {name} (", 1)


def _migrate_v4_eye_and_media(conn: sqlite3.Connection) -> None:
    """v3 -> v4: JoyTag becomes Eye, and audio and video become items.

    * vocab_terms.joytag_alias -> eye_alias, tag_scores.joytag -> eye. The
      old VALUES are dropped, not carried: an alias was a Danbooru tag name
      and a score was JoyTag's sigmoid, and neither means anything to Eye.
      The aliases are re-seeded from vocab/seed.yaml at startup, and every
      image is requeued for tag_visual so Eye's scores replace JoyTag's.
      Human tags are untouched - only machine fields are recomputed.
    * items.kind gains 'audio' and 'video', and items gains `duration`.
      SQLite cannot alter a CHECK constraint, so the table is rebuilt the
      documented way (create, copy, drop, rename) with foreign keys off.
      Row ids are preserved, which matters: fts_items is contentless and
      keyed on them, and every other table references them.

    Idempotent, like v3: each step checks before acting.
    """
    vt = {r["name"] for r in conn.execute("PRAGMA table_info(vocab_terms)")}
    ts = {r["name"] for r in conn.execute("PRAGMA table_info(tag_scores)")}
    retag = False
    if "joytag_alias" in vt and "eye_alias" not in vt:
        conn.execute("ALTER TABLE vocab_terms RENAME COLUMN joytag_alias TO eye_alias")
        conn.execute("UPDATE vocab_terms SET eye_alias = NULL")
        retag = True
    if "joytag" in ts and "eye" not in ts:
        conn.execute("ALTER TABLE tag_scores RENAME COLUMN joytag TO eye")
        conn.execute("UPDATE tag_scores SET eye = NULL")
        retag = True

    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='items'"
                       ).fetchone()
    if row and "'audio'" not in (row["sql"] or ""):
        log.warning("schema v4: rebuilding items to admit audio and video")
        old_cols = [r["name"] for r in conn.execute("PRAGMA table_info(items)")]
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN")
            conn.execute("DROP TABLE IF EXISTS items_v4")
            conn.execute(_items_ddl("items_v4"))
            new_cols = {r["name"] for r in conn.execute("PRAGMA table_info(items_v4)")}
            cols = ", ".join(c for c in old_cols if c in new_cols)
            conn.execute(f"INSERT INTO items_v4({cols}) SELECT {cols} FROM items")
            conn.execute("DROP TABLE items")
            conn.execute("ALTER TABLE items_v4 RENAME TO items")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA.read_text())        # recreate the items indexes
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            log.error("schema v4: %d foreign key violation(s) after rebuild", len(bad))

    if retag:
        n = conn.execute(
            "INSERT INTO ingest_queue(item_id, stage, enqueued_at) "
            "SELECT id, 'tag_visual', unixepoch() FROM items "
            "WHERE kind = 'image' AND blob_path IS NOT NULL AND status != 'error'"
        ).rowcount
        conn.execute("INSERT INTO meta(key, value) VALUES ('vocab_reseed','1') "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
        log.warning("schema v4: JoyTag -> Eye; requeued %d image(s) for tag_visual", n)


def versions(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": conn.execute("SELECT vec_version()").fetchone()[0],
    }
