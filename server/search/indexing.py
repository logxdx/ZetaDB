"""FTS5 maintenance.

Contentless FTS5 is written explicitly, never by triggers: a trigger would
fire mid-enrichment while flow_text and caption are still null (spec 9.2).
"""
from __future__ import annotations

import sqlite3

COLUMNS = ("title", "body", "flow_text", "caption", "tags")


def _row_text(conn: sqlite3.Connection, item_id: int) -> tuple[str, str, str, str, str] | None:
    row = conn.execute(
        "SELECT title, body, flow_text, caption FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    if row is None:
        return None
    tags = " ".join(
        r[0] for r in conn.execute(
            "SELECT v.term FROM item_tags it JOIN vocab_terms v ON v.id = it.term_id "
            "WHERE it.item_id = ? AND it.state = 'accepted'", (item_id,)
        )
    )
    return (row["title"] or "", row["body"] or "", row["flow_text"] or "",
            row["caption"] or "", tags)


def reindex_item(conn: sqlite3.Connection, item_id: int) -> None:
    """Upsert one item into fts_items. Contentless FTS5 needs an explicit
    delete-by-rowid before re-insert; 'delete' commands need the old values,
    so we keep a shadow of the indexed text in fts_shadow."""
    vals = _row_text(conn, item_id)
    if vals is None:
        return
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fts_shadow ("
        "  item_id INTEGER PRIMARY KEY, title TEXT, body TEXT,"
        "  flow_text TEXT, caption TEXT, tags TEXT)"
    )
    old = conn.execute("SELECT * FROM fts_shadow WHERE item_id = ?", (item_id,)).fetchone()
    if old is not None:
        conn.execute(
            "INSERT INTO fts_items(fts_items, rowid, title, body, flow_text, caption, tags) "
            "VALUES ('delete', ?, ?, ?, ?, ?, ?)",
            (item_id, old["title"], old["body"], old["flow_text"], old["caption"], old["tags"]),
        )
    conn.execute(
        "INSERT INTO fts_items(rowid, title, body, flow_text, caption, tags) VALUES (?,?,?,?,?,?)",
        (item_id, *vals),
    )
    conn.execute(
        "INSERT INTO fts_shadow(item_id, title, body, flow_text, caption, tags) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(item_id) DO UPDATE SET "
        "title=excluded.title, body=excluded.body, flow_text=excluded.flow_text, "
        "caption=excluded.caption, tags=excluded.tags",
        (item_id, *vals),
    )


def unindex_item(conn: sqlite3.Connection, item_id: int) -> None:
    old = conn.execute("SELECT * FROM fts_shadow WHERE item_id = ?", (item_id,)).fetchone()
    if old is None:
        return
    conn.execute(
        "INSERT INTO fts_items(fts_items, rowid, title, body, flow_text, caption, tags) "
        "VALUES ('delete', ?, ?, ?, ?, ?, ?)",
        (item_id, old["title"], old["body"], old["flow_text"], old["caption"], old["tags"]),
    )
    conn.execute("DELETE FROM fts_shadow WHERE item_id = ?", (item_id,))
