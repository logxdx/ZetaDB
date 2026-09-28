"""Stage 7 - finalize. Lexical index + semantic edge seeding (spec 4.2, 11.1)."""
from __future__ import annotations

import logging
import sqlite3
import time

from server.config import load
from server.db.writer import writer
from server.models.embeddings import cosine_from_l2
from server.search.indexing import reindex_item

log = logging.getLogger(__name__)


def seed_edges(conn: sqlite3.Connection, item_id: int) -> int:
    """Top-k neighbours above threshold become dotted 'semantic' edges, so
    the graph is populated on day one instead of waiting for manual links."""
    cfg = load()
    k = cfg.get("graph.seed_top_k", 8)
    made = 0
    for table, kind_filter, thresh in (
        ("vec_text", None, cfg.get("graph.seed_min_cos_qwen", 0.70)),
        ("vec_jina", "image", cfg.get("graph.seed_min_cos_jina", 0.62)),
    ):
        try:
            row = conn.execute(
                f"SELECT embedding FROM {table} WHERE item_id = ? LIMIT 1", (item_id,)
            ).fetchone()
            if row is None:
                continue
            where = "embedding MATCH ? AND k = ?"
            params: list = [row["embedding"], k + 1]
            if kind_filter:
                where += " AND kind = ?"
                params.append(kind_filter)
            for n in conn.execute(
                f"SELECT item_id, distance FROM {table} WHERE {where} ORDER BY distance", params
            ):
                other = int(n["item_id"])
                cos = cosine_from_l2(float(n["distance"]))
                if other == item_id or cos < thresh:
                    continue
                a, b = sorted((item_id, other))
                conn.execute(
                    "INSERT INTO links(src, dst, kind, weight, created_at) "
                    "VALUES (?,?,'semantic',?,?) ON CONFLICT(src, dst, kind) "
                    "DO UPDATE SET weight = excluded.weight",
                    (a, b, cos, int(time.time())))
                made += 1
        except sqlite3.OperationalError:
            log.debug("edge seeding skipped for %s", table, exc_info=True)
    return made


def run(item_ids: list[int]) -> None:
    def work(conn: sqlite3.Connection) -> None:
        for i in item_ids:
            reindex_item(conn, i)
            seed_edges(conn, i)
    writer.submit(work)
