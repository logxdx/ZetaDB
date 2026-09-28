"""Hybrid retrieval: tag path + BM25 + two vector spaces, fused with RRF.

The tag path is what implements the original brief's example - searching
"tree" returns items tagged `tree` AND items tagged with terms semantically
near `tree` - and every result records WHY it matched (spec 10).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field

from server.config import load
from server.db.conn import reader
from server.models.embeddings import cosine_from_l2

log = logging.getLogger(__name__)

RETRIEVERS = ("tag", "lexical", "text_vec", "img_vec")
_FTS_SPECIAL = re.compile(r'[":^*(){}\[\]-]')


def fts_query(q: str) -> str:
    """Make user input safe for FTS5 MATCH and prefix-match the last token."""
    tokens = [t for t in _FTS_SPECIAL.sub(" ", q).split() if t]
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " ".join(quoted)


@dataclass
class Hit:
    item_id: int
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, float] = field(default_factory=dict)

    @property
    def why(self) -> list[str]:
        order = sorted(self.ranks, key=lambda r: self.ranks[r])
        return order


def _rrf(groups: dict[str, list[tuple[int, float]]], weights: dict[str, float],
         k: int) -> list[Hit]:
    hits: dict[int, Hit] = {}
    for retriever, ranked in groups.items():
        w = weights.get(retriever, 1.0)
        for rank, (item_id, raw) in enumerate(ranked, start=1):
            h = hits.setdefault(item_id, Hit(item_id))
            h.score += w / (k + rank)
            h.ranks[retriever] = rank
            h.reasons[retriever] = round(raw, 4)
    return sorted(hits.values(), key=lambda h: -h.score)


def _lexical(conn: sqlite3.Connection, q: str, limit: int) -> list[tuple[int, float]]:
    match = fts_query(q)
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT rowid, bm25(fts_items, 3.0, 1.0, 1.0, 1.5, 2.0) AS s "
            "FROM fts_items WHERE fts_items MATCH ? ORDER BY s LIMIT ?", (match, limit)
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("fts query failed for %r: %s", match, exc)
        return []
    # bm25() is negative, more negative = better. Flip for readability.
    return [(r["rowid"], -r["s"]) for r in rows]


def _tag_seeds(conn: sqlite3.Connection, q: str, min_cos: float,
               embed_text) -> list[tuple[int, str, float]]:
    """Resolve a query to vocabulary terms: exact, alias, then embedding-near."""
    seeds: dict[int, tuple[str, float]] = {}
    ql = q.strip().casefold()
    for r in conn.execute(
        "SELECT id, term FROM vocab_terms WHERE state='active' AND lower(term) = ?", (ql,)
    ):
        seeds[r["id"]] = (r["term"], 1.0)
    for r in conn.execute(
        "SELECT t.id, t.term FROM vocab_aliases a JOIN vocab_terms t ON t.id = a.term_id "
        "WHERE lower(a.alias) = ? AND t.state='active'", (ql,)
    ):
        seeds.setdefault(r["id"], (r["term"], 0.95))

    if embed_text is not None:
        try:
            vec = embed_text(q)
            for r in conn.execute(
                "SELECT term_id, distance FROM vec_vocab "
                "WHERE embedding MATCH ? AND k = 12 ORDER BY distance", (vec,)
            ):
                cos = cosine_from_l2(float(r["distance"]))
                if cos >= min_cos and r["term_id"] not in seeds:
                    name = conn.execute(
                        "SELECT term FROM vocab_terms WHERE id = ? AND state='active'",
                        (r["term_id"],)).fetchone()
                    if name:
                        seeds[r["term_id"]] = (name["term"], cos)
        except Exception:
            log.debug("vocab knn unavailable", exc_info=True)
    return [(tid, name, sim) for tid, (name, sim) in seeds.items()]


def _tagged(conn: sqlite3.Connection, seeds, limit: int) -> list[tuple[int, float]]:
    if not seeds:
        return []
    marks = ",".join("?" * len(seeds))
    ids = [s[0] for s in seeds]
    sim = {s[0]: s[2] for s in seeds}
    rows = conn.execute(
        f"SELECT it.item_id, it.term_id, it.confidence FROM item_tags it "
        f"WHERE it.term_id IN ({marks}) AND it.state = 'accepted'", ids
    ).fetchall()
    best: dict[int, float] = {}
    for r in rows:
        s = r["confidence"] * sim.get(r["term_id"], 1.0)
        best[r["item_id"]] = max(best.get(r["item_id"], 0.0), s)
    return sorted(best.items(), key=lambda kv: -kv[1])[:limit]


def _knn(conn: sqlite3.Connection, table: str, vec: bytes, limit: int,
         kind: str | None = None, min_cos: float = 0.0) -> list[tuple[int, float]]:
    where = "embedding MATCH ? AND k = ?"
    params: list = [vec, limit]
    if kind is not None:
        where += " AND kind = ?"
        params.append(kind)
    col = "item_id"
    rows = conn.execute(
        f"SELECT {col}, distance FROM {table} WHERE {where} ORDER BY distance", params
    ).fetchall()
    out: dict[int, float] = {}
    for r in rows:
        cos = cosine_from_l2(float(r["distance"]))
        if cos < min_cos:
            continue      # KNN always returns k rows; without a floor a
        iid = int(r[col])  # small corpus "matches" every query
        out[iid] = max(out.get(iid, -1.0), cos)
    return sorted(out.items(), key=lambda kv: -kv[1])


def search(q: str, *, limit: int = 60, filters: dict | None = None,
           embed_text=None, embed_jina_text=None) -> dict:
    """Run every available retriever and fuse. Vector retrievers are skipped
    silently when their model is not loaded, so search degrades to BM25 +
    tags rather than failing (this is what makes M2 usable before M3)."""
    cfg = load()
    conn = reader()
    k = cfg.get("search.rrf_k", 60)
    weights = {
        "tag": cfg.get("search.w_tag", 1.30),
        "lexical": cfg.get("search.w_lexical", 1.00),
        "text_vec": cfg.get("search.w_text_vec", 1.00),
        "img_vec": cfg.get("search.w_img_vec", 0.90),
    }
    pool = max(limit * 3, 50)
    groups: dict[str, list[tuple[int, float]]] = {}

    seeds = _tag_seeds(conn, q, cfg.get("search.tag_seed_min_cos", 0.75), embed_text)
    if (tagged := _tagged(conn, seeds, pool)):
        groups["tag"] = tagged
    if (lex := _lexical(conn, q, pool)):
        groups["lexical"] = lex
    if embed_text is not None:
        try:
            hits_t = _knn(conn, "vec_text", embed_text(q), 50,
                          min_cos=cfg.get("search.min_cos_text", 0.30))
            if hits_t:
                groups["text_vec"] = hits_t
        except Exception:
            log.debug("vec_text knn skipped", exc_info=True)
    if embed_jina_text is not None:
        try:
            hits_i = _knn(conn, "vec_jina", embed_jina_text(q), 50,
                          min_cos=cfg.get("search.min_cos_jina", 0.18))
            if hits_i:
                groups["img_vec"] = hits_i
        except Exception:
            log.debug("vec_jina knn skipped", exc_info=True)

    hits = _rrf(groups, weights, k)
    hits = _apply_filters(conn, hits, filters or {})
    return {
        "query": q,
        "seeds": [{"term": n, "similarity": round(s, 3)} for _, n, s in seeds],
        "retrievers": {r: len(v) for r, v in groups.items()},
        "hits": hits[:limit],
    }


def _apply_filters(conn: sqlite3.Connection, hits: list[Hit], f: dict) -> list[Hit]:
    """Structured filters compose as SQL predicates over items, never over
    vec0 metadata - that sidesteps the 16-column ceiling (spec 10.4)."""
    if not hits:
        return hits
    where, params = ["near_dup_of IS NULL"], []
    if f.get("kind"):
        where.append("kind = ?"); params.append(f["kind"])
    if f.get("project"):
        where.append("project_id = ?"); params.append(f["project"])
    if f.get("since"):
        where.append("created_at >= ?"); params.append(int(f["since"]))
    if f.get("until"):
        where.append("created_at <= ?"); params.append(int(f["until"]))
    if f.get("has_ocr"):
        where.append("flow_text IS NOT NULL AND flow_text != ''")
    ids = [h.item_id for h in hits]
    marks = ",".join("?" * len(ids))
    ok = {r[0] for r in conn.execute(
        f"SELECT id FROM items WHERE id IN ({marks}) AND {' AND '.join(where)}", (*ids, *params))}
    return [h for h in hits if h.item_id in ok]
