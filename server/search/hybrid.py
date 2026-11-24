"""Hybrid retrieval: tag path + BM25 + three vector views, fused with RRF.

The third vector view is `moment`: KNN over audio/video SEGMENTS. An item
it surfaces carries the window that matched, so a query can land on 12:40
of a recording or the scene at 0:35 of a clip rather than on the file.

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

RETRIEVERS = ("tag", "lexical", "text_vec", "img_vec", "moment")
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
    # Best-matching audio/video window, when the moment retriever found one.
    moment: dict | None = None

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
         kind: str | None = None, min_cos: float = 0.0,
         rel_margin: float = 0.0) -> list[tuple[int, float]]:
    """Top-k, then TWO gates, because they answer different questions and
    neither one covers the other. [M0] measured on a 29-image corpus:

      min_cos (absolute)  "is anything here good enough at all?"
          A query with no answer in the corpus - "recipe for sourdough
          bread" against nature photos - still gets 29 KNN rows back, all
          bunched at ~0.50. Only an absolute floor can say "none of these".

      rel_margin (relative)  "where does this result set fall off?"
          A query with many answers - "mountain landscape" - clears 0.55 for
          26 of 29 images because the whole corpus is outdoor photography.
          Cutting at (best - margin) keeps the 6 that are actually close.

    The relative gate ALONE fails case one outright: when nothing matches,
    the tail is flat, every row sits within the margin of the best, and all
    29 are admitted. The absolute floor alone fails case two. Both, in that
    order, handle both.

    Ranking itself does not depend on either: _rrf fuses by RANK, so these
    only decide who becomes a candidate, and - via `reasons` - which "why"
    badge the UI is entitled to show.
    """
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
            continue
        iid = int(r[col])
        out[iid] = max(out.get(iid, -1.0), cos)
    ranked = sorted(out.items(), key=lambda kv: -kv[1])
    if rel_margin > 0 and ranked:
        cut = ranked[0][1] - rel_margin
        ranked = [x for x in ranked if x[1] >= cut]
    return ranked


def _moments(conn: sqlite3.Connection, vec: bytes, limit: int, min_cos: float,
             rel_margin: float) -> tuple[list[tuple[int, float]], dict[int, dict]]:
    """KNN over segment vectors, reduced to one best window per item.

    Same two gates as _knn, for the same reasons: an absolute floor so a
    query with no answer returns nothing, then a margin relative to the best
    hit so a broad query keeps only what is actually close."""
    rows = conn.execute(
        "SELECT segment_id, item_id, distance FROM vec_segments "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance", (vec, limit)).fetchall()
    best: dict[int, tuple[float, int]] = {}
    for r in rows:
        cos = cosine_from_l2(float(r["distance"]))
        if cos < min_cos:
            continue
        iid = int(r["item_id"])
        if iid not in best or cos > best[iid][0]:
            best[iid] = (cos, int(r["segment_id"]))
    ranked = sorted(((i, c) for i, (c, _) in best.items()), key=lambda kv: -kv[1])
    if rel_margin > 0 and ranked:
        cut = ranked[0][1] - rel_margin
        ranked = [x for x in ranked if x[1] >= cut]
    moments: dict[int, dict] = {}
    if ranked:
        seg_ids = [best[i][1] for i, _ in ranked]
        marks = ",".join("?" * len(seg_ids))
        for s in conn.execute(
                f"SELECT id, item_id, kind, start_s, end_s, frame_path FROM segments "
                f"WHERE id IN ({marks})", seg_ids):
            moments[int(s["item_id"])] = {
                "segment_id": int(s["id"]), "kind": s["kind"],
                "start": round(float(s["start_s"]), 2), "end": round(float(s["end_s"]), 2),
                "frame_url": f"/api/frame/{s['id']}" if s["frame_path"] else None,
                "score": round(best[int(s["item_id"])][0], 4)}
    return ranked, moments


def search(q: str, *, limit: int = 60, filters: dict | None = None,
           embed_query=None) -> dict:
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
        "moment": cfg.get("search.w_moment", 1.00),
    }
    pool = max(limit * 3, 50)
    groups: dict[str, list[tuple[int, float]]] = {}
    moments: dict[int, dict] = {}

    seeds = _tag_seeds(conn, q, cfg.get("search.tag_seed_min_cos", 0.75), embed_query)
    if (tagged := _tagged(conn, seeds, pool)):
        groups["tag"] = tagged
    if (lex := _lexical(conn, q, pool)):
        groups["lexical"] = lex
    if embed_query is not None:
        # One query vector, one space, two views of it. text_vec asks "does
        # this item MEAN what was asked" (kind='item', which for an image
        # already carries its caption and OCR); img_vec asks "does it LOOK
        # like it" (kind='visual', pixels only). Both are the same KNN with
        # a metadata filter - the two separate models and the two separate
        # tables this replaced are gone, but the distinction the UI shows
        # the user as "similar wording" vs "looks similar" survives.
        try:
            qv = embed_query(q)
        except Exception:
            qv = None
            log.debug("query embedding failed", exc_info=True)
        if qv is not None:
            rel = cfg.get("search.rel_margin", 0.08)
            for group, kind, key, default in (
                ("text_vec", "item", "search.min_cos_item", 0.55),
                ("img_vec", "visual", "search.min_cos_visual", 0.55),
            ):
                try:
                    hits = _knn(conn, "vec_items", qv, 50, kind=kind,
                                min_cos=cfg.get(key, default), rel_margin=rel)
                    if hits:
                        groups[group] = hits
                except Exception:
                    log.debug("vec_items knn skipped for kind=%s", kind, exc_info=True)
            try:
                found, moments = _moments(conn, qv, 100, cfg.get("search.min_cos_moment", 0.50),
                                          rel)
                if found:
                    groups["moment"] = found
            except Exception:
                log.debug("moment knn skipped", exc_info=True)

    hits = _rrf(groups, weights, k)
    for h in hits:
        h.moment = moments.get(h.item_id)
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
