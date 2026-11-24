"""Search and related-items endpoints (spec 10, 12)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.api.items import _tags_for, card
import sqlite3

from server.db.conn import reader
from server.models.embeddings import cosine_from_l2
from server.search import hybrid
from server.models.embeddings import get_embedder

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search")
def search(q: str, limit: int = 60, group: bool = False,
           kind: str | None = None, project: int | None = None,
           since: int | None = None, until: int | None = None,
           has_ocr: bool = False) -> dict:
    if not q.strip():
        raise HTTPException(400, "empty query")
    res = hybrid.search(
        q, limit=limit,
        filters={"kind": kind, "project": project, "since": since,
                 "until": until, "has_ocr": has_ocr},
        embed_query=get_embedder(),
    )
    conn = reader()
    ids = [h.item_id for h in res["hits"]]
    if not ids:
        return {**res, "hits": [], "results": [], "groups": {}}

    marks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT {card.__globals__['CARD_FIELDS']} FROM items WHERE id IN ({marks})", ids)}
    tags = _tags_for(conn, ids)

    results = []
    for h in res["hits"]:
        row = rows.get(h.item_id)
        if row is None:
            continue
        c = card(row, tags.get(h.item_id))
        c["score"] = round(h.score, 5)
        c["why"] = h.why
        c["ranks"] = h.ranks
        c["match"] = h.match
        if h.moment is not None:
            c["moment"] = h.moment
        results.append(c)

    out = {"query": res["query"], "seeds": res["seeds"],
           "retrievers": res["retrievers"], "results": results}
    if group:
        buckets: dict[str, list] = {}
        for c in results:
            buckets.setdefault(c["why"][0] if c["why"] else "other", []).append(c)
        out["groups"] = buckets
    return out


@router.get("/items/{item_id}/related")
def related(item_id: int, limit: int = 60, mode: str = "similar") -> dict:
    """Neighbours of one item, two ways.

    `similar` is a live KNN in whichever embedding spaces the item has, so
    it finds things that look or read alike even with no tags in common.
    `tag`  is set overlap on accepted tags, ranked by how many are shared —
    a different question, and often a more explainable answer.
    """
    if mode not in ("similar", "tag"):
        raise HTTPException(400, "mode must be 'similar' or 'tag'")
    conn = reader()
    if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
        raise HTTPException(404, "no such item")
    limit = max(1, min(limit, 200))

    scored: dict[int, float] = {}
    shared: dict[int, list[str]] = {}

    if mode == "tag":
        rows = conn.execute(
            "SELECT b.item_id, v.term FROM item_tags a "
            "JOIN item_tags b ON b.term_id = a.term_id AND b.item_id != a.item_id "
            "JOIN vocab_terms v ON v.id = a.term_id "
            "JOIN items i ON i.id = b.item_id "
            "WHERE a.item_id = ? AND a.state = 'accepted' AND b.state = 'accepted' "
            "AND i.near_dup_of IS NULL",
            (item_id,),
        ).fetchall()
        for r in rows:
            scored[r["item_id"]] = scored.get(r["item_id"], 0.0) + 1.0
            shared.setdefault(r["item_id"], []).append(r["term"])
    else:
        for kind in ("visual", "item"):
            try:
                row = conn.execute(
                    "SELECT embedding FROM vec_items WHERE item_id = ? AND kind = ? LIMIT 1",
                    (item_id, kind)).fetchone()
                if row is None:
                    continue
                qp: list = [row["embedding"], limit + 1, kind]
                for n in conn.execute(
                    "SELECT item_id, distance FROM vec_items "
                    "WHERE embedding MATCH ? AND k = ? AND kind = ? ORDER BY distance", qp
                ):
                    other = int(n["item_id"])
                    if other == item_id:
                        continue
                    cos = cosine_from_l2(float(n["distance"]))
                    scored[other] = max(scored.get(other, 0.0), cos)
            except sqlite3.OperationalError:
                continue

    ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
    ids = [i for i, _ in ranked]
    if not ids:
        return {"item_id": item_id, "mode": mode, "related": []}

    marks = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        f"SELECT {card.__globals__['CARD_FIELDS']} FROM items "
        f"WHERE id IN ({marks}) AND near_dup_of IS NULL", ids)}
    tags = _tags_for(conn, ids)
    out = []
    for i, score in ranked:
        if i not in rows:
            continue
        c = card(rows[i], tags.get(i))
        c["score"] = round(score, 4)
        if mode == "tag":
            c["shared"] = sorted(set(shared.get(i, [])))
        out.append(c)
    return {"item_id": item_id, "mode": mode, "related": out}


@router.get("/items/{item_id}/graph")
def graph(item_id: int, hops: int = 2, limit: int = 60) -> dict:
    """Neighbourhood graph only - the global graph is never rendered (spec 11)."""
    conn = reader()
    if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
        raise HTTPException(404, "no such item")
    hops = max(1, min(hops, 3))
    seen = {item_id}
    frontier = {item_id}
    for _ in range(hops):
        if not frontier or len(seen) >= limit:
            break
        marks = ",".join("?" * len(frontier))
        rows = conn.execute(
            f"SELECT src, dst FROM links WHERE src IN ({marks}) OR dst IN ({marks})",
            (*frontier, *frontier),
        ).fetchall()
        nxt = {r["src"] for r in rows} | {r["dst"] for r in rows}
        frontier = nxt - seen
        seen |= frontier
        if len(seen) > limit:
            seen = set(list(seen)[:limit])
            break

    marks = ",".join("?" * len(seen))
    nodes = conn.execute(
        f"SELECT id, kind, title, thumb_path, caption FROM items WHERE id IN ({marks})", tuple(seen)
    ).fetchall()
    edges = conn.execute(
        f"SELECT src, dst, kind, weight, label FROM links "
        f"WHERE src IN ({marks}) AND dst IN ({marks})", (*seen, *seen)
    ).fetchall()
    return {
        "focus": item_id,
        "nodes": [{"id": n["id"], "kind": n["kind"], "title": n["title"],
                   "thumb_url": f"/api/thumb/{n['id']}" if n["thumb_path"] else None,
                   "preview": (n["caption"] or n["title"] or "")[:90]} for n in nodes],
        "edges": [{"src": e["src"], "dst": e["dst"], "kind": e["kind"],
                   "weight": round(e["weight"], 3), "label": e["label"]} for e in edges],
    }
