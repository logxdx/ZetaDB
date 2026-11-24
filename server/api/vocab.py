"""Vocabulary and review endpoints (spec 12)."""
from __future__ import annotations

import sqlite3
import time

from fastapi import APIRouter, Body, HTTPException

from server.db.conn import reader
from server.db.writer import writer
from server.search import vocab as V
from server.search.indexing import reindex_item

router = APIRouter(prefix="/api", tags=["vocab"])


@router.get("/vocab")
def list_vocab(state: str = "active", limit: int = 500) -> dict:
    rows = reader().execute(
        "SELECT v.id, v.term, v.kind, v.eye_alias, v.threshold, v.state, "
        "v.clip_mu, v.clip_sigma, v.accept_count, v.reject_count, "
        "(SELECT count(*) FROM item_tags it WHERE it.term_id = v.id AND it.state='accepted') AS n "
        "FROM vocab_terms v WHERE v.state = ? ORDER BY n DESC, v.term LIMIT ?",
        (state, limit)).fetchall()
    return {"terms": [dict(r) for r in rows]}


@router.post("/vocab/seed")
def seed() -> dict:
    added = V.seed_from_yaml()
    embedded = V.embed_vocab()
    return {"added": added, "embedded": embedded}


@router.post("/vocab/calibrate")
def calibrate() -> dict:
    return {"calibrated": V.calibrate_clip()}


@router.get("/vocab/merge-suggestions")
def suggestions(limit: int = 20) -> dict:
    return {"suggestions": V.merge_suggestions(limit)}


@router.post("/vocab/{term_id}/merge")
def do_merge(term_id: int, payload: dict = Body(...)) -> dict:
    into = payload.get("into")
    if not into:
        raise HTTPException(400, "body needs {'into': <term_id>}")
    V.merge(term_id, int(into))
    return {"merged": term_id, "into": into}


@router.get("/review")
def review(limit: int = 50) -> dict:
    conn = reader()
    tags = conn.execute(
        "SELECT it.item_id, it.term_id, v.term, it.confidence, i.kind, i.title "
        "FROM item_tags it JOIN vocab_terms v ON v.id = it.term_id "
        "JOIN items i ON i.id = it.item_id "
        "WHERE it.state = 'pending' ORDER BY it.confidence DESC LIMIT ?", (limit,)).fetchall()
    proposals = conn.execute(
        "SELECT id, term, created_at FROM vocab_terms WHERE state = 'pending' "
        "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return {"tags": [dict(r) for r in tags], "proposals": [dict(r) for r in proposals]}


@router.post("/review/tag/{item_id}/{term_id}/{action}")
def review_tag(item_id: int, term_id: int, action: str) -> dict:
    if action not in ("accept", "reject"):
        raise HTTPException(400, "action must be accept or reject")
    accepted = action == "accept"

    def work(conn: sqlite3.Connection) -> None:
        if accepted:
            # A reviewed tag becomes human-sourced and is then terminal.
            conn.execute(
                "UPDATE item_tags SET state='accepted', source='human', confidence=1.0 "
                "WHERE item_id=? AND term_id=?", (item_id, term_id))
        else:
            conn.execute("UPDATE item_tags SET state='rejected' WHERE item_id=? AND term_id=?",
                         (item_id, term_id))
        reindex_item(conn, item_id)
    writer.submit(work)
    V.apply_feedback(term_id, accepted)
    return {"item_id": item_id, "term_id": term_id, "action": action}


@router.post("/review/proposal/{term_id}/{action}")
def review_proposal(term_id: int, action: str) -> dict:
    if action not in ("accept", "reject"):
        raise HTTPException(400, "action must be accept or reject")
    state = "active" if action == "accept" else "retired"
    writer.submit(lambda c: c.execute("UPDATE vocab_terms SET state=? WHERE id=?",
                                      (state, term_id)))
    if action == "accept":
        V.embed_vocab()
    return {"term_id": term_id, "state": state}


@router.post("/items/{item_id}/tags/{term}")
def add_tag(item_id: int, term: str) -> dict:
    tid, how = V.resolve_or_propose(term)
    if tid is None:
        return {"added": False, "action": how, "term": term}

    def work(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO item_tags(item_id, term_id, confidence, source, state, created_at) "
            "VALUES (?,?,1.0,'human','accepted',?) ON CONFLICT(item_id, term_id) "
            "DO UPDATE SET source='human', state='accepted', confidence=1.0",
            (item_id, tid, int(time.time())))
        reindex_item(conn, item_id)
    writer.submit(work)
    return {"added": True, "action": how, "term_id": tid}


@router.delete("/items/{item_id}/tags/{term_id}")
def remove_tag(item_id: int, term_id: int) -> dict:
    def work(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE item_tags SET state='rejected' WHERE item_id=? AND term_id=?",
                     (item_id, term_id))
        reindex_item(conn, item_id)
    writer.submit(work)
    V.apply_feedback(term_id, accepted=False)
    return {"removed": True}
