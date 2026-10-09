"""Item CRUD, media serving, and the SSE stream (spec 12)."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.ingest.events import hub
from server.ingest.stages.intake import (ingest_image, ingest_note, ingest_text_file,
                                         is_image, is_text)

router = APIRouter(prefix="/api", tags=["items"])

CARD_FIELDS = (
    "id, kind, status, title, body, thumb_path, mime, width, height, "
    "caption, caption_model, near_dup_of, project_id, created_at, updated_at"
)


def _tags_for(conn: sqlite3.Connection, ids: list[int]) -> dict[int, list[dict]]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT it.item_id, v.term, it.confidence, it.source, it.state "
        f"FROM item_tags it JOIN vocab_terms v ON v.id = it.term_id "
        f"WHERE it.item_id IN ({marks}) AND it.state != 'rejected' "
        f"ORDER BY it.source ASC, it.confidence DESC", ids   # human before machine
    ).fetchall()
    out: dict[int, list[dict]] = {i: [] for i in ids}
    for r in rows:
        out[r["item_id"]].append(
            {"term": r["term"], "confidence": round(r["confidence"], 3),
             "source": r["source"], "state": r["state"]}
        )
    return out


def card(row: sqlite3.Row, tags: list[dict] | None = None) -> dict[str, Any]:
    # SELECT * carries the dhash BLOB, which is not JSON-serialisable.
    d = {k: v for k, v in dict(row).items() if not isinstance(v, bytes)}
    d["tags"] = tags or []
    if d.get("thumb_path"):
        d["thumb_url"] = f"/api/thumb/{d['id']}"
    if d["kind"] == "image":
        d["file_url"] = f"/api/file/{d['id']}"
    d.pop("thumb_path", None)
    # Notes carry their text; images carry a preview of their OCR/caption.
    if d["kind"] == "note" and d.get("body"):
        d["preview"] = d["body"][:400]
        d.pop("body", None)
    else:
        d.pop("body", None)
        d["preview"] = (d.get("caption") or "")[:400]
    return d


# How the canvas can be ordered. "tags" groups the library by each item's
# strongest tag, so similar things sit together — which is the useful
# reading of "sort by tags" for a visual archive; plain alphabetical on a
# tag name would scatter an item's other tags anyway.
SORTS = {
    "date":  "i.created_at DESC, i.id DESC",
    "title": "i.title COLLATE NOCASE ASC, i.id DESC",
    "tags":  "top_tag IS NULL, top_tag COLLATE NOCASE ASC, i.id DESC",
}


@router.get("/items")
def list_items(cursor: int | None = None, limit: int = 60,
               kind: str | None = None, project: int | None = None,
               status: str | None = None, include_dups: bool = False,
               sort: str = "date") -> dict:
    limit = max(1, min(limit, 200))
    if sort not in SORTS:
        raise HTTPException(400, f"sort must be one of {sorted(SORTS)}")
    conn = reader()
    where, params = ["1=1"], []
    if kind:
        where.append("i.kind = ?"); params.append(kind)
    if project:
        where.append("i.project_id = ?"); params.append(project)
    if status:
        where.append("i.status = ?"); params.append(status)
    if not include_dups:
        where.append("i.near_dup_of IS NULL")

    # The default ordering is stable under insertion, so it keyset-paginates
    # on id. The others cannot, and fall back to an offset — fine for a
    # personal library, and it keeps the ordering honest.
    offset = 0
    if sort == "date":
        if cursor:
            where.append("i.id < ?"); params.append(cursor)
    else:
        offset = max(0, cursor or 0)

    cols = ", ".join(f"i.{c.strip()}" for c in CARD_FIELDS.split(","))
    rows = conn.execute(
        f"SELECT {cols}, ("
        "  SELECT v.term FROM item_tags it JOIN vocab_terms v ON v.id = it.term_id"
        "  WHERE it.item_id = i.id AND it.state = 'accepted'"
        "  ORDER BY it.confidence DESC LIMIT 1"
        f") AS top_tag FROM items i WHERE {' AND '.join(where)} "
        f"ORDER BY {SORTS[sort]} LIMIT ? OFFSET ?",
        (*params, limit + 1, offset),
    ).fetchall()

    more = len(rows) > limit
    rows = rows[:limit]
    tags = _tags_for(conn, [r["id"] for r in rows])
    if not more or not rows:
        next_cursor = None
    elif sort == "date":
        next_cursor = rows[-1]["id"]
    else:
        next_cursor = offset + limit
    return {
        "items": [card(r, tags.get(r["id"])) for r in rows],
        "next_cursor": next_cursor,
        "sort": sort,
    }


@router.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    conn = reader()
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such item")
    d = card(row, _tags_for(conn, [item_id]).get(item_id))
    full = {k: v for k, v in dict(row).items() if not isinstance(v, bytes)}
    d["body"] = full.get("body")
    d["layout_text"] = full.get("layout_text")
    d["flow_text"] = full.get("flow_text")
    d["boxes"] = json.loads(full["boxes_json"]) if full.get("boxes_json") else None
    d["entities"] = [dict(r) for r in conn.execute(
        "SELECT kind, value, confidence FROM entities WHERE item_id = ?", (item_id,))]
    d["todos"] = [dict(r) for r in conn.execute(
        "SELECT id, text, due_date, done FROM todos WHERE item_id = ?", (item_id,))]
    d["errors"] = [dict(r) for r in conn.execute(
        "SELECT stage, error, ts FROM item_errors WHERE item_id = ? ORDER BY ts DESC", (item_id,))]
    d["near_dups"] = [r["id"] for r in conn.execute(
        "SELECT id FROM items WHERE near_dup_of = ?", (item_id,))]
    return d


@router.post("/items")
async def create_items(files: list[UploadFile] = File(default=[]),
                       body: str | None = Form(default=None),
                       title: str | None = Form(default=None)) -> dict:
    """Drag-drop / paste (multipart files) or the quick-note composer (body)."""
    created: list[dict] = []
    if body and body.strip():
        r = ingest_note(body, title=title, source="compose")
        created.append({"id": r.item_id, "created": r.created, "duplicate_of": r.duplicate_of})

    if files:
        tmp_root = load().path("paths.data") / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        for up in files:
            if not up.filename:
                continue
            tmp = tmp_root / f"{int(time.time()*1000)}_{Path(up.filename).name}"
            tmp.write_bytes(await up.read())
            try:
                if is_image(tmp):
                    r = ingest_image(tmp, source="upload", title=Path(up.filename).stem)
                elif is_text(tmp):
                    r = ingest_text_file(tmp, source="upload")
                else:
                    created.append({"filename": up.filename,
                                    "error": "unsupported type — images, .txt or .md"})
                    continue
                created.append({"id": r.item_id, "created": r.created,
                                "duplicate_of": r.duplicate_of, "near_dup_of": r.near_dup_of})
            except Exception as exc:
                created.append({"filename": up.filename, "error": str(exc)[:120]})
            finally:
                tmp.unlink(missing_ok=True)

    if not created:
        raise HTTPException(400, "nothing to ingest: send files or a body")
    for c in created:
        if c.get("id") and c.get("created"):
            hub.emit("item.created", {"id": c["id"]})
    return {"created": created}


@router.patch("/items/{item_id}")
def patch_item(item_id: int, payload: dict = Body(...)) -> dict:
    allowed = {"title", "body", "project_id", "caption"}
    fields = {k: v for k, v in payload.items() if k in allowed}
    if not fields:
        raise HTTPException(400, f"no editable fields; allowed: {sorted(allowed)}")

    def work(conn: sqlite3.Connection) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE items SET {sets}, updated_at = ? WHERE id = ?",
                     (*fields.values(), int(time.time()), item_id))
        # Keep the lexical index in step with a manual edit.
        from server.search.indexing import reindex_item
        reindex_item(conn, item_id)

    writer.submit(work)
    hub.emit("item.updated", {"id": item_id})
    return get_item(item_id)


@router.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict:
    def work(conn: sqlite3.Connection) -> None:
        for t in ("vec_items", "vec_phash"):
            conn.execute(f"DELETE FROM {t} WHERE item_id = ?", (item_id,))
        conn.execute("DELETE FROM fts_items WHERE rowid = ?", (item_id,))
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    writer.submit(work)
    hub.emit("item.deleted", {"id": item_id})
    return {"deleted": item_id}


def _media(item_id: int, column: str) -> FileResponse:
    row = reader().execute(f"SELECT {column} FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None or not row[column]:
        raise HTTPException(404, "no such media")
    path = load().path("paths.data") / row[column]
    if not path.exists():
        raise HTTPException(404, "media missing on disk")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/thumb/{item_id}")
def thumb(item_id: int) -> FileResponse:
    return _media(item_id, "thumb_path")


@router.get("/file/{item_id}")
def file(item_id: int) -> FileResponse:
    return _media(item_id, "blob_path")


@router.get("/events")
async def events() -> StreamingResponse:
    hub.bind(asyncio.get_running_loop())
    q = hub.subscribe()

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {payload}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            hub.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
