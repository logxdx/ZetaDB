"""Stage 6 - embeddings. CPU-side models, so no GPU lease is taken.

Two spaces, never mixed (spec 9.3):
  vec_text  Qwen3-Embedding-0.6B, notes and (by config) image text
  vec_jina  jina-clip-v2, image tower and its own text tower

The jina image tower wants the GPU; it is requested through the arbiter
only if pool A is free, otherwise it runs on CPU rather than blocking the
whole batch behind a lease.
"""
from __future__ import annotations

import logging
import sqlite3

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models.jina_clip import JinaClip, engine as clip_engine
from server.models.qwen_embed import QwenEmbedder, engine as qwen_engine

log = logging.getLogger(__name__)

def _models() -> tuple[QwenEmbedder, JinaClip]:
    return qwen_engine(), clip_engine()


def searchable_text(row: sqlite3.Row) -> str:
    """What an item means, as text. Notes are their body; images are their
    caption plus OCR reading order (flow_text, never the padded grid)."""
    if row["kind"] == "note":
        return (row["body"] or row["title"] or "").strip()
    parts = [row["caption"] or "", row["flow_text"] or "", row["title"] or ""]
    return "\n".join(p for p in parts if p.strip()).strip()


def run(item_ids: list[int]) -> None:
    cfg = load()
    qwen, jina = _models()
    conn = reader()
    root = cfg.path("paths.data")
    cover_images = cfg.get("embeddings.qwen_covers_image_text", True)

    rows = []
    for iid in item_ids:
        r = conn.execute(
            "SELECT id, kind, title, body, caption, flow_text, blob_path "
            "FROM items WHERE id = ?", (iid,)).fetchone()
        if r is not None:
            rows.append(r)
    if not rows:
        return

    text_vecs: dict[int, bytes] = {}
    jina_txt: dict[int, bytes] = {}

    # --- Qwen text space
    if qwen.available:
        targets = [r for r in rows
                   if searchable_text(r) and (r["kind"] == "note" or cover_images)]
        if targets:
            try:
                vecs = qwen.embed_many([searchable_text(r) for r in targets])
                for r, v in zip(targets, vecs, strict=True):
                    text_vecs[r["id"]] = v
            except Exception:
                log.exception("qwen embedding failed for %d item(s)", len(targets))

    # --- jina shared space. Image vectors are written by the tagging stage,
    # which already computes them for zero-shot tagging; only the text tower
    # runs here, on CPU.
    if jina.available:
        imgs = [r for r in rows if r["kind"] == "image" and r["blob_path"]]
        texts = [r for r in imgs if searchable_text(r)]
        if texts:
            try:
                vecs = jina.embed_texts([searchable_text(r) for r in texts])
                for r, v in zip(texts, vecs, strict=True):
                    jina_txt[r["id"]] = v
            except Exception:
                log.exception("jina text embedding failed")

    if not (text_vecs or jina_txt):
        return

    def work(c: sqlite3.Connection) -> None:
        for iid, v in text_vecs.items():
            c.execute("DELETE FROM vec_text WHERE item_id = ?", (iid,))
            c.execute("INSERT INTO vec_text(item_id, embedding) VALUES (?, ?)", (iid, v))
        for iid, v in jina_txt.items():
            c.execute("DELETE FROM vec_jina WHERE item_id = ? AND kind = 'imgtext'", (iid,))
            c.execute("INSERT INTO vec_jina(item_id, kind, embedding) VALUES (?,'imgtext',?)",
                      (iid, v))
    writer.submit(work)
    log.info("embedded: %d text, %d image-text", len(text_vecs), len(jina_txt))
