"""Stage 6 - embeddings. CPU-side model, so no GPU lease is taken.

ONE space, 768-d, EmbeddingGemma 2 (spec 9.3). This stage owns the
kind='item' row in vec_items - the canonical searchable vector:

  note   -> its text, with the document task prefix
  image  -> the image INTERLEAVED with its own caption and OCR text, as a
            single vector

The image case is the part that changed shape. It used to take three rows
across two tables that could not be compared with each other: a jina image
vector, a jina text vector over the caption, and a Qwen vector over the
same caption written only so that notes and images shared a space at all
(the spec 9.4 deviation). Now the pixels and the words go into the backbone
together and come out as one vector.

The kind='visual' row - image with no text, which zero-shot tag scoring
needs uncontaminated - is written by the tagging stage, which already runs
the vision encoder under a pool A lease. This stage never touches the GPU:
the EG2 backbone is CPU-resident so that a search query never waits on
pool A, and an image reaching here has its soft tokens recomputed on the
CPU rather than taking a lease for a sub-second job.
"""
from __future__ import annotations

import logging
import sqlite3

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models import eg2

log = logging.getLogger(__name__)


def searchable_text(row: sqlite3.Row) -> str:
    """What an item means, as text. Notes are their body; images are their
    caption plus OCR reading order (flow_text, never the padded grid)."""
    if row["kind"] == "note":
        return (row["body"] or row["title"] or "").strip()
    parts = [row["caption"] or "", row["flow_text"] or ""]
    return "\n".join(p for p in parts if p.strip()).strip()


def run(item_ids: list[int]) -> None:
    cfg = load()
    e = eg2.engine()
    if not e.available:
        log.debug("eg2 weights absent; skipping embed stage")
        return
    conn = reader()
    root = cfg.path("paths.data")

    rows = []
    for iid in item_ids:
        r = conn.execute(
            "SELECT id, kind, title, body, caption, flow_text, blob_path "
            "FROM items WHERE id = ?", (iid,)).fetchone()
        if r is not None:
            rows.append(r)
    if not rows:
        return

    vecs: dict[int, bytes] = {}

    # --- notes, and any image whose blob has gone missing: text only.
    texts = [r for r in rows
             if (r["kind"] == "note" or not r["blob_path"]) and searchable_text(r)]
    if texts:
        try:
            # Document side of the asymmetric pair, NOT the query side. The
            # card is explicit that mixing them costs retrieval quality, and
            # the title field is always present even when it is `none`.
            payload = [eg2.format_document(searchable_text(r), r["title"]) for r in texts]
            for r, v in zip(texts, e.embed_texts(payload), strict=True):
                vecs[r["id"]] = v
        except Exception:
            log.exception("text embedding failed for %d item(s)", len(texts))

    # --- images: pixels and words in one vector.
    for r in (r for r in rows if r["kind"] == "image" and r["blob_path"]):
        try:
            v = e.embed_image(root / r["blob_path"],
                              text=searchable_text(r) or None, title=r["title"])
            if v is not None:
                vecs[r["id"]] = v
        except Exception:
            log.exception("image embedding failed for item %d", r["id"])

    if not vecs:
        return

    def work(c: sqlite3.Connection) -> None:
        for iid, v in vecs.items():
            c.execute("DELETE FROM vec_items WHERE item_id = ? AND kind = 'item'", (iid,))
            c.execute("INSERT INTO vec_items(item_id, kind, embedding) VALUES (?,'item',?)",
                      (iid, v))
    writer.submit(work)
    log.info("embedded %d item vector(s)", len(vecs))
