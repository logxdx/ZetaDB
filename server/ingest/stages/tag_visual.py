"""Stage 3 - visual tagging. GPU pool A.

Scores every active vocabulary term against each image from two
independent signals and fuses them (spec 7.2-7.4):

  JoyTag  terms with a Danbooru alias, independent per-tag sigmoid
  EG2     every term, zero-shot cosine calibrated per term

This stage owns the kind='visual' row in vec_items: the image with NO text
in it. That is deliberate. Scoring the term "chart" against the kind='item'
vector - which interleaves the image with its own OCR text - would match
any screenshot whose text merely says "chart", which is the exact leak
vocab_terms.requires_text exists to prevent.

The term vectors come from the same model and live in the same space, so
there is no separate CLIP text tower any more: a term string and an image
are directly comparable.
"""
from __future__ import annotations

import logging
import sqlite3
import struct
import time

import numpy as np

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models.eg2 import Eg2, engine as eg2_engine
from server.models.joytag import engine as joytag_engine
from server.search import vocab as V

log = logging.getLogger(__name__)

_term_vecs: dict[int, np.ndarray] = {}
_term_vec_stamp = 0.0


def _unpack(b: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"<{len(b) // 4}f", b), dtype=np.float32)


def _term_vectors(terms: list[V.Term], e: Eg2) -> dict[int, np.ndarray]:
    """Vectors for each term prompt, cached across batches.

    Query side, not document side: a term is being used to retrieve images,
    which is the asymmetric case the SearchQuery prefix is trained for.
    """
    global _term_vecs, _term_vec_stamp
    missing = [t for t in terms if t.id not in _term_vecs]
    if missing:
        vecs = e.embed_texts([t.prompt() for t in missing], is_query=True)
        for t, v in zip(missing, vecs, strict=True):
            _term_vecs[t.id] = _unpack(v)
        _term_vec_stamp = time.time()
    return _term_vecs


def run(item_ids: list[int]) -> None:
    cfg = load()
    conn = reader()
    root = cfg.path("paths.data")

    rows = [r for r in (
        conn.execute("SELECT id, kind, blob_path, flow_text, boxes_json FROM items "
                     "WHERE id = ?", (i,)).fetchone()
        for i in item_ids) if r is not None and r["kind"] == "image" and r["blob_path"]]
    if not rows:
        return

    terms = V.active_terms(conn)
    if not terms:
        log.info("vocabulary is empty; skipping visual tagging")
        return

    paths = [root / r["blob_path"] for r in rows]

    # --- JoyTag: full 5813-way sigmoid, we read only our own aliases
    jt = joytag_engine()
    jt_scores: list[dict[str, float] | None] = [None] * len(rows)
    if jt.available:
        try:
            jt_scores = jt.score_batch(paths)
        except Exception:
            log.exception("joytag scoring failed")

    # --- EG2 zero-shot: covers every term, including those Danbooru lacks
    e = eg2_engine()
    img_vecs: list[np.ndarray | None] = [None] * len(rows)
    packed_img: list[bytes | None] = [None] * len(rows)
    tvecs: dict[int, np.ndarray] = {}
    if e.vision_available:
        try:
            packed_img = [e.embed_image(p) for p in paths]   # no text: visual only
            img_vecs = [_unpack(p) if p else None for p in packed_img]
            tvecs = _term_vectors(terms, e)
        except Exception:
            log.exception("eg2 visual tagging failed")

    review_floor = cfg.get("tagging.review_floor", 0.25)
    now = int(time.time())
    tag_rows: list[tuple] = []
    score_rows: list[tuple] = []

    for i, r in enumerate(rows):
        iid = r["id"]
        # Does this image contain text at all? Terms that only make sense
        # on text-bearing images are suppressed when it does not: CLIP will
        # otherwise rank the most "whiteboard-like" photograph in a corpus
        # of photographs as a whiteboard.
        has_text = bool((r["flow_text"] or "").strip())
        for t in terms:
            if t.requires_text and not has_text:
                continue
            j = None
            if jt_scores[i] is not None and t.joytag_alias:
                j = jt_scores[i].get(t.joytag_alias)
            c = None
            if img_vecs[i] is not None and t.id in tvecs:
                c = V.clip_score(t, float(img_vecs[i] @ tvecs[t.id]))
            if j is None and c is None:
                continue
            score_rows.append((iid, t.id, j, c, None))
            fused = V.fuse(j, c, None)
            if fused >= t.threshold:
                state = "accepted"
            elif fused >= review_floor:
                state = "pending"
            else:
                continue
            tag_rows.append((iid, t.id, fused, "machine", state, now))

    def work(conn2: sqlite3.Connection) -> None:
        # Persist the visual vector here: it is computed anyway for tag
        # scoring, and this stage already holds the pool A lease the vision
        # encoder needs (spec 7.2).
        for i, r in enumerate(rows):
            if packed_img[i] is not None:
                conn2.execute("DELETE FROM vec_items WHERE item_id = ? AND kind = 'visual'",
                              (r["id"],))
                conn2.execute(
                    "INSERT INTO vec_items(item_id, kind, embedding) VALUES (?,'visual',?)",
                    (r["id"], packed_img[i]))
        for row in score_rows:
            conn2.execute(
                "INSERT INTO tag_scores(item_id, term_id, joytag, clip, vlm) VALUES (?,?,?,?,?) "
                "ON CONFLICT(item_id, term_id) DO UPDATE SET "
                "joytag=excluded.joytag, clip=excluded.clip", row)
        for row in tag_rows:
            # Human tags are terminal: never overwritten by a re-run (spec 7.6).
            conn2.execute(
                "INSERT INTO item_tags(item_id, term_id, confidence, source, state, created_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(item_id, term_id) DO UPDATE SET "
                "confidence=excluded.confidence, state=excluded.state "
                "WHERE item_tags.source = 'machine'", row)
    writer.submit(work)

    acc = sum(1 for r in tag_rows if r[4] == "accepted")
    log.info("tagged %d image(s): %d accepted, %d pending (of %d term scores)",
             len(rows), acc, len(tag_rows) - acc, len(score_rows))
