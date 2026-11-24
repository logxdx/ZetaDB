"""Stage 3 - visual tagging. GPU pool A.

Scores every active vocabulary term against each image - and each video, via
its keyframes - from two independent signals and fuses them (spec 7.2-7.4):

  Eye     terms with an Eye alias: Desert Ant Labs' calibrated photo tags
          (16 photo types, 558 scenes), where 0.5 is the recommended cut
  EG2     every term, zero-shot cosine calibrated per term

Eye replaces JoyTag. JoyTag's Danbooru schema had no word for most of what an
idea dump holds - screenshot, whiteboard, receipt, chart - and its sigmoid
meant something different for every tag. Eye's `type` head names exactly
those, and its per-tag calibration curve makes a score of 0.6 mean the same
thing for "screenshot" as for "tree".

This stage owns the kind='visual' row in vec_items: the image with NO text
in it. That is deliberate. Scoring the term "chart" against the kind='item'
vector - which interleaves the image with its own OCR text - would match
any screenshot whose text merely says "chart", which is the exact leak
vocab_terms.requires_text exists to prevent.

Videos are tagged from up to three keyframes written by the media stage;
a term's score is its best over those frames, and the middle keyframe's
image vector becomes the video's kind='visual' row ("looks like").
"""
from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models.eg2 import Eg2, engine as eg2_engine
from server.models.eye import engine as eye_engine
from server.search import vocab as V

log = logging.getLogger(__name__)

MAX_KEYFRAMES = 3

# Machine tags never override a person. A human tag is terminal (spec 7.6),
# and so is a REJECTION: the review queue rejects by setting state='rejected'
# on the machine row, and before [M8] a re-run quietly flipped those back to
# accepted - harmless while re-runs were rare, not once the v4 migration
# re-tags every image with Eye.
TAG_UPSERT = (
    "INSERT INTO item_tags(item_id, term_id, confidence, source, state, created_at) "
    "VALUES (?,?,?,?,?,?) ON CONFLICT(item_id, term_id) DO UPDATE SET "
    "confidence=excluded.confidence, state=excluded.state "
    "WHERE item_tags.source = 'machine' AND item_tags.state != 'rejected'"
)

_term_vecs: dict[int, np.ndarray] = {}
_term_vec_stamp = 0.0


def _unpack(b: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"<{len(b) // 4}f", b), dtype=np.float32)


# Eye's type head names the KIND of image. Its scene tags were learned on
# photographs, and on anything else they misfire: [M8] a dark bar chart got
# "food", "person" and "chair" from Eye at 0.67-0.92 while EG2 put them at
# 0.01-0.03, and the fused average still cleared 0.40. So when Eye's own
# type head says the image is not a photograph, its scene tags carry no
# evidence; its type tags ("screenshot", "whiteboard", ...) still do.
EYE_PHOTO_TYPES = ("photo", "selfie")
EYE_NON_PHOTO_TYPES = ("screenshot", "photo_of_screen", "receipt", "document", "id_card",
                       "whiteboard", "map", "chart_or_slide", "meme_or_graphic",
                       "illustration_or_art")


def retract_unsupported(conn: sqlite3.Connection, item_ids: list[int],
                        tag_rows: list[tuple]) -> int:
    """Drop machine tags a re-score no longer supports.

    The upsert only ever adds, so a tag accepted on stale evidence (an
    uncalibrated term, Eye's photo scenes on a chart) stayed forever. A
    person's decisions (rejected, or source='human') and the captioner's
    proposals (tag_scores.vlm) are not this stage's to drop."""
    kept: dict[int, set[int]] = {}
    for iid, tid, *_ in tag_rows:
        kept.setdefault(iid, set()).add(tid)
    n = 0
    for iid in item_ids:
        for (tid,) in conn.execute(
                "SELECT t.term_id FROM item_tags t LEFT JOIN tag_scores s "
                "ON s.item_id = t.item_id AND s.term_id = t.term_id "
                "WHERE t.item_id = ? AND t.source = 'machine' AND t.state != 'rejected' "
                "AND s.vlm IS NULL", (iid,)).fetchall():
            if tid not in kept.get(iid, set()):
                conn.execute("DELETE FROM item_tags WHERE item_id = ? AND term_id = ?", (iid, tid))
                n += 1
    return n


def eye_scenes_apply(scores: dict[str, float]) -> bool:
    """False when Eye itself says this is not a photograph."""
    photo = max((scores.get(k, 0.0) for k in EYE_PHOTO_TYPES), default=0.0)
    other = max((scores.get(k, 0.0) for k in EYE_NON_PHOTO_TYPES), default=0.0)
    return photo >= other


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


def _frames_for(conn: sqlite3.Connection, item_id: int, root: Path) -> list[Path]:
    """Up to MAX_KEYFRAMES keyframes of a video, evenly spread, middle first."""
    rows = conn.execute(
        "SELECT frame_path FROM segments WHERE item_id = ? AND kind = 'frames' "
        "AND frame_path IS NOT NULL ORDER BY start_s", (item_id,)).fetchall()
    paths = [root / r["frame_path"] for r in rows if (root / r["frame_path"]).exists()]
    if len(paths) <= MAX_KEYFRAMES:
        picks = paths
    else:
        step = (len(paths) - 1) / (MAX_KEYFRAMES - 1)
        picks = [paths[round(i * step)] for i in range(MAX_KEYFRAMES)]
    if picks:
        mid = picks[len(picks) // 2]
        picks = [mid] + [p for p in picks if p != mid]
    return picks


def _uncalibrated(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM vocab_terms WHERE state = 'active' "
                        "AND (clip_mu IS NULL OR clip_sigma IS NULL)").fetchone()[0]


def run(item_ids: list[int]) -> None:
    """Tag a batch. Terms without a CLIP calibration (a new library, or a
    reprompted term) are calibrated here once enough image vectors exist -
    including the ones this batch just wrote - and the batch is then
    re-scored, so the first images are not tagged on the uncalibrated
    fallback while every later one gets the real thresholds (spec 7.3)."""
    before = _uncalibrated(reader())
    _tag(item_ids)
    if before and V.calibrate_clip() and _uncalibrated(reader()) < before:
        # Everything scored so far used the fallback: earlier batches too
        # small to calibrate on (a new library's first few images), and
        # this one. Re-score them all, once.
        earlier = [r[0] for r in reader().execute("SELECT DISTINCT item_id FROM tag_scores")]
        ids = sorted(set(item_ids) | set(earlier))
        log.info("calibrated %d term(s); re-scoring %d item(s) tagged without it",
                 before, len(ids))
        _tag(ids)


def _tag(item_ids: list[int]) -> None:
    cfg = load()
    conn = reader()
    root = cfg.path("paths.data")

    rows = [r for r in (
        conn.execute("SELECT id, kind, blob_path, flow_text FROM items WHERE id = ?",
                     (i,)).fetchone()
        for i in item_ids) if r is not None and r["kind"] in ("image", "video")]
    # Each entry: (item row, [image paths to score]); the first path is the
    # one whose vector becomes the item's kind='visual' row.
    work_items: list[tuple[sqlite3.Row, list[Path]]] = []
    for r in rows:
        if r["kind"] == "image" and r["blob_path"]:
            work_items.append((r, [root / r["blob_path"]]))
        elif r["kind"] == "video":
            frames = _frames_for(conn, r["id"], root)
            if frames:
                work_items.append((r, frames))
    if not work_items:
        return

    terms = V.active_terms(conn)
    if not terms:
        log.info("vocabulary is empty; skipping visual tagging")
        return

    # --- Eye: calibrated per-tag confidence; read only our own aliases
    eye = eye_engine()
    eye_scores: list[dict[str, float] | None] = [None] * len(work_items)
    if eye.available:
        try:
            for i, (_, paths) in enumerate(work_items):
                per_frame = [s for s in eye.score_batch(paths) if s is not None]
                if per_frame:
                    eye_scores[i] = {k: max(s.get(k, 0.0) for s in per_frame)
                                     for k in per_frame[0]}
        except Exception:
            log.exception("eye scoring failed")

    # --- EG2 zero-shot: covers every term, including those Eye lacks
    e = eg2_engine()
    best_cos: list[dict[int, float] | None] = [None] * len(work_items)
    packed_img: list[bytes | None] = [None] * len(work_items)
    tvecs: dict[int, np.ndarray] = {}
    if e.vision_available:
        try:
            tvecs = _term_vectors(terms, e)
            ids = [t.id for t in terms if t.id in tvecs]
            tmat = np.stack([tvecs[i] for i in ids]) if ids else None
            for i, (_, paths) in enumerate(work_items):
                vecs = [e.embed_image(p) for p in paths]        # no text: visual only
                if not vecs or vecs[0] is None:
                    continue
                packed_img[i] = vecs[0]
                if tmat is None:
                    continue
                cos = np.max(np.stack([tmat @ _unpack(v) for v in vecs if v is not None]), axis=0)
                best_cos[i] = dict(zip(ids, cos.tolist()))
        except Exception:
            log.exception("eg2 visual tagging failed")

    review_floor = cfg.get("tagging.review_floor", 0.25)
    now = int(time.time())
    tag_rows: list[tuple] = []
    score_rows: list[tuple] = []

    for i, (r, _) in enumerate(work_items):
        iid = r["id"]
        # Does this image contain text at all? Terms that only make sense
        # on text-bearing images are suppressed when it does not: zero-shot
        # scoring will otherwise rank the most "whiteboard-like" photograph
        # in a corpus of photographs as a whiteboard.
        has_text = bool((r["flow_text"] or "").strip())
        scenes_ok = eye_scores[i] is None or eye_scenes_apply(eye_scores[i])
        for t in terms:
            if t.requires_text and not has_text:
                continue
            ey = None
            if eye_scores[i] is not None and t.eye_alias:
                ey = eye_scores[i].get(t.eye_alias)
                if ey is not None and not scenes_ok and eye.head_of(t.eye_alias) == "scenes":
                    ey = None
            c = None
            if best_cos[i] is not None and t.id in best_cos[i]:
                c = V.clip_score(t, best_cos[i][t.id])
            if ey is None and c is None:
                continue
            score_rows.append((iid, t.id, ey, c, None))
            fused = V.fuse(ey, c, None)
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
        for i, (r, _) in enumerate(work_items):
            if packed_img[i] is not None:
                conn2.execute("DELETE FROM vec_items WHERE item_id = ? AND kind = 'visual'",
                              (r["id"],))
                conn2.execute(
                    "INSERT INTO vec_items(item_id, kind, embedding) VALUES (?,'visual',?)",
                    (r["id"], packed_img[i]))
        for row in score_rows:
            conn2.execute(
                "INSERT INTO tag_scores(item_id, term_id, eye, clip, vlm) VALUES (?,?,?,?,?) "
                "ON CONFLICT(item_id, term_id) DO UPDATE SET "
                "eye=excluded.eye, clip=excluded.clip", row)
        for row in tag_rows:
            conn2.execute(TAG_UPSERT, row)
        retract_unsupported(conn2, [r["id"] for r, _ in work_items], tag_rows)
    writer.submit(work)

    acc = sum(1 for r in tag_rows if r[4] == "accepted")
    log.info("tagged %d item(s): %d accepted, %d pending (of %d term scores)",
             len(work_items), acc, len(tag_rows) - acc, len(score_rows))
