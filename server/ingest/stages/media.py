"""Stage 2b - audio and video: windows, segment vectors, keyframes. GPU pool A.

Every audio or video item is cut into windows (server.ingest.media) and each
window becomes one row in `segments` and one vector in `vec_segments`:

  video   10 s windows of 5 frames -> EG2 vision encoder, video layout
          + the soundtrack as 30 s audio windows -> EG2 audio encoder
  audio   30 s windows every 25 s -> EG2 audio encoder

Segment vectors carry no title and no task prefix: like an image's visual
row, they describe the content and nothing else, so a moment is found by
what happens in it. The title still reaches search through FTS and through
the item vector.

The item's own kind='item' vector is the normalised mean of its segment
vectors - "what this file is about overall" - so related-items, the graph's
semantic edges and the text_vec retriever treat media like everything else.
The moment retriever (search.hybrid) is what finds WHERE in the file.

Each frame window's middle frame is saved as a keyframe: it is what the
tagging stage scores (Eye + zero-shot) and what the UI shows for a moment.
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
from server.ingest import media as M
from server.models import eg2
from server.models.embeddings import pack

log = logging.getLogger(__name__)


def _unpack(b: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"<{len(b) // 4}f", b), dtype=np.float32)


def run(item_ids: list[int]) -> None:
    cfg = load()
    conn = reader()
    root = cfg.path("paths.data")
    e = eg2.engine()
    if not e.available:
        log.debug("eg2 weights absent; skipping media stage")
        return

    rows = [r for r in (
        conn.execute("SELECT id, kind, title, blob_path, content_hash FROM items WHERE id = ?",
                     (i,)).fetchone()
        for i in item_ids) if r is not None and r["kind"] in ("audio", "video") and r["blob_path"]]
    if not rows:
        return
    if not M.have_ffmpeg():
        raise RuntimeError("media stage needs ffmpeg and ffprobe on PATH")

    frames_dir = cfg.path("paths.data") / "frames"
    for r in rows:
        t0 = time.monotonic()
        path = root / r["blob_path"]
        segs: list[dict] = []
        try:
            pr = M.probe(path)
            if r["kind"] == "video" and e.vision_available:
                for k, w in enumerate(M.video_windows(path, pr)):
                    v = e.embed_frames(w.frames)
                    if v is None:
                        continue
                    key = frames_dir / r["content_hash"][:2] / f"{r['content_hash']}_{k:04d}.webp"
                    key.parent.mkdir(parents=True, exist_ok=True)
                    w.frames[len(w.frames) // 2].save(key, "WEBP", quality=80, method=4)
                    segs.append({"kind": "frames", "start": w.start, "end": w.end,
                                 "frame": str(key.relative_to(root)), "vec": v})
            if pr.has_audio and e.audio_available:
                for w in M.audio_windows(path, pr.duration,
                                         keep_if_silent=(r["kind"] == "audio")):
                    v = e.embed_audio(w.wave)
                    if v is not None:
                        segs.append({"kind": "sound", "start": w.start, "end": w.end,
                                     "frame": None, "vec": v})
        except Exception:
            log.exception("media segmentation failed for item %d", r["id"])
            raise

        item_vec = None
        if segs:
            mean = np.mean(np.stack([_unpack(s["vec"]) for s in segs]), axis=0)
            item_vec = pack(mean)

        def work(c: sqlite3.Connection, iid=r["id"], segs=segs, item_vec=item_vec) -> None:
            old = [x["id"] for x in c.execute("SELECT id FROM segments WHERE item_id = ?", (iid,))]
            for sid in old:
                c.execute("DELETE FROM vec_segments WHERE segment_id = ?", (sid,))
            c.execute("DELETE FROM segments WHERE item_id = ?", (iid,))
            for s in segs:
                cur = c.execute(
                    "INSERT INTO segments(item_id, kind, start_s, end_s, frame_path) "
                    "VALUES (?,?,?,?,?)", (iid, s["kind"], s["start"], s["end"], s["frame"]))
                c.execute("INSERT INTO vec_segments(segment_id, item_id, kind, embedding) "
                          "VALUES (?,?,?,?)", (int(cur.lastrowid), iid, s["kind"], s["vec"]))
            if item_vec is not None:
                c.execute("DELETE FROM vec_items WHERE item_id = ? AND kind = 'item'", (iid,))
                c.execute("INSERT INTO vec_items(item_id, kind, embedding) VALUES (?,'item',?)",
                          (iid, item_vec))
        writer.submit(work)
        log.info("media item %d: %d segment(s) (%d frames, %d sound) in %.1fs",
                 r["id"], len(segs), sum(s["kind"] == "frames" for s in segs),
                 sum(s["kind"] == "sound" for s in segs), time.monotonic() - t0)
