"""Stage 1 - OCR. GPU pool A.

Runs the whole pending batch under one lease; the scheduler holds the lease
for the batch, not per item (spec 4.2).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models.ppocr import engine

log = logging.getLogger(__name__)


def run(item_ids: list[int]) -> None:
    ocr = engine()
    if not ocr.available:
        log.info("PP-OCR weights absent; skipping OCR for %d item(s)", len(item_ids))
        return
    ocr.load_models()
    root = load().path("paths.data")
    conn = reader()
    results: dict[int, list[dict]] = {}

    for iid in item_ids:
        row = conn.execute("SELECT kind, blob_path FROM items WHERE id = ?", (iid,)).fetchone()
        if row is None or row["kind"] != "image" or not row["blob_path"]:
            continue
        path = root / row["blob_path"]
        if not path.exists():
            continue
        try:
            boxes = ocr.read(path)
        except Exception:
            log.exception("OCR failed for item %d", iid)
            continue
        results[iid] = [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
                         "text": b.text, "score": round(b.score, 4)} for b in boxes]
        log.info("item %d: %d text box(es)", iid, len(boxes))

    if not results:
        return

    def work(c: sqlite3.Connection) -> None:
        now = int(time.time())
        for iid, boxes in results.items():
            c.execute("UPDATE items SET boxes_json = ?, updated_at = ? WHERE id = ?",
                      (json.dumps(boxes, ensure_ascii=False), now, iid))
    writer.submit(work)
