"""Stage 2 - layout. CPU.

Turns OCR boxes into reading order via grid projection, producing the two
renditions the rest of the pipeline needs (spec 6.2).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from server.db.conn import reader
from server.db.writer import writer
from server.layout.grid import project
from server.layout.types import TextBox

log = logging.getLogger(__name__)


def run(item_ids: list[int]) -> None:
    conn = reader()
    out: dict[int, tuple[str, str]] = {}
    for iid in item_ids:
        row = conn.execute(
            "SELECT boxes_json, width FROM items WHERE id = ?", (iid,)).fetchone()
        if row is None or not row["boxes_json"]:
            continue
        raw = json.loads(row["boxes_json"])
        if not raw:
            continue
        boxes = [TextBox(b["x0"], b["y0"], b["x1"], b["y1"], b["text"], b.get("score", 1.0))
                 for b in raw]
        try:
            res = project(boxes, page_width=row["width"])
        except Exception:
            log.exception("layout failed for item %d", iid)
            continue
        out[iid] = (res.layout_text, res.flow_text)
        log.info("item %d: %d line(s), %d anchor(s), flowing=%s",
                 iid, res.n_lines, res.n_anchors, res.flowing)

    if not out:
        return

    def work(c: sqlite3.Connection) -> None:
        now = int(time.time())
        for iid, (layout_text, flow_text) in out.items():
            c.execute("UPDATE items SET layout_text = ?, flow_text = ?, updated_at = ? "
                      "WHERE id = ?", (layout_text, flow_text, now, iid))
    writer.submit(work)
