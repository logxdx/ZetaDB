"""Stage 4 - captioning. GPU pool B.

Qwen3.5-4B (Unsloth UD-Q3_K_XL decoder + F16 mmproj) describes each image
and, in the same call, emits the structured CaptionTags object under a GBNF
grammar.

CaptionTags lives here rather than on Needle 3 because M0 measured Needle
producing duplicated, near-useless tags on generative fields (spec 8.1).
llama.cpp's grammar gives the same guaranteed-parse property from a model
that can actually see the image.

Escalation is by EFFORT, not by a second model: a 12B does not fit in 6 GB,
so flagged items are re-run at higher image detail with a stricter prompt
(spec 17.1).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sqlite3
import time
from pathlib import Path

from server.config import load
from server.db.conn import reader
from server.db.writer import writer
from server.models.llama_server import LlamaServer, find_binary

log = logging.getLogger(__name__)

_srv: LlamaServer | None = None

# Guaranteed-parse output. Keeps the model from prose-wrapping the JSON.
# Guaranteed-parse output. Kept to plain GBNF: llama.cpp rejected both
# {m,n} repetition and \xNN escapes inside character classes.
GRAMMAR = r"""
root    ::= "{" ws "\"caption\":" ws string "," ws "\"subject\":" ws string "," ws "\"tags\":" ws taglist "," ws "\"is_chart\":" ws bool "," ws "\"is_screenshot\":" ws bool ws "}"
taglist ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
string  ::= "\"" chars "\""
chars   ::= ([^"\\] | "\\" ["\\bfnrt/])*
bool    ::= "true" | "false"
ws      ::= [ \t\n]*
"""

PROMPT = (
    "Describe this image for a searchable personal archive. "
    "Write one or two factual sentences in `caption` covering what it shows, "
    "any visible text or interface, and its overall look. "
    "Put the single main subject in `subject`. "
    "In `tags`, give 3 to 10 short lowercase keywords. "
    "Set the booleans honestly. Respond with JSON only."
)

ESCALATED_PROMPT = PROMPT + (
    " This image is text-dense: read the interface carefully and name the "
    "specific screens, columns, labels and numbers you can see."
)


def _server() -> LlamaServer | None:
    global _srv
    if _srv is not None:
        return _srv
    cfg = load()
    d = cfg.path("paths.models") / "vlm_default"
    if not d.exists() or not find_binary():
        return None
    model = next((p for p in sorted(d.glob("*.gguf")) if "mmproj" not in p.name.lower()), None)
    mmproj = next((p for p in sorted(d.glob("*.gguf")) if "mmproj" in p.name.lower()), None)
    if model is None:
        return None
    _srv = LlamaServer(model, mmproj=mmproj,
                       ctx=cfg.get("models.vlm_default.ctx", 16384), ngl=99,
                       cache_type=cfg.get("models.vlm_default.cache_type"),
                       flash_attn=cfg.get("models.vlm_default.flash_attn", False))
    return _srv


def unload() -> None:
    global _srv
    if _srv is not None:
        _srv.stop()
        _srv = None


def _encode(path: Path, long_edge: int) -> tuple[str, str]:
    """Downscale before sending to the VLM.

    The originals in a real library are 25-50 MP. Sending one whole tiles
    into a vast number of vision tokens: it made captioning the dominant
    cost of ingest and a 4000x6000 image killed the llama-server
    connection outright ("Server disconnected without sending a
    response"). The model downsamples internally anyway, so nothing is
    gained by shipping full resolution.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if max(im.size) > long_edge:
            im.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _describe(srv: LlamaServer, path: Path, escalate: bool) -> dict | None:
    cfg = load()
    long_edge = cfg.get("vlm.image_long_edge", 1024)
    if escalate:
        long_edge = cfg.get("vlm.escalate_image_long_edge", 1536)
    b64, mime = _encode(path, long_edge)
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": ESCALATED_PROMPT if escalate else PROMPT},
    ]}]
    raw = srv.chat(msg, max_tokens=400, temperature=0.2, grammar=GRAMMAR)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("caption did not parse despite grammar: %.160s", raw)
        return None


def should_escalate(row: sqlite3.Row, cfg) -> bool:
    boxes = json.loads(row["boxes_json"]) if row["boxes_json"] else []
    return (len(boxes) > cfg.get("vlm.escalate_on_box_count", 40)
            or len(row["flow_text"] or "") > cfg.get("vlm.escalate_on_text_len", 600))


def run(item_ids: list[int]) -> None:
    cfg = load()
    srv = _server()
    if srv is None:
        log.info("VLM unavailable; skipping captions for %d item(s)", len(item_ids))
        return
    conn = reader()
    root = cfg.path("paths.data")

    rows = [r for r in (conn.execute(
        "SELECT id, kind, blob_path, boxes_json, flow_text FROM items WHERE id = ?", (i,)
    ).fetchone() for i in item_ids)
        if r is not None and r["kind"] == "image" and r["blob_path"]]
    if not rows:
        return

    srv.start()   # one load for the whole batch (spec 4.2)
    out: dict[int, tuple[str, str]] = {}
    proposals: list[tuple[int, list[str]]] = []
    for r in rows:
        path = root / r["blob_path"]
        if not path.exists():
            continue
        try:
            got = _describe(srv, path, should_escalate(r, cfg))
        except Exception:
            log.exception("caption failed for item %d", r["id"])
            continue
        if not got:
            continue
        # The subject is usually already inside the caption; appending it
        # produced "...evergreen trees. snow-capped mountain".
        body = (got.get("caption") or "").strip()
        subj = (got.get("subject") or "").strip()
        caption = body if (not subj or subj.lower() in body.lower()) else f"{body} {subj}".strip()
        out[r["id"]] = (caption, "qwen3.5-4b")
        tags = [t for t in (got.get("tags") or []) if isinstance(t, str)]
        if tags:
            proposals.append((r["id"], tags))
        log.info("item %d captioned: %.90s", r["id"], caption)

    if out:
        def work(c: sqlite3.Connection) -> None:
            now = int(time.time())
            for iid, (cap, model) in out.items():
                c.execute("UPDATE items SET caption = ?, caption_model = ?, updated_at = ? "
                          "WHERE id = ?", (cap, model, now, iid))
        writer.submit(work)

    # VLM-proposed tags go through the same vocabulary gate as anything else
    # (spec 7.5): near an existing term -> aliased, else queued for review.
    if proposals:
        from server.search import vocab as V
        for iid, tags in proposals:
            for name in tags[:10]:
                tid, action = V.resolve_or_propose(name)
                if tid is None:
                    continue

                def work(c: sqlite3.Connection, i=iid, t=tid) -> None:
                    c.execute(
                        "INSERT INTO tag_scores(item_id, term_id, vlm) VALUES (?,?,1.0) "
                        "ON CONFLICT(item_id, term_id) DO UPDATE SET vlm = 1.0", (i, t))
                    c.execute(
                        "INSERT INTO item_tags(item_id, term_id, confidence, source, state, "
                        "created_at) VALUES (?,?,?,'machine','pending',?) "
                        "ON CONFLICT(item_id, term_id) DO NOTHING",
                        (i, t, 0.6, int(time.time())))
                writer.submit(work)
