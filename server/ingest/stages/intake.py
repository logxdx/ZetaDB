"""Stage 0 - intake. CPU only.

Hash, dedupe, sniff, thumbnail, perceptual hash. The item row is written
here and becomes visible in the UI immediately with status='processing';
everything else streams in later (spec 4.2).
"""
from __future__ import annotations

import logging
import mimetypes
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from server.config import load
from server.ingest.hashing import content_hash, dhash, text_hash
from server.db.writer import writer

log = logging.getLogger(__name__)

THUMB_LONG_EDGE = 400
IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".rst", ".org"}
MAX_TEXT_BYTES = 2_000_000


@dataclass
class IntakeResult:
    item_id: int
    created: bool
    duplicate_of: int | None = None
    near_dup_of: int | None = None


def _thumb(src: Path, dest: Path) -> tuple[int, int]:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        im.thumbnail((THUMB_LONG_EDGE, THUMB_LONG_EDGE), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGB").save(dest, "WEBP", quality=82, method=4)
    return w, h


def _near_dup(conn: sqlite3.Connection, digest: bytes, max_ham: int) -> int | None:
    row = conn.execute(
        "SELECT item_id, distance FROM vec_phash "
        "WHERE hash MATCH vec_bit(?) AND k = 1 ORDER BY distance",
        (digest,),
    ).fetchone()
    if row and row["distance"] <= max_ham:
        return int(row["item_id"])
    return None


def ingest_image(path: Path, source: str, *, title: str | None = None) -> IntakeResult:
    cfg = load()
    digest = content_hash(path)

    def work(conn: sqlite3.Connection) -> IntakeResult:
        existing = conn.execute(
            "SELECT id FROM items WHERE content_hash = ?", (digest,)
        ).fetchone()
        if existing:
            # Exact duplicate: no new item, just record where we saw it again.
            conn.execute(
                "UPDATE items SET source_path = COALESCE(source_path || char(10) || ?, ?), "
                "updated_at = ? WHERE id = ?",
                (str(path), str(path), int(time.time()), existing["id"]),
            )
            return IntakeResult(int(existing["id"]), created=False, duplicate_of=int(existing["id"]))

        blob_dir = cfg.path("paths.blobs") / digest[:2]
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob = blob_dir / f"{digest}{path.suffix.lower()}"
        if not blob.exists():
            shutil.copy2(path, blob)

        thumb = cfg.path("paths.thumbs") / digest[:2] / f"{digest}.webp"
        w, h = _thumb(blob, thumb)
        with Image.open(blob) as im:
            digest_p = dhash(im)

        near = _near_dup(conn, digest_p, cfg.get("dedupe.phash_hamming_max", 6))
        now = int(time.time())
        root = cfg.path("paths.data")
        cur = conn.execute(
            "INSERT INTO items(kind,status,title,blob_path,thumb_path,mime,width,height,"
            "bytes,content_hash,dhash,near_dup_of,source,source_path,created_at,updated_at) "
            "VALUES ('image','processing',?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (title or path.stem,
             str(blob.relative_to(root)), str(thumb.relative_to(root)),
             mimetypes.guess_type(path.name)[0] or "application/octet-stream",
             w, h, blob.stat().st_size, digest, digest_p, near, source, str(path), now, now),
        )
        item_id = int(cur.lastrowid)
        conn.execute("INSERT INTO vec_phash(item_id, hash) VALUES (?, vec_bit(?))",
                     (item_id, digest_p))
        conn.execute("INSERT INTO ingest_queue(item_id, stage, enqueued_at) VALUES (?,'ocr',?)",
                     (item_id, now))
        return IntakeResult(item_id, created=True, near_dup_of=near)

    return writer.submit(work)


def ingest_note(body: str, *, title: str | None = None, source: str = "compose") -> IntakeResult:
    digest = text_hash(body)

    def work(conn: sqlite3.Connection) -> IntakeResult:
        existing = conn.execute(
            "SELECT id FROM items WHERE content_hash = ?", (digest,)
        ).fetchone()
        if existing:
            return IntakeResult(int(existing["id"]), created=False, duplicate_of=int(existing["id"]))
        now = int(time.time())
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "untitled")
        cur = conn.execute(
            "INSERT INTO items(kind,status,title,body,content_hash,source,created_at,updated_at) "
            "VALUES ('note','processing',?,?,?,?,?,?)",
            (title or first[:120], body, digest, source, now, now),
        )
        item_id = int(cur.lastrowid)
        # Notes skip ocr/layout/tag_visual/caption; they go straight to extraction.
        conn.execute("INSERT INTO ingest_queue(item_id, stage, enqueued_at) VALUES (?,'extract',?)",
                     (item_id, now))
        return IntakeResult(item_id, created=True)

    return writer.submit(work)


def is_image(path: Path) -> bool:
    mime = mimetypes.guess_type(path.name)[0]
    return mime in IMAGE_MIMES


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def ingest_text_file(path: Path, source: str = "upload") -> IntakeResult:
    """A dropped .md or .txt becomes a note, titled from its filename.

    Decoded leniently: a stray byte in a hand-written note should not cost
    you the note.
    """
    raw = path.read_bytes()[:MAX_TEXT_BYTES]
    body = raw.decode("utf-8", errors="replace").strip()
    if not body:
        raise ValueError(f"{path.name} is empty")
    return ingest_note(body, title=path.stem, source=source)
