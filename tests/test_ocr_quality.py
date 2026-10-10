"""Regression tests for the PP-OCR under-detection check (spec 17.10).

Each case encodes something the first cut of this got wrong:
  - a photograph's texture bands up exactly like text, and an ungated band
    count fired on 5 of the 29 real items, every one a photo with no text
  - a wide table must not look better covered than a paragraph merely for
    having more columns, so boxes on one visual line collapse to one row
  - "PP-OCR found nothing" is a different question from "PP-OCR under-read
    a page", and only the second one is worth a VLM re-read
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from server.ingest.ocr_quality import detected_rows, profile, underdetected


def _page(tmp_path, lines=8, w=900, h=620, fg=20, bg=250):
    """A page-like image: light, mostly empty, with evenly spaced text rows.

    Drawn as glyph-sized marks with gaps, not solid bars. A solid bar is
    ~0.14 ink coverage, and even 5px marks reach 0.069, where the
    real documents in this corpus measure 0.003-0.036, which is dense enough to
    read as a photograph - the first version of this fixture failed the
    page-likeness gate for exactly that reason.
    """
    img = Image.new("L", (w, h), bg)
    d = ImageDraw.Draw(img)
    for i in range(lines):
        y = 40 + i * 38
        for x in range(40, w - 120, 16):
            d.rectangle([x, y, x + 3, y + 12], fill=fg)
    p = tmp_path / f"page_{lines}.png"
    img.save(p)
    return p


def _photo(tmp_path, w=900, h=620):
    """Dense random texture: not page-like under any threshold."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    p = tmp_path / "photo.png"
    Image.fromarray(a, mode="L").save(p)
    return p


def box(y0, y1, x0=40.0, x1=600.0):
    return {"x0": x0, "y0": float(y0), "x1": x1, "y1": float(y1)}


def test_boxes_on_one_line_collapse_to_one_row():
    # A four-column table row is ONE row, not four. Counting them
    # separately would make a wide table look fully covered.
    cells = [box(100, 120, 40, 120), box(100, 120, 200, 280),
             box(100, 120, 360, 440), box(100, 120, 520, 600)]
    assert detected_rows(cells) == 1
    assert detected_rows(cells + [box(160, 180)]) == 2


def test_empty_boxes_are_zero_rows():
    assert detected_rows([]) == 0


def test_page_is_page_like_and_photo_is_not(tmp_path):
    assert profile(_page(tmp_path)).page_like is True
    assert profile(_photo(tmp_path)).page_like is False


def test_fires_when_most_lines_were_missed(tmp_path):
    # The measured failure: 8 lines on the page, 3 rows detected.
    p = _page(tmp_path, lines=8)
    found = [box(40 + i * 38, 53 + i * 38) for i in range(3)]
    fire, why = underdetected(p, found)
    assert fire is True
    assert why["ocr_rows"] == 3
    assert why["est_lines"] >= 7


def test_quiet_when_detection_covered_the_page(tmp_path):
    p = _page(tmp_path, lines=8)
    found = [box(40 + i * 38, 53 + i * 38) for i in range(8)]
    fire, _ = underdetected(p, found)
    assert fire is False


def test_photograph_never_fires_however_few_boxes(tmp_path):
    # The false-positive that gated this behind page-likeness: a photo with
    # one incidental sign, or none at all, must never queue a re-read.
    p = _photo(tmp_path)
    for found in ([], [box(100, 130)]):
        fire, why = underdetected(p, found)
        assert fire is False, why


def test_no_boxes_at_all_is_not_an_under_read(tmp_path):
    # Zero boxes means "no text here" far more often than "missed it all",
    # and a re-read of every photograph would cost a VLM pass per item.
    p = _page(tmp_path, lines=8)
    fire, _ = underdetected(p, [])
    assert fire is False


def test_short_pages_are_left_alone(tmp_path):
    # Below a handful of lines the band count carries no information.
    p = _page(tmp_path, lines=3)
    fire, _ = underdetected(p, [box(40, 53)], min_lines=4)
    assert fire is False


def test_unreadable_file_is_not_an_under_read(tmp_path):
    bad = tmp_path / "truncated.png"
    bad.write_bytes(b"not an image")
    fire, _ = underdetected(bad, [box(10, 30)])
    assert fire is False
