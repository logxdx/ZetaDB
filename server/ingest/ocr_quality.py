"""Did PP-OCR actually read this image, or did it miss most of the lines?

spec 17.10 found a reproducible hole: on a flowing prose paragraph PP-OCRv6
detection returns 4-5 boxes for 8 lines and garbage text for some of those,
for CER 0.908, while reading a discrete-field invoice at 11 boxes for 11
lines with every score at 0.98-1.00. It is not box merging and not a font
size effect - dense uniform prose is simply a shape its detector drops.

There is no ground truth at ingest time, so the signal has to come from the
pixels. The cheapest honest one is a row-ink profile: binarise against the
page background, and a page of text shows up as alternating bands of inked
and empty rows. Counting those bands estimates the number of text LINES
without detecting anything, and comparing it with the number of distinct
rows PP-OCR returned says whether the detector saw the page or a third of
it.

The trap is photographs. Their texture bands up exactly like text - a first
cut of this fired on 5 of the 29 real items, every one of them a photo that
genuinely contains no text. So band counting is gated behind a page-likeness
test first. [M7c] measured on this corpus, the two populations do not
overlap:

                  blank rows      ink coverage
  7 documents    0.769 - 0.933    0.003 - 0.036
  6 photographs  0.047 - 0.594    0.064 - 0.411

A page is mostly empty and lightly inked; a photograph is neither. The
thresholds below sit in those gaps.

This deliberately judges coverage only, never recognition quality: a
caption-stage re-read is expensive enough to fire on "half the page is
missing", not on "one word looks odd".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# A band thinner than this is noise - a rule, a border, a stray mark.
_MIN_BAND_PX = 4
# A band taller than this is not a text line: a filled panel, a plot area.
_MAX_BAND_FRAC = 0.22
# Fraction of a row's pixels differing from background before the row counts
# as inked. Low, because a line of text is mostly background.
_ROW_INK_FRAC = 0.004
# Page-likeness, from the measured gaps above.
_MIN_BLANK_ROWS = 0.65
_MAX_INK_FRAC = 0.06


@dataclass(frozen=True)
class PageProfile:
    lines: int          # estimated text lines, from ink bands
    blank_rows: float   # fraction of rows with no ink
    ink: float          # fraction of inked pixels overall

    @property
    def page_like(self) -> bool:
        """Mostly empty and lightly inked - a document, not a photograph."""
        return self.blank_rows >= _MIN_BLANK_ROWS and self.ink <= _MAX_INK_FRAC


def profile(path, max_width: int = 1000) -> PageProfile:
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("L")
            if im.width > max_width:
                im = im.resize((max_width, max(1, int(im.height * max_width / im.width))),
                               Image.Resampling.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
    except Exception:
        log.exception("could not read %s for line estimation", path)
        return PageProfile(0, 0.0, 1.0)
    if a.size == 0:
        return PageProfile(0, 0.0, 1.0)

    # Background is the dominant tone, which handles dark themes as well as
    # light ones without a separate branch.
    bg = float(np.median(a))
    ink = np.abs(a - bg) > 48.0
    rowfrac = ink.mean(axis=1)
    rows = rowfrac > _ROW_INK_FRAC

    h = a.shape[0]
    max_band = max(_MIN_BAND_PX + 1, int(h * _MAX_BAND_FRAC))
    bands, run = 0, 0
    for r in rows:
        if r:
            run += 1
            continue
        if _MIN_BAND_PX <= run <= max_band:
            bands += 1
        run = 0
    if _MIN_BAND_PX <= run <= max_band:
        bands += 1
    return PageProfile(bands, float((rowfrac <= _ROW_INK_FRAC).mean()),
                       float(ink.mean()))


def detected_rows(boxes: list[dict], tol: float = 0.6) -> int:
    """Distinct text rows among PP-OCR's boxes.

    Boxes on one visual line must collapse to one row, or a wide table would
    look better covered than a paragraph purely for having more columns.
    """
    if not boxes:
        return 0
    items = []
    for b in boxes:
        y0, y1 = float(b.get("y0", 0.0)), float(b.get("y1", 0.0))
        items.append(((y0 + y1) / 2.0, max(1.0, y1 - y0)))
    items.sort()
    rows, last_cy, last_h = 1, items[0][0], items[0][1]
    for cy, hgt in items[1:]:
        if abs(cy - last_cy) > tol * max(last_h, hgt):
            rows += 1
        last_cy, last_h = cy, hgt
    return rows


def underdetected(path, boxes: list[dict], *, min_lines: int = 4,
                  ratio: float = 0.6) -> tuple[bool, dict]:
    """True when the pixels show far more text lines than PP-OCR returned.

    Three gates, each earning its place:
      page_like   a photograph's texture bands like text; without this the
                  check fired on 5 of 29 real items, all of them photos.
      rows >= 1   PP-OCR must have found SOME text. "Found nothing" is a
                  different question from "under-read a page", and the
                  measured failure mode leaves boxes behind.
      min_lines   below a handful of lines the band count means nothing.
    """
    p = profile(path)
    rows = detected_rows(boxes)
    why = {"est_lines": p.lines, "ocr_rows": rows, "blank_rows": round(p.blank_rows, 3),
           "ink": round(p.ink, 3), "page_like": p.page_like,
           "ratio": round(rows / p.lines, 3) if p.lines else None}
    if not p.page_like or rows < 1 or p.lines < min_lines:
        return False, why
    return rows < p.lines * ratio, why
