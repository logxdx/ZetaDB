#!/usr/bin/env python
"""Generate a calibration corpus for INT8 quantisation.

Static quantisation is only as good as its calibration data: the ranges it
records become the fixed scales at inference. Four images is not enough, so
this synthesises a spread across the content types this archive is for -
dark and light UI, dense tables, documents, prose pages, charts, and
photographic texture - at a range of sizes and contrasts.

Real images are always preferred; these exist so the first quantisation
before the library has grown is not calibrated on a handful of samples.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import load  # noqa: E402

WORDS = ("model params det rec latency tokens batch score epoch vocab index "
         "query embed cosine recall invoice total subtotal tax due date portal "
         "contact project atlas threshold calibrate dashboard render layout").split()


def _ui(d, w, h, dark, rng):
    bg = (18, 18, 20) if dark else (250, 249, 246)
    fg = (236, 236, 238) if dark else (26, 26, 28)
    mut = (138, 138, 147) if dark else (110, 110, 118)
    d.rectangle([0, 0, w, h], fill=bg)
    d.text((28, 22), " ".join(rng.sample(WORDS, 3)).title(), fill=fg)
    d.line([(0, 56), (w, 56)], fill=mut)
    cols = rng.randint(3, 5)
    for r in range(rng.randint(3, 8)):
        y = 78 + r * 30
        if y > h - 30:
            break
        for c in range(cols):
            x = 28 + c * ((w - 56) // cols)
            txt = rng.choice(WORDS) if c == 0 else f"{rng.uniform(0.5, 99):.1f}"
            d.text((x, y), txt, fill=fg if c == 0 else mut)
    return bg


def _doc(d, w, h, rng):
    d.rectangle([0, 0, w, h], fill=(252, 251, 248))
    d.text((36, 26), " ".join(rng.sample(WORDS, 2)).upper(), fill=(20, 20, 22))
    for i in range(rng.randint(4, 9)):
        y = 80 + i * 34
        if y > h - 34:
            break
        d.text((36, y), rng.choice(WORDS), fill=(110, 110, 118))
        d.text((240, y), rng.choice([f"{rng.randint(1,9)},{rng.randint(100,999)}.{rng.randint(10,99)}",
                                     f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                                     f"user{rng.randint(1,99)}@example.com"]), fill=(20, 20, 22))


def _prose(d, w, h, rng):
    d.rectangle([0, 0, w, h], fill=(255, 255, 255))
    x, y = 30, 30
    for _ in range(rng.randint(40, 90)):
        word = rng.choice(WORDS)
        if x + len(word) * 7 > w - 30:
            x, y = 30, y + 22
        if y > h - 24:
            break
        d.text((x, y), word, fill=(30, 30, 34))
        x += len(word) * 7 + 7


def _chart(d, w, h, rng):
    d.rectangle([0, 0, w, h], fill=(11, 11, 12))
    n = rng.randint(4, 9)
    bw = (w - 80) // n
    for i in range(n):
        bh = rng.randint(30, h - 110)
        d.rectangle([40 + i * bw, h - 50 - bh, 40 + i * bw + bw - 10, h - 50],
                    fill=(255, 90, 31) if i % 3 == 0 else (60, 60, 66))
        d.text((40 + i * bw, h - 44), rng.choice(WORDS)[:5], fill=(138, 138, 147))
    d.text((36, 22), " ".join(rng.sample(WORDS, 2)).title(), fill=(236, 236, 238))


def _photo(im, rng):
    px = im.load()
    base = (rng.randint(20, 120), rng.randint(40, 160), rng.randint(20, 140))
    for x in range(0, im.size[0], 2):
        for y in range(0, im.size[1], 2):
            v = tuple(max(0, min(255, c + rng.randint(-45, 45))) for c in base)
            for dx in (0, 1):
                for dy in (0, 1):
                    if x + dx < im.size[0] and y + dy < im.size[1]:
                        px[x + dx, y + dy] = v


def main() -> int:
    out = load().path("paths.data") / "calib"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    sizes = [(900, 520), (1280, 720), (820, 460), (640, 480), (1100, 680), (760, 1000)]
    made = 0
    for i in range(30):
        w, h = rng.choice(sizes)
        im = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(im)
        kind = i % 6
        if kind == 0:
            _ui(d, w, h, True, rng)
        elif kind == 1:
            _ui(d, w, h, False, rng)
        elif kind == 2:
            _doc(d, w, h, rng)
        elif kind == 3:
            _prose(d, w, h, rng)
        elif kind == 4:
            _chart(d, w, h, rng)
        else:
            _photo(im, rng)
        im.save(out / f"calib_{i:02d}.png")
        made += 1
    print(f"wrote {made} calibration images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
