#!/usr/bin/env python3
"""Does EmbeddingGemma 2 actually work here, and is it better than what we have?

Nothing in the vec_* migration is safe to write until this prints sane
numbers, because every failure mode in this swap is silent:

  * fp16 returns NaN or degraded vectors instead of erroring (Google's card)
  * a wrong <|image|> splice still produces a plausible-looking vector
  * every [M0] threshold in config.toml was measured against jina/Qwen
    cosines and means nothing in a 768-d Gemma space

So this measures rather than asserts. Run it before touching schema.sql.

    uv run scripts/probe.py --download eg2
    uv run scripts/eg2_check.py
    uv run scripts/eg2_check.py --images 12 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run before onnxruntime is imported anywhere: ORT's CUDA provider
# dlopen()s the unversioned "libcudnn.so", which no wheel ships under that
# name. Without this the vision encoder dies at the first attention node.
from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

from server.config import load                      # noqa: E402
from server.db import conn as dbconn                # noqa: E402
from server.models import eg2, vram                 # noqa: E402

# Pairs chosen to separate "retrieves the right thing" from "retrieves
# anything": a usable model needs a MARGIN, not a high absolute score.
# jina's absolute cosines sit at 0.17-0.37 and Qwen's at 0.55-0.71, which
# is exactly why spec 7.3 calibration exists.
TEXT_PROBES = [
    ("how do I rotate the API signing key",
     "Key rotation: generate a new signing key, deploy it alongside the old one, "
     "then revoke the previous key after 24 hours.",
     "The office coffee machine is on the third floor next to the lifts."),
    ("what did we decide about the pricing tiers",
     "Decision: three tiers at 9, 29 and 99 a month, with annual billing "
     "discounted 20 percent.",
     "Remember to water the plants before the long weekend."),
    ("dark dashboard screenshot with a table",
     "A dark-themed analytics dashboard showing a sortable table of request "
     "latencies by endpoint.",
     "A photograph of a wooded hillside at sunrise, mist in the valley."),
]


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)          # pack() L2-normalises, so dot == cosine


def check_text(e, report: dict) -> None:
    print("\n── text retrieval ─────────────────────────────────────────")
    print(f"{'query':<42} {'right':>8} {'wrong':>8} {'margin':>8}")
    rows = []
    for q, right, wrong in TEXT_PROBES:
        qv = unpack(e.embed_text(q))
        dv = [unpack(b) for b in e.embed_texts(
            [eg2.format_document(right), eg2.format_document(wrong)])]
        if not np.isfinite(qv).all():
            print("  !! non-finite query vector - this is the fp16 failure mode")
        cr, cw = cos(qv, dv[0]), cos(qv, dv[1])
        rows.append({"query": q, "right": cr, "wrong": cw, "margin": cr - cw})
        flag = "" if cr > cw else "   <-- WRONG ORDER"
        print(f"{q[:42]:<42} {cr:8.3f} {cw:8.3f} {cr - cw:8.3f}{flag}")
    report["text"] = rows
    print(f"\n  mean margin {np.mean([r['margin'] for r in rows]):.3f}  "
          f"(a margin near zero means no usable threshold exists)")


def check_images(e, limit: int, report: dict) -> None:
    """Cross-modal, which is the whole reason for the swap.

    Before schema v3 this comparison could not be made at all: an image
    lived in vec_jina and a note in vec_text, and spec 9.4 existed precisely
    to paper over that. Now both sides are the same space.
    """
    cfg = load()
    # blob_path is stored relative to paths.data ("blobs/ab/<hash>.png"),
    # not to paths.blobs - every ingest stage resolves it as root / blob_path.
    root = cfg.path("paths.data")
    with dbconn.connect() as c:
        rows = c.execute(
            "SELECT id, title, caption, flow_text, blob_path FROM items "
            "WHERE kind = 'image' AND blob_path IS NOT NULL AND caption IS NOT NULL "
            "ORDER BY id LIMIT ?", (limit,)).fetchall()
    if not rows:
        print("\n── cross-modal ────────────────────────────────────────────")
        print("  no captioned images in the db; ingest some first")
        return

    print(f"\n── cross-modal ({len(rows)} images) ───────────────────────────")
    with vram.measure() as m:
        bare, joint = [], []
        for r in rows:
            p = root / r["blob_path"]
            bare.append(e.embed_image(p))
            text = " ".join(x for x in (r["caption"], r["flow_text"] or "") if x).strip()
            joint.append(e.embed_image(p, text=text, title=r["title"]))
    print(f"  vision encoder VRAM: {m.delta_mb} MB peak, {m.leaked_mb} MB resident after")

    # Each image's own caption as a query should rank its own image first.
    # This is the single most informative number in the script: it is the
    # retrieval path the UI actually exercises.
    caps = [unpack(b) for b in e.embed_texts(
        [r["caption"] for r in rows], is_query=True)]
    out = {"n": len(rows), "vram_mb": m.delta_mb, "modes": {}}
    for label, vecs in (("image only", bare), ("image + caption + ocr", joint)):
        ok = [v is not None for v in vecs]
        if not all(ok):
            print(f"  {label}: {ok.count(False)} image(s) failed to embed")
        M = np.stack([unpack(v) if v is not None else np.zeros(768, "f4") for v in vecs])
        sims = np.stack(caps) @ M.T                 # [query, image]
        rank1 = float(np.mean(np.argmax(sims, axis=1) == np.arange(len(rows))))
        self_c = float(np.mean(np.diag(sims)))
        off = float((sims.sum() - np.trace(sims)) / max(1, sims.size - len(rows)))
        out["modes"][label] = {"recall@1": rank1, "self_cos": self_c, "other_cos": off}
        print(f"  {label:<24} recall@1 {rank1:5.1%}   "
              f"self {self_c:.3f}  others {off:.3f}  margin {self_c - off:.3f}")
    report["cross_modal"] = out


def check_baseline(report: dict) -> None:
    """The same text probes through Qwen3-Embedding-0.6B, so the swap is a
    comparison and not a leap. MTEB multilingual has Qwen at 64.33 against
    EG2's 61.36, so a small text regression is the expected result - what
    matters is whether it shows up on THIS corpus."""
    from server.models.qwen_embed import engine as qwen_engine
    q = qwen_engine()
    if not q.available:
        print("\n  (qwen_embed weights absent - no baseline)")
        return
    print("\n── baseline: Qwen3-Embedding-0.6B ─────────────────────────")
    rows = []
    for query, right, wrong in TEXT_PROBES:
        qv = unpack(q.embed(query))
        dv = [unpack(b) for b in q.embed_many([right, wrong])]
        cr, cw = cos(qv, dv[0]), cos(qv, dv[1])
        rows.append({"query": query, "right": cr, "wrong": cw, "margin": cr - cw})
        print(f"{query[:42]:<42} {cr:8.3f} {cw:8.3f} {cr - cw:8.3f}")
    report["baseline_qwen"] = rows
    print(f"\n  mean margin {np.mean([r['margin'] for r in rows]):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=8, help="how many images to test")
    ap.add_argument("--soft-tokens", type=int, choices=(70, 140, 280, 560, 1120),
                    help="override models.eg2.image_soft_tokens; attention is "
                         "quadratic in patch count, so this is the VRAM knob")
    ap.add_argument("--no-baseline", action="store_true", help="skip the Qwen comparison")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    e = eg2.engine()
    if args.soft_tokens:
        e.soft_tokens = args.soft_tokens
    if not e.available:
        print(f"eg2 weights missing in {e.dir}\n"
              f"  uv run scripts/probe.py --download eg2", file=sys.stderr)
        return 2

    report: dict = {"model": e.describe(), "vision": e.vision_available}
    print(f"eg2: {e.describe()}  vision={'yes' if e.vision_available else 'NO'}")
    check_text(e, report)
    check_images(e, args.images, report)
    if not args.no_baseline:
        check_baseline(report)

    if args.json:
        print("\n" + json.dumps(report, indent=2))
    print("\nNext: these cosines are the inputs to every [M0] threshold in "
          "config.toml.\nNone of them carry over from jina/Qwen - see spec 7.3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
