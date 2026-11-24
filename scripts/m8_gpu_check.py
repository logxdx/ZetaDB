"""[M8] GPU validation - the measurements spec 20.6 still owes.

    uv run scripts/m8_gpu_check.py --before data/pre_m8.db [--media ~/clips]

1. Engine on the GPU: load Eye, the EG2 vision + audio encoders and PP-OCR
   through server.engine one pool at a time; record provider, load/build
   time, per-item latency and VRAM delta (nvidia-smi, as the arbiter does).
2. Eye vs JoyTag: score every library image with Eye, fuse exactly as
   tag_visual does, and diff the accepted machine tags against the JoyTag-era
   snapshot (`--before`, a copy of the database taken BEFORE the v4 migration).
   Prints a per-image table for a human to judge - tag quality is not a number
   this script can decide.
3. Media (optional, `--media DIR` of real recordings and clips): windows/s
   through the media path, and the cosine band of a few queries against the
   segments, which is what search.min_cos_moment has to sit inside.

Writes data/m8_gpu_report.json. Read-only against the library: nothing is
written to the database.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

import numpy as np  # noqa: E402

from server.config import load  # noqa: E402
from server.engine.core import engine as model_engine  # noqa: E402
from server.models import vram  # noqa: E402


def _unpack(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<f4")


def measure_models(images: list[Path], report: dict) -> None:
    from server.models.eg2 import engine as eg2_engine
    from server.models.eye import engine as eye_engine, prepare
    from server.models.ppocr import engine as ppocr_engine

    out = report.setdefault("models", {})
    with model_engine.use("A", "m8-check"):
        # Eye: the first session build includes the TensorRT engine build.
        eye = eye_engine()
        with vram.measure() as m:
            t = time.monotonic()
            eye.load_model()
            load_s = time.monotonic() - t
            lat = []
            for p in images:
                x = prepare(p)
                t = time.monotonic()
                eye.score_image(x)
                lat.append(time.monotonic() - t)
                m.sample()
        out["eye"] = {"provider": eye.provider, "load_or_build_s": round(load_s, 1),
                      "ms_per_image_p50": round(1000 * float(np.median(lat[1:] or lat)), 1),
                      "vram_delta_mb": m.delta_mb}

        e = eg2_engine()
        with vram.measure() as m:
            t = time.monotonic()
            for p in images:
                e.embed_image(p)
                m.sample()
            img_s = (time.monotonic() - t) / max(1, len(images))
            sr = 16_000
            wave = (0.3 * np.sin(2 * np.pi * 440 * np.arange(30 * sr) / sr)).astype(np.float32)
            t = time.monotonic()
            e.embed_audio(wave)
            aud_s = time.monotonic() - t
            m.sample()
        out["eg2_encoders"] = {"provider": e._encoder_providers(),
                               "s_per_image": round(img_s, 3),
                               "s_per_30s_audio_window": round(aud_s, 3),
                               "vram_delta_mb": m.delta_mb}

        oc = ppocr_engine()
        with vram.measure() as m:
            t = time.monotonic()
            oc.load_models()
            load_s = time.monotonic() - t
            t = time.monotonic()
            for p in images[:10]:
                oc.read(p)
                m.sample()
            per = (time.monotonic() - t) / max(1, min(10, len(images)))
        out["ppocr"] = {"provider": f"{oc.provider} / {oc.rec_provider}",
                        "load_s": round(load_s, 1), "s_per_image": round(per, 3),
                        "vram_delta_mb": m.delta_mb}
    model_engine.unload_all()


def eye_vs_joytag(conn: sqlite3.Connection, before: Path | None, report: dict) -> None:
    from server.models.eg2 import engine as eg2_engine
    from server.models.eye import engine as eye_engine
    from server.search import vocab as V

    cfg = load()
    root = cfg.path("paths.data")
    terms = V.active_terms(conn)
    rows = conn.execute("SELECT id, title, blob_path, flow_text FROM items "
                        "WHERE kind='image' AND blob_path IS NOT NULL ORDER BY id").fetchall()
    old: dict[int, set[str]] = {}
    if before and before.exists():
        b = sqlite3.connect(f"file:{before}?mode=ro", uri=True)
        for iid, term in b.execute(
                "SELECT t.item_id, v.term FROM item_tags t JOIN vocab_terms v ON v.id=t.term_id "
                "WHERE t.state='accepted' AND t.source='machine'"):
            old.setdefault(iid, set()).add(term)
    eye, e = eye_engine(), eg2_engine()
    with model_engine.use("A", "m8-check"):
        tvec = {t.id: _unpack(v) for t, v in zip(
            terms, e.embed_texts([t.prompt() for t in terms], is_query=True))}
        table = []
        for r in rows:
            path = root / r["blob_path"]
            scores = eye.score_batch([path])[0] or {}
            iv = e.embed_image(path)
            has_text = bool((r["flow_text"] or "").strip())
            new = set()
            for t in terms:
                if t.requires_text and not has_text:
                    continue
                ey = scores.get(t.eye_alias) if t.eye_alias else None
                c = V.clip_score(t, float(_unpack(iv) @ tvec[t.id])) if iv else None
                if (ey is not None or c is not None) and V.fuse(ey, c, None) >= t.threshold:
                    new.add(t.term)
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
            table.append({"id": r["id"], "title": r["title"],
                          "joytag_era": sorted(old.get(r["id"], set())),
                          "eye_era": sorted(new),
                          "eye_top5": [(k, round(v, 2)) for k, v in top]})
    model_engine.unload_all()
    report["eye_vs_joytag"] = table
    print("\n== Eye vs JoyTag: accepted machine tags per image")
    for row in table:
        gained = sorted(set(row["eye_era"]) - set(row["joytag_era"]))
        lost = sorted(set(row["joytag_era"]) - set(row["eye_era"]))
        print(f"#{row['id']:<3} {str(row['title'])[:30]:<30} +{gained} -{lost}")
        print(f"     eye top5: {row['eye_top5']}")


def media(dir_: Path, report: dict) -> None:
    from server.ingest import media as M
    from server.models.eg2 import engine as eg2_engine

    e = eg2_engine()
    files = sorted(p for p in dir_.iterdir() if M.is_audio(p) or M.is_video(p))
    vecs, segs, t0 = [], 0, time.monotonic()
    with model_engine.use("A", "m8-check"):
        for p in files:
            pr = M.probe(p)
            if M.is_video(p):
                for w in M.video_windows(p, pr):
                    vecs.append((p.name, "frames", w.start, _unpack(e.embed_frames(w.frames))))
            if pr.has_audio:
                for w in M.audio_windows(p, pr.duration, keep_if_silent=M.is_audio(p)):
                    vecs.append((p.name, "sound", w.start, _unpack(e.embed_audio(w.wave))))
        segs = len(vecs)
    model_engine.unload_all()
    elapsed = time.monotonic() - t0
    report["media"] = {"files": len(files), "segments": segs,
                       "segments_per_s": round(segs / elapsed, 2) if elapsed else None}
    if vecs:
        mat = np.stack([v for *_, v in vecs])
        cos_all = []
        for q in ("someone talking", "music playing", "a person walking outside", "a screen"):
            s = mat @ _unpack(e.embed_text(q))
            cos_all.append(s)
            best = int(np.argmax(s))
            print(f"  {q!r}: best {s[best]:.3f} at {vecs[best][0]} {vecs[best][1]} "
                  f"{vecs[best][2]:.0f}s; median {np.median(s):.3f}")
        allc = np.concatenate(cos_all)
        report["media"]["query_cosine_percentiles"] = {
            p: round(float(np.percentile(allc, p)), 3) for p in (10, 50, 90, 99)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, help="database copy from BEFORE the v4 migration")
    ap.add_argument("--media", type=Path, help="folder of real recordings and clips")
    ap.add_argument("--skip-models", action="store_true")
    a = ap.parse_args()

    cfg = load()
    db = cfg.path("paths.db")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    root = cfg.path("paths.data")
    images = [root / r[0] for r in conn.execute(
        "SELECT blob_path FROM items WHERE kind='image' AND blob_path IS NOT NULL")]

    report: dict = {"gpu": vram.info().__dict__ if vram.available() else None}
    if not a.skip_models:
        measure_models(images, report)
        print(json.dumps(report["models"], indent=2))
    eye_vs_joytag(conn, a.before, report)
    if a.media:
        media(a.media.expanduser(), report)
    out = root / "m8_gpu_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
