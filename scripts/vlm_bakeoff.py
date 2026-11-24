#!/usr/bin/env python3
"""Which VLM should caption this library? Measure it, do not read a leaderboard.

A captioner here is not judged on how pleasant its prose is. Its output is
fed to EmbeddingGemma 2 and becomes part of how an image is FOUND again, so
the question that matters is: does this caption retrieve its own image?

That is measurable without a human, and this script measures it.

  recall@1   Embed the generated caption as a SearchQuery and KNN it against
             the kind='visual' vectors - the image with NO text in it. A
             caption that describes the picture well ranks its own image
             first. The text-free target is deliberate: scoring against
             kind='item' would reward a caption for echoing OCR text that is
             already baked into that vector, which flatters text-dense
             screenshots and tells us nothing about the picture.
  margin     cos(own image) - cos(best other image). Recall@1 saturates at
             100% on a small library (spec 10 found exactly this), so the
             margin is the number that still separates models once they all
             "pass".
  ocr_recall Of the distinctive words PP-OCRv6 actually read off the image,
             how many appear in the caption. This is the screenshot-reading
             axis, and it is the one that motivated the escalation tier.
  json_ok    The caption stage runs under a GBNF grammar, so malformed JSON
             is impossible - but a model can still satisfy the grammar with
             empty strings and an empty tag list. This counts captions where
             every field actually carries content.

Everything runs through the PRODUCTION path: the same prompt, the same
grammar, the same downscale, the same LlamaServer. A bake-off against a
different prompt measures the prompt.

    uv run scripts/vlm_bakeoff.py --list
    uv run scripts/vlm_bakeoff.py --download                  # ~10 GB
    uv run scripts/vlm_bakeoff.py --models gemma3_4b incumbent
    uv run scripts/vlm_bakeoff.py --images 12
    uv run scripts/vlm_bakeoff.py --score-only                # re-score, no GPU

Captions are written to data/bakeoff/<model>.json so scoring can be redone
without paying for generation again.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run before onnxruntime is imported anywhere (see eg2_check.py).
from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

from server.config import load                              # noqa: E402
from server.db.conn import reader                           # noqa: E402
from server.ingest.stages import caption as C               # noqa: E402
from server.models.llama_server import LlamaServer          # noqa: E402
from server.models import vram                              # noqa: E402

BAKEOFF_DIR = "_bakeoff"
OUT_DIR = "bakeoff"


@dataclass
class Candidate:
    key: str
    repo: str
    model_file: str
    mmproj_file: str
    note: str = ""
    # Qwen3.5 is a hybrid reasoner and needs enable_thinking=False or it
    # spends the whole budget on reasoning_content. Every other candidate
    # here has no such switch, and sending the kwarg to a template that
    # does not declare it is a needless way to earn a 400.
    thinking_kwarg: bool = False
    extra: list[str] = field(default_factory=list)
    # Resolved for the incumbent, which lives outside the bake-off dir.
    subdir: str = BAKEOFF_DIR
    # A model already on this machine, outside data/models entirely. Nothing
    # to download and nothing to clean up afterwards.
    abs_dir: str = ""
    # Publisher-recommended sampling, merged into the request. The defaults
    # are llama.cpp's, which are not always what a model card asks for.
    sampling: dict = field(default_factory=dict)


CANDIDATES: list[Candidate] = [
    Candidate("incumbent", "", "Qwen3.5-4B-UD-Q3_K_XL.gguf", "mmproj-F16.gguf",
              note="Qwen3.5-4B, what ships today", thinking_kwarg=True,
              subdir="vlm_default"),
    Candidate("gemma3_4b", "unsloth/gemma-3-4b-it-GGUF",
              "gemma-3-4b-it-UD-Q3_K_XL.gguf", "mmproj-F16.gguf",
              note="Gemma 3 4B IT, same UD quant as the incumbent"),
    Candidate("moondream2", "ggml-org/moondream2-20250414-GGUF",
              "moondream2-text-model-f16_ct-vicuna.gguf",
              "moondream2-mmproj-f16-20250414.gguf",
              note="1.86B, captioning-first; only f16 is published"),
    Candidate("smolvlm2_2b", "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
              "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
              "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
              note="2.2B, the lightest credible captioner"),
    Candidate("qwen3vl_4b", "unsloth/Qwen3-VL-4B-Instruct-GGUF",
              "Qwen3-VL-4B-Instruct-UD-Q3_K_XL.gguf", "mmproj-F16.gguf",
              note="Qwen3-VL 4B, a vision SPECIALIST at the incumbent's quant"),
    # Already built on this machine. --jinja is not optional here: the stock
    # template detection does not render LFM2.5-VL's chat template, and the
    # existing serve script for it passes --jinja for exactly that reason.
    Candidate("lfm25_vl_3b", "", "text-q6.gguf", "mmproj-q8.gguf",
              note="LFM2.5-VL 3B, Q6 text + Q8 mmproj, already on disk",
              extra=["--jinja"],
              abs_dir="~/llama.cpp/models/lfm25-vl-3b"),
    # The same weights under Liquid's documented settings. [M7] The default
    # image budget is 256 tokens for a 1024px screenshot; --image-max-tokens
    # 1024 lifts it to ~512, which is the ceiling this image's own tiling
    # allows - asking for 2048 produces an identical 531-token prompt.
    Candidate("lfm25_vl_tuned", "", "text-q6.gguf", "mmproj-q8.gguf",
              note="LFM2.5-VL 3B at Liquid's documented sampling + 2x image tokens",
              extra=["--jinja", "--image-max-tokens", "1024"],
              sampling={"top_k": 50, "repeat_penalty": 1.0},
              abs_dir="~/llama.cpp/models/lfm25-vl-3b"),
    # [M8] The captioner since LFM2.5-VL left this machine. A hybrid
    # reasoner: thinking must be OFF under the grammar, or the constrained
    # output lands in reasoning_content and content comes back empty.
    Candidate("lfm3_vl_2b", "", "LFM3-VL-2B-Q6_K.gguf", "mmproj-LFM3-VL-2B-Q8_0.gguf",
              note="LFM3-VL 2B, Q6 text + Q8 mmproj, in ~/models",
              extra=["--jinja"], thinking_kwarg=True,
              sampling={"top_k": 50, "repeat_penalty": 1.0},
              abs_dir="~/models/lfm3-vl-2b"),
    Candidate("internvl3_2b", "ggml-org/InternVL3-2B-Instruct-GGUF",
              "InternVL3-2B-Instruct-Q4_K_M.gguf",
              "mmproj-InternVL3-2B-Instruct-f16.gguf",
              note="2B, strongest small model on document/chart suites"),
]

BY_KEY = {c.key: c for c in CANDIDATES}

# Words too common to prove a caption read anything off the image.
_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "has",
    "have", "not", "you", "your", "all", "any", "can", "its", "but", "our",
    "image", "text", "page", "data", "view", "name", "type", "list", "new",
}


def model_dir(cfg, c: Candidate) -> Path:
    if c.abs_dir:
        return Path(c.abs_dir).expanduser()
    return cfg.path("paths.models") / (
        c.subdir if c.subdir != BAKEOFF_DIR else f"{BAKEOFF_DIR}/{c.key}")


def paths_for(cfg, c: Candidate) -> tuple[Path, Path] | None:
    d = model_dir(cfg, c)
    m, p = d / c.model_file, d / c.mmproj_file
    return (m, p) if m.exists() and p.exists() else None


def download(keys: list[str]) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("need huggingface-hub:  uv pip install -e '.[hf]'", file=sys.stderr)
        return 2
    cfg = load()
    for key in keys:
        c = BY_KEY[key]
        if not c.repo:
            continue
        d = model_dir(cfg, c)
        d.mkdir(parents=True, exist_ok=True)
        for f in (c.model_file, c.mmproj_file):
            if (d / f).exists():
                print(f"  {key}/{f}  already here")
                continue
            print(f"↓ {c.repo}  {f}")
            hf_hub_download(repo_id=c.repo, filename=f, local_dir=str(d))
    return 0


def pick_images(n: int) -> list[dict]:
    """A spread, not the first n. Captioning a photo and captioning a
    screenshot are different jobs and a model can be good at one of them."""
    conn = reader()
    rows = conn.execute(
        "SELECT id, blob_path, flow_text, "
        "       COALESCE(json_array_length(boxes_json), 0) AS boxes "
        "FROM items WHERE kind = 'image' AND blob_path IS NOT NULL "
        "ORDER BY boxes DESC").fetchall()
    rows = [dict(r) for r in rows]
    texty = [r for r in rows if r["boxes"] > 0]
    plain = [r for r in rows if r["boxes"] == 0]
    # Half text-dense, half not, as far as the library allows.
    want = max(1, n // 2)
    out = texty[:want] + plain[: n - min(want, len(texty))]
    return out[:n]


def resolve_blob(cfg, blob_path: str) -> Path:
    p = Path(blob_path)
    return p if p.is_absolute() else (cfg.path("paths.data") / p)


# Several candidates open nearly every caption with "The image shows" or
# "The image depicts". That is ~18 characters of constant text in a field
# whose whole job is to be embedded, so it dilutes every vector identically.
# This variant asks for it to stop, which separates a model that cannot see
# from a model that merely preambles.
NO_PREAMBLE = (
    " Do not begin with \"The image shows\", \"The image depicts\", \"This is\" "
    "or any similar opener - start with the subject itself.")


def generate(c: Candidate, images: list[dict], escalate: bool,
             max_tokens: int = 400, no_preamble: bool = False) -> dict:
    cfg = load()
    found = paths_for(cfg, c)
    if not found:
        return {"key": c.key, "status": "missing", "captions": []}
    model, mmproj = found

    srv = LlamaServer(model, mmproj=mmproj,
                      ctx=cfg.get("models.vlm_default.ctx", 16384), ngl=99,
                      cache_type=cfg.get("models.vlm_default.cache_type"),
                      flash_attn=cfg.get("models.vlm_default.flash_attn", False),
                      extra=c.extra)
    rec: dict = {"key": c.key, "note": c.note, "model_file": c.model_file,
                 "status": "ok", "captions": []}
    t0 = time.time()
    try:
        srv.start()
    except Exception as exc:
        return {"key": c.key, "status": f"start failed: {type(exc).__name__}: {exc}",
                "captions": []}
    rec["load_s"] = round(time.time() - t0, 1)
    try:
        rec["vram_mb"] = vram.info().used_mb
    except Exception:
        rec["vram_mb"] = None

    long_edge = cfg.get("vlm.escalate_image_long_edge" if escalate
                        else "vlm.image_long_edge", 1024)
    for row in images:
        blob = resolve_blob(cfg, row["blob_path"])
        t1 = time.time()
        try:
            b64, mime = C._encode(blob, long_edge)
            prompt = C.ESCALATED_PROMPT if escalate else C.PROMPT
            if no_preamble:
                prompt += NO_PREAMBLE
            msg = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}]
            raw = _chat(srv, msg, c, max_tokens)
            obj = json.loads(raw)
            err = None
        except Exception as exc:
            obj, err = None, f"{type(exc).__name__}: {exc}"[:200]
        rec["captions"].append({
            "item_id": row["id"], "boxes": row["boxes"],
            "s": round(time.time() - t1, 2), "out": obj, "error": err,
        })
        print(f"    item {row['id']:>3}  {time.time()-t1:5.1f}s  "
              f"{(obj or {}).get('caption', err or '')[:72]}", flush=True)
    srv.stop()
    time.sleep(2)
    return rec


def _chat(srv: LlamaServer, messages: list[dict], c: Candidate,
          max_tokens: int = 400) -> str:
    """srv.chat() always sends enable_thinking, which only Qwen declares."""
    if c.thinking_kwarg:
        return srv.chat(messages, max_tokens=max_tokens, temperature=0.2,
                        grammar=C.GRAMMAR)
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
               "stream": False, "grammar": C.GRAMMAR, **c.sampling}
    r = srv._client.post("/v1/chat/completions", json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"llama-server {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _ocr_recall(caption: str, flow_text: str | None) -> float | None:
    if not flow_text:
        return None
    words = {w.lower().strip(".,:;()[]\"'") for w in flow_text.split()}
    words = {w for w in words if len(w) > 3 and w not in _STOP and not w.isdigit()}
    if not words:
        return None
    low = caption.lower()
    return round(sum(1 for w in words if w in low) / len(words), 3)


def score(rec: dict) -> dict:
    """Embed each caption as a query, KNN against the text-free image vectors."""
    import numpy as np
    from server.models import eg2
    from server.models.embeddings import pack

    conn = reader()
    vis = {}
    for r in conn.execute("SELECT item_id, embedding FROM vec_items WHERE kind='visual'"):
        v = np.frombuffer(r["embedding"], dtype="float32")
        vis[r["item_id"]] = v / (np.linalg.norm(v) or 1.0)
    flow = {r["id"]: r["flow_text"] for r in conn.execute(
        "SELECT id, flow_text FROM items WHERE kind='image'")}

    e = eg2.engine()
    ok = [c for c in rec["captions"] if c["out"] and c["out"].get("caption")]
    if not ok:
        return {**rec, "scored": False}
    qs = [c["out"]["caption"] for c in ok]
    vecs = [np.frombuffer(b, dtype="float32") for b in e.embed_texts(qs, is_query=True)]

    hits, margins, ocrs, lens, tags = 0, [], [], [], []
    for c, q in zip(ok, vecs, strict=True):
        q = q / (np.linalg.norm(q) or 1.0)
        sims = sorted(((float(q @ v), iid) for iid, v in vis.items()), reverse=True)
        own = next((s for s, iid in sims if iid == c["item_id"]), None)
        best_other = next((s for s, iid in sims if iid != c["item_id"]), 0.0)
        if own is None:
            continue
        c["rank"] = [iid for _, iid in sims].index(c["item_id"]) + 1
        c["margin"] = round(own - best_other, 4)
        hits += c["rank"] == 1
        margins.append(c["margin"])
        lens.append(len(c["out"]["caption"]))
        tags.append(len(c["out"].get("tags") or []))
        r = _ocr_recall(c["out"]["caption"], flow.get(c["item_id"]))
        if r is not None:
            ocrs.append(r)

    full = [c for c in ok if c["out"].get("caption") and c["out"].get("subject")
            and (c["out"].get("tags") or [])]
    return {
        **rec, "scored": True,
        "n": len(rec["captions"]),
        "json_ok": len(full),
        "recall_at_1": round(hits / len(ok), 3),
        "margin": round(statistics.mean(margins), 4) if margins else None,
        "ocr_recall": round(statistics.mean(ocrs), 3) if ocrs else None,
        "chars": int(statistics.mean(lens)) if lens else 0,
        "tags": round(statistics.mean(tags), 1) if tags else 0,
        "s_per_caption": round(statistics.median(
            [c["s"] for c in rec["captions"]]), 2),
    }


def table(recs: list[dict]) -> None:
    print(f"\n  {'model':<13} {'vram':>6} {'load':>6} {'s/cap':>6} {'json':>6} "
          f"{'r@1':>6} {'margin':>7} {'ocr':>6} {'chars':>6} {'tags':>5}")
    print("  " + "-" * 82)
    for r in recs:
        if not r.get("scored"):
            print(f"  {r['key']:<13} {r.get('status', 'no output')}")
            continue
        vm = f"{r['vram_mb']}" if r.get("vram_mb") else "-"
        ocr = f"{r['ocr_recall']:.3f}" if r.get("ocr_recall") is not None else "-"
        print(f"  {r['key']:<13} {vm:>6} {r.get('load_s', 0):>5}s "
              f"{r['s_per_caption']:>6} {r['json_ok']:>3}/{r['n']:<2} "
              f"{r['recall_at_1']:>6.2f} {r['margin']:>7.4f} {ocr:>6} "
              f"{r['chars']:>6} {r['tags']:>5}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--escalate", action="store_true",
                    help="use the escalated prompt and 1536px long edge")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="a truncated string is a JSON failure, so this is the\ndifference between a model that cannot comply and one that rambles")
    ap.add_argument("--no-preamble", action="store_true",
                    help="ask the model not to open with \"The image shows\"")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    keys = a.models or [c.key for c in CANDIDATES]
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        print(f"unknown model(s): {unknown}; have {list(BY_KEY)}", file=sys.stderr)
        return 2

    if a.list:
        cfg = load()
        for c in CANDIDATES:
            here = "present" if paths_for(cfg, c) else "NOT DOWNLOADED"
            print(f"  {c.key:<13} {here:<15} {c.note}")
        return 0

    if a.download:
        return download(keys)

    cfg = load()
    out_dir = cfg.path("paths.data") / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    images = pick_images(a.images)
    print(f"\n{len(images)} image(s): "
          f"{sum(1 for i in images if i['boxes'] > 0)} text-dense, "
          f"{sum(1 for i in images if i['boxes'] == 0)} plain\n")

    recs = []
    for key in keys:
        c = BY_KEY[key]
        # Every experimental condition gets its own file. Without this an
        # --escalate run silently overwrites the default-condition captions
        # and the saved record no longer matches the table it produced.
        suffix = ("-escalate" if a.escalate else "") + \
                 ("-nopreamble" if a.no_preamble else "")
        f = out_dir / f"{key}{suffix}.json"
        if a.score_only:
            if not f.exists():
                print(f"  {key}: no saved captions, skipping")
                continue
            rec = json.loads(f.read_text())
        else:
            print(f"[{key}] {c.note}")
            rec = generate(c, images, a.escalate, a.max_tokens,
                           a.no_preamble)
            f.write_text(json.dumps(rec, indent=2))
            if rec["status"] != "ok":
                print(f"  {key}: {rec['status']}\n")
        recs.append(score(rec) if rec.get("captions") else rec)

    table(recs)
    if a.json:
        print(json.dumps(recs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
