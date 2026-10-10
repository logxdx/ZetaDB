#!/usr/bin/env python3
"""Can a VLM replace PP-OCRv6 as the OCR authority? Measure, do not assume.

spec 1.1 asserts that PP-OCRv6_medium (34.5M params) outperforms billion-scale
VLMs on recognition, and the whole two-jobs-two-models split rests on it. That
assertion was inherited, not measured here, and LFM2.5-VL 3B advertises OCR and
document understanding, so it deserves a real test.

The comparison is end-to-end and like-for-like. PP-OCR is scored on what the
PIPELINE actually produces - detection, recognition, then the grid projection
of stage 2 - not on raw boxes, because `flow_text` is what reaches FTS5,
embeddings and extraction. The VLM is scored on a plain transcription of the
same image.

Ground truth is SYNTHETIC by default, and that is the point: an image rendered
from a known string has exact ground truth, free of the labelling error that
makes small hand-labelled OCR sets untrustworthy. The cases mirror this corpus
- an invoice, a metrics table, a code block, dense prose, small UI chrome - and
include the things extraction depends on: emails, URLs, ISO dates, decimal
amounts.

  CER   character error rate, Levenshtein / len(truth). The headline.
  WER   word error rate, on whitespace tokens.
  crit  exact-match rate over CRITICAL tokens - emails, URLs, dates, amounts.
        A 2% CER that eats the decimal point in an invoice total is not a 2%
        problem, and spec 5.1 already found INT8 recognition turning
        "finance@acme.example" into "finiance@acme.examp".

    uv run scripts/ocr_bakeoff.py --make          # render the synthetic set
    uv run scripts/ocr_bakeoff.py                 # both engines, full report
    uv run scripts/ocr_bakeoff.py --engines ppocr
    uv run scripts/ocr_bakeoff.py --real          # the library's own images
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

from server.config import load                              # noqa: E402
from server.db.conn import reader                           # noqa: E402
from server.layout.grid import project                      # noqa: E402
from server.layout.types import TextBox                     # noqa: E402
from server.models.llama_server import LlamaServer          # noqa: E402
from server.models import vram                              # noqa: E402

OUT = "ocr_bakeoff"

# Text that mirrors what this library actually holds, with the token types
# extraction depends on deliberately present.
CASES: list[dict] = [
    {
        "key": "invoice",
        "size": (1000, 700),
        "bg": (252, 252, 250), "fg": (20, 20, 24), "size_px": 22,
        "lines": [
            "ACME SUPPLY CO",
            "Invoice INV-2026-0314",
            "Issued: 2026-03-14    Due: 2026-04-13",
            "Portal: https://portal.acme.example/inv/0314",
            "Contact: finance@acme.example",
            "",
            "Subtotal      1,240.00",
            "Tax (8%)         99.20",
            "Total         1,339.20",
        ],
    },
    {
        "key": "metrics",
        "size": (1000, 520),
        "bg": (18, 18, 22), "fg": (235, 235, 240), "size_px": 22,
        "lines": [
            "Tag Fusion Dashboard",
            "",
            "PP-OCRv6_tiny      1.5M    80.6    73.5",
            "PP-OCRv6_small     7.7M    84.1    81.3",
            "PP-OCRv6_medium   34.5M    86.2    83.2",
            "",
            "updated 2026-03-14T09:12:44Z",
        ],
    },
    {
        "key": "code",
        "size": (1100, 560),
        "bg": (24, 26, 32), "fg": (210, 215, 225), "size_px": 19, "mono": True,
        "lines": [
            "def pack(vec) -> bytes:",
            "    a = np.asarray(vec, dtype='float32').ravel()",
            "    n = float(np.linalg.norm(a))",
            "    if n > 0:",
            "        a = a / n",
            "    return struct.pack(f'<{a.size}f', *a.tolist())",
            "",
            "# see spec 9.3; truncation requires re-normalising",
        ],
    },
    {
        "key": "prose",
        "size": (980, 620),
        "bg": (255, 255, 253), "fg": (28, 28, 30), "size_px": 20,
        "lines": [
            "The design pressure throughout is a hard 6 GB VRAM ceiling,",
            "which makes model scheduling a first-class architectural",
            "concern rather than an implementation detail. The machine is",
            "hybrid-graphics: the Intel UHD iGPU drives the Wayland",
            "session, so the dGPU is compute-only and the budget does not",
            "fluctuate with desktop use. Retrieval value only appears",
            "after a few hundred items, so two frictionless capture paths",
            "beat six half-built ones.",
        ],
    },
    {
        "key": "ui_small",
        "size": (1120, 460),
        "bg": (244, 245, 248), "fg": (60, 62, 70), "size_px": 14,
        "lines": [
            "Settings / Indexing / Advanced",
            "Watch directory: ~/dump          Debounce: 500 ms",
            "Embedding dim: 768               Truncate: off",
            "Review floor: 0.25               Threshold step: 0.02",
            "Last calibration: 2026-03-14 09:12   Items: 1,284",
            "Contact support at help@iwashere.example for assistance",
        ],
    },
]

# The library's own text-dense images, transcribed by eye from the originals.
# Two only: the rest of the corpus is photographs whose "text" is a single
# incidental sign, which measures nothing about an OCR engine.
REAL_TRUTH: dict[str, str] = {
    "item1": (
        "Tag Fusion Dashboard\n"
        "PP-OCRv6_tiny 1.5M 80.6 73.5\n"
        "PP-OCRv6_small 7.7M 84.1 81.3\n"
        "PP-OCRv6_medium 34.5M 86.2 83.2"
    ),
    "item4": (
        "ACME SUPPLY CO - INVOICE\n"
        "Issued 2026-03-14\n"
        "Due 2026-04-13\n"
        "Portal https://billing.acme.example/inv/8813\n"
        "Contact finance@acme.example\n"
        "Subtotal 1,240.00\n"
        "Tax 99.20\n"
        "Total 1,339.20"
    ),
}
REAL_IDS = {"item1": 1, "item4": 4}


def real_images() -> tuple[dict[str, str], list[Path]]:
    cfg = load()
    conn = reader()
    root = cfg.path("paths.data")
    truths, paths = {}, []
    for key, iid in REAL_IDS.items():
        row = conn.execute(
            "SELECT blob_path FROM items WHERE id = ?", (iid,)).fetchone()
        if row is None or not row["blob_path"]:
            continue
        src = root / row["blob_path"]
        if not src.exists():
            continue
        # Copy under the case name so results key the same way as synthetic.
        dest = cfg.path("paths.data") / OUT / f"{key}{src.suffix}"
        dest.write_bytes(src.read_bytes())
        truths[key] = REAL_TRUTH[key]
        paths.append(dest)
    return truths, paths


CRITICAL = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"                 # emails
    r"|https?://[^\s]+"                        # urls
    r"|\d{4}-\d{2}-\d{2}(?:T[\d:]+Z)?"         # iso dates
    r"|\d{1,3}(?:,\d{3})*\.\d{2}"              # money
)

OCR_PROMPT = (
    "Transcribe every piece of text in this image exactly as written, "
    "line by line, in reading order. Preserve punctuation, capitalisation, "
    "digits and symbols exactly. Output only the transcribed text, with no "
    "commentary, no headings and no markdown."
)


def render(case: dict, dest: Path) -> str:
    """Render one case and return its exact ground-truth text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", case["size"], case["bg"])
    d = ImageDraw.Draw(img)
    mono = case.get("mono")
    candidates = (
        ["/usr/share/fonts/TTF/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"] if mono else
        ["/usr/share/fonts/TTF/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    font = None
    for c in candidates:
        if Path(c).exists():
            font = ImageFont.truetype(c, case["size_px"])
            break
    if font is None:                       # pragma: no cover - font-less box
        font = ImageFont.load_default()
    y, step = 34, int(case["size_px"] * 1.9)
    for line in case["lines"]:
        if line:
            d.text((40, y), line, fill=case["fg"], font=font)
        y += step
    img.save(dest, "PNG")
    return "\n".join(x for x in case["lines"] if x)


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    """Collapse whitespace. Neither engine is being judged on line breaks."""
    return re.sub(r"\s+", " ", s or "").strip()


def score(truth: str, hyp: str) -> dict:
    t, h = _norm(truth), _norm(hyp)
    cer = _lev(t, h) / max(len(t), 1)
    tw, hw = t.split(), h.split()
    wer = _lev_words(tw, hw) / max(len(tw), 1)
    crit = CRITICAL.findall(truth)
    hit = sum(1 for c in crit if c in h)
    return {"cer": round(cer, 4), "wer": round(wer, 4),
            "crit_total": len(crit), "crit_hit": hit,
            "crit_rate": round(hit / len(crit), 3) if crit else None}


def _lev_words(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_ppocr(paths: list[Path], backend: str | None = None) -> dict[str, dict]:
    """PP-OCR as the PIPELINE uses it: detect, recognise, then project.

    `backend` overrides config for this run. Recognition accuracy is a
    property of the GRAPH, not of the execution provider, so onnx-cpu
    measures the same text as the TensorRT path - only slower. It is the
    honest fallback when cuDNN is absent, and if anything it flatters
    PP-OCR slightly: CPU runs the graph at fp32 where TensorRT runs fp16.
    """
    from server.models.ppocr import engine
    if backend:
        cfg = load()
        cfg._d.setdefault("ocr", {})
        for k in ("det_backend", "rec_backend", "backend"):
            cfg._d["ocr"][k] = backend
    ocr = engine()
    if not ocr.available:
        return {}
    ocr.load_models()
    out: dict[str, dict] = {}
    for p in paths:
        t0 = time.time()
        boxes: list[TextBox] = ocr.read(p)
        res = project(boxes, page_width=None)
        out[p.stem] = {"text": res.flow_text, "s": round(time.time() - t0, 2),
                       "boxes": len(boxes)}
    try:
        ocr.unload()
    except Exception:
        pass
    return out


def run_vlm(paths: list[Path], long_edge: int, image_max_tokens: int | None,
            cpu_only: bool = False) -> dict[str, dict]:
    """cpu_only is for accuracy work when something else holds the card.
    A transcription is the same string either way; only the clock changes,
    so the s/image column must not be read across devices."""
    import base64
    import io
    from PIL import Image, ImageOps

    d = Path("~/llama.cpp/models/lfm25-vl-3b").expanduser()
    model, mmproj = d / "text-q6.gguf", d / "mmproj-q8.gguf"
    if not model.exists():
        return {}
    extra = ["--jinja"]
    if image_max_tokens:
        extra += ["--image-max-tokens", str(image_max_tokens)]
    srv = LlamaServer(model, mmproj=mmproj, ctx=16384,
                      ngl=0 if cpu_only else 99, cpu_only=cpu_only,
                      cache_type="q8_0", flash_attn=True, extra=extra)
    srv.start()
    out: dict[str, dict] = {}
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            if max(im.size) > long_edge:
                im.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": OCR_PROMPT}]}]
        t0 = time.time()
        payload = {"messages": msg, "max_tokens": 1200, "temperature": 0.2,
                   "stream": False, "top_k": 50, "repeat_penalty": 1.0}
        r = srv._client.post("/v1/chat/completions", json=payload)
        txt = (r.json()["choices"][0]["message"]["content"]
               if r.status_code < 400 else f"<HTTP {r.status_code}>")
        out[p.stem] = {"text": txt, "s": round(time.time() - t0, 2)}
    try:
        rec_vram = vram.info().used_mb
    except Exception:
        rec_vram = None
    srv.stop()
    for v in out.values():
        v["vram_mb"] = rec_vram
    return out


def report(truths: dict[str, str], runs: dict[str, dict[str, dict]]) -> None:
    print(f"\n  {'engine':<22} {'case':<10} {'CER':>7} {'WER':>7} {'crit':>8} {'s':>6}")
    print("  " + "-" * 66)
    for eng, res in runs.items():
        cers, wers, ch, ct, secs = [], [], 0, 0, []
        for key, truth in truths.items():
            got = res.get(key)
            if not got:
                continue
            sc = score(truth, got["text"])
            cers.append(sc["cer"]); wers.append(sc["wer"]); secs.append(got["s"])
            ch += sc["crit_hit"]; ct += sc["crit_total"]
            print(f"  {eng:<22} {key:<10} {sc['cer']:>7.3f} {sc['wer']:>7.3f} "
                  f"{sc['crit_hit']:>3}/{sc['crit_total']:<4} {got['s']:>6.2f}")
        if cers:
            print(f"  {eng:<22} {'MEAN':<10} {sum(cers)/len(cers):>7.3f} "
                  f"{sum(wers)/len(wers):>7.3f} {ch:>3}/{ct:<4} "
                  f"{sum(secs)/len(secs):>6.2f}")
            print("  " + "-" * 66)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true", help="render the synthetic set and stop")
    ap.add_argument("--engines", nargs="*",
                    default=["ppocr", "lfm_1024", "lfm_1536"])
    ap.add_argument("--real", action="store_true",
                    help="also run the library's own text-dense images")
    ap.add_argument("--vlm-cpu", action="store_true",
                    help="run the VLM on CPU; same text, different clock")
    ap.add_argument("--ocr-backend", default=None,
                    help="override [ocr] backend, e.g. onnx-cpu when cuDNN is absent")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    cfg = load()
    d = cfg.path("paths.data") / OUT
    d.mkdir(parents=True, exist_ok=True)

    truths: dict[str, str] = {}
    paths: list[Path] = []
    for case in CASES:
        p = d / f"{case['key']}.png"
        truths[case["key"]] = render(case, p)
        paths.append(p)
    if a.real:
        rt, rp = real_images()
        truths.update(rt)
        paths.extend(rp)
    (d / "truth.json").write_text(json.dumps(truths, indent=2))
    print(f"{len(paths)} synthetic image(s) in {d}")
    if a.make:
        return 0

    runs: dict[str, dict[str, dict]] = {}
    if "ppocr" in a.engines:
        print("\n[ppocr] PP-OCRv6 detect + recognise + grid projection")
        runs["PP-OCRv6 + layout"] = run_ppocr(paths, a.ocr_backend)
    if "lfm_1024" in a.engines:
        print("[lfm_1024] LFM2.5-VL at the production 1024 px downscale")
        runs["LFM2.5-VL @1024"] = run_vlm(paths, 1024, 1024, a.vlm_cpu)
    if "lfm_1536" in a.engines:
        print("[lfm_1536] LFM2.5-VL at the escalated 1536 px")
        runs["LFM2.5-VL @1536"] = run_vlm(paths, 1536, 1024, a.vlm_cpu)

    report(truths, runs)
    # Merge, never overwrite. Engines are usually run separately - one needs
    # cuDNN, another needs the card free - and a plain dump of the current
    # run silently discards every engine that is not in it.
    f = d / "results.json"
    merged = {}
    if f.exists():
        try:
            merged = json.loads(f.read_text())
        except Exception:
            merged = {}
    merged.update(runs)
    f.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"  raw transcriptions -> {d / 'results.json'}\n")
    if a.json:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
