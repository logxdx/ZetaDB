#!/usr/bin/env python
"""Static INT8 quantisation of the ONNX models.

    uv run scripts/quantize.py              # quantise everything missing
    uv run scripts/quantize.py ppocr_det    # one model
    uv run scripts/quantize.py --force      # redo

Two recipes, chosen by architecture:

  STATIC QDQ (PP-OCR) - these graphs are convolution-heavy, and ORT's
  dynamic quantisation only covers MatMul/Gemm, so a CNN comes out barely
  smaller and no faster. Static needs calibration data, which we have, and
  produces a graph TensorRT can consume directly to build an INT8 engine,
  so one artefact serves both backends.

  DYNAMIC - for MatMul-heavy ViTs, where dynamic quantisation covers most
  of the work. It was used for JoyTag (static calibration of a 366 MB ViT
  exhausted 15 GB of RAM). No current model needs it: Eye, which replaced
  JoyTag, already ships int8 weights and runs at FP16 on TensorRT, so the
  set below is empty but the recipe is kept.

Each model gets its own calibration reader because each has its own
preprocessing; they reuse the exact functions the runtime uses, so
calibration statistics match inference.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

import numpy as np  # noqa: E402

from server.config import load  # noqa: E402

CALIB_LIMIT = 48


def calib_images() -> list[Path]:
    """Real images: ingested blobs first, then anything in the watch dirs."""
    cfg = load()
    out: list[Path] = []
    # Purpose-built calibration corpus first, then anything really ingested.
    calib = cfg.path("paths.data") / "calib"
    if calib.exists():
        out += sorted(p for p in calib.glob("*.png"))
    blobs = cfg.path("paths.blobs")
    if blobs.exists():
        out += [p for p in blobs.rglob("*") if p.is_file()]
    for d in cfg.paths("paths.watch_dirs"):
        if d.exists():
            out += [p for p in d.rglob("*") if p.is_file()
                    and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    seen, uniq = set(), []
    for p in out:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq[:CALIB_LIMIT]


class _Reader:
    """onnxruntime CalibrationDataReader protocol."""

    def __init__(self, batches: list[dict]):
        self._batches = batches
        self._i = 0

    def get_next(self):
        if self._i >= len(self._batches):
            return None
        b = self._batches[self._i]
        self._i += 1
        return b

    def rewind(self):
        self._i = 0


def _det_batches(paths: list[Path], input_name: str) -> list[dict]:
    import cv2
    from server.models.ppocr import _DET_MEAN, _DET_STD
    out = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = min(960 / max(h, w), 1.0)
        nh = max(32, int(round(h * s / 32)) * 32)
        nw = max(32, int(round(w * s / 32)) * 32)
        x = (cv2.resize(img, (nw, nh)).astype(np.float32) / 255.0 - _DET_MEAN) / _DET_STD
        out.append({input_name: x.transpose(2, 0, 1)[None]})
    return out


def _rec_batches(paths: list[Path], input_name: str) -> list[dict]:
    """Calibrate on real text crops: run detection first so the statistics
    come from the crops recognition will actually see, not whole pages."""
    import cv2
    from server.models.ppocr import engine
    ocr = engine()
    ocr.load_models()
    out = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            quads = ocr._detect(img)
        except Exception:
            continue
        for quad in quads[:8]:
            crop = ocr._crop(img, quad)
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            nw = min(320, max(8, int(round(w * 48 / max(1, h)))))
            r = cv2.resize(crop, (nw, 48)).astype(np.float32)
            r = (r / 255.0 - 0.5) / 0.5
            batch = np.zeros((1, 3, 48, 320), dtype=np.float32)
            batch[0, :, :, :nw] = r.transpose(2, 0, 1)
            out.append({input_name: batch})
    ocr.unload()
    return out


BUILDERS = {
    "ppocr_det": _det_batches,
    "ppocr_rec": _rec_batches,
}
DYNAMIC: set[str] = set()     # see the module docstring

# Nodes kept at full precision. PP-OCR recognition ends in an 18710-way CTC
# classifier, and quantising that head is what wrecked accuracy: measured
# recall fell from 15/15 to 5/15 with characters degrading into each other
# ("2026-03-14" -> "202631", "PP-OCRv6_tiny" -> "PP00R36_in"). Detection is
# a binary segmentation map and is unaffected. The backbone still
# quantises, so most of the size and speed win survives.
HEAD_TAIL_NODES = {"ppocr_rec": 6}


def excluded_nodes(key: str, model_path) -> list[str]:
    n = HEAD_TAIL_NODES.get(key)
    if not n:
        return []
    import onnx
    g = onnx.load(str(model_path)).graph
    keep = [nd.name for nd in list(g.node)[-n:] if nd.name]
    return keep


def quantise(key: str, force: bool = False) -> bool:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    cfg = load()
    d = cfg.path("paths.models") / key
    srcs = [p for p in sorted(d.rglob("*.onnx")) if "_int8" not in p.name]
    if not srcs:
        print(f"  {key}: no source .onnx found"); return False
    src = srcs[0]
    dst = src.with_name(src.stem + "_int8.onnx")
    if dst.exists() and not force:
        print(f"  {key}: already int8 ({dst.stat().st_size / 1e6:.0f} MB)"); return True

    if key in DYNAMIC:
        t0 = time.time()
        print(f"  {key}: dynamic int8 (no calibration pass)…", flush=True)
        quantize_dynamic(model_input=str(src), model_output=str(dst),
                         weight_type=QuantType.QInt8, per_channel=True,
                         extra_options={"MatMulConstBOnly": True})
        print(f"  {key}: {src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB "
              f"in {time.time() - t0:.0f}s  [{dst.name}]")
        return True

    imgs = calib_images()
    if len(imgs) < 4:
        print(f"  {key}: only {len(imgs)} calibration image(s); ingest some first"); return False

    import onnxruntime as ort
    name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    print(f"  {key}: preparing graph…", flush=True)
    prepped = src.with_name(src.stem + "_prep.onnx")
    try:
        quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
    except Exception as exc:
        print(f"    pre-process failed ({type(exc).__name__}); using the raw graph")
        shutil.copy(src, prepped)

    print(f"  {key}: building calibration set from {len(imgs)} image(s)…", flush=True)
    batches = BUILDERS[key](imgs, name)
    if not batches:
        print(f"  {key}: calibration produced no samples"); prepped.unlink(missing_ok=True); return False

    skip = excluded_nodes(key, prepped)
    if skip:
        print(f"  {key}: keeping {len(skip)} output-head node(s) at full precision")

    def _build(per_channel: bool) -> None:
        quantize_static(
            model_input=str(prepped), model_output=str(dst),
            calibration_data_reader=_Reader(batches),
            quant_format=QuantFormat.QDQ,      # TRT consumes QDQ to build INT8
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=skip,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )

    def _loads() -> bool:
        """A quantised graph that ORT refuses is worse than none: verify."""
        import onnxruntime as ort
        try:
            ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
            return True
        except Exception as exc:
            print(f"    per-channel graph rejected ({str(exc)[:80]}…)")
            return False

    t0 = time.time()
    print(f"  {key}: quantising on {len(batches)} sample(s)…", flush=True)
    _build(per_channel=True)
    if not _loads():
        # Per-channel weight quantisation emits a DequantizeLinear whose axis
        # does not match some conv weights in these graphs. Per-tensor is
        # slightly less accurate and always valid.
        print(f"  {key}: retrying per-tensor…", flush=True)
        dst.unlink(missing_ok=True)
        _build(per_channel=False)
        if not _loads():
            dst.unlink(missing_ok=True)
            prepped.unlink(missing_ok=True)
            print(f"  {key}: quantisation produced no loadable graph")
            return False
    prepped.unlink(missing_ok=True)
    print(f"  {key}: {src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB "
          f"in {time.time() - t0:.0f}s  [{dst.name}]")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    keys = a.models or (list(BUILDERS) + sorted(DYNAMIC))
    print(f"calibration images: {len(calib_images())}")
    ok = True
    for k in keys:
        if k not in BUILDERS and k not in DYNAMIC:
            print(f"  {k}: no quantisation recipe"); ok = False; continue
        ok = quantise(k, a.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
