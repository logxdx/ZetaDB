"""PP-OCRv6 detection + recognition over ONNX Runtime.

Backend order is TensorRT, then CUDA, then CPU (spec 5), decided by the
model engine (server.engine) like every other ONNX model; the chosen provider
is reported by /api/health so a silent fall back to CPU is visible.

Post-process parameters and the 18708-entry character dictionary come from
the model's own inference.yml rather than being hardcoded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server.config import load
from server.engine import ModelSpec, Profile, max_batch, register_profile, session
from server.engine.core import engine as model_engine
from server.layout.types import TextBox

log = logging.getLogger(__name__)

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Explicit optimisation profiles. Without them TensorRT either refuses the
# dynamic axes or builds for the model's declared maximum - PP-OCR's own
# inference.yml advertises 4000x4000 detection, which produces an enormous
# engine and a very long build for shapes we never feed it. These are our
# real working ranges: detection is capped at ocr.det_long_edge, and
# recognition at the largest width bucket.
# Optimisation profiles.
#
# Two lessons, one learned the hard way and one taken from TurboOCR
# (github.com/aiptimizer/TurboOCR), which runs the same PP-OCRv6 models on
# TensorRT in production:
#
#   1. RANGE costs more than maximum. A 32x32 minimum against 960x960 is a
#      900x area span and TensorRT must find kernels valid across all of
#      it; the medium INT8 detector was still searching after 45 minutes
#      where the small one built in 188 s. Detection input is always a
#      multiple of 32 scaled to ocr.det_long_edge, so the floor is ~320.
#   2. TurboOCR uses NINE discrete width profiles for recognition rather
#      than one wide one. ORT's TRT EP exposes a single profile, so the
#      equivalent here is to keep the range tight around the width buckets
#      we actually feed it.
register_profile("ppocr_det", Profile(min="x:1x3x320x320", opt="x:1x3x960x960",
                                     max="x:1x3x960x960"))
register_profile("ppocr_rec", Profile(min="x:1x3x48x64", opt="x:8x3x48x320",
                                     max="x:8x3x48x512"))


def trt_max_batch(model_key: str) -> int:
    """Max batch the model's TensorRT profile was built for."""
    return max_batch(model_key)


@dataclass
class OcrLine:
    box: tuple[float, float, float, float]
    text: str
    score: float


class PPOCR:
    def __init__(self) -> None:
        cfg = load()
        self.cfg = cfg
        root = cfg.path("paths.models")
        # Always take the most-quantised graph present (spec 5).
        from server.models._native import pick_onnx
        # FP16-on-TensorRT is the production path for both OCR models, so
        # neither prefers the INT8 graph by default. The INT8 artefacts are
        # still built and kept; ocr.det_quantised switches detection over.
        self.det_path = pick_onnx(root / "ppocr_det",
                                  quantised=cfg.get("ocr.det_quantised", False))
        # Recognition stays at full precision by default: INT8 PTQ preserved
        # only 13 of 28 text lines even with the classifier head excluded
        # (spec 5.1). Flip ocr.rec_quantised to trade accuracy for 9 MB.
        self.rec_path = pick_onnx(root / "ppocr_rec",
                                  quantised=cfg.get("ocr.rec_quantised", False))
        self.available = bool(self.det_path and self.rec_path)
        self._det = self._rec = None
        self._charset: list[str] = []
        self.provider = "unavailable"
        self.det_post = {"thresh": 0.2, "box_thresh": 0.45,
                         "unclip_ratio": 1.4, "max_candidates": 3000}
        model_engine.register(ModelSpec(
            name="ppocr", pool="A", backend="tensorrt", unload=self.unload,
            provider=lambda: self.provider, loaded=lambda: self._det is not None))

    # ------------------------------------------------------------- lifecycle

    def load_models(self) -> None:
        if self._det is not None or not self.available:
            return
        import yaml

        default = self.cfg.get("ocr.backend", "auto")
        self._det = self._session(str(self.det_path), None,
                                  self.cfg.get("ocr.det_backend", default), "ppocr_det")
        self._rec = self._session(str(self.rec_path), None,
                                  self.cfg.get("ocr.rec_backend", default), "ppocr_rec")
        self.provider = self._det.get_providers()[0]
        self.rec_provider = self._rec.get_providers()[0]

        det_yml = self.det_path.parent / "inference.yml"
        if det_yml.exists():
            pp = (yaml.safe_load(det_yml.read_text()) or {}).get("PostProcess") or {}
            self.det_post.update({k: pp[k] for k in self.det_post if k in pp})

        rec_yml = self.rec_path.parent / "inference.yml"
        chars = []
        if rec_yml.exists():
            pp = (yaml.safe_load(rec_yml.read_text()) or {}).get("PostProcess") or {}
            chars = pp.get("character_dict") or []
        # CTCLabelDecode convention: index 0 is the blank, a space is appended.
        self._charset = ["<blank>"] + list(chars) + [" "]
        out_dim = self._rec.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != len(self._charset):
            log.warning("charset %d != rec output %d; decoding may be offset",
                        len(self._charset), out_dim)
        log.info("PP-OCRv6 ready · det=%s on %s · rec=%s on %s · charset %d",
                 self.det_path.name, self.provider,
                 self.rec_path.name, self.rec_provider, len(self._charset))

    def unload(self) -> None:
        self._det = self._rec = None

    def _session(self, path: str, so, pref: str, key: str):
        """TensorRT FP16 -> CUDA -> CPU, degrading per session, via the
        engine (spec 5.1). Recognition's TRT build can fail where detection's
        succeeds, and one model's failure must not take OCR down with it."""
        return session(path, key, pref)

    # ------------------------------------------------------------- detection

    def _detect(self, img: np.ndarray) -> list[np.ndarray]:
        import cv2
        limit = self.cfg.get("ocr.det_long_edge", 960)
        h, w = img.shape[:2]
        scale = min(limit / max(h, w), 1.0)
        nh, nw = max(32, int(round(h * scale / 32)) * 32), max(32, int(round(w * scale / 32)) * 32)
        resized = cv2.resize(img, (nw, nh))
        x = ((resized.astype(np.float32) / 255.0 - _DET_MEAN) / _DET_STD)
        x = x.transpose(2, 0, 1)[None]
        prob = self._det.run(None, {self._det.get_inputs()[0].name: x})[0][0, 0]
        return self._db_boxes(prob, (w / nw, h / nh))

    def _db_boxes(self, prob: np.ndarray, ratio: tuple[float, float]) -> list[np.ndarray]:
        import cv2
        import pyclipper
        from shapely.geometry import Polygon

        p = self.det_post
        mask = (prob > p["thresh"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rx, ry = ratio
        boxes: list[np.ndarray] = []
        for c in contours[: int(p["max_candidates"])]:
            if len(c) < 4:
                continue
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype(np.float32)
            if min(rect[1]) < 3:
                continue
            # Score the region by mean probability inside its bounding box.
            xs, ys = pts[:, 0], pts[:, 1]
            x0, x1 = int(max(0, xs.min())), int(min(prob.shape[1] - 1, xs.max()))
            y0, y1 = int(max(0, ys.min())), int(min(prob.shape[0] - 1, ys.max()))
            if x1 <= x0 or y1 <= y0:
                continue
            if float(prob[y0:y1 + 1, x0:x1 + 1].mean()) < p["box_thresh"]:
                continue
            # Unclip: DB shrinks regions during training, so grow them back.
            poly = Polygon(pts)
            if poly.area <= 0:
                continue
            dist = poly.area * p["unclip_ratio"] / poly.length
            off = pyclipper.PyclipperOffset()
            off.AddPath(pts.astype(np.int64).tolist(),
                        pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = off.Execute(dist)
            if not expanded:
                continue
            quad = np.array(expanded[0], dtype=np.float32)
            quad[:, 0] *= rx
            quad[:, 1] *= ry
            boxes.append(quad)
        return boxes

    # ----------------------------------------------------------- recognition

    @staticmethod
    def _crop(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        import cv2
        xs, ys = quad[:, 0], quad[:, 1]
        x0, x1 = int(max(0, xs.min())), int(min(img.shape[1], xs.max()))
        y0, y1 = int(max(0, ys.min())), int(min(img.shape[0], ys.max()))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.zeros((8, 8, 3), np.uint8)
        return img[y0:y1, x0:x1]

    def _recognise(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import cv2
        if not crops:
            return []
        buckets = self.cfg.get("ocr.rec_width_buckets", [64, 128, 256, 512])
        H = self.cfg.get("ocr.rec_height", 48)
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)

        # Bucket by aspect ratio so one engine profile serves many widths.
        groups: dict[int, list[int]] = {}
        for i, c in enumerate(crops):
            h, w = c.shape[:2]
            want = max(8, int(round(w * H / max(1, h))))
            bucket = next((b for b in buckets if want <= b), buckets[-1])
            groups.setdefault(bucket, []).append(i)

        name = self._rec.get_inputs()[0].name
        # Chunk to the TensorRT profile's max batch. A group can hold any
        # number of crops, and a batch of 9 against a profile built for
        # [1..8] fails at RUN time with "does not satisfy any optimization
        # profiles" - which the per-session fallback then masks by quietly
        # dropping to the CUDA EP. Derived from the registered engine profile so the two
        # cannot drift apart.
        max_b = trt_max_batch("ppocr_rec") if "Tensorrt" in self.rec_provider else len(crops)
        for bucket, idxs in groups.items():
            for start in range(0, len(idxs), max_b):
                chunk = idxs[start:start + max_b]
                batch = np.zeros((len(chunk), 3, H, bucket), dtype=np.float32)
                for slot, i in enumerate(chunk):
                    c = crops[i]
                    h, w = c.shape[:2]
                    nw = min(bucket, max(8, int(round(w * H / max(1, h)))))
                    r = cv2.resize(c, (nw, H)).astype(np.float32)
                    r = (r / 255.0 - 0.5) / 0.5
                    batch[slot, :, :, :nw] = r.transpose(2, 0, 1)
                logits = self._rec.run(None, {name: batch})[0]
                for slot, i in enumerate(chunk):
                    out[i] = self._ctc_decode(logits[slot])
        return out

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        idx = logits.argmax(axis=-1)
        probs = logits.max(axis=-1)
        chars, confs, prev = [], [], -1
        for t, k in enumerate(idx):
            if k != prev and k != 0:
                if k < len(self._charset):
                    chars.append(self._charset[k])
                    confs.append(float(probs[t]))
            prev = int(k)
        text = "".join(chars)
        return text, float(np.mean(confs)) if confs else 0.0

    # ------------------------------------------------------------------- api

    def read(self, image_path: Path) -> list[TextBox]:
        import cv2
        self.load_models()
        if not self.available:
            return []
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        quads = self._detect(img)
        if not quads:
            return []
        crops = [self._crop(img, q) for q in quads]
        texts = self._recognise(crops)
        min_score = self.cfg.get("ocr.min_rec_score", 0.5)
        boxes: list[TextBox] = []
        for quad, (text, score) in zip(quads, texts, strict=True):
            if not text.strip() or score < min_score:
                continue
            boxes.append(TextBox(float(quad[:, 0].min()), float(quad[:, 1].min()),
                                 float(quad[:, 0].max()), float(quad[:, 1].max()),
                                 text.strip(), score))
        return boxes


_engine: PPOCR | None = None


def engine() -> PPOCR:
    global _engine
    if _engine is None:
        _engine = PPOCR()
    return _engine
