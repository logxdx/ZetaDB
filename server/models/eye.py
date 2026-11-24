"""Eye (Desert Ant Labs) - calibrated photo tags, scored against our vocabulary.

Eye replaces JoyTag. Where JoyTag gave an uncalibrated sigmoid over 5813
Danbooru tags - an anime-art schema that had no word for "screenshot",
"whiteboard" or "receipt" - Eye gives:

  type    16-way single-label: photo, screenshot, receipt, document,
          whiteboard, map, chart_or_slide, illustration_or_art, ...
          (plus `reference`, the summed probability of the non-photo types)
  scenes  558 Open Images scene/object labels, multi-label
  sensitive / reject / rotation  (kept for completeness)

and, crucially, a CALIBRATION CURVE per tag (eye_meta.json): a piecewise
linear map from the raw head output to a confidence where 0.5 is Desert Ant's
recommended cut and 0.9 their strict one. So an Eye score means the same
thing for every tag, which JoyTag's never did.

Like JoyTag it is NOT used as a tag generator: the vocabulary aliases terms
onto Eye tag ids (`vocab_terms.eye_alias`) and only those are read. Terms
with no Eye counterpart fall through to EmbeddingGemma 2 zero-shot.

Runtime: `eye.tflite` is converted once to `eye_photo.onnx` by
scripts/convert_eye.py, which also checks the ONNX outputs against the LiteRT
reference (max |diff| 1.3e-4 over 24 real images, embedding cosine 1.000000)
and records that every head already emits probabilities. From then on Eye is
an ordinary engine model: TensorRT FP16, then CUDA, then CPU.

Used with permission from Desert Ant Labs; the licence and third-party
notices ship next to the weights. Credit: Powered by Desert Ant Labs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server.config import load
from server.engine import ModelSpec, session
from server.engine.core import engine as model_engine

log = logging.getLogger(__name__)

SIDE = 384


@dataclass(frozen=True)
class EyeTag:
    id: str
    output: str
    indices: tuple[int, ...]
    xs: np.ndarray
    ys: np.ndarray
    calibrated: bool
    turned: bool = False     # rotation: P(not upright) = 1 - P(0 degrees)

    def raw(self, heads: dict[str, np.ndarray]) -> float:
        v = heads[self.output]
        if self.turned:
            return float(1.0 - v[0])
        return float(sum(v[i] for i in self.indices))

    def confidence(self, raw: float) -> float:
        return float(np.interp(raw, self.xs, self.ys))


def load_tags(meta: dict) -> dict[str, EyeTag]:
    out: dict[str, EyeTag] = {}
    for t in meta.get("tags", {}).get("model", []):
        idx = t.get("indices")
        if idx is None and "index" in t:
            idx = [t["index"]]
        curve = np.asarray(t.get("curve") or [[0, 0], [1, 1]], dtype=np.float32)
        out[t["id"]] = EyeTag(
            id=t["id"], output=t["output"], indices=tuple(idx or ()),
            xs=curve[:, 0], ys=curve[:, 1], calibrated=bool(t.get("calibrated")),
            turned=bool(t.get("turned")))
    return out


def prepare(path: Path) -> np.ndarray:
    """Eye's documented preparation: upright (EXIF), short side to 384
    bicubic, centre crop 384, RGB in 0..1, NCHW. Normalisation is inside the
    model. Identical to the reference used for the parity check."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = SIDE / min(w, h)
        im = im.resize((max(SIDE, round(w * s)), max(SIDE, round(h * s))),
                       Image.Resampling.BICUBIC)
        w, h = im.size
        left, top = (w - SIDE) // 2, (h - SIDE) // 2
        im = im.crop((left, top, left + SIDE, top + SIDE))
        a = np.asarray(im, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None]


def _activate(x: np.ndarray, how: str) -> np.ndarray:
    if how == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if how == "softmax":
        e = np.exp(x - x.max())
        return e / e.sum()
    return x


class Eye:
    def __init__(self) -> None:
        cfg = load()
        d = cfg.path("paths.models") / "eye"
        self.dir = d
        self.onnx = d / "eye_photo.onnx"
        self.meta_path = d / "eye_meta.json"
        self.side_path = d / "eye_photo.json"
        self.backend = cfg.get("models.eye.backend", "auto")
        self.available = self.onnx.exists() and self.meta_path.exists()
        self.tags: dict[str, EyeTag] = {}
        self.labels: dict[str, list[str]] = {}
        self.activations: dict[str, str] = {}
        self._sess = None
        self.provider = "unavailable"
        model_engine.register(ModelSpec(
            name="eye", pool="A", backend="onnx", unload=self.unload,
            provider=lambda: self.provider, loaded=lambda: self._sess is not None))

    def load_model(self) -> None:
        if self._sess is not None or not self.available:
            return
        meta = json.loads(self.meta_path.read_text())
        self.tags = load_tags(meta)
        self.labels = meta.get("labels", {})
        if self.side_path.exists():
            self.activations = json.loads(self.side_path.read_text()).get("activations", {})
        self._sess = session(self.onnx, "eye", self.backend)
        self.provider = self._sess.get_providers()[0]
        log.info("Eye ready on %s · %d calibrated tags", self.provider, len(self.tags))

    def unload(self) -> None:
        self._sess = None
        self.provider = "unavailable" if not self.available else "unloaded"

    def covers(self, tag_id: str) -> bool:
        self.load_tags_only()
        return tag_id in self.tags

    def load_tags_only(self) -> None:
        if not self.tags and self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            self.tags = load_tags(meta)
            self.labels = meta.get("labels", {})

    # --------------------------------------------------------------- scoring

    def heads(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Raw head outputs for one prepared image, activated if the graph
        emits logits (it does not, per the parity run - kept for safety)."""
        names = [o.name for o in self._sess.get_outputs()]
        outs = dict(zip(names, self._sess.run(None, {"image": image})))
        return {n: _activate(np.asarray(v, dtype=np.float32).ravel(),
                             self.activations.get(n, "none"))
                for n, v in outs.items()}

    def score_image(self, image: np.ndarray) -> dict[str, float]:
        """tag id -> calibrated confidence (0..1, 0.5 = recommended cut)."""
        h = self.heads(image)
        return {tid: t.confidence(t.raw(h)) for tid, t in self.tags.items()
                if t.output in h}

    def score_batch(self, paths: list[Path]) -> list[dict[str, float] | None]:
        """Calibrated scores per image, or None where the image failed.
        The exported graph is batch-1, so this is a loop - one image is
        ~2.5k patches through a ViT and saturates the GPU on its own."""
        self.load_model()
        if not self.available or self._sess is None:
            return [None] * len(paths)
        out: list[dict[str, float] | None] = []
        for p in paths:
            try:
                out.append(self.score_image(prepare(p)))
            except Exception:
                log.warning("eye failed on %s", p, exc_info=True)
                out.append(None)
        return out

    def top(self, scores: dict[str, float], n: int = 12) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


_engine: Eye | None = None


def engine() -> Eye:
    """Process-wide singleton (two sessions on one GPU fail to allocate - the
    same lesson as EG2's engine(), spec 7.2 [M3])."""
    global _engine
    if _engine is None:
        _engine = Eye()
    return _engine
