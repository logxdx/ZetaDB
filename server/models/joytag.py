"""JoyTag - per-tag applicability scoring against our own vocabulary.

JoyTag is NOT used as a tag generator with a global cutoff. It emits an
independent sigmoid score for each of its 5813 Danbooru tags, so we look up
only the terms our vocabulary aliases into and read their confidence, with
a per-term threshold that adapts from feedback (spec 7.1).

Terms with no Danbooru alias - "screenshot", "wireframe", "invoice" - are
not scorable here at all and fall through to the CLIP zero-shot path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from server.config import load

log = logging.getLogger(__name__)

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_RES = 448


class JoyTag:
    def __init__(self) -> None:
        d = load().path("paths.models") / "joytag"
        from server.models._native import pick_onnx
        self.onnx = pick_onnx(d, stem_hint="model")
        self.tags_file = d / "top_tags.txt"
        self.available = bool(self.onnx) and self.tags_file.exists()
        self.tags: list[str] = []
        self.index: dict[str, int] = {}
        self._sess = None

    def load_model(self) -> None:
        if self._sess is not None or not self.available:
            return
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        have = set(ort.get_available_providers())
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in have else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(str(self.onnx), so, providers=provs)
        self.tags = self.tags_file.read_text().splitlines()
        self.index = {t: i for i, t in enumerate(self.tags)}
        log.info("JoyTag ready on %s · %s · %d tags",
                 self._sess.get_providers()[0], self.onnx.name, len(self.tags))

    def unload(self) -> None:
        self._sess = None

    def covers(self, danbooru_tag: str) -> bool:
        return danbooru_tag in self.index

    def _prep(self, path: Path) -> np.ndarray:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # Pad to square on white, then resize - JoyTag's own recipe.
            side = max(im.size)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
            canvas = canvas.resize((_RES, _RES), Image.Resampling.BICUBIC)
            a = np.asarray(canvas, dtype=np.float32) / 255.0
        return ((a - _MEAN) / _STD).transpose(2, 0, 1)

    def score_batch(self, paths: list[Path], batch: int = 8) -> list[dict[str, float] | None]:
        """Returns the full tag->probability map per image, or None on failure."""
        self.load_model()
        if not self.available:
            return [None] * len(paths)
        out: list[dict[str, float] | None] = [None] * len(paths)
        name = self._sess.get_inputs()[0].name
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            arrs, idxs = [], []
            for i, p in enumerate(chunk):
                try:
                    arrs.append(self._prep(p))
                    idxs.append(start + i)
                except Exception:
                    log.warning("joytag prep failed: %s", p, exc_info=True)
            if not arrs:
                continue
            logits = self._sess.run(None, {name: np.stack(arrs)})[0]
            probs = 1.0 / (1.0 + np.exp(-logits))
            for slot, gi in enumerate(idxs):
                out[gi] = {self.tags[i]: float(probs[slot, i]) for i in range(len(self.tags))}
        return out

    def top(self, scores: dict[str, float], n: int = 12) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


_engine: JoyTag | None = None


def engine() -> JoyTag:
    global _engine
    if _engine is None:
        _engine = JoyTag()
    return _engine
