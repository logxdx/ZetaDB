"""jina-clip-v2 - split into two single-tower ONNX graphs.

The published export declares input_ids AND pixel_values as required
inputs. Requesting only the text output does NOT prune the vision branch:
ORT still executes it over whatever dummy pixels you supply, which both
wasted the compute and OOMed the 6 GB card on a batch of 512x512 dummies.

So on first use each tower is extracted into its own graph with
onnx.utils.extract_model. The image tower (312 MB) runs on the GPU in pool
A; the text tower (563 MB) runs on the CPU, where it costs no VRAM and
never contends with pool A or B.

Both towers share one embedding space, which is what makes plain-language
search over images work (spec 9.3).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from server.config import load
from server.models.embeddings import pack

log = logging.getLogger(__name__)

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_RES = 512
_MAX_TOK = 512


class JinaClip:
    def __init__(self) -> None:
        cfg = load()
        d = cfg.path("paths.models") / "jina_clip"
        self.dir = d
        self.combined = d / "onnx" / "model_int8.onnx"   # int8 by policy
        self.text_onnx = d / "onnx" / "model_int8_text.onnx"
        self.image_onnx = d / "onnx" / "model_int8_image.onnx"
        self.tok_path = d / "tokenizer.json"
        self.dim = cfg.get("models.jina_clip.dim", 1024)
        self.available = bool(self.combined.exists() and self.tok_path.exists())
        self._img_sess = None
        self._txt_sess = None
        self._tok = None

    def _ensure_split(self) -> None:
        """Extract the two single-tower graphs once, beside the original."""
        if self.text_onnx.exists() and self.image_onnx.exists():
            return
        import onnx.utils
        for dst, ins, outs in ((self.text_onnx, ["input_ids"], ["l2norm_text_embeddings"]),
                               (self.image_onnx, ["pixel_values"], ["l2norm_image_embeddings"])):
            if dst.exists():
                continue
            log.info("extracting %s tower from jina-clip-v2 (one time)", dst.stem.split("_")[-1])
            onnx.utils.extract_model(str(self.combined), str(dst), ins, outs, check_model=False)

    def _session(self, path: Path, gpu: bool):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        have = set(ort.get_available_providers())
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if gpu and "CUDAExecutionProvider" in have else ["CPUExecutionProvider"])
        s = ort.InferenceSession(str(path), so, providers=provs)
        log.info("jina-clip %s tower on %s", path.stem.split("_")[-1], s.get_providers()[0])
        return s

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self.tok_path))
            self._tok.enable_truncation(max_length=_MAX_TOK)
        return self._tok

    def load_model(self, prefer_gpu: bool = True) -> None:
        if not self.available:
            return
        self._ensure_split()
        if self._img_sess is None:
            self._img_sess = self._session(self.image_onnx, prefer_gpu)

    def load_text(self) -> None:
        if not self.available:
            return
        self._ensure_split()
        if self._txt_sess is None:
            # CPU always: the text tower must never compete for the budget.
            self._txt_sess = self._session(self.text_onnx, gpu=False)

    def unload(self) -> None:
        """Only the GPU-resident tower is released; the CPU text tower stays
        so search does not pay a reload on every query."""
        self._img_sess = None

    # --------------------------------------------------------------- towers

    def _prep_image(self, path: Path) -> np.ndarray:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # resize_mode "shortest" then centre crop, per preprocessor_config
            w, h = im.size
            s = _RES / min(w, h)
            im = im.resize((max(_RES, int(round(w * s))), max(_RES, int(round(h * s)))),
                           Image.Resampling.BICUBIC)
            w, h = im.size
            left, top = (w - _RES) // 2, (h - _RES) // 2
            im = im.crop((left, top, left + _RES, top + _RES))
            a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - _MEAN) / _STD
        return a.transpose(2, 0, 1)[None]

    def embed_images(self, paths: list[Path], batch: int = 4) -> list[bytes | None]:
        self.load_model()
        if not self.available:
            return [None] * len(paths)
        out: list[bytes | None] = [None] * len(paths)
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            arrs, idxs = [], []
            for i, p in enumerate(chunk):
                try:
                    arrs.append(self._prep_image(p))
                    idxs.append(start + i)
                except Exception:
                    log.warning("image prep failed: %s", p, exc_info=True)
            if not arrs:
                continue
            px = np.concatenate(arrs, axis=0)
            vecs = self._img_sess.run(None, {"pixel_values": px})[0]
            for slot, gi in enumerate(idxs):
                out[gi] = pack(vecs[slot])
        return out

    def embed_texts(self, texts: list[str], batch: int = 16) -> list[bytes]:
        self.load_text()
        if not self.available:
            return []
        out: list[bytes] = []
        for start in range(0, len(texts), batch):
            chunk = [t[:4000] for t in texts[start:start + batch]]
            enc = [self._tokenizer().encode(t) for t in chunk]
            width = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), width), dtype=np.int64)
            for i, e in enumerate(enc):
                ids[i, :len(e.ids)] = e.ids
            vecs = self._txt_sess.run(None, {"input_ids": ids})[0]
            out.extend(pack(v) for v in vecs)
        return out

    def embed_text(self, text: str) -> bytes:
        v = self.embed_texts([text])
        if not v:
            raise RuntimeError("jina-clip unavailable")
        return v[0]


_engine: JinaClip | None = None


def engine() -> JinaClip:
    """Process-wide singleton.

    The model is 874 MB on the GPU; instantiating it per stage loaded two
    copies and the second InferenceSession failed to allocate. One instance,
    reused by the tagging and embedding stages.
    """
    global _engine
    if _engine is None:
        _engine = JinaClip()
    return _engine
