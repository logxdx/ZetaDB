"""EmbeddingGemma 2 - one model, one space, for text and images.

Replaces jina-clip-v2 (vec_jina) and Qwen3-Embedding-0.6B (vec_text). Every
modality lands in the same 768-d space, which is what retires the spec 9.4
workaround: today an image's caption is embedded a *second* time through
Qwen purely so that one space exists containing both notes and images.

Shape of the ONNX export (onnx-community/embeddinggemma-2-ONNX), which
drives every design choice below:

  vision_encoder.onnx   pixel_values       [B, patches, 768]  -> image_features [tokens, 512]
                        pixel_position_ids [B, patches, 2]
  model.onnx            input_ids, attention_mask,            -> sentence_embedding [B, 768]
                        image_features / video_features / audio_features

Two consequences:

  * Pooling is inside the graph. `sentence_embedding` is the finished
    vector - there is no mean-pool step to get wrong, and no hand-split of
    a combined graph the way jina-clip needed (spec 7.2 [M3]).
  * An image embedding needs BOTH graphs. The vision encoder only emits
    soft tokens; they are spliced into the backbone's sequence at the
    <|image|> placeholders and the backbone produces the vector. There is
    no standalone image tower, so "load the image side only" is not a
    thing here.

Precision: int8 by default, because it is smallest and fastest. [M0]
against the unquantised graphs, cosine agreement with fp32 on identical
input is 0.9954-1.0000 for int8 and 0.9999-1.0000 for fp16, with retrieval
margin 0.212 vs 0.216 - all three are interchangeable on quality, and int8
is 509 MB against fp16's 878 MB.

Google's card warns that fp16 gives NaN or silently degraded embeddings
because the activation range exceeds fp16's dynamic range. That does not
reproduce on THIS export, on CPU or CUDA: onnx-community ships a mixed
precision graph (320 fp16 weight tensors, 7 kept fp32, and ReduceSum / Max
/ Sin / Cos still emitting fp32), so the sensitive ops never run in fp16.
The warning applies to a torch_dtype=float16 load, which this is not.

Placement: vision encoder on the GPU (pool A), backbone on the CPU. The
backbone is the only part search needs, and keeping it off the GPU means a
query never waits for pool A to wake - the same reasoning that put jina's
text tower and Qwen on the CPU (spec 7.2). The soft tokens crossing between
them are [~280, 512] floats, about 570 KB, so the split costs nothing.

Audio and video are deliberately not wired up: intake accepts images and
text only, so the 300M audio encoder is never downloaded.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from server.config import load
from server.models.embeddings import pack

log = logging.getLogger(__name__)

# Task instruction prefixes, from the model card's prefix table. These apply
# to TEXT ONLY - images are passed with no prefix at all. Omitting them
# still works but measurably costs precision, and query/document asymmetry
# is the whole point: mixing the two sides measurably hurt retrieval with
# the Qwen embedder this replaced, too.
QUERY_PREFIX = "task: search result | query: "
SIMILARITY_PREFIX = "task: sentence similarity | query: "

# Placeholder understood by format_document(); expanded to the real
# begin-of-image / soft-token / end-of-image run at tokenisation time.
IMAGE_MARK = "<|image|>"

# Fallbacks only; the real values are read from the checkpoint's config.json.
_DEFAULT_IDS = {"bos": 2, "boi": 255999, "image": 258880, "eoi": 258882}


def format_query(text: str) -> str:
    """Query side of an asymmetric retrieval pair."""
    return QUERY_PREFIX + text


def format_document(content: str, title: str | None = None) -> str:
    """Document side. The card requires the title field to always be
    present, with the literal `none` when an item has no title - and
    `prompt_name="Document"` in sentence-transformers does exactly this."""
    t = title.strip() if title and title.strip() else "none"
    return f"title: {t} | text: {content}"


class Eg2:
    def __init__(self, key: str = "eg2") -> None:
        """`key` selects the config block, so an alternate build (the fp32
        reference, say) can be instantiated beside the default one to
        measure what quantisation costs. Everything in the serving path
        uses engine(), which only ever builds the default."""
        cfg = load()
        d = cfg.path("paths.models") / key
        self.key = key
        self.dir = d
        dtype = cfg.get(f"models.{key}.dtype", "quantized")
        sfx = f"_{dtype}" if dtype else ""      # "" selects the fp32 graphs
        self.backbone_path = d / "onnx" / f"model{sfx}.onnx"
        self.vision_path = d / "onnx" / f"vision_encoder{sfx}.onnx"
        self.tok_path = d / "tokenizer.json"
        self.dim = cfg.get(f"models.{key}.dim", 768)
        # Matryoshka: 768 native, truncatable to 512/256/128. 0 = no
        # truncation. Truncating REQUIRES re-normalising, which pack() does.
        self.truncate_dim = cfg.get(f"models.{key}.truncate_dim", 0) or None
        self.soft_tokens = cfg.get(f"models.{key}.image_soft_tokens", 280)
        self.backbone_gpu = cfg.get(f"models.{key}.backbone_device", "cpu") == "cuda"
        self.max_tok = cfg.get(f"models.{key}.max_text_tokens", 8192)
        self.available = bool(self.backbone_path.exists() and self.tok_path.exists())
        self.vision_available = self.available and self.vision_path.exists()
        self._bb = None
        self._vis = None
        self._tok = None
        self._proc = None
        self._ids = dict(_DEFAULT_IDS)
        cfg_json = d / "config.json"
        if cfg_json.exists():
            try:
                c = json.loads(cfg_json.read_text())
                self._ids = {
                    "bos": c.get("text_config", {}).get("bos_token_id", _DEFAULT_IDS["bos"]),
                    "boi": c.get("boi_token_id", _DEFAULT_IDS["boi"]),
                    "image": c.get("image_token_id", _DEFAULT_IDS["image"]),
                    "eoi": c.get("eoi_token_id", _DEFAULT_IDS["eoi"]),
                }
            except Exception:
                log.warning("eg2 config.json unreadable; using default token ids", exc_info=True)

    def describe(self) -> str:
        where = "cuda" if self.backbone_gpu else "cpu"
        d = self.truncate_dim or self.dim
        return f"{self.backbone_path.name} ({where}, {d}d)"

    # ------------------------------------------------------------- sessions

    def _session(self, path: Path, gpu: bool):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        have = set(ort.get_available_providers())
        provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if gpu and "CUDAExecutionProvider" in have else ["CPUExecutionProvider"])
        s = ort.InferenceSession(str(path), so, providers=provs)
        log.info("eg2 %s on %s", path.stem, s.get_providers()[0])
        return s

    def load_backbone(self) -> None:
        if self.available and self._bb is None:
            self._bb = self._session(self.backbone_path, self.backbone_gpu)

    def load_vision(self, prefer_gpu: bool = True) -> None:
        if self.vision_available and self._vis is None:
            self._vis = self._session(self.vision_path, prefer_gpu)

    def unload(self) -> None:
        """Release only the GPU-resident vision encoder. The backbone is on
        the CPU and search needs it on every query, so it stays."""
        self._vis = None
        if self.backbone_gpu:
            self._bb = None

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self.tok_path))
        return self._tok

    def _processor(self):
        """Reference preprocessor, PIL/numpy backend - no torch in the
        serving path. Gemma 4 resizes aspect-preserving to a multiple of
        patch_size * pooling_kernel_size and emits flattened 16x16 patches
        plus their (row, col) ids, which is not something worth
        reimplementing from the config file by hand."""
        if self._proc is None:
            from transformers.models.gemma4.image_processing_pil_gemma4 import (
                Gemma4ImageProcessorPil,
            )
            self._proc = Gemma4ImageProcessorPil(
                patch_size=16,
                pooling_kernel_size=3,
                max_soft_tokens=self.soft_tokens,
                do_convert_rgb=True,
                do_rescale=True,
                rescale_factor=1.0 / 255.0,
                do_normalize=False,   # image_mean 0 / image_std 1 per processor_config
            )
        return self._proc

    # ------------------------------------------------------------- encoding

    def _encode_ids(self, text: str, n_soft: int = 0) -> list[int]:
        """Token ids for `text`, expanding every IMAGE_MARK into the real
        begin/soft-token/end run. Segments are encoded separately because
        the marker is not a token in the text vocabulary."""
        tok = self._tokenizer()
        ids: list[int] = [self._ids["bos"]]
        parts = text.split(IMAGE_MARK)
        for i, part in enumerate(parts):
            if part:
                ids.extend(tok.encode(part, add_special_tokens=False).ids)
            if i < len(parts) - 1:
                ids.append(self._ids["boi"])
                ids.extend([self._ids["image"]] * n_soft)
                ids.append(self._ids["eoi"])
        return ids[:self.max_tok]

    def _run_backbone(self, batch_ids: list[list[int]],
                      image_features: np.ndarray | None = None) -> np.ndarray:
        """Right-pad to a rectangle and run. Unused modality inputs are
        required by the graph, so they are passed as empty [0, 512]."""
        self.load_backbone()
        width = max(len(i) for i in batch_ids)
        ids = np.zeros((len(batch_ids), width), dtype=np.int64)
        mask = np.zeros((len(batch_ids), width), dtype=np.int64)
        for r, row in enumerate(batch_ids):
            ids[r, :len(row)] = row
            mask[r, :len(row)] = 1
        empty = np.zeros((0, 512), dtype=np.float32)
        feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "image_features": empty if image_features is None else image_features,
            "video_features": empty,
            "audio_features": empty,
        }
        out = self._bb.run(["sentence_embedding"], feeds)[0]
        return out[:, :self.truncate_dim] if self.truncate_dim else out

    # --------------------------------------------------------------- public

    def embed_texts(self, texts: list[str], *, is_query: bool = False,
                    batch: int = 8) -> list[bytes]:
        if not self.available or not texts:
            return []
        out: list[bytes] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            prepped = [format_query(t) if is_query else t for t in chunk]
            vecs = self._run_backbone([self._encode_ids(t) for t in prepped])
            out.extend(pack(v) for v in vecs)
        return out

    def embed_text(self, text: str) -> bytes:
        """Query-side embedding, used by search."""
        out = self.embed_texts([text], is_query=True)
        if not out:
            raise RuntimeError("eg2 embedder unavailable")
        return out[0]

    def _vision_pass(self, path: Path) -> np.ndarray | None:
        """Image -> soft tokens. The expensive half, and the only half that
        touches the GPU, so it runs once per image even when both the
        interleaved and the bare vector are wanted."""
        if not self.vision_available:
            return None
        self.load_vision()
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                feat = self._processor()(images=[im], return_tensors="np")
        except Exception:
            log.warning("eg2 image prep failed: %s", path, exc_info=True)
            return None
        px = np.asarray(feat["pixel_values"], dtype=np.float32)
        pos = np.asarray(feat["image_position_ids"], dtype=np.int64)
        if px.ndim == 2:          # [patches, 768] -> [1, patches, 768]
            px, pos = px[None], pos[None]
        return self._vis.run(
            ["image_features"], {"pixel_values": px, "pixel_position_ids": pos})[0]

    def _from_soft_tokens(self, feats: np.ndarray, text: str | None,
                          title: str | None) -> bytes:
        body = IMAGE_MARK if not text else f"{IMAGE_MARK} {text}"
        ids = self._encode_ids(format_document(body, title), int(feats.shape[0]))
        return pack(self._run_backbone([ids], feats)[0])

    def embed_image_pair(self, path: Path, text: str | None = None,
                         title: str | None = None) -> tuple[bytes | None, bytes | None]:
        """(item, visual) for one image, from a single vision pass.

        `item` interleaves the image with its own caption and OCR text at
        the <|image|> placeholder. [M0] on 8 captioned images that is worth
        0.291 of margin against 0.224 for the image alone - the text says
        things the pixels do not, and both end up in one vector instead of
        the three rows spec 9.3 used to write.

        `visual` is the image with no text at all. Zero-shot tag scoring
        must use it: scoring the term "chart" against a vector that already
        contains the image's OCR text would match any screenshot whose text
        merely says "chart" (spec 7.2, and the reason requires_text exists).

        The second vector costs one extra backbone pass on the CPU, not a
        second vision pass - the soft tokens are computed once and reused.
        """
        feats = self._vision_pass(path)
        if feats is None:
            return None, None
        item = self._from_soft_tokens(feats, text, title) if text else None
        visual = self._from_soft_tokens(feats, None, None)
        return (item or visual), visual

    def embed_image(self, path: Path, text: str | None = None,
                    title: str | None = None) -> bytes | None:
        """One vector for an image, optionally interleaved with its text."""
        feats = self._vision_pass(path)
        return None if feats is None else self._from_soft_tokens(feats, text, title)


_engine: Eg2 | None = None


def engine() -> Eg2:
    """Process-wide singleton.

    Same reason as jina-clip (spec 7.2 [M3]): the tagging stage and the
    embed stage each used to construct their own model, the second CUDA
    session failed to allocate, and the only visible symptom was a silent
    drop in the term-score count.
    """
    global _engine
    if _engine is None:
        _engine = Eg2()
    return _engine
