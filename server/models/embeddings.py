"""Embedding model access.

Two spaces, never mixed (spec 9.3):
  - Qwen3-Embedding-0.6B  -> vec_text   (notes, and image text when enabled)
  - jina-clip-v2          -> vec_jina   (image tower + its own text tower)

Both are lazily loaded and both return None when their weights are absent,
so search degrades to BM25 + tags instead of failing.
"""
from __future__ import annotations

import logging
import struct
from typing import Callable

log = logging.getLogger(__name__)

_text_embedder: Callable[[str], bytes] | None = None
_jina_text: Callable[[str], bytes] | None = None
_jina_image = None
_probed = {"text": False, "jina": False}


def pack(vec) -> bytes:
    """float32 little-endian blob, L2-normalised, for sqlite-vec."""
    import numpy as np
    a = np.asarray(vec, dtype="float32").ravel()
    n = float(np.linalg.norm(a))
    if n > 0:
        a = a / n
    return struct.pack(f"<{a.size}f", *a.tolist())


def get_text_embedder():
    global _text_embedder
    if _probed["text"]:
        return _text_embedder
    _probed["text"] = True
    try:
        from server.models.qwen_embed import engine as qwen_engine
        emb = qwen_engine()
        if emb.available:
            _text_embedder = emb.embed
            log.info("text embedder ready: %s", emb.describe())
    except Exception:
        log.debug("text embedder unavailable", exc_info=True)
    return _text_embedder


def get_jina_text_embedder():
    global _jina_text
    if _probed["jina"]:
        return _jina_text
    _probed["jina"] = True
    try:
        from server.models.jina_clip import engine as clip_engine
        clip = clip_engine()
        if clip.available:
            _jina_text = clip.embed_text
            log.info("jina-clip text tower ready")
    except Exception:
        log.debug("jina-clip unavailable", exc_info=True)
    return _jina_text


def reset() -> None:
    """Force re-probe after a model download."""
    global _text_embedder, _jina_text
    _text_embedder = _jina_text = None
    _probed["text"] = _probed["jina"] = False


def cosine_from_l2(distance: float) -> float:
    """Convert a vec0 L2 distance to cosine similarity.

    sqlite-vec's default metric for float columns is L2, not cosine. Every
    vector we store is L2-normalised by pack(), so the identity
        d^2 = 2(1 - cos)  =>  cos = 1 - d^2/2
    is exact. Reading `1 - distance` as a cosine (the original bug) reported
    a true 0.671 as 0.189 and silently disabled every similarity threshold.
    """
    return 1.0 - (distance * distance) / 2.0
