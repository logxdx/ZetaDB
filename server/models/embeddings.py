"""Embedding model access.

ONE space (spec 9.3). EmbeddingGemma 2 embeds notes, images and vocabulary
terms into the same 768-d vectors, so any of them can be compared with any
other. The two-space arrangement this replaced - Qwen3-Embedding for text,
jina-clip-v2 for images - could not compare a note with an image at all,
which is what spec 9.4's duplicate caption embedding existed to paper over.

The embedder is lazily loaded and returns None when its weights are absent,
so search degrades to BM25 + tags instead of failing.
"""
from __future__ import annotations

import logging
import struct
from typing import Callable

log = logging.getLogger(__name__)

_embedder: Callable[[str], bytes] | None = None
_probed = False


def pack(vec) -> bytes:
    """float32 little-endian blob, L2-normalised, for sqlite-vec.

    Normalisation is not optional and not only for cosine: Matryoshka
    truncation slices a unit vector, and a slice of a unit vector is not
    unit length. Skipping the re-normalise degrades ranking silently -
    plausible scores, wrong order.
    """
    import numpy as np
    a = np.asarray(vec, dtype="float32").ravel()
    n = float(np.linalg.norm(a))
    if n > 0:
        a = a / n
    return struct.pack(f"<{a.size}f", *a.tolist())


def get_embedder():
    """Query-side text embedder: applies the SearchQuery task prefix.

    Document-side text and images go through server.models.eg2 directly -
    they need the asymmetric document prefix, and images need no prefix at
    all, which a single `str -> bytes` callable cannot express.
    """
    global _embedder, _probed
    if _probed:
        return _embedder
    _probed = True
    try:
        from server.models.eg2 import engine as eg2_engine
        e = eg2_engine()
        if e.available:
            _embedder = e.embed_text
            log.info("embedder ready: %s", e.describe())
    except Exception:
        log.debug("embedder unavailable", exc_info=True)
    return _embedder


# Kept as the old name so callers that only want "embed a query" keep
# working; there is no longer a second space to distinguish it from.
get_text_embedder = get_embedder


def reset() -> None:
    """Force re-probe after a model download."""
    global _embedder, _probed
    _embedder = None
    _probed = False


def cosine_from_l2(distance: float) -> float:
    """Convert a vec0 L2 distance to cosine similarity.

    sqlite-vec's default metric for float columns is L2, not cosine. Every
    vector we store is L2-normalised by pack(), so the identity
        d^2 = 2(1 - cos)  =>  cos = 1 - d^2/2
    is exact. Reading `1 - distance` as a cosine (the original bug) reported
    a true 0.671 as 0.189 and silently disabled every similarity threshold.
    """
    return 1.0 - (distance * distance) / 2.0
