"""Content hashing and perceptual hashing for dedupe (spec 10.5)."""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

_CHUNK = 1 << 20


def content_hash(path: Path) -> str:
    """BLAKE2b over file bytes. Exact-duplicate detection.

    blake3 is the spec's choice and is faster, but hashlib.blake2b is in the
    stdlib and this is not the bottleneck - the pipeline is model-bound.
    """
    h = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    norm = unicodedata.normalize("NFKC", text).strip().casefold()
    norm = " ".join(norm.split())
    return hashlib.blake2b(norm.encode(), digest_size=32).hexdigest()


def dhash(img: Image.Image, size: int = 8) -> bytes:
    """64-bit difference hash. Returns 8 bytes, MSB-first per row.

    Stored in vec_phash as bit[64] and queried with vec_bit(); near-duplicates
    are Hamming distance <= dedupe.phash_hamming_max.
    """
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = bytearray(size * size // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))
