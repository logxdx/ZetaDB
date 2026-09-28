"""Vector-space regressions (spec 9.3, 10)."""
from __future__ import annotations

import math
import sqlite3
import struct

import pytest

from server.models.embeddings import cosine_from_l2, pack


def _norm(v):
    import numpy as np
    a = np.asarray(v, dtype="float32")
    return a / np.linalg.norm(a)


def test_pack_normalises():
    b = pack([3.0, 4.0, 0.0, 0.0])
    vals = struct.unpack("<4f", b)
    assert math.isclose(sum(v * v for v in vals), 1.0, rel_tol=1e-5)


@pytest.mark.parametrize("cos", [1.0, 0.9, 0.671, 0.5, 0.0, -0.5])
def test_cosine_from_l2_roundtrip(cos):
    """Regression: sqlite-vec's default metric is L2, not cosine. Reading
    `1 - distance` as a cosine reported a true 0.671 as 0.189, which
    silently disabled every similarity threshold in search and edge
    seeding."""
    d = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
    assert math.isclose(cosine_from_l2(d), cos, abs_tol=1e-6)


def test_cosine_conversion_against_sqlite_vec():
    """The identity must hold against the real extension, not just algebra."""
    import numpy as np
    import sqlite_vec

    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    db.execute("CREATE VIRTUAL TABLE v USING vec0(id INTEGER PRIMARY KEY, e FLOAT[4])")

    a = _norm([1.0, 0.2, 0.0, 0.0])
    b = _norm([0.8, 0.6, 0.0, 0.0])
    db.execute("INSERT INTO v(id, e) VALUES (1, ?)", (pack(b),))
    row = db.execute("SELECT distance FROM v WHERE e MATCH ? AND k = 1", (pack(a),)).fetchone()

    expected = float(np.dot(a, b))
    assert math.isclose(cosine_from_l2(row["distance"]), expected, abs_tol=1e-5)
    # and the naive reading really is wrong, so the test has teeth
    assert not math.isclose(1.0 - row["distance"], expected, abs_tol=1e-2)


def test_model_wrappers_are_singletons():
    """Regression: each QwenEmbedder owns a llama-server subprocess, and
    vocab.resolve_or_propose() constructed one per proposed tag name. A
    single captioning batch then tried to spawn ~40 servers and killed the
    process. Same class of bug loaded jina-clip twice onto a 6 GB card."""
    from server.models.jina_clip import engine as clip_engine
    from server.models.joytag import engine as joytag_engine
    from server.models.ppocr import engine as ppocr_engine
    from server.models.qwen_embed import engine as qwen_engine

    for factory in (clip_engine, joytag_engine, ppocr_engine, qwen_engine):
        assert factory() is factory(), f"{factory.__module__} is not a singleton"


def test_no_module_constructs_a_model_directly():
    """The singletons only help if nothing bypasses them."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "server"
    offenders = []
    for py in root.rglob("*.py"):
        for n, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r"\b(QwenEmbedder|JinaClip|JoyTag|PPOCR)\(\)", line) \
                    and "_engine = " not in line:
                offenders.append(f"{py.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "construct via engine() instead:\n" + "\n".join(offenders)
