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
    process. Same class of bug loaded jina-clip twice onto a 6 GB card, and
    EG2 inherits the risk: its vision encoder is the pool A tenant and a
    second CUDA session would fail to allocate just as silently."""
    from server.models.eg2 import engine as eg2_engine
    from server.models.joytag import engine as joytag_engine
    from server.models.ppocr import engine as ppocr_engine

    for factory in (eg2_engine, joytag_engine, ppocr_engine):
        assert factory() is factory(), f"{factory.__module__} is not a singleton"


def test_no_module_constructs_a_model_directly():
    """The singletons only help if nothing bypasses them."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "server"
    offenders = []
    for py in root.rglob("*.py"):
        for n, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r"\b(Eg2|QwenEmbedder|JinaClip|JoyTag|PPOCR)\(\)", line) \
                    and "_engine = " not in line:
                offenders.append(f"{py.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "construct via engine() instead:\n" + "\n".join(offenders)


# --------------------------------------------------------------- schema v3

def test_one_vector_space_at_768d(tmp_path):
    """Three 1024-d spaces became one 768-d space (schema v3).

    Guards the dimension and the kind split together: a vec0 table silently
    accepts nothing and errors on everything if the declared width drifts
    from what pack() produces, and the kind values are what the search API
    exposes as "similar wording" vs "looks similar".
    """
    import numpy as np
    from server.db.conn import init_db
    from server.models.embeddings import pack

    conn = init_db(tmp_path / "v3.db")
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_items'").fetchone()
    assert "768" in ddl["sql"]
    for gone in ("vec_text", "vec_jina"):
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (gone,)).fetchone() is None

    conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                 "VALUES ('note','ready','compose',0,0)")
    iid = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
    for kind in ("item", "visual"):
        conn.execute("INSERT INTO vec_items(item_id, kind, embedding) VALUES (?,?,?)",
                     (iid, kind, pack(np.random.rand(768))))
    # Both kinds coexist for one item, and the filter separates them.
    assert conn.execute("SELECT count(*) AS n FROM vec_items WHERE item_id = ?",
                        (iid,)).fetchone()["n"] == 2
    row = conn.execute(
        "SELECT item_id FROM vec_items WHERE embedding MATCH ? AND k = 1 AND kind = 'visual'",
        (pack(np.random.rand(768)),)).fetchone()
    assert row["item_id"] == iid


def test_v2_vectors_are_dropped_and_requeued(tmp_path):
    """The v2 -> v3 migration cannot convert 1024-d vectors, so it must drop
    them AND requeue the items - otherwise search goes quietly empty with no
    error anywhere, which is the failure mode this whole swap keeps hitting.

    The v2 database is built by taking a real one and putting the old vector
    tables back, rather than hand-rolling a fixture that drifts from the
    actual schema.
    """
    from server.db.conn import connect, init_db

    path = tmp_path / "v2.db"
    conn = init_db(path)
    conn.executescript("""
        DROP TABLE vec_items;
        DROP TABLE vec_vocab;
        CREATE VIRTUAL TABLE vec_text USING vec0(
            item_id INTEGER PRIMARY KEY, embedding FLOAT[1024]);
        CREATE VIRTUAL TABLE vec_jina USING vec0(
            rowid INTEGER PRIMARY KEY, item_id INTEGER, kind TEXT,
            embedding FLOAT[1024]);
        CREATE VIRTUAL TABLE vec_vocab USING vec0(
            term_id INTEGER PRIMARY KEY, embedding FLOAT[1024]);
    """)
    for kind, blob in (("note", None), ("image", "blobs/a.png"), ("image", "blobs/b.png")):
        conn.execute("INSERT INTO items(kind, status, blob_path, source, "
                     "created_at, updated_at) VALUES (?,'ready',?,'compose',0,0)",
                     (kind, blob))
    conn.execute("DELETE FROM ingest_queue")
    conn.close()

    conn = init_db(path)
    for gone in ("vec_text", "vec_jina"):
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (gone,)).fetchone() is None
    assert "768" in conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_vocab'").fetchone()["sql"]

    queued = {r["stage"]: r["n"] for r in conn.execute(
        "SELECT stage, count(*) AS n FROM ingest_queue GROUP BY stage")}
    assert queued == {"embed": 3, "tag_visual": 2}, queued

    # Idempotent: a second startup must not requeue everything again.
    conn.close()
    conn = init_db(path)
    assert conn.execute("SELECT count(*) AS n FROM ingest_queue").fetchone()["n"] == 5
