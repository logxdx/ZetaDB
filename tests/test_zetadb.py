"""ZetaDB: the model engine, Eye, the v4 migration, and audio/video moments.

None of these load a model or touch the GPU: they pin down the arithmetic and
the plumbing that a model run would otherwise only exercise indirectly.
"""
from __future__ import annotations

import json
import sqlite3
import struct

import numpy as np
import pytest


# ------------------------------------------------------------------- engine

def test_engine_pools_are_exclusive_and_evict_registered_models(monkeypatch):
    """Taking pool B must unload every model registered in pool A - the
    whole point of the registry is that nothing has to remember to."""
    from server.engine import core
    from server.models import vram

    monkeypatch.setattr(vram, "available", lambda: False)
    eng = core.Engine()
    calls: list[str] = []
    eng.register(core.ModelSpec("a1", "A", "onnx", unload=lambda: calls.append("a1")))
    eng.register(core.ModelSpec("a2", "A", "onnx", unload=lambda: calls.append("a2")))
    eng.register(core.ModelSpec("vlm", "B", "llama.cpp", unload=lambda: calls.append("vlm")))

    with eng.use("A", "ocr"):
        pass
    assert calls == []
    with eng.use("B", "caption"):
        pass
    assert sorted(calls) == ["a1", "a2"]          # pool A went when B came in
    with eng.use("A", "tag_visual"):
        pass
    assert "vlm" in calls

    # CPU work never takes the GPU lease.
    lease = eng.lease("CPU", "embed")
    assert eng.status()["held"] is None
    eng.release(lease)


def test_ort_quantised_graphs_never_go_to_tensorrt(tmp_path):
    """TensorRT cannot run ORT-only quantisation ops (MatMulInteger, or the
    MatMulNBits that EG2's published int8 graphs use); sending such a graph
    there shatters it into tiny partitions. 'auto' must pick CUDA instead."""
    import onnx
    from onnx import TensorProto, helper

    from server.engine import onnx as eo

    def graph(op: str, path):
        if op == "MatMulInteger":
            node = helper.make_node("MatMulInteger", ["a", "b"], ["y"])
            ins = [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 2]),
                   helper.make_tensor_value_info("b", TensorProto.UINT8, [2, 2])]
            out = helper.make_tensor_value_info("y", TensorProto.INT32, [1, 2])
        else:
            node = helper.make_node("MatMul", ["a", "b"], ["y"])
            ins = [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2]),
                   helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 2])]
            out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        onnx.save(helper.make_model(helper.make_graph([node], "g", ins, [out])), str(path))
        return path

    q = graph("MatMulInteger", tmp_path / "q.onnx")
    f = graph("MatMul", tmp_path / "f.onnx")
    assert eo.is_ort_quantised(q) and not eo.is_ort_quantised(f)
    assert eo.backend_for(q, "auto") == "onnx-cuda"
    assert eo.backend_for(q, "onnx-cpu") == "onnx-cpu"      # explicit wins


def test_trt_profile_max_batch():
    from server.engine import Profile, max_batch, register_profile
    register_profile("unit_test", Profile(min="x:1x3", opt="x:4x3", max="x:8x3"))
    assert max_batch("unit_test") == 8
    assert max_batch("never_registered") > 1000


# --------------------------------------------------------------------- Eye

def _eye_meta():
    return {"tags": {"model": [
        {"id": "screenshot", "output": "type", "index": 2,
         "curve": [[0.0, 0.0], [0.42, 0.5], [0.54, 0.9], [1.0, 1.0]], "calibrated": True},
        {"id": "reference", "output": "type", "indices": [2, 3],
         "curve": [[0.0, 0.0], [0.77, 0.5], [0.99, 0.9], [1.0, 1.0]], "calibrated": True},
        {"id": "tree", "output": "scenes", "index": 4,
         "curve": [[0.0, 0.0], [0.6, 0.5], [1.0, 0.85]], "calibrated": False},
        {"id": "rotated", "output": "rotation", "turned": True,
         "curve": [[0.0, 0.0], [0.76, 0.5], [0.93, 0.9], [1.0, 1.0]], "calibrated": True},
    ]}}


def test_eye_calibration_curves():
    """The curve maps raw head output to a confidence where 0.5 is Desert
    Ant's recommended cut. Summed and 'turned' tags read the head their own
    way - getting either wrong tags every screenshot as a reference shot or
    every photo as rotated."""
    from server.models.eye import load_tags

    tags = load_tags(_eye_meta())
    heads = {"type": np.zeros(16, np.float32), "scenes": np.zeros(565, np.float32),
             "rotation": np.array([0.9, 0.05, 0.03, 0.02], np.float32)}
    heads["type"][2], heads["type"][3] = 0.42, 0.30
    heads["scenes"][4] = 1.0

    t = tags["screenshot"]
    assert t.confidence(t.raw(heads)) == pytest.approx(0.5, abs=1e-6)
    r = tags["reference"]
    assert r.raw(heads) == pytest.approx(0.72, abs=1e-6)        # 0.42 + 0.30
    tree = tags["tree"]
    assert tree.confidence(tree.raw(heads)) == pytest.approx(0.85)  # uncalibrated tops out
    rot = tags["rotated"]
    assert rot.raw(heads) == pytest.approx(0.1, abs=1e-6)       # 1 - P(upright)
    assert rot.confidence(rot.raw(heads)) < 0.5


def test_eye_preparation_matches_the_documented_recipe(tmp_path):
    from PIL import Image

    from server.models.eye import prepare
    p = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600), (255, 0, 0)).save(p)
    x = prepare(p)
    assert x.shape == (1, 3, 384, 384) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0                      # no mean/std: in-model
    assert x[0, 0].mean() == pytest.approx(1.0) and x[0, 1].mean() == pytest.approx(0.0)


# ------------------------------------------------------------ v4 migration

def _downgrade_to_v3(conn: sqlite3.Connection) -> None:
    """Put a fresh v4 database back into the v3 shape: JoyTag columns, and an
    items table whose CHECK knows only notes and images."""
    from server.db.conn import _items_ddl
    ddl = (_items_ddl("items_v3")
           .replace("('note','image','audio','video')", "('note','image')"))
    ddl = "\n".join(ln for ln in ddl.splitlines() if "duration" not in ln)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(ddl)
    conn.execute("DROP TABLE items")
    conn.execute("ALTER TABLE items_v3 RENAME TO items")
    conn.execute("ALTER TABLE vocab_terms RENAME COLUMN eye_alias TO joytag_alias")
    conn.execute("ALTER TABLE tag_scores RENAME COLUMN eye TO joytag")
    conn.execute("PRAGMA foreign_keys = ON")


def test_v3_to_v4_migration(tmp_path):
    from server.db.conn import init_db

    path = tmp_path / "v3.db"
    conn = init_db(path)
    _downgrade_to_v3(conn)
    ids = []
    for kind, blob in (("note", None), ("image", "blobs/a.png"), ("image", "blobs/b.png")):
        ids.append(conn.execute(
            "INSERT INTO items(kind, status, title, blob_path, source, created_at, updated_at) "
            "VALUES (?,'ready','t',?,'compose',0,0)", (kind, blob)).lastrowid)
    tid = conn.execute("INSERT INTO vocab_terms(term, joytag_alias, created_at) "
                       "VALUES ('tree','tree',0)").lastrowid
    conn.execute("INSERT INTO tag_scores(item_id, term_id, joytag) VALUES (?,?,0.91)",
                 (ids[1], tid))
    conn.execute("INSERT INTO item_tags(item_id, term_id, confidence, source, state, created_at) "
                 "VALUES (?,?,1.0,'human','accepted',0)", (ids[1], tid))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                     "VALUES ('video','ready','upload',0,0)")
    conn.execute("DELETE FROM ingest_queue")
    conn.close()

    conn = init_db(path)
    # Ids survive the rebuild - FTS and every foreign key depend on them.
    assert [r["id"] for r in conn.execute("SELECT id FROM items ORDER BY id")] == ids
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)")}
    assert "duration" in cols
    conn.execute("INSERT INTO items(kind, status, duration, source, created_at, updated_at) "
                 "VALUES ('video','processing',12.5,'upload',0,0)")
    # JoyTag's values are gone; the human tag is not.
    assert conn.execute("SELECT eye_alias FROM vocab_terms WHERE id = ?",
                        (tid,)).fetchone()["eye_alias"] is None
    assert conn.execute("SELECT eye FROM tag_scores").fetchone()["eye"] is None
    assert conn.execute("SELECT source FROM item_tags").fetchone()["source"] == "human"
    queued = {r["stage"]: r["n"] for r in conn.execute(
        "SELECT stage, count(*) AS n FROM ingest_queue GROUP BY stage")}
    assert queued == {"tag_visual": 2}, queued
    assert conn.execute("SELECT value FROM meta WHERE key='vocab_reseed'").fetchone()
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()

    conn.close()
    conn = init_db(path)                                  # idempotent
    assert conn.execute("SELECT count(*) AS n FROM ingest_queue").fetchone()["n"] == 2
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'"
                        ).fetchone()["value"] == "4"


# --------------------------------------------------------- EG2 token layout

def test_eg2_audio_and_video_token_layout():
    """The backbone splices one feature row per placeholder token, so the
    layout must be exactly the processor's: per video frame <boi> <video>*n
    <eoi>, and <boa> <audio>*n <eoa>. A count off by one is a runtime
    shape error at best and a silently shifted embedding at worst."""
    from server.models.eg2 import AUDIO_MARK, VIDEO_MARK, Eg2

    e = Eg2()
    if not e.tok_path.exists():
        pytest.skip("eg2 tokenizer not downloaded")
    ids = e._ids
    v = e._encode_ids(VIDEO_MARK, video_frames=[[3, 2]])
    assert v == [ids["bos"], ids["boi"], *[ids["video"]] * 3, ids["eoi"],
                 ids["boi"], *[ids["video"]] * 2, ids["eoi"]]
    a = e._encode_ids(AUDIO_MARK, audio_tokens=[4])
    assert a == [ids["bos"], ids["boa"], *[ids["audio"]] * 4, ids["eoa"]]
    # The real ids, from the checkpoint - not the fallbacks.
    assert (ids["boa"], ids["eoa"], ids["video"]) == (256000, 258883, 258884)


# ------------------------------------------------------------ media windows

def test_audio_window_plan():
    from server.ingest.media import window_starts
    assert window_starts(12, 30, 25, 240) == [0.0]
    # A tail window is kept only when it adds >= 2 s that no window covers:
    # at 90 s the 75 s window adds 80..90; at 81 s it would add only 1 s.
    assert window_starts(90, 30, 25, 240) == [0, 25, 50, 75]
    assert window_starts(81, 30, 25, 240) == [0, 25, 50]
    long = window_starts(10_000, 30, 25, 240)
    assert len(long) == 240 and long[0] == 0 and long[-1] == pytest.approx(10_000 - 30)


def test_moment_retriever_keeps_one_best_window_per_item(tmp_path):
    from server.db.conn import init_db
    from server.models.embeddings import pack
    from server.search.hybrid import _moments

    conn = init_db(tmp_path / "m.db")
    rng = np.random.default_rng(0)
    q = rng.normal(size=768).astype(np.float32)
    q /= np.linalg.norm(q)

    def near(cos):
        r = rng.normal(size=768).astype(np.float32)
        r -= (r @ q) * q
        r /= np.linalg.norm(r)
        return cos * q + np.sqrt(1 - cos * cos) * r

    item = conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                        "VALUES ('audio','ready','upload',0,0)").lastrowid
    other = conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                         "VALUES ('video','ready','upload',0,0)").lastrowid
    for iid, kind, start, cos in ((item, "sound", 0, 0.62), (item, "sound", 25, 0.81),
                                  (other, "frames", 10, 0.30)):
        sid = conn.execute("INSERT INTO segments(item_id, kind, start_s, end_s) VALUES (?,?,?,?)",
                           (iid, kind, start, start + 30)).lastrowid
        conn.execute("INSERT INTO vec_segments(segment_id, item_id, kind, embedding) "
                     "VALUES (?,?,?,?)", (sid, iid, kind, pack(near(cos))))

    ranked, moments = _moments(conn, pack(q), 10, min_cos=0.5, rel_margin=0.08)
    assert [i for i, _ in ranked] == [item]               # 0.30 is under the floor
    assert moments[item]["start"] == 25.0                  # the better window wins
    assert moments[item]["score"] == pytest.approx(0.81, abs=1e-3)


# ------------------------------------------------------------------- rename

def test_legacy_iwashere_database_is_adopted(tmp_path):
    """ZetaDB was iwashere: an existing data/iwashere.db (and its WAL) must
    become data/zetadb.db on first start, not be left behind for an empty
    library."""
    from server.db.conn import init_db

    old = tmp_path / "iwashere.db"
    conn = init_db(old)
    conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                 "VALUES ('note','ready','compose',0,0)")
    conn.close()

    new = tmp_path / "zetadb.db"
    conn = init_db(new)
    assert not old.exists()
    assert conn.execute("SELECT count(*) AS n FROM items").fetchone()["n"] == 1


# ------------------------------------------------------------------ dimming

def test_match_strength_dims_by_distance_from_the_best():
    """The UI dims a result by `match`. It must follow similarity, not RRF
    rank: a flat tail 0.1 below the answer is mostly dimmed, a keyword hit
    never is, and nothing goes negative."""
    from server.search.hybrid import Hit, match_strength

    best = Hit(1, reasons={"moment": 0.70})
    near = Hit(2, reasons={"img_vec": 0.66})
    tail = Hit(3, reasons={"img_vec": 0.60})
    far = Hit(4, reasons={"text_vec": 0.40})
    word = Hit(5, reasons={"lexical": 1.0, "img_vec": 0.40})
    tag = Hit(6, reasons={"tag": 0.45})
    other_tag = Hit(7, reasons={"tag": 0.90})
    hits = [best, near, tail, far, word, tag, other_tag]
    match_strength(hits, dim_range=0.15)

    assert best.match == 1.0
    assert near.match == pytest.approx(1 - 0.04 / 0.15, abs=1e-3)
    assert tail.match == pytest.approx(1 - 0.10 / 0.15, abs=1e-3)
    assert far.match == 0.0
    assert word.match >= 0.6
    assert tag.match == pytest.approx(0.5) and other_tag.match == 1.0


# ------------------------------------------------------------ tag authority

def test_retagging_never_overturns_a_rejection_or_a_human_tag(tmp_path):
    """[M8] the v4 migration re-tags every image; a tag the user rejected in
    review must stay rejected, and a human tag must stay human."""
    from server.db.conn import init_db
    from server.ingest.stages.tag_visual import TAG_UPSERT

    conn = init_db(tmp_path / "t.db")
    iid = conn.execute("INSERT INTO items(kind, status, source, created_at, updated_at) "
                       "VALUES ('image','ready','upload',0,0)").lastrowid
    tids = [conn.execute("INSERT INTO vocab_terms(term, created_at) VALUES (?,0)",
                         (t,)).lastrowid for t in ("rejected", "human", "machine")]
    conn.execute("INSERT INTO item_tags VALUES (?,?,0.5,'machine','rejected',0)", (iid, tids[0]))
    conn.execute("INSERT INTO item_tags VALUES (?,?,1.0,'human','accepted',0)", (iid, tids[1]))
    conn.execute("INSERT INTO item_tags VALUES (?,?,0.3,'machine','pending',0)", (iid, tids[2]))
    for tid in tids:
        conn.execute(TAG_UPSERT, (iid, tid, 0.95, "machine", "accepted", 1))
    got = {r["term_id"]: (r["source"], r["state"]) for r in conn.execute(
        "SELECT term_id, source, state FROM item_tags")}
    assert got[tids[0]] == ("machine", "rejected")
    assert got[tids[1]] == ("human", "accepted")
    assert got[tids[2]] == ("machine", "accepted")


def test_reprompting_a_term_clears_its_calibration(tmp_path, monkeypatch):
    """A term's mu/sigma describe ITS prompt's cosines; a new prompt must not
    be scored against them ([M8]: indoors/night/portrait were)."""
    import server.search.vocab as V
    from server.db.conn import init_db

    db = tmp_path / "v.db"
    conn = init_db(db)
    conn.execute("INSERT INTO vocab_terms(term, prompt_template, clip_mu, clip_sigma, created_at) "
                 "VALUES ('night','a photo of {term}',0.56,0.03,0), "
                 "('tree','a photo of {term}',0.57,0.04,0)")
    seed = tmp_path / "seed.yaml"
    seed.write_text("- {term: night, prompt: 'a photo taken at night'}\n- {term: tree}\n")
    monkeypatch.setattr(V.writer, "submit", lambda fn, *a, **k: fn(conn, *a, **k))
    V.seed_from_yaml(seed)
    rows = {r["term"]: (r["clip_mu"], r["prompt_template"]) for r in conn.execute(
        "SELECT term, clip_mu, prompt_template FROM vocab_terms")}
    assert rows["night"] == (None, "a photo taken at night")
    assert rows["tree"][0] == pytest.approx(0.57)          # unchanged prompt keeps it


def test_lease_holds_is_thread_scoped():
    """Only the thread holding pool A may borrow GPU sessions under it, so a
    search query running beside ingest stays on the CPU."""
    import threading
    from server.engine.core import Engine
    e = Engine()
    assert not e.holds("A")
    lease = e.lease("A", "t")
    try:
        assert e.holds("A") and not e.holds("B")
        seen = []
        t = threading.Thread(target=lambda: seen.append(e.holds("A")))
        t.start(); t.join()
        assert seen == [False]
    finally:
        e.release(lease)
    assert not e.holds("A")
