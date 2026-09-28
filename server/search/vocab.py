"""Controlled vocabulary: seeding, the write-time merge gate, CLIP
calibration and score fusion (spec 7).

Three defences against tag rot, as specified:
  1. write-time gate    - a proposed term near an existing one is aliased,
                          not added
  2. rest-time merge    - clustering plus co-occurrence proposes merges
  3. human authority    - a human-sourced tag is never touched by a re-run
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from server.config import ROOT, load
from server.db.conn import reader
from server.db.writer import writer
from server.models.embeddings import cosine_from_l2

log = logging.getLogger(__name__)


@dataclass
class Term:
    id: int
    term: str
    joytag_alias: str | None
    prompt_template: str
    threshold: float
    clip_mu: float | None
    clip_sigma: float | None
    requires_text: bool = False

    def prompt(self) -> str:
        return (self.prompt_template or "a photo of {term}").format(term=self.term)


def active_terms(conn: sqlite3.Connection | None = None) -> list[Term]:
    c = conn or reader()
    return [Term(r["id"], r["term"], r["joytag_alias"], r["prompt_template"],
                 r["threshold"], r["clip_mu"], r["clip_sigma"], bool(r["requires_text"]))
            for r in c.execute(
                "SELECT id, term, joytag_alias, prompt_template, threshold, clip_mu, "
                "clip_sigma, requires_text "
                "FROM vocab_terms WHERE state = 'active' ORDER BY term")]


# --------------------------------------------------------------------- seed

def seed_from_yaml(path: Path | None = None) -> int:
    import yaml
    p = path or (ROOT / "vocab" / "seed.yaml")
    if not p.exists():
        return 0
    entries = yaml.safe_load(p.read_text()) or []

    def work(conn: sqlite3.Connection) -> int:
        now, added = int(time.time()), 0
        for e in entries:
            term = str(e["term"]).strip().lower()
            if not term:
                continue
            cur = conn.execute(
                "INSERT INTO vocab_terms(term, kind, joytag_alias, prompt_template, "
                "threshold, requires_text, state, created_at) VALUES (?,?,?,?,?,?, 'active', ?) "
                "ON CONFLICT(term) DO UPDATE SET joytag_alias = excluded.joytag_alias, "
                "prompt_template = excluded.prompt_template, kind = excluded.kind, "
                "requires_text = excluded.requires_text",
                (term, e.get("kind"), e.get("joytag"),
                 e.get("prompt", "a photo of {term}"),
                 load().get("tagging.default_threshold", 0.40),
                 int(bool(e.get("requires_text"))), now))
            if cur.rowcount:
                added += 1
            tid = conn.execute("SELECT id FROM vocab_terms WHERE term = ?", (term,)).fetchone()["id"]
            for alias in e.get("aliases") or []:
                conn.execute("INSERT OR IGNORE INTO vocab_aliases(term_id, alias) VALUES (?,?)",
                             (tid, str(alias).strip().lower()))
        return added

    n = writer.submit(work)
    log.info("vocabulary seeded: %d term(s) from %s", n, p.name)
    return n


def embed_vocab() -> int:
    """Populate vec_vocab so the merge gate and tag-seed expansion work."""
    from server.models.embeddings import get_text_embedder
    emb = get_text_embedder()
    if emb is None:
        return 0
    conn = reader()
    rows = conn.execute(
        "SELECT t.id, t.term FROM vocab_terms t LEFT JOIN vec_vocab v ON v.term_id = t.id "
        "WHERE t.state = 'active' AND v.term_id IS NULL").fetchall()
    if not rows:
        return 0
    from server.models.qwen_embed import engine as qwen_engine
    vecs = qwen_engine().embed_many([r["term"] for r in rows])

    def work(c: sqlite3.Connection) -> None:
        for r, v in zip(rows, vecs, strict=True):
            c.execute("INSERT INTO vec_vocab(term_id, embedding) VALUES (?,?)", (r["id"], v))
    writer.submit(work)
    log.info("embedded %d vocabulary term(s)", len(rows))
    return len(rows)


# ----------------------------------------------------------- write-time gate

def resolve_or_propose(name: str) -> tuple[int | None, str]:
    """Map a model-proposed tag onto the vocabulary.

    Returns (term_id, action) where action is exact | alias | merged |
    review | proposed. Thresholds come from config (spec 7.5).
    """
    cfg = load()
    name = name.strip().lower()
    if not name:
        return None, "empty"
    conn = reader()

    row = conn.execute("SELECT id FROM vocab_terms WHERE term = ? AND state='active'",
                       (name,)).fetchone()
    if row:
        return int(row["id"]), "exact"
    row = conn.execute(
        "SELECT t.id FROM vocab_aliases a JOIN vocab_terms t ON t.id = a.term_id "
        "WHERE a.alias = ? AND t.state='active'", (name,)).fetchone()
    if row:
        return int(row["id"]), "alias"

    from server.models.embeddings import get_text_embedder
    emb = get_text_embedder()
    if emb is not None:
        try:
            from server.models.qwen_embed import engine as qwen_engine
            vec = qwen_engine().embed_many([name])[0]
            near = conn.execute(
                "SELECT term_id, distance FROM vec_vocab WHERE embedding MATCH ? AND k = 1 "
                "ORDER BY distance", (vec,)).fetchone()
            if near:
                cos = cosine_from_l2(float(near["distance"]))
                if cos >= cfg.get("vocab.merge_auto", 0.86):
                    tid = int(near["term_id"])
                    writer.submit(lambda c, t=tid, n=name: c.execute(
                        "INSERT OR IGNORE INTO vocab_aliases(term_id, alias) VALUES (?,?)", (t, n)))
                    return tid, "merged"
                if cos >= cfg.get("vocab.merge_review", 0.72):
                    _propose(name, near_term_id=int(near["term_id"]))
                    return None, "review"
        except Exception:
            log.debug("merge gate embedding unavailable", exc_info=True)

    _propose(name)
    return None, "proposed"


def _propose(name: str, near_term_id: int | None = None) -> None:
    def work(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO vocab_terms(term, state, threshold, created_at) "
            "VALUES (?, 'pending', ?, ?) ON CONFLICT(term) DO NOTHING",
            (name, load().get("tagging.default_threshold", 0.40), int(time.time())))
        conn.execute("INSERT INTO audit_log(ts, action, payload) VALUES (?,?,?)",
                     (int(time.time()), "vocab.proposed",
                      f'{{"term": "{name}", "near": {near_term_id or "null"}}}'))
    writer.submit(work)


# ------------------------------------------------------------- calibration

def calibrate_clip(sample: int | None = None) -> int:
    """Per-term mu/sigma over a sample of images.

    Raw CLIP cosines sit in a compressed, term-dependent band (measured here
    at roughly 0.17-0.36), so one global threshold is meaningless. Each term
    stores its own distribution and scores by z-score (spec 7.3).
    """
    import numpy as np
    import struct
    cfg = load()
    n = sample or cfg.get("tagging.clip_calibration_sample", 500)
    conn = reader()
    rows = conn.execute(
        "SELECT embedding FROM vec_jina WHERE kind = 'image' LIMIT ?", (n,)).fetchall()
    if len(rows) < 8:
        log.info("only %d image vector(s); skipping CLIP calibration", len(rows))
        return 0
    mat = np.stack([np.array(struct.unpack(f"<{len(r['embedding'])//4}f", r["embedding"]))
                    for r in rows])

    from server.models.jina_clip import engine as clip_engine
    clip = clip_engine()
    if not clip.available:
        return 0
    terms = active_terms(conn)
    vecs = clip.embed_texts([t.prompt() for t in terms])
    updates = []
    for t, v in zip(terms, vecs, strict=True):
        tv = np.array(struct.unpack(f"<{len(v)//4}f", v))
        sims = mat @ tv
        updates.append((float(sims.mean()), float(sims.std() or 1e-3), t.id))

    def work(c: sqlite3.Connection) -> None:
        for mu, sigma, tid in updates:
            c.execute("UPDATE vocab_terms SET clip_mu = ?, clip_sigma = ? WHERE id = ?",
                      (mu, sigma, tid))
    writer.submit(work)
    log.info("calibrated %d term(s) over %d image vector(s)", len(updates), len(rows))
    return len(updates)


def clip_score(term: Term, cosine: float) -> float:
    """Calibrated z-score squashed to 0-1.

    The squash is centred on z = clip_z_center (default 2.0), not on 0. A
    tag should apply when an image is an OUTLIER for that term, not when
    it is merely typical - and logistic(0) = 0.5, so centring on the mean
    meant an average term already cleared a 0.40 threshold. Measured on a
    29-photo corpus that accepted 53 of 54 terms for a single image.
    """
    if term.clip_mu is None or not term.clip_sigma:
        return max(0.0, min(1.0, (cosine - 0.15) / 0.25))   # uncalibrated fallback
    cfg = load()
    # A relative score alone says "unusually X for THIS corpus", not "is X".
    # On a homogeneous library every term finds its outliers regardless of
    # whether the concept is present, so an absolute floor is needed too.
    if cosine < cfg.get("tagging.clip_abs_floor", 0.22):
        return 0.0
    z0 = cfg.get("tagging.clip_z_center", 2.0)
    slope = cfg.get("tagging.clip_z_slope", 1.0) or 1.0
    z = (cosine - term.clip_mu) / term.clip_sigma
    return 1.0 / (1.0 + math.exp(-(z - z0) / slope))


def fuse(joytag: float | None, clip: float | None, vlm: float | None) -> float:
    cfg = load()
    w = {"j": cfg.get("tagging.w_joytag", 0.45), "c": cfg.get("tagging.w_clip", 0.30),
         "v": cfg.get("tagging.w_vlm", 0.25)}
    num = den = 0.0
    for key, val in (("j", joytag), ("c", clip), ("v", vlm)):
        if val is not None:
            num += w[key] * val
            den += w[key]
    return num / den if den else 0.0


def apply_feedback(term_id: int, accepted: bool) -> None:
    """Per-term threshold adapts from accept/reject (spec 7.4)."""
    cfg = load()
    step = cfg.get("tagging.threshold_step", 0.02)

    def work(conn: sqlite3.Connection) -> None:
        delta = -step if accepted else step
        conn.execute(
            "UPDATE vocab_terms SET threshold = max(0.15, min(0.85, threshold + ?)), "
            "accept_count = accept_count + ?, reject_count = reject_count + ? WHERE id = ?",
            (delta, 1 if accepted else 0, 0 if accepted else 1, term_id))
    writer.submit(work)


# ------------------------------------------------------------ merge at rest

def merge_suggestions(limit: int = 20) -> list[dict]:
    """Embedding-near pairs plus high co-occurrence pairs."""
    import struct
    import numpy as np
    conn = reader()
    cfg = load()
    out: list[dict] = []

    rows = conn.execute(
        "SELECT v.term_id, v.embedding, t.term FROM vec_vocab v "
        "JOIN vocab_terms t ON t.id = v.term_id WHERE t.state='active'").fetchall()
    if len(rows) > 1:
        mat = np.stack([np.array(struct.unpack(f"<{len(r['embedding'])//4}f", r["embedding"]))
                        for r in rows])
        sim = mat @ mat.T
        np.fill_diagonal(sim, -1)
        thresh = 1.0 - cfg.get("vocab.cluster_distance", 0.18)
        for i, jx in zip(*np.where(sim >= thresh)):
            if i < jx:
                out.append({"reason": "embedding", "a": rows[i]["term"], "b": rows[jx]["term"],
                            "a_id": rows[i]["term_id"], "b_id": rows[jx]["term_id"],
                            "score": round(float(sim[i, jx]), 3)})

    co = conn.execute(
        "SELECT a.term_id AS a, b.term_id AS b, count(*) AS n FROM item_tags a "
        "JOIN item_tags b ON a.item_id = b.item_id AND a.term_id < b.term_id "
        "WHERE a.state='accepted' AND b.state='accepted' GROUP BY a.term_id, b.term_id "
        "HAVING n >= 3").fetchall()
    names = {r["id"]: r["term"] for r in conn.execute("SELECT id, term FROM vocab_terms")}
    counts = {r["term_id"]: r["n"] for r in conn.execute(
        "SELECT term_id, count(*) AS n FROM item_tags WHERE state='accepted' GROUP BY term_id")}
    frac = cfg.get("vocab.cooccurrence_merge", 0.60)
    for r in co:
        smaller = min(counts.get(r["a"], 1), counts.get(r["b"], 1))
        if smaller and r["n"] / smaller >= frac:
            out.append({"reason": "co-occurrence", "a": names.get(r["a"]), "b": names.get(r["b"]),
                        "a_id": r["a"], "b_id": r["b"],
                        "score": round(r["n"] / smaller, 3)})
    return sorted(out, key=lambda d: -d["score"])[:limit]


def merge(loser_id: int, winner_id: int) -> None:
    def work(conn: sqlite3.Connection) -> None:
        loser = conn.execute("SELECT term FROM vocab_terms WHERE id = ?", (loser_id,)).fetchone()
        if loser is None:
            return
        conn.execute(
            "INSERT INTO item_tags(item_id, term_id, confidence, source, state, created_at) "
            "SELECT item_id, ?, confidence, source, state, created_at FROM item_tags "
            "WHERE term_id = ? ON CONFLICT(item_id, term_id) DO NOTHING", (winner_id, loser_id))
        conn.execute("DELETE FROM item_tags WHERE term_id = ?", (loser_id,))
        conn.execute("INSERT OR IGNORE INTO vocab_aliases(term_id, alias) VALUES (?,?)",
                     (winner_id, loser["term"]))
        conn.execute("UPDATE vocab_terms SET state='merged', merged_into=? WHERE id = ?",
                     (winner_id, loser_id))
        conn.execute("DELETE FROM vec_vocab WHERE term_id = ?", (loser_id,))
        conn.execute("INSERT INTO audit_log(ts, action, payload) VALUES (?,?,?)",
                     (int(time.time()), "vocab.merged",
                      f'{{"loser": {loser_id}, "winner": {winner_id}}}'))
    writer.submit(work)
