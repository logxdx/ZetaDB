"""
LanceDB-backed vector index.

Each *collection* (LanceDB table) holds any mix of text, code, images, audio and
video. Files are stored as **segments** (text chunks, audio windows, video
windows, images), one row and one vector per segment, all in the same
EmbeddingGemma 2 space. Search ranks *files* by their best segment and reports
where in the file the match is.

Ranking:

* cosine similarities are turned into per-modality z-scores — text vectors
  score systematically higher than image/audio vectors for the same topic, so
  raw cosine would let notes crowd out photos in an "everything" search;
* a keyword-coverage bonus (LanceDB full-text search over file names, folder
  names and text) rewards results that literally contain the query words —
  exact names and numbers matter;
* query words naming a kind ("photo of…", "video of…") softly boost that kind.
"""

import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union

import lancedb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

from .config import (
    ANN_INDEX_THRESHOLD,
    DEFAULT_SEARCH_K,
    EMBEDDING_DIMENSION,
    INDEX_FILE,
    MODALITIES,
    RANK_MEANING_WEIGHT,
    RANK_MIN_STATS,
    RANK_STATS_SAMPLE,
    RANK_WORDS_WEIGHT,
    WEAK_RESULT_RATIO,
)

logger = logging.getLogger(__name__)

Modality = Literal["text", "image", "audio", "video"]

# File-level columns (repeated on every segment of a file)
FILE_COLUMNS = ["file_id", "modality", "path", "name", "text", "size", "width", "height", "duration", "added_at"]
# Segment-level columns
SEGMENT_COLUMNS = ["id", "segment", "kind", "start", "end", "snippet"]
META_COLUMNS = SEGMENT_COLUMNS + FILE_COLUMNS

_TABLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,63}$")

STOPWORDS = set(
    """a an and are as at be by for from has have in is it its of on or that the this to was were
    with what which who where when how i me my we our you your show find search look looking for
    any some all one two""".split()
)
# Query words that name a kind of result: soft boost instead of a hard filter
KIND_WORDS = {
    "image": {"photo", "photos", "picture", "pictures", "image", "images", "pic", "pics", "screenshot", "screenshots"},
    "video": {"video", "videos", "clip", "clips", "movie", "movies", "footage", "film"},
    "audio": {"audio", "song", "songs", "music", "sound", "sounds", "recording", "recordings", "podcast", "track"},
    "text": {"note", "notes", "doc", "docs", "document", "documents", "text", "code", "file", "article"},
}
_ALL_KIND_WORDS = set().union(*KIND_WORDS.values())
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def validate_table_name(name: str) -> str:
    name = (name or "").strip()
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            "Collection names must start with a letter or digit and contain only "
            "letters, digits, '_', '-' or '.' (max 64 characters)."
        )
    return name


def _quote(value: str) -> str:
    """Quote a string literal for a LanceDB SQL filter."""
    return "'" + str(value).replace("'", "''") + "'"


def _normalize_word(word: str) -> str:
    word = word.lower()
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def query_words(text: Optional[str]) -> List[str]:
    """Content words of a query (lower-cased, light plural stemming, no stopwords / kind words)."""
    if not text:
        return []
    words = []
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 2 or w in STOPWORDS or w in _ALL_KIND_WORDS:
            continue
        w = _normalize_word(w)
        if w not in words:
            words.append(w)
    return words


def kind_hints(text: Optional[str]) -> set:
    words = set(_WORD_RE.findall((text or "").lower()))
    return {m for m, kw in KIND_WORDS.items() if words & kw}


class VectorObject(BaseModel):
    """Schema for one segment stored in LanceDB."""

    id: str
    file_id: str
    segment: int = 0
    kind: str = "chunk"  # chunk | image | sound | frames
    start: float = 0.0
    end: float = 0.0
    snippet: str = ""
    keywords: str = ""
    vector: List[float]
    modality: Modality = "text"
    path: str = ""
    name: str = ""
    text: str = ""  # preview of the file (text documents)
    size: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LanceDBIndex:
    """Wraps lancedb for storing segments and ranking files."""

    def __init__(
        self,
        index_path: Union[str, Path] = INDEX_FILE,
        embedding_dim: int = EMBEDDING_DIMENSION,
    ):
        self.db = lancedb.connect(str(index_path), read_consistency_interval=timedelta(seconds=1))
        self.embedding_dim = embedding_dim
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("file_id", pa.string()),
                pa.field("segment", pa.int32()),
                pa.field("kind", pa.string()),
                pa.field("start", pa.float64()),
                pa.field("end", pa.float64()),
                pa.field("snippet", pa.string()),
                pa.field("keywords", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
                pa.field("modality", pa.string()),
                pa.field("path", pa.string()),
                pa.field("name", pa.string()),
                pa.field("text", pa.string()),
                pa.field("size", pa.int64()),
                pa.field("width", pa.int32()),
                pa.field("height", pa.int32()),
                pa.field("duration", pa.float64()),
                pa.field("added_at", pa.timestamp("us", tz="UTC")),
            ]
        )
        self._stats_cache: Dict[str, tuple] = {}
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------- tables
    def get_tables(self) -> List[str]:
        if hasattr(self.db, "list_tables"):
            result = self.db.list_tables()
            names = getattr(result, "tables", result)
        else:  # older lancedb
            names = self.db.table_names()
        return sorted(names)

    def has_table(self, table_name: str) -> bool:
        return table_name in self.get_tables()

    def _open(self, table_name: str):
        if not self.has_table(table_name):
            raise KeyError(f"Collection '{table_name}' does not exist.")
        table = self.db.open_table(table_name)
        if "file_id" not in table.schema.names:
            raise ValueError(f"Collection '{table_name}' uses an older format. Re-index it into a new collection.")
        dim = table.schema.field("vector").type.list_size
        if dim != self.embedding_dim:
            raise ValueError(
                f"Collection '{table_name}' stores {dim}-d vectors but the model "
                f"produces {self.embedding_dim}-d vectors. Re-index it into a new collection."
            )
        return table

    def create_table(self, table_name: str):
        table_name = validate_table_name(table_name)
        if self.has_table(table_name):
            return self._open(table_name)
        return self.db.create_table(table_name, schema=self.schema)

    def drop_table(self, table_name: str) -> None:
        self.db.drop_table(table_name)
        self._stats_cache.pop(table_name, None)

    # -------------------------------------------------------------- write
    def insert_documents(self, docs: Sequence[Dict], table_name: str) -> int:
        """Insert segment rows (dicts matching VectorObject)."""
        if not docs:
            return 0
        table = self.create_table(table_name)
        table.add(pa.Table.from_pylist(list(docs), schema=self.schema))
        return len(docs)

    def optimize(self, table_name: str) -> None:
        """
        Compact fragments left by inserts, (re)build the keyword index and fold
        new rows into the vector index. An ANN index is created once the
        collection is large enough that brute-force search starts to cost time.
        """
        table = self._open(table_name)
        try:
            table.optimize()
        except Exception as e:  # optimisation is best-effort
            logger.warning(f"Could not optimize '{table_name}': {e}")
        try:
            table.create_fts_index("keywords", replace=True, use_tantivy=False, stem=True, ascii_folding=True)
        except Exception as e:
            logger.warning(f"Could not build keyword index for '{table_name}': {e}")
        try:
            rows = table.count_rows()
            has_index = any("vector" in (getattr(i, "columns", None) or []) for i in table.list_indices())
            if rows >= ANN_INDEX_THRESHOLD and not has_index:
                logger.info(f"Building ANN index for '{table_name}' ({rows} segments) ...")
                table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_HNSW_SQ")
        except Exception as e:
            logger.warning(f"Could not build ANN index for '{table_name}': {e}")

    def delete_item(self, table_name: str, file_id: str) -> None:
        self._open(table_name).delete(f"file_id = {_quote(file_id)}")

    # --------------------------------------------------------------- read
    @staticmethod
    def _file_row(row: Dict) -> Dict:
        """Shape a segment row as a file-level item (``id`` is the file id)."""
        out = {k: row.get(k) for k in FILE_COLUMNS}
        out["id"] = out.pop("file_id")
        return out

    def get_item(self, table_name: str, file_id: str) -> Optional[Dict]:
        table = self._open(table_name)
        rows = (
            table.search()
            .where(f"file_id = {_quote(file_id)} AND segment = 0")
            .select(META_COLUMNS)
            .limit(1)
            .to_list()
        )
        return self._file_row(rows[0]) if rows else None

    def existing_paths(self, table_name: str) -> set:
        """Paths already indexed in a collection (used to skip duplicates)."""
        if not self.has_table(table_name):
            return set()
        table = self._open(table_name)
        count = table.count_rows("segment = 0")
        if count == 0:
            return set()
        rows = table.search().where("segment = 0").select(["path"]).limit(count).to_arrow()
        return set(rows.column("path").to_pylist())

    def list_items(
        self,
        table_name: str,
        modalities: Optional[Iterable[str]] = None,
        limit: int = 60,
        offset: int = 0,
    ) -> List[Dict]:
        """Browse files in a collection, newest first."""
        table = self._open(table_name)
        where = " AND ".join(filter(None, ["segment = 0", self._modality_filter(modalities)]))
        total = table.count_rows(where)
        rows = table.search().where(where).select(META_COLUMNS).limit(max(total, 1)).to_list()
        rows.sort(key=lambda r: r["added_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return [self._file_row(r) for r in rows[offset : offset + limit]]

    def stats(self, table_name: str) -> Dict:
        table = self._open(table_name)
        counts = {m: table.count_rows(f"segment = 0 AND modality = {_quote(m)}") for m in MODALITIES}
        return {
            "name": table_name,
            "total": sum(counts.values()),
            "counts": counts,
            "segments": table.count_rows(),
        }

    def get_vector(self, table_name: str, file_id: str) -> Optional[np.ndarray]:
        """A file's vector: the normalised mean of its segment vectors."""
        table = self._open(table_name)
        rows = table.search().where(f"file_id = {_quote(file_id)}").select(["vector"]).limit(100_000).to_arrow()
        if rows.num_rows == 0:
            return None
        mat = self._vectors(rows.column("vector"))
        mean = mat.mean(axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)

    # ------------------------------------------------------------- search
    @staticmethod
    def _modality_filter(modalities: Optional[Iterable[str]]) -> Optional[str]:
        if not modalities:
            return None
        selected = [m for m in modalities if m in MODALITIES]
        if not selected or set(selected) == set(MODALITIES):
            return None
        return "modality IN (" + ", ".join(_quote(m) for m in selected) + ")"

    def _vectors(self, column) -> np.ndarray:
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        return column.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, self.embedding_dim)

    def _score_stats(self, table_name: str, table, query: np.ndarray):
        """
        Per-modality mean similarity of the query to the collection and the
        overall standard deviation, estimated on a cached sample of vectors.
        """
        version = getattr(table, "version", None)
        with self._stats_lock:
            cached = self._stats_cache.get(table_name)
            if cached is None or cached[0] != version:
                samples = {}
                for m in MODALITIES:
                    rows = (
                        table.search()
                        .where(f"modality = {_quote(m)}")
                        .select(["vector"])
                        .limit(RANK_STATS_SAMPLE)
                        .to_arrow()
                    )
                    if rows.num_rows:
                        samples[m] = self._vectors(rows.column("vector"))
                cached = (version, samples)
                self._stats_cache[table_name] = cached
        samples = cached[1]
        if not samples:
            return {}, 0.0, 1.0
        scores = {m: mat @ query for m, mat in samples.items()}
        everything = np.concatenate(list(scores.values()))
        overall_mean = float(everything.mean())
        std = max(float(everything.std()), 0.02)
        means = {
            m: float(s.mean()) if len(s) >= RANK_MIN_STATS else overall_mean for m, s in scores.items()
        }
        return means, overall_mean, std

    def search(
        self,
        query_vector: Sequence[float],
        table_name: str,
        top_k: int = DEFAULT_SEARCH_K,
        modalities: Optional[Iterable[str]] = None,
        exclude_id: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid search returning *files* ranked by their best segment.

        Each result carries ``score`` (cosine of the best segment), ``relevance``
        (the combined ranking score), ``strong`` (False for "weaker matches") and
        ``match`` = {kind, start, end, snippet} describing where it matched.
        """
        table = self._open(table_name)
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        filters = [f for f in [self._modality_filter(modalities)] if f]
        if exclude_id:
            filters.append(f"file_id != {_quote(exclude_id)}")
        where = " AND ".join(filters) or None
        n_candidates = max(top_k * 8, 120)

        # 1. Semantic candidates
        q = (
            table.search(query.tolist())
            .distance_type("cosine")
            .select([*META_COLUMNS, "keywords", "_distance"])
            .limit(n_candidates)
        )
        if where:
            q = q.where(where, prefilter=True)
        candidates: Dict[str, Dict] = {}
        for row in q.to_list():
            row["score"] = 1.0 - float(row.pop("_distance", 1.0))
            candidates[row["id"]] = row

        # 2. Keyword candidates (exact names, numbers, rare words)
        words = query_words(query_text)
        if words:
            try:
                fq = table.search(" ".join(words), query_type="fts").select(
                    [*META_COLUMNS, "keywords", "vector", "_score"]
                )
                if where:
                    fq = fq.where(where, prefilter=True)
                for row in fq.limit(n_candidates).to_list():
                    if row["id"] in candidates:
                        continue
                    vec = np.asarray(row.pop("vector"), dtype=np.float32)
                    row.pop("_score", None)
                    row["score"] = float(vec @ query)
                    candidates[row["id"]] = row
            except Exception as e:  # no keyword index yet
                logger.debug(f"Keyword search unavailable for '{table_name}': {e}")

        if not candidates:
            return []

        # 3. Calibrate per modality, group by file, add keyword coverage
        means, overall_mean, std = self._score_stats(table_name, table, query)
        hinted = kind_hints(query_text)
        files: Dict[str, Dict] = {}
        file_words: Dict[str, set] = {}
        for row in candidates.values():
            z = (row["score"] - means.get(row["modality"], overall_mean)) / std
            row["z"] = z
            fid = row["file_id"]
            tokens = file_words.setdefault(fid, set())
            tokens.update(_normalize_word(w) for w in _WORD_RE.findall((row.get("keywords") or "").lower()))
            best = files.get(fid)
            if best is None or z > best["z"]:
                files[fid] = row

        results = []
        for fid, row in files.items():
            coverage = (sum(w in file_words[fid] for w in words) / len(words)) if words else 0.0
            relevance = RANK_MEANING_WEIGHT * row["z"] + RANK_WORDS_WEIGHT * coverage**1.5
            if row["modality"] in hinted:
                relevance += 1.0
            item = self._file_row(row)
            item.update(
                score=row["score"],
                relevance=round(float(relevance), 4),
                match={
                    "kind": row["kind"],
                    "start": row["start"],
                    "end": row["end"],
                    "snippet": row.get("snippet") or "",
                    "words": round(coverage, 3),
                },
            )
            results.append(item)

        results.sort(key=lambda r: r["relevance"], reverse=True)
        results = results[:top_k]
        if results:
            best = results[0]["relevance"]
            for r in results:
                r["strong"] = bool(best <= 0 or r["relevance"] >= WEAK_RESULT_RATIO * best)
        return results
