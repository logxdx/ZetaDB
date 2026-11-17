"""
Application engine: owns the embedding model, the vector index and the
background indexing jobs. Shared by the web API and the CLI.

Ingestion pipeline
------------------
1. Scan (skipping dependency / build folders and secrets) and de-duplicate:
   files already in the collection are skipped up front.
2. Group files by modality into small file batches (quick modalities first).
3. A thread pool prepares upcoming file batches — reading, JPEG draft
   decoding, chunking text, cutting audio into 30 s windows and videos into
   frame windows — while the model embeds the current one.
4. Segments of a file batch are grouped by model input kind and embedded in
   batches (``BATCH_SIZES``); text is sorted by length to minimise padding.
   ``GemmaEmbedder.embed_batch`` halves batches on CUDA OOM and isolates
   corrupt inputs by bisection.
5. Rows are written to LanceDB in large chunks; afterwards the table is
   compacted, the keyword index rebuilt and (for big collections) an ANN
   index built.
"""

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .config import (
    BATCH_SIZES,
    DEFAULT_QUERY_INTENT,
    DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    ENABLED_MODALITIES,
    FILES_PER_BATCH,
    INSERT_BATCH,
    LOADER_WORKERS,
    PREFETCH_BATCHES,
    modality_for,
)
from .index import LanceDBIndex, VectorObject, validate_table_name
from .segments import PreparedFile, prepare_file
from .utils import normalize_vectors

logger = logging.getLogger(__name__)

MAX_JOBS_KEPT = 50
# Order in which modalities are processed: fast ones first
MODALITY_ORDER = ("text", "image", "audio", "video")

# Folders never worth indexing (dependencies, caches, build output)
SKIP_DIRS = {
    "node_modules", "venv", ".venv", "env", "site-packages", "__pycache__", "dist", "build", "target",
    "DerivedData", "Pods", ".git", ".hg", ".svn", ".idea", ".vscode", ".cache", ".tox", ".mypy_cache",
}
# Files that must never be read into an index
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".der", ".jks", ".keystore")
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock", "uv.lock"}
MAX_TEXT_FILE_BYTES = 5 * 1024 * 1024


@dataclass
class IndexJob:
    id: str
    collection: str
    source: str
    total: int = 0
    done: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    segments: int = 0
    status: str = "queued"  # queued | running | completed | failed
    current: str = ""
    errors: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    embed_started_at: Optional[float] = None
    embedded: int = 0  # files that went through the model (for throughput)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["errors"] = self.errors[-20:]
        end = self.finished_at or time.time()
        elapsed = end - self.embed_started_at if self.embed_started_at else 0.0
        rate = self.embedded / elapsed if elapsed > 0 and self.embedded else 0.0
        remaining = max(self.total - self.done, 0)
        data["elapsed"] = round(end - self.started_at, 2)
        data["rate"] = round(rate, 2)  # files / second
        data["eta"] = round(remaining / rate, 1) if rate and self.status == "running" else None
        return data


def _skip_file(path: Path) -> bool:
    name = path.name
    if name.startswith(".") or name in SKIP_NAMES or name.lower().endswith(SECRET_SUFFIXES):
        return True
    if ".min." in name:
        return True
    modality = modality_for(path)
    if modality is None:
        return True
    if modality == "text":
        try:
            return path.stat().st_size > MAX_TEXT_FILE_BYTES
        except OSError:
            return True
    return False


def scan_folder(folder: Path) -> List[Path]:
    """All supported files below ``folder`` (recursively), sorted."""
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for name in names:
            path = Path(root) / name
            if not _skip_file(path):
                files.append(path)
    return sorted(files)


def make_batches(files: Iterable[Path]) -> List[Tuple[str, List[Path]]]:
    """Group files by modality and chunk them into file batches."""
    groups: Dict[str, List[Path]] = {m: [] for m in MODALITY_ORDER}
    for path in files:
        modality = modality_for(path)
        if modality:
            groups[modality].append(path)
    batches = []
    for modality in MODALITY_ORDER:
        paths = groups[modality]
        size = max(1, FILES_PER_BATCH.get(modality, 4))
        batches.extend((modality, paths[i : i + size]) for i in range(0, len(paths), size))
    return batches


class Engine:
    def __init__(self):
        self.index = LanceDBIndex()
        self.embedder = None
        self.state = "idle"  # idle | loading | ready | error
        self.error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._jobs: "OrderedDict[str, IndexJob]" = OrderedDict()
        self._jobs_lock = threading.Lock()
        # Index jobs run one at a time so they don't fight over the GPU
        self._job_runner_lock = threading.Lock()

    # ------------------------------------------------------------- model
    def load_model(self) -> None:
        with self._load_lock:
            if self.embedder is not None:
                return
            self.state = "loading"
            try:
                from .embedding import GemmaEmbedder

                self.embedder = GemmaEmbedder()
                # Warm the kernels so the first real query is fast
                self.embedder.embed_query("warm up")
                self.state = "ready"
                self.error = None
            except Exception as e:  # pragma: no cover - depends on environment
                logger.exception("Failed to load embedding model")
                self.state = "error"
                self.error = str(e)
                raise

    def load_model_async(self) -> None:
        def _load():
            try:
                self.load_model()
            except Exception:
                pass

        threading.Thread(target=_load, name="model-loader", daemon=True).start()

    def require_model(self):
        if self.embedder is None:
            if self.state == "error":
                raise RuntimeError(f"Embedding model failed to load: {self.error}")
            if self.state == "loading":
                raise RuntimeError("The embedding model is still loading. Try again in a moment.")
            self.load_model()
        return self.embedder

    def status(self) -> Dict:
        return {
            "state": self.state,
            "error": self.error,
            "model": EMBEDDING_MODEL_NAME,
            "device": DEVICE,
            "dimension": EMBEDDING_DIMENSION,
            "modalities": list(ENABLED_MODALITIES),
        }

    # ------------------------------------------------------------- search
    def build_query_vector(
        self,
        text: Optional[str] = None,
        file_path: Optional[Path] = None,
        collection: Optional[str] = None,
        item_id: Optional[str] = None,
        intent: str = DEFAULT_QUERY_INTENT,
    ) -> np.ndarray:
        """
        Build a query vector from any combination of text, a file and an
        existing item. Multiple inputs are fused by averaging their unit vectors.

        ``intent`` picks the task prompt for the text part (search, question,
        code, fact, similarity) as described in the EmbeddingGemma 2 guide.
        """
        vectors = []
        if text and text.strip():
            vectors.append(self.require_model().embed_query(text, intent=intent))
        if file_path is not None:
            vectors.append(self.require_model().embed_file(file_path))
        if item_id and collection:
            vec = self.index.get_vector(collection, item_id)
            if vec is None:
                raise KeyError(f"Item '{item_id}' not found.")
            vectors.append(vec)
        if not vectors:
            raise ValueError("Provide a text query, a file, or an item to search with.")
        return normalize_vectors(np.mean(np.stack(vectors), axis=0))

    def search(
        self,
        collection: str,
        top_k: int,
        modalities: Optional[Iterable[str]] = None,
        text: Optional[str] = None,
        file_path: Optional[Path] = None,
        item_id: Optional[str] = None,
        intent: str = DEFAULT_QUERY_INTENT,
    ) -> List[Dict]:
        vector = self.build_query_vector(text, file_path, collection, item_id, intent=intent)
        return self.index.search(
            vector.tolist(),
            table_name=collection,
            top_k=top_k,
            modalities=modalities,
            exclude_id=item_id,
            query_text=text,
        )

    # ----------------------------------------------------------- indexing
    def start_index_job(self, collection: str, files: List[Path], source: str) -> IndexJob:
        collection = validate_table_name(collection)
        self.index.create_table(collection)
        job = IndexJob(id=uuid.uuid4().hex[:12], collection=collection, source=source, total=len(files))
        with self._jobs_lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
        threading.Thread(
            target=self._run_job, args=(job, files), name=f"index-{job.id}", daemon=True
        ).start()
        return job

    def index_folder(self, collection: str, folder: str) -> IndexJob:
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError(f"Not a folder: {folder}")
        return self.start_index_job(collection, scan_folder(path), source=str(path.resolve()))

    def _run_job(self, job: IndexJob, files: List[Path]) -> None:
        with self._job_runner_lock:
            job.status = "running"
            try:
                self.require_model()

                # 1. De-duplicate against the collection and within the request
                existing = self.index.existing_paths(job.collection)
                todo: List[Path] = []
                seen = set()
                for path in files:
                    key = str(path.resolve())
                    if key in existing or key in seen:
                        job.skipped += 1
                        job.done += 1
                    else:
                        seen.add(key)
                        todo.append(path)

                # 2-5. Batched, prefetching pipeline
                pending: List[Dict] = []
                job.embed_started_at = time.time()
                for prepared, vectors in self._pipeline(make_batches(todo), job):
                    rows = self._to_rows(prepared, vectors, job)
                    pending.extend(rows)
                    job.done += 1
                    job.embedded += 1
                    if len(pending) >= INSERT_BATCH:
                        job.indexed += self._flush(pending, job.collection)
                        pending = []
                job.indexed += self._flush(pending, job.collection)

                if job.indexed:
                    self.index.optimize(job.collection)
                job.status = "completed"
            except Exception as e:
                logger.exception(f"Index job {job.id} failed")
                job.status = "failed"
                job.errors.append(str(e))
            finally:
                job.current = ""
                job.finished_at = time.time()
                info = job.to_dict()
                logger.info(
                    f"Job {job.id}: {job.indexed} files ({job.segments} segments) indexed, "
                    f"{job.skipped} skipped, {job.failed} failed in {info['elapsed']:.1f}s "
                    f"({info['rate']:.2f} files/s)"
                )

    def _flush(self, rows: List[Dict], collection: str) -> int:
        """Insert rows and return the number of *files* written."""
        if not rows:
            return 0
        self.index.insert_documents(rows, collection)
        return sum(1 for r in rows if r["segment"] == 0)

    def _pipeline(
        self, batches: List[Tuple[str, List[Path]]], job: IndexJob
    ) -> Iterator[Tuple[PreparedFile, List]]:
        """
        Yield ``(prepared_file, vectors_or_errors)`` per file while loader
        threads prepare the next ``PREFETCH_BATCHES`` file batches.
        """
        embedder = self.require_model()
        queue: Deque[List[Tuple[Path, Future]]] = deque()
        it = iter(batches)

        with ThreadPoolExecutor(max_workers=LOADER_WORKERS, thread_name_prefix="zeta-load") as pool:

            def submit_next() -> None:
                nxt = next(it, None)
                if nxt is not None:
                    modality, paths = nxt
                    queue.append([(p, pool.submit(prepare_file, p, modality)) for p in paths])

            for _ in range(PREFETCH_BATCHES + 1):
                submit_next()

            while queue:
                futures = queue.popleft()
                submit_next()  # keep the loaders busy while the model works

                prepared: List[PreparedFile] = []
                for path, fut in futures:
                    try:
                        prepared.append(fut.result())
                    except Exception as e:
                        self._fail(job, path, e)
                        job.done += 1
                if not prepared:
                    continue
                job.current = prepared[0].path.name

                # Embed all segments of this file batch, grouped by model input kind
                results: Dict[Tuple[int, int], object] = {}
                by_kind: Dict[str, List[Tuple[int, int]]] = {}
                for fi, pf in enumerate(prepared):
                    for si, seg in enumerate(pf.segments):
                        by_kind.setdefault(seg.embed_as, []).append((fi, si))
                for kind, keys in by_kind.items():
                    if kind == "text":  # similar lengths together -> less padding
                        keys.sort(key=lambda k: len(prepared[k[0]].segments[k[1]].payload))
                    size = max(1, BATCH_SIZES.get(kind, 8))
                    for i in range(0, len(keys), size):
                        chunk = keys[i : i + size]
                        outs = embedder.embed_batch(kind, [prepared[f].segments[s].payload for f, s in chunk])
                        results.update(zip(chunk, outs))

                for fi, pf in enumerate(prepared):
                    yield pf, [results[(fi, si)] for si in range(len(pf.segments))]
                    # Free decoded media (frames, waveforms) as soon as the file is done
                    for seg in pf.segments:
                        seg.payload = None

    def _to_rows(self, pf: PreparedFile, vectors: List, job: IndexJob) -> List[Dict]:
        file_id = str(uuid.uuid4())
        resolved = str(pf.path.resolve())
        rows = []
        errors = []
        for seg, vec in zip(pf.segments, vectors):
            if isinstance(vec, Exception):
                errors.append(vec)
                continue
            rows.append(
                VectorObject(
                    id=str(uuid.uuid4()),
                    file_id=file_id,
                    segment=len(rows),
                    kind=seg.kind,
                    start=seg.start,
                    end=seg.end,
                    snippet=seg.snippet,
                    keywords=seg.keywords,
                    vector=vec.tolist(),
                    modality=pf.modality,
                    path=resolved,
                    name=pf.path.name,
                    text=pf.preview,
                    size=pf.size,
                    **pf.meta,
                ).model_dump()
            )
        if not rows:
            self._fail(job, pf.path, errors[0] if errors else ValueError("Nothing to embed."))
        else:
            job.segments += len(rows)
            if errors:
                job.errors.append(f"{pf.path.name}: {len(errors)} segment(s) skipped ({errors[0]})")
        return rows

    @staticmethod
    def _fail(job: IndexJob, path: Path, error: Exception) -> None:
        job.failed += 1
        job.errors.append(f"{path.name}: {error}")
        logger.warning(f"Failed to index {path}: {error}")

    def get_job(self, job_id: str) -> Optional[IndexJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[IndexJob]:
        return list(reversed(self._jobs.values()))

    def wait(self, job: IndexJob, poll: float = 0.5) -> IndexJob:
        while job.status in ("queued", "running"):
            time.sleep(poll)
        return job
