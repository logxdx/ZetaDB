import os
from pathlib import Path

import torch

# --- Device ---
# Automatically detect CUDA, fallback to CPU. Override with ZETA_DEVICE=cpu
DEVICE = os.environ.get("ZETA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# --- Project Root ---
# Goes up two levels from src/config.py to the project root
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
DATA_DIR = Path(os.environ.get("ZETA_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True, parents=True)

# Files added through the web UI are stored here (one sub-folder per collection)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# The LanceDB database directory. A new name is used because EmbeddingGemma 2
# vectors (768-d) are incompatible with the old jina-clip-v2 tables (1024-d).
INDEX_FILE = INDEX_DIR / "zeta_db_v2"

# --- Frontend ---
WEB_DIR = PROJECT_ROOT / "web"

# --- Embedding Model ---
# EmbeddingGemma 2 maps text, images, audio and video into one shared space.
EMBEDDING_MODEL_NAME = os.environ.get("ZETA_MODEL", "google/embeddinggemma-2")
# Native size is 768. Matryoshka truncation to 512 / 256 / 128 is supported.
# Changing this requires re-indexing (and a fresh INDEX_FILE).
EMBEDDING_DIMENSION = int(os.environ.get("ZETA_EMBEDDING_DIM", 768))

# Task prompts from the EmbeddingGemma 2 developer guide. Prompts are applied to
# *text queries* only; images, audio and video are embedded without a prompt,
# and indexed text/code documents use the "Document" format
# ("title: {title | none} | text: {content}").
QUERY_PROMPTS = {
    "search": "SearchQuery",  # task: search result | query:  (also used for cross-modal search)
    "question": "QuestionAnswering",  # task: question answering | query:
    "code": "CodeRetrieval",  # task: code retrieval | query:
    "fact": "FactChecking",  # task: fact checking | query:
    "similarity": "SentenceSimilarity",  # task: sentence similarity | query:
}
DEFAULT_QUERY_INTENT = "search"
TEXT_QUERY_PROMPT = QUERY_PROMPTS[DEFAULT_QUERY_INTENT]
TEXT_DOCUMENT_PROMPT = "Document"  # title: none | text:

# Modular loading: only load the encoders you need (all share one vector space).
#   text only  ~270M params · + images/video ~440M · + audio ~570M · everything ~740M
ENABLED_MODALITIES = tuple(
    m.strip()
    for m in os.environ.get("ZETA_MODALITIES", "text,image,audio,video").split(",")
    if m.strip()
)

# --- Segmentation ---
# Long files are split into segments; each segment gets its own vector and
# results point at the best-matching moment / passage.
#
# Text & code: ~450-token chunks on line boundaries with a little overlap.
TEXT_CHUNK_CHARS = 1800
TEXT_CHUNK_OVERLAP = 200
MAX_TEXT_CHUNKS = 200
# Max characters read from a text document (larger files are cut)
MAX_TEXT_CHARS = TEXT_CHUNK_CHARS * MAX_TEXT_CHUNKS
# Characters of a text document kept in the index for previews / snippets
TEXT_PREVIEW_CHARS = 600
SNIPPET_CHARS = 400

# Audio: the model only hears the first 30 s of an input (verified), so audio
# (and the soundtrack of videos) is embedded as 30 s windows every 25 s.
AUDIO_WINDOW = 30.0
AUDIO_HOP = 25.0
AUDIO_SAMPLE_RATE = 16_000
SILENCE_RMS = 0.003  # ~ -50 dBFS; silent windows are skipped
MAX_AUDIO_WINDOWS = 240  # beyond ~100 min the hop widens to spread windows evenly

# Video: windows of frames (vision only — the soundtrack is embedded as audio).
VIDEO_WINDOW = 10.0
VIDEO_FRAMES_PER_WINDOW = 5
VIDEO_FRAME_MAX_SIDE = 512
MAX_VIDEO_WINDOWS = 120  # beyond 20 min windows widen to cover the whole video

# Images are decoded at (roughly) this max side. The default budget of 280
# soft tokens corresponds to ~800 px, so larger decodes only cost time.
MAX_IMAGE_SIDE = 1024

# --- Ingestion pipeline ---
_ON_GPU = DEVICE.startswith("cuda")
# Segments are embedded in batches of these sizes (upper bounds: on CUDA OOM a
# batch is split in half automatically). Batching pays off on GPUs; on CPU the
# model is compute-bound and padding costs more than batching saves.
BATCH_SIZES = {
    "text": int(os.environ.get("ZETA_BATCH_TEXT", 32 if _ON_GPU else 8)),
    "image": int(os.environ.get("ZETA_BATCH_IMAGE", 16 if _ON_GPU else 2)),
    "audio": int(os.environ.get("ZETA_BATCH_AUDIO", 8 if _ON_GPU else 8)),
    "video": int(os.environ.get("ZETA_BATCH_VIDEO", 4 if _ON_GPU else 1)),
}
# Files prepared together (segments of these files are then batched as above)
FILES_PER_BATCH = {"text": 16, "image": 16, "audio": 2, "video": 1}
# Threads that read / decode files while the model is busy. On CPU they compete
# with PyTorch for cores, so fewer are used there.
LOADER_WORKERS = int(
    os.environ.get("ZETA_LOADER_WORKERS", min(8, os.cpu_count() or 4) if _ON_GPU else 2)
)
# File batches prepared ahead of the one being embedded
PREFETCH_BATCHES = 2
# Rows written to LanceDB per insert (fewer, larger fragments)
INSERT_BATCH = 256
# Build an approximate-nearest-neighbour index once a collection reaches this size
ANN_INDEX_THRESHOLD = int(os.environ.get("ZETA_ANN_THRESHOLD", 50_000))

# --- Search ---
DEFAULT_SEARCH_K = 12  # Default number of results (files) to return
DEFAULT_TABLE = "library"
# Ranking (after DigUp): cosine scores are turned into per-modality z-scores so
# text, images, audio and video compete fairly, then a keyword-coverage bonus
# rewards results that literally contain the query words.
RANK_MEANING_WEIGHT = 1.0
RANK_WORDS_WEIGHT = 2.0
RANK_STATS_SAMPLE = 512  # vectors per modality used to estimate score statistics
RANK_MIN_STATS = 20  # modalities with fewer vectors use the overall mean
WEAK_RESULT_RATIO = 0.5  # results below this fraction of the best are "weaker matches"
# Cached query embeddings (text + intent)
QUERY_CACHE_SIZE = 256

# --- Ingestion ---
SUPPORTED_EXTENSIONS = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"),
    "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"),
    "text": (
        # prose / passages
        ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html", ".xml", ".yaml", ".yml", ".toml",
        # source code (embedded as documents; query it with the "code" intent)
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h",
        ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".sql", ".lua", ".scala", ".r", ".css",
    ),
}
MODALITIES = tuple(SUPPORTED_EXTENSIONS)

# Kept for backwards compatibility with older imports
SUPPORTED_IMAGE_EXTENSIONS = SUPPORTED_EXTENSIONS["image"]


def modality_for(path: str | Path) -> str | None:
    """Return the modality of a file based on its extension, or None if unsupported/disabled."""
    suffix = Path(path).suffix.lower()
    for modality, extensions in SUPPORTED_EXTENSIONS.items():
        if suffix in extensions:
            return modality if modality in ENABLED_MODALITIES else None
    return None


# --- Server ---
HOST = os.environ.get("ZETA_HOST", "127.0.0.1")
PORT = int(os.environ.get("ZETA_PORT", 7860))