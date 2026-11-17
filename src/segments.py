"""
Turn a file into embeddable *segments*.

Each segment becomes one vector and remembers where it lives in the file
(time range or chunk), so search results can point at the matching moment or
passage instead of just the file:

* text / code  -> line-aware ~1800-char chunks with overlap ("chunk")
* image        -> one segment ("image")
* audio        -> 30 s windows every 25 s, silence skipped ("sound")
* video        -> windows of frames ("frames") + the soundtrack as audio windows ("sound")

The model only hears the first 30 s of an audio input and video inputs are
vision-only, which is why long media is windowed here.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from .config import (
    AUDIO_HOP,
    AUDIO_SAMPLE_RATE,
    AUDIO_WINDOW,
    MAX_AUDIO_WINDOWS,
    MAX_TEXT_CHARS,
    MAX_TEXT_CHUNKS,
    MAX_VIDEO_WINDOWS,
    SILENCE_RMS,
    SNIPPET_CHARS,
    TEXT_CHUNK_CHARS,
    TEXT_CHUNK_OVERLAP,
    VIDEO_FRAME_MAX_SIDE,
    VIDEO_FRAMES_PER_WINDOW,
    VIDEO_WINDOW,
)

logger = logging.getLogger(__name__)

# Which model input each segment kind is embedded as
EMBED_AS = {"chunk": "text", "image": "image", "sound": "audio", "frames": "video"}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".sql", ".lua", ".scala", ".r", ".css",
}


@dataclass
class Segment:
    kind: str  # chunk | image | sound | frames
    payload: Any  # model input (formatted text, PIL image, float32 audio, list of PIL frames)
    start: float = 0.0  # seconds (media) or chunk index (text)
    end: float = 0.0
    snippet: str = ""  # text shown for "why it matched"
    keywords: str = ""  # extra words for keyword search

    @property
    def embed_as(self) -> str:
        return EMBED_AS[self.kind]


@dataclass
class FileSegments:
    segments: List[Segment] = field(default_factory=list)
    preview: str = ""
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- keywords
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def name_keywords(path: Path) -> str:
    """File name + parent folder as plain words (``Trip_Paris-2024.jpg`` -> ``trip paris 2024 …``)."""
    parts = [path.stem, path.parent.name]
    words = []
    for part in parts:
        # split camelCase too
        part = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", part)
        words.extend(w.lower() for w in _WORD_RE.findall(part))
    return " ".join(words)


# -------------------------------------------------------------------- text
_HEADING_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)
_GENERIC_STEM_RE = re.compile(r"^(?:untitled|new|file|document|doc|notes?|readme|index|main|test)?[\W_]*\d*$", re.I)


def document_title(path: Path, text: str) -> Optional[str]:
    """
    A meaningful title for a document, or None (-> ``title: none``).

    Markdown H1 / RST headings win. Code uses its path inside the project
    (e.g. ``billing/payments/retry.py``), which retrieves better than the bare
    file name. Otherwise a cleaned-up file name, unless it is generic
    (``untitled``, ``IMG_0001``, ``notes2`` …).
    """
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        match = _HEADING_RE.search(text[:4000])
        if match:
            return match.group(1).strip()[:200]
    if suffix == ".rst":
        lines = [line.rstrip() for line in text[:4000].splitlines()]
        for first, second in zip(lines, lines[1:]):
            if first.strip() and second and set(second) <= set("=-~^*#") and len(second) >= len(first.strip()):
                return first.strip()[:200]
    if suffix in CODE_EXTENSIONS:
        parts = [p for p in path.parts[-3:] if p not in ("/", "")]
        return "/".join(parts)
    stem = re.sub(r"[_\-.]+", " ", path.stem).strip()
    if not stem or _GENERIC_STEM_RE.match(stem.replace(" ", "")) or re.fullmatch(r"[A-Z]{2,4} ?\d+", stem):
        return None
    return stem[:200]


def format_document(text: str, title: Optional[str]) -> str:
    """The EmbeddingGemma 2 "Document" format: ``title: {title | none} | text: {content}``."""
    return f"title: {title or 'none'} | text: {text.strip()}"


def chunk_text(text: str, size: int = TEXT_CHUNK_CHARS, overlap: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into ~``size``-char chunks on line boundaries (code stays intact
    line-by-line); consecutive chunks share up to ``overlap`` chars of context.
    Over-long lines are split on whitespace.
    """
    lines: List[str] = []
    for line in text.splitlines():
        while len(line) > size:
            cut = line.rfind(" ", size // 2, size)
            cut = cut if cut > 0 else size
            lines.append(line[:cut])
            line = line[cut:].lstrip()
        lines.append(line)

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > size:
            chunks.append("\n".join(current).strip())
            # carry trailing lines (up to `overlap` chars) into the next chunk
            carry: List[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, length = carry, carried
        current.append(line)
        length += len(line) + 1
    if current and "\n".join(current).strip():
        chunks.append("\n".join(current).strip())
    # Drop chunks that are only the carried overlap of the previous chunk
    return [c for i, c in enumerate(chunks) if c and (i == 0 or c not in chunks[i - 1])]


def text_segments(path: Path) -> FileSegments:
    text = path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_CHARS]
    if not text.strip():
        raise ValueError("Text file is empty.")
    title = document_title(path, text)
    chunks = chunk_text(text)[:MAX_TEXT_CHUNKS]
    base_keywords = name_keywords(path)
    segments = [
        Segment(
            kind="chunk",
            payload=format_document(chunk, title),
            start=float(i),
            end=float(i),
            snippet=chunk[:SNIPPET_CHARS],
            keywords=f"{base_keywords} {(title or '').lower()} {chunk}",
        )
        for i, chunk in enumerate(chunks)
    ]
    return FileSegments(segments=segments, preview=text[:600], meta={})


# ------------------------------------------------------------------- image
def load_image(path, max_side: int) -> Image.Image:
    """Decode an image quickly: JPEG draft mode + downscale + EXIF orientation."""
    from PIL import ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            # JPEG "draft" decodes at 1/2, 1/4 or 1/8 scale — several times faster for photos
            img.draft("RGB", (max_side, max_side))
            img = ImageOps.exif_transpose(img)
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            return img.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Unsupported or corrupt image: {path}") from e


# ------------------------------------------------------------------- audio
def _window_starts(duration: float, window: float, hop: float, max_windows: int) -> List[float]:
    if duration <= window:
        return [0.0]
    count = int(math.ceil((duration - window) / hop)) + 1
    if count > max_windows:
        # Spread windows evenly over very long media
        step = (duration - window) / (max_windows - 1)
        return [i * step for i in range(max_windows)]
    starts = [i * hop for i in range(count)]
    # Keep a short tail only if it adds at least 2 s of new audio
    if len(starts) > 1 and duration - (starts[-2] + window) < 2.0:
        starts.pop()
    return starts


def audio_windows(path: Path, keywords: str, keep_if_silent: bool = True) -> List[Segment]:
    """30 s windows (25 s hop) of 16 kHz mono audio; silent windows are skipped."""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(str(path), sample_rate=AUDIO_SAMPLE_RATE, num_channels=1)
    duration = float(decoder.metadata.duration_seconds or 0.0)
    if duration <= 0:
        samples = decoder.get_all_samples().data.numpy()[0]
        duration = len(samples) / AUDIO_SAMPLE_RATE
    segments: List[Segment] = []
    first: Optional[Segment] = None
    for start in _window_starts(duration, AUDIO_WINDOW, AUDIO_HOP, MAX_AUDIO_WINDOWS):
        stop = min(start + AUDIO_WINDOW, duration)
        data = decoder.get_samples_played_in_range(start_seconds=start, stop_seconds=stop).data
        wave = data.numpy()[0].astype(np.float32, copy=False)
        if wave.size < AUDIO_SAMPLE_RATE // 4:
            continue
        segment = Segment(kind="sound", payload=wave, start=start, end=stop, keywords=keywords)
        first = first or segment
        if float(np.sqrt(np.mean(np.square(wave)))) > SILENCE_RMS:
            segments.append(segment)
    if not segments and first is not None and keep_if_silent:
        segments.append(first)  # all silent: still index the file once
    return segments


def audio_segments(path: Path) -> FileSegments:
    segments = audio_windows(path, name_keywords(path))
    if not segments:
        raise ValueError("No decodable audio.")
    return FileSegments(segments=segments)


# ------------------------------------------------------------------- video
def _to_pil(frame: np.ndarray, max_side: int) -> Image.Image:
    img = Image.fromarray(frame)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return img


def video_segments(path: Path) -> FileSegments:
    """Windows of frames (vision) plus the soundtrack as audio windows."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path), dimension_order="NHWC")
    meta = decoder.metadata
    duration = float(meta.duration_seconds or 0.0)
    if duration <= 0:
        raise ValueError("Video has no duration.")
    keywords = name_keywords(path)

    window = VIDEO_WINDOW
    if duration / window > MAX_VIDEO_WINDOWS:
        window = duration / MAX_VIDEO_WINDOWS
    starts = [i * window for i in range(max(1, int(math.ceil(duration / window))))]
    last_frame = max(0.0, duration - 1.0 / float(meta.average_fps or 30.0))

    segments: List[Segment] = []
    for start in starts:
        end = min(start + window, duration)
        span = end - start
        n = max(1, min(VIDEO_FRAMES_PER_WINDOW, int(round(span)) or 1))
        times = [min(start + span * (i + 0.5) / n, last_frame) for i in range(n)]
        frames = decoder.get_frames_played_at(seconds=times).data.numpy()
        segments.append(
            Segment(
                kind="frames",
                payload=[_to_pil(f, VIDEO_FRAME_MAX_SIDE) for f in frames],
                start=start,
                end=end,
                keywords=keywords,
            )
        )

    try:
        segments.extend(audio_windows(path, keywords, keep_if_silent=False))
    except Exception:
        pass  # no audio stream (or undecodable): vision only

    return FileSegments(segments=segments)


# -------------------------------------------------------------------- file
@dataclass
class PreparedFile:
    """A file read, decoded and segmented by a loader thread, ready for the model."""

    path: Path
    modality: str
    segments: List[Segment]
    preview: str
    size: int
    meta: dict


def prepare_file(path: Path, modality: Optional[str] = None) -> PreparedFile:
    """CPU-side work for one file (safe to run in worker threads)."""
    from .config import MAX_IMAGE_SIDE, modality_for
    from .utils import probe_media

    path = Path(path)
    modality = modality or modality_for(path)
    if modality is None:
        raise ValueError(f"Unsupported file type: {path.suffix or path.name}")
    if modality == "text":
        fs = text_segments(path)
    elif modality == "image":
        fs = FileSegments(
            segments=[Segment(kind="image", payload=load_image(path, MAX_IMAGE_SIDE), keywords=name_keywords(path))]
        )
    elif modality == "audio":
        fs = audio_segments(path)
    else:
        fs = video_segments(path)
    return PreparedFile(
        path=path,
        modality=modality,
        segments=fs.segments,
        preview=fs.preview,
        size=path.stat().st_size,
        meta=probe_media(path, modality),
    )
