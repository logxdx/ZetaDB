"""Audio and video decoding for ingest - ffprobe/ffmpeg, no Python codec deps.

Why windows (spec 20): EmbeddingGemma 2 hears only the first 30 s of an audio
input - its feature extractor truncates at 480 000 samples - and a video is
vision-only, a handful of frames. Embedding a 40-minute podcast or a long
clip as ONE vector would describe its first half-minute and nothing else. So
long media is cut into windows, each its own vector, and search can point at
the moment that matched:

  audio   30 s windows every 25 s (5 s overlap), silent windows skipped
  video   10 s windows of 5 frames (one every 2 s), vision only,
          plus the soundtrack as audio windows

ffmpeg is a hard runtime dependency for audio/video only; images and notes
never touch it.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server.config import load

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wmv", ".mpg", ".mpeg"}
SR = 16_000


def _cfg(key: str, default):
    return load().get(f"media.{key}", default)


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


# ------------------------------------------------------------------ probing

@dataclass
class Probe:
    duration: float
    width: int | None
    height: int | None
    has_audio: bool
    has_video: bool


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60, check=True).stdout
    info = json.loads(out or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float((info.get("format") or {}).get("duration") or 0.0)
    if not dur:
        for s in (video, audio):
            if s and s.get("duration"):
                dur = float(s["duration"])
                break
    w = h = None
    if video:
        w, h = int(video.get("width") or 0) or None, int(video.get("height") or 0) or None
        # Rotated phone video: width/height are stored pre-rotation.
        rot = abs(int(((video.get("tags") or {}).get("rotate")) or 0))
        for sd in video.get("side_data_list") or []:
            if "rotation" in sd:
                rot = abs(int(sd["rotation"]))
        if rot in (90, 270) and w and h:
            w, h = h, w
    return Probe(duration=dur, width=w, height=h,
                 has_audio=audio is not None, has_video=video is not None)


# -------------------------------------------------------------------- audio

def window_starts(duration: float, window: float, hop: float, max_windows: int) -> list[float]:
    """Window start times covering `duration`. Past `max_windows` the hop
    widens so the windows still span the whole file evenly."""
    if duration <= window:
        return [0.0]
    count = int(math.ceil((duration - window) / hop)) + 1
    if count > max_windows:
        step = (duration - window) / (max_windows - 1)
        return [i * step for i in range(max_windows)]
    starts = [i * hop for i in range(count)]
    # Keep a short tail only if it adds at least 2 s of new audio.
    if len(starts) > 1 and duration - (starts[-2] + window) < 2.0:
        starts.pop()
    return starts


def decode_audio(path: Path, start: float, length: float) -> np.ndarray:
    """`length` seconds from `start`, as 16 kHz mono float32 in -1..1."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, timeout=300, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class AudioWindow:
    start: float
    end: float
    wave: np.ndarray


def audio_windows(path: Path, duration: float, *, keep_if_silent: bool = True
                  ) -> list[AudioWindow]:
    """30 s windows every 25 s; windows quieter than ~-50 dBFS are dropped.
    If everything is silent the first window is kept (keep_if_silent), so a
    quiet recording is still findable by its title - but a video whose
    soundtrack is silent gets no sound segments at all."""
    win = float(_cfg("audio_window_s", 30.0))
    hop = float(_cfg("audio_hop_s", 25.0))
    floor = float(_cfg("silence_rms", 0.003))
    cap = int(_cfg("max_audio_windows", 240))
    out: list[AudioWindow] = []
    first: AudioWindow | None = None
    for s in window_starts(duration, win, hop, cap):
        length = min(win, max(0.0, duration - s)) or win
        wave = decode_audio(path, s, length)
        if wave.size < SR // 4:
            continue
        w = AudioWindow(start=s, end=s + wave.size / SR, wave=wave)
        first = first or w
        if float(np.sqrt(np.mean(np.square(wave)))) > floor:
            out.append(w)
    if not out and first is not None and keep_if_silent:
        out.append(first)
    return out


# -------------------------------------------------------------------- video

@dataclass
class FrameWindow:
    start: float
    end: float
    frames: list        # PIL images, time order


def _scaled_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    s = min(1.0, max_side / max(w, h))
    # even dimensions, as most pixel formats require
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def video_windows(path: Path, pr: Probe) -> list[FrameWindow]:
    """Frames every `frame_every_s` grouped into `video_window_s` windows,
    decoded in ONE ffmpeg pass (seeking per frame is far slower on long
    files). Past `max_video_windows` the sampling interval stretches so the
    windows still cover the whole video."""
    from PIL import Image
    if not pr.has_video or not pr.width or not pr.height:
        return []
    win = float(_cfg("video_window_s", 10.0))
    per = int(_cfg("frames_per_window", 5))
    cap = int(_cfg("max_video_windows", 120))
    side = int(_cfg("frame_max_side", 512))
    duration = max(pr.duration, 0.1)
    if duration / win > cap:
        win = duration / cap
    every = win / per
    w, h = _scaled_size(pr.width, pr.height, side)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an",
         "-vf", f"fps=1/{every:.4f}:round=near,scale={w}:{h}:flags=bilinear",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    frames: list[tuple[float, object]] = []
    try:
        i = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            t = i * every
            frames.append((t, Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", 0, 1).copy()))
            i += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=60)
    if not frames:
        return []
    out: list[FrameWindow] = []
    n_win = max(1, int(math.ceil(duration / win)))
    for k in range(n_win):
        s, e = k * win, min((k + 1) * win, duration)
        group = [f for t, f in frames if s <= t < e] or ([frames[-1][1]] if k == n_win - 1 else [])
        if group:
            out.append(FrameWindow(start=s, end=e, frames=group))
    return out


# --------------------------------------------------------------- thumbnails

def poster_score(luma: np.ndarray) -> float:
    """How good a frame is as a poster: detail (contrast) and light. A night
    launch, a fade or a title card scores low; a lit scene scores high."""
    mean, std = float(luma.mean()), float(luma.std())
    score = std + 0.35 * min(mean, 150.0)
    if mean < 40 or mean > 235:        # near black or blown out
        score *= 0.3
    return score


def video_poster(path: Path, pr: Probe, dest: Path, long_edge: int,
                 candidates: int = 7) -> tuple[int, int]:
    """The item's thumbnail: the best-lit, most detailed of a few frames
    spread over 10-85% of the film. One fixed frame at 10% gave a grey fade
    for an animated film and a black sky for a night launch."""
    import io

    from PIL import Image
    best: tuple[float, bytes] | None = None
    d = max(pr.duration, 0.0)
    for k in range(candidates):
        frac = 0.10 + 0.75 * k / max(1, candidates - 1)
        t = min(d * frac, max(d - 0.1, 0.0))
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, timeout=120).stdout
        if not raw:
            continue
        with Image.open(io.BytesIO(raw)) as im:
            small = im.convert("L")
            small.thumbnail((160, 160))
            s = poster_score(np.asarray(small, dtype=np.float32))
        if best is None or s > best[0]:
            best = (s, raw)
        if d < 2.0:
            break
    if best is None:
        raise RuntimeError(f"no frame decodable from {path.name}")
    with Image.open(io.BytesIO(best[1])) as im:
        im = im.convert("RGB")
        size = im.size
        im.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "WEBP", quality=82, method=4)
    return size


def audio_waveform(path: Path, duration: float, dest: Path, width: int = 480,
                   height: int = 120) -> None:
    """A waveform card as the audio item's thumbnail, drawn from a coarse
    peak envelope of the whole file (decoded at 4 kHz to stay cheap)."""
    from PIL import Image, ImageDraw
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
         "-ar", "4000", "-f", "f32le", "-"],
        capture_output=True, timeout=600, check=True).stdout
    a = np.abs(np.frombuffer(raw, dtype=np.float32))
    bars = 64
    if a.size == 0:
        env = np.zeros(bars)
    else:
        chunks = np.array_split(a, bars)
        env = np.array([float(c.max()) if c.size else 0.0 for c in chunks])
        env = env / (env.max() or 1.0)
    # Transparent ground, opaque bars: the UI uses this as a MASK and paints
    # it with the theme's own colour, so one file serves light and dark.
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad, gap = 0, 3
    bw = (width - 2 * pad - gap * (bars - 1)) / bars
    mid = height / 2
    for i, v in enumerate(env):
        x0 = pad + i * (bw + gap)
        hh = max(bw / 2, v * (height / 2 - 2))
        d.rounded_rectangle([x0, mid - hh, x0 + bw, mid + hh], radius=bw / 2,
                            fill=(255, 255, 255, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "WEBP", quality=85, method=4)
