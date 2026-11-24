# Setup & Usage

Single-user and local-only. Built and measured on Linux with an RTX 4050 Laptop GPU (6 GB); every
model also falls back to the CPU. The full design, and the record of what was measured rather than
assumed, is in [spec.md](../spec.md).

## Install

```sh
uv sync                        # base deps (Python 3.12)
uv pip install -e '.[onnx,ocr,hf]'   # model runtimes, OCR, weight downloads
uv pip install -e '.[needle]'  # structured extraction
cd web && npm install && npm run build
```

Audio and video need `ffmpeg` and `ffprobe` on `PATH`. Captioning needs a CUDA build of
`llama-server` (see `[llama]` in `config.toml`).

### Model weights

```sh
uv run scripts/probe.py --download eg2        # EmbeddingGemma 2 (text, image, audio encoders)
uv run scripts/probe.py --download ppocr_det
uv run scripts/probe.py --download ppocr_rec
uv run scripts/probe.py --download eye        # Eye: eye.tflite, metadata, licence notices
```

**Eye** (image tags) ships as LiteRT and is converted to ONNX once so it runs in the same engine
as every other model:

```sh
uv run --no-project --with tensorflow-cpu --with tf2onnx --with ai-edge-litert \
    --with onnx --with onnxruntime --with pillow \
    python scripts/convert_eye.py data/blobs
```

This writes `eye_photo.onnx` and `eye_photo.json`, and checks every head against the LiteRT
reference on your own images before you rely on it. Keep Eye's licence and notice files next to the
weights.

## Run

```sh
uv run -m server.app           # http://127.0.0.1:8777
cd web && npm run dev          # hot-reload UI on :5173, proxies /api
```

An existing `data/iwashere.db` from before the rename is adopted as `data/zetadb.db` on first start.

Ways in: drag files onto the window, paste (`Ctrl+V`) an image or text, `⌘K` → new note, or save
into a watched folder (`paths.watch_dirs`, default `~/dump`). Supported: images, `.txt`/`.md`
notes, audio (mp3, wav, flac, ogg, m4a, aac, opus) and video (mp4, mov, webm, mkv, avi).

## Testing

Three layers, cheapest first.

```sh
# 1. unit + regression (no server, no GPU, ~1 s)
uv run --with pytest pytest tests/ -q

# 2. API contract against a running server — every endpoint the SPA calls,
#    checking response SHAPE matches what the frontend destructures
uv run scripts/smoke.py

# 3. what this machine can actually load, and what it costs
uv run scripts/probe.py
```

`scripts/smoke.py` catches the failure mode with no server-side error: a mismatch between
`web/src/lib/api.ts` and `server/api/*` breaks the UI silently. It also verifies SSE delivers and
that thumbnails and full files serve.

### Testing the UI

The SPA has to be exercised in a real browser — nothing above covers rendering, layout or
interaction. At http://127.0.0.1:8777:

| check | how |
|---|---|
| canvas renders | images, notes, audio and video as equal-weight cards, no horizontal scroll |
| live ingest | `cp shot.png ~/dump` — a card appears within ~1 s and fills in as stages land |
| drag/paste | drop a file on the window; `Ctrl+V` an image or a block of text |
| search | results show `why` badges (tag / lexical / similar / scene / heard) |
| moments | search for something said in a recording: the card shows the time, Focus plays from it |
| command palette | `⌘K` / `Ctrl-K`, `Tab` switches search ↔ new note, `⌘↵` saves |
| focus view | tags with confidences, OCR tab, graph tab, related strip, segment timeline |
| theme | the `system` button cycles system → dark → light; both must be legible |
| responsive | narrow to ~400px: one column, full-screen overlay, still no h-scroll |
| reduced motion | with `prefers-reduced-motion`, card settle and focus zoom stop |

## Probes

`scripts/probe.py` answers "what can this machine actually load, and what does it cost?"

```sh
uv run scripts/probe.py                    # table of every component
uv run scripts/probe.py needle sqlite      # a subset
uv run scripts/probe.py --json             # machine-readable
```

Results are cached to `data/probe_results.json` and served by `/api/health`, together with the
engine's live view of every model: its pool, backend, and the provider actually running it.

## Layout

```
server/engine/   the model engine: registry, VRAM leases, ONNX sessions (TensorRT → CUDA → CPU)
server/          FastAPI app, SQLite + sqlite-vec, ingest pipeline, model wrappers
web/             Svelte 5 + Vite SPA
vocab/           controlled tag vocabulary (seed with Eye aliases, suppressions)
scripts/         probes, smoke test, bake-offs, Eye conversion, quantisation
data/            gitignored: db, blobs, thumbnails, keyframes, model weights
```
