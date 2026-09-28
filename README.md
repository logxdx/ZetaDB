# iwashere

Local idea dump with hybrid keyword + semantic search over text and images.
Single-user, local-only. See [spec.md](spec.md) for the full design.

## Setup

```sh
uv sync                        # base deps (Python 3.12)
uv pip install -e '.[needle]'  # structured extraction
cd web && npm install && npm run build
```

## Run

```sh
uv run -m server.app           # http://127.0.0.1:8777
cd web && npm run dev          # hot-reload UI on :5173, proxies /api
```

## Testing

Three layers, cheapest first.

```sh
# 1. unit + regression (no server, no GPU, <1s)
uv run pytest tests/ -q

# 2. API contract against a running server — every endpoint the SPA calls,
#    checking response SHAPE matches what the frontend destructures
uv run scripts/smoke.py

# 3. what this machine can actually load, and what it costs
uv run scripts/probe.py
```

`scripts/smoke.py` is the one that catches the failure mode with no server-side
error: a mismatch between `web/src/lib/api.ts` and `server/api/*` breaks the UI
silently. It also verifies SSE delivers and that thumbnails and full images serve.

### Testing the UI

The SPA has to be exercised in a real browser — nothing above covers rendering,
layout or interaction.

```sh
uv run -m server.app                 # backend + built SPA on :8777
cd web && npm run dev                # OR hot-reload UI on :5173, proxying /api
```

Then, at http://127.0.0.1:8777:

| check | how |
|---|---|
| canvas renders | images and notes as equal-weight cards, no horizontal scroll |
| live ingest | `cp shot.png ~/dump` — a card should appear within ~1s and fill in as stages land |
| drag/paste | drop a file on the window; `Ctrl+V` an image or a block of text |
| search | type in the header bar; results should show `why` badges (tag/lexical/similar) |
| command palette | `⌘K` / `Ctrl-K`, `Tab` switches search ↔ new note, `⌘↵` saves |
| focus view | click a card → tags with confidences, OCR tab, graph tab, related strip |
| theme | the `system` button cycles system → dark → light; both must be legible |
| responsive | narrow to ~400px: one column, full-screen overlay, still no h-scroll |
| reduced motion | with `prefers-reduced-motion`, card settle and focus zoom stop |

## Probes

`scripts/probe.py` answers "what can this machine actually load, and what does it cost?"

```sh
uv run scripts/probe.py                    # table of every component
uv run scripts/probe.py needle sqlite      # a subset
uv run scripts/probe.py --json             # machine-readable
uv run scripts/probe.py --download joytag  # fetch one model's weights
```

Results are cached to `data/probe_results.json` and served by `/api/health`.

## Layout

```
server/    FastAPI app, SQLite + sqlite-vec, ingest pipeline, model runtimes
web/       Svelte 5 + Vite SPA
vocab/     controlled tag vocabulary (seed, JoyTag aliases, suppressions)
data/      gitignored: db, blobs, thumbnails, model weights
```
