# How Zeta works

## Architecture

```
ZetaDB/
├── src/
│   ├── config.py          # All settings (model, prompts, batch sizes, paths, server)
│   ├── embedding.py       # EmbeddingGemma 2 wrapper: task-prompted queries, batched segments
│   ├── segments.py        # Files → segments: text chunks, audio windows, video windows
│   ├── engine.py          # Model lifecycle, search, batched ingestion pipeline + jobs
│   ├── index.py           # LanceDB segments, hybrid calibrated ranking, FTS + ANN indexes
│   ├── api.py             # FastAPI JSON API + serves the built web app
│   └── utils.py           # Vector normalisation, media probing
├── frontend/              # React + Vite + Tailwind v4 + shadcn/ui source
├── web/                   # Built frontend (served by FastAPI)
└── zeta.py                # Entry point: serve | index | search
```

## Ingestion

1. **Scan and deduplicate.** Dependency, cache and build folders (`node_modules`, `venv`, `__pycache__`, …), lock files, minified files and key/certificate files are skipped. Files already in the collection are skipped before any work is done.
2. **Segment.** Each file becomes one or more segments, each with its own vector:
   - **Text and code:** about 1800-character chunks on line boundaries with 200 characters of overlap. Each chunk is embedded as `title: … | text: …`. The title is the Markdown/RST heading, the project-relative path for code, or a descriptive filename.
   - **Audio:** 30 s windows every 25 s, with silence skipped. EmbeddingGemma 2 only hears the first 30 s of an audio input (verified), so without windows the rest of a song or podcast would be invisible.
   - **Video:** 10 s windows of 5 frames each (vision only), plus the soundtrack as audio windows.
   - **Images:** decoded at up to 1024 px using JPEG draft mode. The default 280-token budget corresponds to about 800 px.
3. **Prefetch in parallel.** A thread pool reads, decodes and segments the next file batches while the model embeds the current one.
4. **Embed in batches.** Segments are grouped by kind and batched, with text sorted by length to minimise padding. A CUDA OOM halves the batch and retries. A corrupt input is isolated by bisection, so only it fails.
5. **Write efficiently.** Rows are written to LanceDB 256 at a time. Afterwards the table is compacted, the keyword index is rebuilt, and large collections get an ANN index.

## Ranking

1. The query is embedded once and cached. Text uses the chosen task prompt, a file is embedded segment by segment and averaged, and "more like this" uses the item's mean vector.
2. Candidates come from vector search over segments, plus LanceDB full-text search over file names, folder names and text.
3. Each cosine score becomes a **per-modality z-score**: the query's mean similarity to that modality is subtracted, and the result is divided by the overall spread, both estimated on a cached sample. Text scores systematically higher than images or audio for the same topic, so raw cosine would let notes crowd out photos.
4. Files are ranked by their best segment: `relevance = z + 2 × keyword_coverage^1.5`, plus 1 when the query names the item's kind ("a *video* of …"). Results below half the best relevance fold under **Weaker matches**.

## Measured performance

RTX 4050 Laptop GPU (6 GB), bf16. Segments per second, model only:

| Kind | Batch 1 | Best batch |
| --- | --- | --- |
| Text chunks | 45 | 95 (batch 8) |
| Audio windows (30 s) | 17 | 40 (batch 4) |
| Images (280 tokens) | 9.2 | none better: batch 16 was 6.2 |
| Video windows (5 frames) | 3.4 | none better: batch 4 was 2.8 |

The vision tower runs on the GPU. One image at the default budget is about 2,500 vision tokens and already saturates it, so images and video aren't batched. The image token budget is the lever: about 111 ms per image at 280, 60 ms at 140 and 43 ms at 70.

End to end, on 129 mixed files (48 photos, 64 notes, 12 audio, 4 video): 10.0 s unbatched, 8.8 s with the tuned defaults, and 5.9 s with `ZETA_IMAGE_TOKENS=140`. The same set takes about 5 minutes on CPU.
