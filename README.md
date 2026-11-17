# Zeta DB: Multimodal Vector Database

Zeta DB is a local, multimodal semantic search engine. It embeds **text, code, images, audio and video** into one shared vector space with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), stores the vectors in [LanceDB](https://lancedb.com), and ships a polished, Apple-style web app built with React and [shadcn/ui](https://ui.shadcn.com).

Describe anything in words, or drop in a photo, a sound clip or a video, and Zeta finds what looks, sounds or reads like it, across every modality at once.

## Features

- **One space for all media.** Text, code, images, audio and video are embedded by a single model, so any modality can search any other: "ocean waves at sunset" finds photos, videos and recordings.
- **Finds the moment, not just the file.** Long files are indexed as segments: 30 s audio windows, video scene windows plus their soundtrack, and text and code passages. Results say why they matched ("Matches what's heard at 7:55"), and the viewer jumps to that moment or highlights that passage.
- **Task-aware text queries.** Queries use EmbeddingGemma 2's task prompts: `SearchQuery`, `QuestionAnswering`, `CodeRetrieval`, `FactChecking` and `SentenceSimilarity`. Documents use the `Document` format (`title: … | text: …`). Media is embedded without a prompt, as the developer guide recommends.
- **Fair hybrid ranking.** Cosine scores are calibrated per modality, so photos aren't crowded out by notes. A keyword side (LanceDB full-text search) rewards exact names and numbers like `INV-2047`. Weaker matches fold away.
- **Query by example and fused queries.** Search with a file, find items similar to an indexed item, or combine words with a file ("this photo, but at night").
- **Fast batched ingestion.** Per-kind segment batches tuned for GPU or CPU, parallel decoding and prefetching, OOM-safe batch splitting, deduplication, bulk LanceDB writes and automatic compaction. An ANN index is built for large collections.
- **Collections.** Keep separate libraries, each in its own LanceDB table.
- **Web app.** A justified photo grid, hover-to-preview video, generated artwork for audio, shared-element lightbox, a ⌘K command palette, drag-and-drop anywhere, live indexing progress, light and dark themes.
- **CLI.** Index and search from the terminal.

## Project Structure

```
ZetaDB/
├── data/                  # LanceDB index + files uploaded through the UI (git-ignored)
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
├── zeta.py                # Entry point: serve | index | search
└── requirements.txt
```

## Setup

1. **Clone and create a virtual environment**

   ```bash
   git clone https://github.com/logxdx/ZetaDB.git
   cd ZetaDB
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install PyTorch** for your hardware (see [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:

   ```bash
   pip install -r requirements.txt
   ```

   `torchcodec` decodes audio and video and needs FFmpeg on your system. Install the `torchcodec` build that matches your `torch` (for example, CPU-only torch needs the CPU wheel from the PyTorch index).

3. **(Optional) rebuild the frontend.** The built app is already in `web/`. To change it, use Node 20+:

   ```bash
   cd frontend && npm install && npm run build   # outputs to ../web
   npm run dev                                    # hot-reload dev server, proxies /api to :7860
   ```

## Usage

### Web app

```bash
python zeta.py            # http://127.0.0.1:7860
```

Drop files anywhere on the window. Drop on the left half to **add** them to the current collection, or on the right half to **search** with them. You can also index a whole folder from **Add → Index a folder…**.

| Shortcut | Action |
| --- | --- |
| `⌘K` / `Ctrl+K` | Command palette |
| `/` | Focus search |
| `←` `→` | Previous / next in the viewer |
| `I` | Toggle info panel |
| `Space` | Play / pause audio in the viewer |
| `Esc` | Close |

### CLI

```bash
python zeta.py index ~/Pictures -c photos
python zeta.py search "a dog running on the beach" -c photos
python zeta.py search --file clip.wav -c sounds --only audio
python zeta.py search "how do I reset the router?" -c manuals --intent question
python zeta.py search "parse a CSV file into a dataframe" -c repo --intent code
```

### Query intents

The text part of a query is embedded with one of the EmbeddingGemma 2 task prompts:

| Intent | `prompt_name` | Prefix | Use for |
| --- | --- | --- | --- |
| `search` (default) | `SearchQuery` | `task: search result \| query:` | General and cross-modal search (photos, audio, video, passages) |
| `question` | `QuestionAnswering` | `task: question answering \| query:` | Questions answered by passages |
| `code` | `CodeRetrieval` | `task: code retrieval \| query:` | Finding source files from a description |
| `fact` | `FactChecking` | `task: fact checking \| query:` | Evidence for or against a claim |
| `similarity` | `SentenceSimilarity` | `task: sentence similarity \| query:` | Text that means the same thing |

Indexed text and code files are always embedded as documents: `title: {title} | text: {content}`. The title comes from the Markdown H1 or RST heading, a code file's name, or a descriptive filename. Otherwise it is `none`. In the web app, **Auto** uses the question prompt for questions when filtering to Notes, and the search prompt everywhere else.

## Configuration

Everything lives in `src/config.py`. The most useful settings can also be set with environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZETA_DEVICE` | `cuda` if available | `cuda` or `cpu` |
| `ZETA_MODALITIES` | `text,image,audio,video` | Load only these encoders. All variants share one vector space, so you can add encoders later without re-indexing. `text` alone needs about 270M params, `text,image,video` about 440M. |
| `ZETA_EMBEDDING_DIM` | `768` | Matryoshka size: 768 / 512 / 256 / 128. Changing it needs a fresh index. Use 768 or 512 for multimodal retrieval. |
| `ZETA_BATCH_TEXT` / `_IMAGE` / `_AUDIO` / `_VIDEO` | GPU `16 / 1 / 4 / 1`, CPU `8 / 1 / 8 / 1` | Max segments per forward pass, tuned from the measurements below. Batches halve automatically on CUDA OOM. |
| `ZETA_IMAGE_TOKENS` | `280` | Vision budget per image (70 / 140 / 280 / 560 / 1120). `140` is about 1.9× faster per image at cos 0.957 to the default. Re-index images after changing it. |
| `ZETA_LOADER_WORKERS` | GPU `min(8, CPUs)`, CPU `2` | Threads that read and decode files while the model runs. On CPU they would compete with PyTorch for cores. |
| `ZETA_ANN_THRESHOLD` | `50000` | Build an IVF-HNSW-SQ index once a collection reaches this many rows |
| `ZETA_DATA_DIR` | `./data` | Where the index and uploads live |
| `ZETA_HOST` / `ZETA_PORT` | `127.0.0.1` / `7860` | Web server address |

The model runs in `bfloat16` on GPUs that support it, otherwise `float32`. EmbeddingGemma 2 does not support `float16`.

## How ingestion works

1. **Scan and deduplicate.** Dependency, cache and build folders (`node_modules`, `venv`, `__pycache__`, …), lock files, minified files and key/certificate files are skipped. Files already in the collection are skipped before any work is done.
2. **Segment.** Each file becomes one or more segments with its own vector:
   - **Text and code:** about 1800-character chunks on line boundaries with 200 characters of overlap. Each is embedded as `title: … | text: …`, where the title is the Markdown/RST heading, the project-relative path for code, or a descriptive filename.
   - **Audio:** 30 s windows every 25 s, with silent windows skipped. EmbeddingGemma 2 only hears the first 30 s of an audio input (we verified this), so without windows the rest of a song or podcast would be invisible.
   - **Video:** 10 s windows of 5 frames each (vision only), plus the soundtrack as audio windows.
   - **Images:** decoded at up to 1024 px using JPEG draft mode. The default 280-token image budget corresponds to about 800 px, so larger decodes only cost time.
3. **Prefetch in parallel.** A thread pool reads, decodes and segments the next file batches while the model embeds the current one.
4. **Embed in batches.** Segments are grouped by kind and embedded in batches. Text is sorted by length to minimise padding. Batching pays off on GPUs. On CPU the model is compute-bound, so the CPU defaults are small. A CUDA OOM halves the batch and retries, and a corrupt input is isolated by bisection, so only it fails.
5. **Write efficiently.** Rows are written to LanceDB 256 at a time. Afterwards the table is compacted, the keyword index is rebuilt, and large collections get an ANN index.

Progress, throughput and ETA are shown live in the web app and returned by `GET /api/jobs/{id}`.

### Measured performance (RTX 4050 Laptop, 6 GB, bf16)

Segments per second, model only:

| Kind | Batch 1 | Best batch |
| --- | --- | --- |
| Text chunks | 45 | 95 (batch 8) |
| Audio windows (30 s) | 17 | 40 (batch 4) |
| Images (280 tokens) | 9.2 | none better: batch 16 was 6.2 |
| Video windows (5 frames) | 3.4 | none better: batch 4 was 2.8 |

The vision tower runs on the GPU (all parameters on `cuda:0`; `pixel_values` arrive on the device). One image at the default budget is already about 2,500 vision tokens and saturates the GPU, which is why images and video aren't batched. The image token budget is the lever: about 111 ms per image at 280, 60 ms at 140 and 43 ms at 70.

End to end, on 129 mixed files (48 photos, 64 notes, 12 audio, 4 video): about 10.0 s unbatched, 8.8 s with the tuned defaults, and 5.9 s with `ZETA_IMAGE_TOKENS=140`. The same set takes about 5 minutes on CPU.

## How search ranks results

1. The query is embedded once and cached. Text uses the chosen task prompt, a file is embedded segment by segment and averaged, and "more like this" uses the item's mean vector.
2. Candidates come from vector search over segments, plus LanceDB full-text search over file names, folder names and text, so exact names and numbers are found even when their meaning is vague.
3. Each segment's cosine score becomes a **per-modality z-score**. The mean similarity of the query to that modality is subtracted and the result divided by the overall spread, estimated on a cached sample of vectors. Text vectors score systematically higher than image or audio vectors for the same topic, so raw cosine would let notes crowd out photos.
4. Files are ranked by their best segment: `relevance = z + 2 × keyword_coverage^1.5`, plus 1 if the query names the item's kind ("a *video* of …"). Results below half the best relevance are folded under **Weaker matches**.

The ranking approach follows [DigUp](https://github.com/ARahim3/DigUp)'s measured design.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/status` | Model state, device, dimension, supported extensions and intents |
| `GET` / `POST` | `/api/collections` | List collections with per-modality counts / create one |
| `DELETE` | `/api/collections/{name}` | Delete a collection |
| `GET` | `/api/collections/{name}/items` | Browse items (`?modality=image,video`) |
| `DELETE` | `/api/collections/{name}/items/{id}` | Remove an item |
| `POST` | `/api/index/folder` | `{collection, path}`: index a folder in the background |
| `POST` | `/api/index/upload` | Multipart `collection` + `files[]`: upload and index |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | Indexing progress |
| `POST` | `/api/search` | Multipart: `collection`, `text?`, `file?`, `item_id?`, `intent?`, `modalities?`, `top_k?`. Each result has `score` (cosine), `relevance`, `strong` and `match` = `{kind, start, end, snippet, words}`. |
| `GET` | `/api/media/{collection}/{id}` | Stream the original file of an indexed item |

## Upgrading from the CLIP version

Earlier versions used `jinaai/jina-clip-v2` (1024-d vectors), which are not compatible with EmbeddingGemma 2. The new index lives in `data/index/zeta_db_v2`, so old indexes are left untouched. Re-index your folders to use the new model.

## Notes

- If you use the Dark Reader browser extension, Zeta opts out of it via `<meta name="darkreader-lock">`, because it has its own light and dark themes.
