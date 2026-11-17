# Setup & Usage

## Install

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

   `torchcodec` decodes audio and video and needs FFmpeg on your system. Install the `torchcodec` build that matches your `torch`. For example, CPU-only torch needs the CPU wheel from the PyTorch index.

3. **(Optional) rebuild the frontend.** The built app is already in `web/`. To change it, use Node 20+:

   ```bash
   cd frontend && npm install && npm run build   # outputs to ../web
   npm run dev                                    # hot-reload dev server, proxies /api to :7860
   ```

The model (`google/embeddinggemma-2`, about 1.5 GB) downloads from Hugging Face on first run.

## Web app

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

## CLI

```bash
python zeta.py index ~/Pictures -c photos
python zeta.py search "a dog running on the beach" -c photos
python zeta.py search --file clip.wav -c sounds --only audio
python zeta.py search "how do I reset the router?" -c manuals --intent question
python zeta.py search "parse a CSV file into a dataframe" -c repo --intent code
```

## Query intents

The words in a query are embedded with one of EmbeddingGemma 2's task prompts:

| Intent | `prompt_name` | Prefix | Use for |
| --- | --- | --- | --- |
| `search` (default) | `SearchQuery` | `task: search result \| query:` | General and cross-modal search |
| `question` | `QuestionAnswering` | `task: question answering \| query:` | Questions answered by passages |
| `code` | `CodeRetrieval` | `task: code retrieval \| query:` | Finding source code from a description |
| `fact` | `FactChecking` | `task: fact checking \| query:` | Evidence for or against a claim |
| `similarity` | `SentenceSimilarity` | `task: sentence similarity \| query:` | Text that means the same thing |

Indexed text and code are embedded as documents: `title: {title} | text: {content}`. In the web app, **Auto** uses the question prompt for questions when filtering to Notes, and the search prompt everywhere else.

## Configuration

Everything lives in `src/config.py`. The most useful settings can also be set with environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZETA_DEVICE` | `cuda` if available | `cuda` or `cpu` |
| `ZETA_MODALITIES` | `text,image,audio,video` | Load only these encoders. All variants share one vector space. `text` alone needs about 270M params, `text,image,video` about 440M. |
| `ZETA_EMBEDDING_DIM` | `768` | Matryoshka size: 768 / 512 / 256 / 128. Needs a fresh index. Use 768 or 512 for multimodal. |
| `ZETA_IMAGE_TOKENS` | `280` | Vision budget per image (70 / 140 / 280 / 560 / 1120). `140` is about 1.9× faster per image. Re-index images after changing it. |
| `ZETA_BATCH_TEXT` / `_IMAGE` / `_AUDIO` / `_VIDEO` | GPU `16 / 1 / 4 / 1`, CPU `8 / 1 / 8 / 1` | Max segments per forward pass. Batches halve automatically on CUDA OOM. |
| `ZETA_LOADER_WORKERS` | GPU `min(8, CPUs)`, CPU `2` | Threads that read and decode files while the model runs |
| `ZETA_ANN_THRESHOLD` | `50000` | Build an IVF-HNSW-SQ index once a collection reaches this many segments |
| `ZETA_DATA_DIR` | `./data` | Where the index and uploads live |
| `ZETA_HOST` / `ZETA_PORT` | `127.0.0.1` / `7860` | Web server address |

The model runs in `bfloat16` on GPUs that support it, otherwise `float32`. EmbeddingGemma 2 does not support `float16`.

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
| `GET` | `/api/jobs`, `/api/jobs/{id}` | Indexing progress, throughput and ETA |
| `POST` | `/api/search` | Multipart: `collection`, `text?`, `file?`, `item_id?`, `intent?`, `modalities?`, `top_k?`. Results include `score`, `relevance`, `strong` and `match` = `{kind, start, end, snippet, words}`. |
| `GET` | `/api/media/{collection}/{id}` | Stream the original file of an indexed item |

## Upgrading from the CLIP version

Earlier versions used `jinaai/jina-clip-v2` (1024-d vectors), which are not compatible with EmbeddingGemma 2. The new index lives in `data/index/zeta_db_v2`, so old indexes are left untouched. Re-index your folders to use the new model.

## Notes

- Zeta opts out of the Dark Reader browser extension (`<meta name="darkreader-lock">`), because it has its own light and dark themes.
