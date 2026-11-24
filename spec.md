# ZetaDB — Technical Specification

> **[M8]** This project was called *iwashere*; it is now **ZetaDB**, and audio, video, a single
> model engine and Eye (replacing JoyTag) were added. Sections 1–19 are the record as built and
> measured; where they are superseded a note points to **§20**, which describes the current state.

**Version:** 1.0 (v1 scope)
**Status:** Draft for implementation
**Target:** Single-user, local-only, Linux (CachyOS, x86_64), RTX 4050 Laptop 6 GB
**M0 measured:** 2026-09-27 — see §3 and §17 for what changed

---

## 1. What this is

A local idea-dump: you throw text notes and images at it, and it makes them findable again. Retrieval is hybrid — exact keyword, semantic similarity, and a controlled tag vocabulary that the AI populates but never owns. A neighbourhood graph shows what a given item is connected to.

The design pressure throughout is a **hard 6 GB VRAM ceiling** (5853 MiB usable), which makes model
scheduling a first-class architectural concern rather than an implementation detail. The machine is
hybrid-graphics: the Intel UHD iGPU drives the Wayland session, so the dGPU is compute-only and the
budget does not fluctuate with desktop use.

### 1.1 Design decisions and their reasons

| Decision | Reason |
|---|---|
| Capture is a watched folder + the SPA, nothing else | Retrieval value only appears after a few hundred items. Two frictionless paths beat six half-built ones. |
| Stage-batched ingest, not a per-item pipeline | A 12B Q4 GGUF costs 10–20 s to load. Swapping models per item makes ingest unusable. The queue drains one *stage* at a time. |
| PP-OCRv6 is the OCR authority; the VLM never does OCR | PP-OCRv6_medium (34.5M params) outperforms billion-scale VLMs on recognition. Two jobs, two models, no overlap. |
| Two renditions of every OCR result | Grid-projected `layout_text` for VLM/extraction/display; plain `flow_text` for FTS5 and embeddings. Indexing padded grid text wrecks BM25 tokenisation. |
| Graph is neighbourhood-only | Global graphs are demo features. They hairball past a few hundred nodes and nobody uses them twice. |
| Semantic edges seeded automatically | A manual-links-only graph is empty on day one, so nobody ever builds the first link. |
| Every machine-written value carries a confidence | Confidence is what makes auto-accept / review / discard possible, which is what stops tag rot. |
| Human edits are never overwritten by a re-run | `source='human'` is terminal. Re-ingestion updates machine fields only. |

### 1.2 In scope for v1

Text notes and images. Keyword + semantic search over both. Auto-tagging against a controlled vocabulary with review. Auto-categorisation into projects. Neighbourhood graph with manual and semantic links. Dedupe.

### 1.3 Out of scope for v1

PDFs (the layout engine is built so they slot in later — see §6.4). ~~Audio, video~~ — **in scope since [M8], see §20.4**. Multi-user, auth, sync. Mobile. Browser extension. Global hotkey capture.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Browser SPA — Svelte 5 + Vite + TS      http://127.0.0.1:8777 │
└────────────────────────────┬───────────────────────────────────┘
                             │ REST + SSE
┌────────────────────────────┴───────────────────────────────────┐
│  FastAPI (uvicorn, single process)                             │
│    /api/items  /api/search  /api/graph  /api/vocab  /api/events│
├────────────────────────────────────────────────────────────────┤
│  Ingest worker (single thread, sole DB writer)                 │
│    watcher → queue → stage scheduler → GPU arbiter             │
├──────────────────────────────────┬─────────────────────────────┤
│  GPU pools (mutually exclusive)  │  CPU-resident (always)      │
│                                  │                             │
│  Pool A  "vision-small"          │   Needle 3      (own rt)    │
│    PP-OCRv6 det+rec  TensorRT    │   EmbeddingGemma 2 backbone │
│    JoyTag            ONNX int8   │     (ONNX int8) — the half  │
│    EmbeddingGemma 2              │     a query needs, so a     │
│      vision encoder  ONNX int8   │     search never waits on   │
│                                  │     pool A                  │
│  Pool B  "vlm"  (exactly one)    │                             │
│    LFM2.5-VL 3B  Q6 + Q8 mmproj  │                             │
│    via llama-server subprocess   │                             │
│    escalation is EFFORT, not a   │                             │
│    second model (§17.1)          │                             │
├──────────────────────────────────┴─────────────────────────────┤
│  SQLite (WAL)  +  sqlite-vec  +  FTS5      data/zetadb.db      │
│  Blobs and thumbnails on disk              data/blobs/ thumbs/ │
└────────────────────────────────────────────────────────────────┘
```

**Language:** Python 3.12 backend (every model runtime here is Python-first), TypeScript frontend.

---

## 3. Model inventory and VRAM budget

### 3.0 Quantisation policy

No model is loaded at fp32 or fp16 where a quantised form exists.

| Family | Form | Note |
|---|---|---|
| ONNX (PP-OCR) | **static QDQ INT8** | calibrated locally, see §5.1 (JoyTag, which also used this row, was replaced by Eye in [M8] — §20.3) |
| ONNX (EmbeddingGemma 2) | **int8** | published `*_quantized`; [M0] cosine agreement with fp32 is 0.9954–1.0000 and retrieval margin 0.212 against 0.216, at 509 MB instead of 878 MB |
| GGUF decoder | **Unsloth UD-Q3_K_XL** | dynamic per-layer quant |
| GGUF vision projector | **F16** | smallest form Unsloth publishes; quantising an mmproj below Q8 degrades image grounding |
| KV cache | **q8_0** | at 16k context the f16 cache is the largest non-weight allocation |

`server/models/_native.py::pick_onnx` enforces this at load time: it ranks the `.onnx` files in a
model directory and always returns the most-quantised one, so dropping a quantised graph beside the
original is enough to switch a runtime over.

**[M0 measured]** RTX 4050 Laptop (AD107M): 6141 MiB total − 76 idle − 288 driver-reserved =
**5853 MiB usable**. The Intel iGPU drives the display, so nothing else competes for the card.
Budget is set to **5400 MiB**, leaving ~450 MiB for allocator fragmentation.

Weight-dependent VRAM figures below remain estimates until the weights are downloaded; every
other row is measured. Per the quantisation policy, decoders use Unsloth Dynamic (UD) quants and
vision projectors stay at F16/Q8_0 — quantising an mmproj below Q8 measurably degrades grounding.

| Model | Role | Runtime | Precision | VRAM | Pool |
|---|---|---|---|---|---|
| PP-OCRv6_medium det+rec | OCR | ONNX Runtime **TensorRT fp16** | fp16 engines from fp32 graphs | see §5.1 | A |
| ~~JoyTag~~ **Eye** [M8] | vocab tag scoring | ONNX Runtime **TensorRT fp16** (converted from LiteRT, §20.3) | int8 weights, fp16 execution | not yet measured | A |
| EmbeddingGemma 2 audio encoder [M8] | 30 s audio → soft tokens | ONNX Runtime CUDA | int8 | not yet measured | A |
| EmbeddingGemma 2 vision encoder | image → soft tokens | ONNX Runtime CUDA | int8 | **624 MB** at 140 soft tokens | A |
| EmbeddingGemma 2 backbone | text + image → 768-d | ONNX Runtime **CPU** | int8 | **0** (CPU, on purpose) | — |
| **Pool A peak** | | | | **~1.2 GB** measured, one stage at a time | |
| **LFM2.5-VL 3B** | captioner | llama.cpp CUDA | Q6 text + Q8 mmproj, q8_0 KV | **3079 MB** at 16K ctx | B |
| ~~Qwen3.5-4B~~ | ~~captioner~~ | — | — | 3747 MB — **superseded [M7], kept as `models.vlm_qwen35`** | — |
| ~~Gemma 4 12B~~ | ~~escalation~~ | — | — | **~6.3 GB — dropped, see §17.1** | — |
| Needle 3 | structured extraction | Cactus runtime CPU | CQ2 | **0** (34 MB weights) | — |

**[M7] The captioner was chosen by measurement, not reputation** — see §17.9 for the bake-off
against Gemma 3 4B, Qwen3-VL 4B, InternVL3 2B, SmolVLM2 2.2B and Moondream2, and §17.10 for why
the VLM still does not own OCR.

**[M7c] TensorRT is the OCR default and it works here**, at 0.16 s mean per image against 1.72 s on
the CPU EP. What used to make it look unreachable was a missing `libcudnn.so`, which ships inside a
wheel — see the correction in §17.2.

Three VRAM lessons from making this fit:

1. **`-ngl 0` is not CPU-only.** llama.cpp still initialises a CUDA context worth a few hundred MB.
   The embedder only reaches zero VRAM with `CUDA_VISIBLE_DEVICES=""` in its environment.
2. **Releasing a lease frees nothing** unless the arbiter has an unload hook registered for that
   pool. Without them pool A stayed resident, and the next allocation failed.
3. **llama-server outlives its parent — three separate leaks.** (a) no `stop()` on shutdown;
   (b) a model wrapper constructed per call site, so `resolve_or_propose` spawned one server per
   proposed tag; (c) a SIGKILLed or OOM-killed parent never runs `atexit`, and the child is
   reparented to init still holding memory. Only `PR_SET_PDEATHSIG` covers (c). Arm it in
   `preexec_fn`, and resolve libc **at import time**: `preexec_fn` runs in the forked child
   before `exec`, where only async-signal-safe work is allowed, and this process is
   multithreaded, so a `dlopen` there can deadlock on the loader lock.
   `tests/manual_orphan_check.py` hard-kills a parent and asserts the child dies.

### 3.1 The GPU arbiter

> **[M8]** The arbiter is now part of the model engine (`server/engine/core.py`); the rules below
> are unchanged, but pools unload every *registered* model instead of a hand-kept list — §20.2.

A single `GpuArbiter` holds a VRAM budget and hands out **leases**. Rules:

1. Pool A and Pool B are **never co-resident**. Pool A (~2.3 GB) plus Qwen3.5-4B (~3.5 GB) would fit in 7.2 GB on paper, but not with headroom for the desktop compositor and image-token KV growth. Serialise instead of gambling.
2. A lease is held for a whole **stage batch**, never a single item.
3. Pool B is one `llama-server` subprocess. Switching models means kill, wait for VRAM to actually free (poll `nvidia-smi`, don't trust process exit), spawn the other.
   *[M0] Implemented in `GpuArbiter._evict` / `vram.wait_until_free`.*
4. Idle TTL: Pool B unloads after 120 s idle. Pool A after 300 s.
5. If a lease request would exceed budget, the arbiter blocks the requester rather than OOMing. There is no preemption.

```python
class GpuArbiter:
    def lease(self, pool: Literal["A", "B"], model: str | None = None) -> Lease: ...
    def release(self, lease: Lease) -> None: ...
    def free_vram_mb(self) -> int: ...     # polls nvidia-smi, cached 500ms
```

### 3.2 Model licences — verify before committing

- **Qwen3.5-2B/4B** — Apache-2.0. Clear.
- **Gemma 4 12B** — Apache-2.0. Clear.
- **EmbeddingGemma 2** — Apache-2.0. Clear, and it is now the only embedding model.
- **Needle 3** — Apache-2.0. Clear. **[M0]** Note: the package sends anonymous telemetry to a
  Supabase endpoint unless `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1` is set. `server/app.py` sets
  both before import.
- **PP-OCRv6** — Apache-2.0 (PaddleOCR). Confirm the model weights carry the same.
- ~~**jina-clip-v2** — ⚠️ CC-BY-NC-4.0, which blocks any commercial path.~~ **Moot [M6]:** it was replaced by EmbeddingGemma 2, so the licence question went with it.
- **LFM2.5-VL 3B** — the captioner since [M7]. Check Liquid's licence before any commercial path; nothing in §17.9 depends on it, and `models.vlm_qwen35` is a one-line swap back.
- **JoyTag** — ⚠️ README states no licence. Treat as unlicensed; personal use only until clarified.

---

## 4. Ingest pipeline

### 4.1 Sources

1. **Watched folder** — `config.ingest.watch_dirs`, default `~/dump`. `watchdog` observer, 500 ms debounce, waits for file size to stabilise across two polls before enqueueing (avoids reading half-written screenshots).
2. **SPA** — drag-and-drop onto the window, `Ctrl+V` paste (image blob or text), and the quick-note composer.

Both funnel into the same `ingest_queue` table. Nothing else creates items.

### 4.2 Stages

Each stage processes the **entire pending batch** before the next begins. Items are visible in the UI from Stage 0 onward with `status='processing'`; enrichment streams in over SSE.

| # | Stage | Device | Work |
|---|---|---|---|
| 0 | intake | CPU | BLAKE3 content hash, dedupe check, MIME sniff, EXIF, thumbnail (WebP, 400px long edge), dHash. Row written, item appears in UI. |
| 1 | ocr | GPU A | PP-OCRv6 det+rec → quads + text + per-box score. Images only. |
| 2 | layout | CPU | Grid projection (§6) → `layout_text`, `flow_text`, `boxes` JSON. |
| 3 | tag_visual | GPU A | JoyTag → full 5813-tag score vector; persist only the vocab-relevant slice. EmbeddingGemma 2 vision encoder → the `kind='visual'` 768-d vector: the image with **no text in it**, which is what keeps zero-shot term scoring from matching a screenshot whose text merely says "chart". |
| 4 | caption | GPU B | VLM description + `CaptionTags` under one GBNF grammar. **Also re-reads any page PP-OCR under-read (§17.10)**, because pool B is already resident here and doing it at stage 1 would load a second model under a pool A lease. |
| 5 | extract | CPU | Regex `scan` over the full text, then Needle 3 as a guarded supplement (§8.3). |
| 6 | embed | CPU | EmbeddingGemma 2 backbone → the `kind='item'` vector. A note takes its text; an image takes the pixels **interleaved with its own caption and OCR text**, as one vector. |
| 7 | finalize | CPU | Tag fusion + vocab gating (§7), FTS5 upsert, semantic edge seeding (§10), `status='ready'`. |

Stages 1–3 share one Pool A lease. Stage 4 takes a Pool B lease. Stages 0, 2, 5, 6, 7 need no GPU and run while a GPU stage is blocked.

### 4.3 VLM escalation policy

Qwen3.5-4B captions everything. An item is queued for Gemma 4 12B re-description when **any** of:

- OCR box count > 40, or `flow_text` length > 600 chars → text-dense screenshot or document
- image contains a chart/diagram per the Qwen pass's own `is_chart` field
- Needle 3 confidence on `CaptionTags` < 0.50
- user pressed "re-describe" in the UI

Escalation is **batched**: candidates accumulate until the sub-batch has ≥ 8 items or 10 minutes pass, then Gemma loads once, drains, and unloads. Never load a 6.4 GB model for one screenshot.

### 4.4 Failure handling

Per-stage try/except. A failed stage writes `item_errors(item_id, stage, error, ts)` and the item continues to the next stage with that field null. Nothing blocks the batch. The UI shows a small mono `!` badge on partially-enriched cards; "Retry enrichment" re-queues from the failed stage.

---

## 5. OCR

PP-OCRv6_medium, detection + recognition, 50 languages.

**Backend order:** TensorRT primary; ONNX Runtime (CUDA EP, then CPU EP) fallback. Backend is chosen at startup by probing, and the choice is logged and shown in Settings.

**[M2 superseded] TensorRT 10 is installable — from NVIDIA's index, not PyPI.** The earlier
conclusion was wrong because it only checked each package's *latest* version. PyPI's `tensorrt` is
at 11.3 (sdist only), but `tensorrt-cu13-libs==10.16.1.11`, which carries `libnvinfer.so.10`, is
published on `https://pypi.nvidia.com` and installs with
`--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match`. PyPI hosts only the
small `*-bindings` wheels, which is why a PyPI-only search finds nothing usable.

### 5.1 Quantisation recipes, by architecture

Two recipes, because one does not fit both (`scripts/quantize.py`):

| Model | Recipe | Why |
|---|---|---|
| PP-OCRv6 det + rec | **FP16 via TensorRT** — weights untouched | INT8 weights cost accuracy on rec and hours of build time. FP16 is lossless here (**28/28 lines identical to fp32**) and still gets TensorRT's speedup. The INT8 QDQ graphs are still produced and kept, behind `ocr.det_quantised` / `ocr.rec_quantised` |
| JoyTag | **dynamic INT8** | a ViT is MatMul-heavy, exactly what dynamic covers |

Three things that had to be learned the hard way:

- **Static calibration OOMs on the ViT.** Calibrating JoyTag augments the graph to capture every
  intermediate tensor; at 366 MB and 448×448 inputs that exhausted 15 GB of RAM and was killed.
  Dynamic needs no calibration pass and finished in 7 s.
- **Per-channel weights can emit an invalid graph.** `ppocr_rec` quantised per-channel produced a
  `DequantizeLinear` whose axis did not match its conv weight, and ORT refused to load it. The
  quantiser now *verifies its own output* by opening it, and retries per-tensor on failure — a
  quantised graph that will not load is worse than none.
- **Calibration data is the quantisation.** Four ingested images is not a calibration set;
  `scripts/make_calib.py` synthesises 30 spanning dark/light UI, dense tables, documents, prose,
  charts and photographic texture, so the recorded ranges are representative before the real
  library has grown.

#### Recognition is the one model that stays at full precision

INT8 **wrecked OCR accuracy**, and it did so quietly — the detector still found exactly the right
number of boxes, so nothing looks wrong until you read the text:

| exact lines vs fp32 | dashboard | invoice | total |
|---|---|---|---|
| full INT8 | 7/13 | 5/15 | 12/28 |
| INT8, classifier head excluded | 7/13 | 6/15 | **13/28** |

```
'Tag Fusion Dashboard'                  ->  'Tag Fusion Dash0ar'
'2026-03-14'                            ->  '202631'
'finance@acme.example'                  ->  'finiance@acme.examp'
'https//billing.acme.example/inv/8813'  ->  'htps)2bi)lingacme.example)inv½813'
'Subtotal' -> 'Subtota'    'Tax' -> 'ax'    'Due' -> 'Du'
```

Excluding the 18710-way CTC classifier was the obvious fix and it **barely helped** — 12/28 to
13/28. The degradation is spread through the SVTR sequence encoder, not concentrated in the output
layer, so there is no small set of nodes to protect.

Detection is a binary segmentation map and quantises cleanly: INT8 + TensorRT, no measured loss.
Recognition is pinned to full precision (`ocr.rec_quantised = false`). OCR text is what search,
layout ordering and entity extraction are all built on; trading it for 9 MB on a 21 MB model is a
bad deal. The INT8 graph is still built and kept, so flipping one flag switches it on.

Sizes: JoyTag 366 → 93 MB. The OCR INT8 graphs exist (det 62 → 16 MB, rec 21 → 12 MB) but are
not the default: FP16-on-TensorRT is both lossless and faster.

**[M5] Measured on PP-OCRv6 medium, FP16 TensorRT vs fp32 CUDA:**

| | load | VRAM load | VRAM peak | cold read | warm read | accuracy |
|---|---|---|---|---|---|---|
| fp32 CUDA | 3.9 s | 367 MB | 647 MB | 4.59 s | 0.35 s | baseline |
| FP16 TensorRT | 344 s first / **5.0 s cached** | 716 MB | 764 MB | 2.48 s | 0.32 s | **28/28 identical** |

TensorRT costs ~120 MB more VRAM (engine + execution context) and halves cold latency; warm
latency is a wash at this batch size because two images are dominated by Python-side preprocessing,
not GPU time. The 344 s build is one-time and cached per GPU arch (`sm89`).

**A batch larger than the profile fails at RUN time, not load time.** Recognition grouped crops by
width bucket and sent each group whole; a group of 9 against a profile built for `[1..8]` threw
`does not satisfy any optimization profiles` mid-inference — and the per-session fallback then
masked it by dropping to the CUDA EP, so the logs still said TensorRT while TensorRT was not
running. `trt_max_batch()` now derives the cap from `TRT_PROFILES` so the two cannot drift.

**Measure quantisation against a task-level baseline, never just size and latency.** Both looked
excellent — 9× faster, 40% less VRAM — while the output was quietly turning to mush.

### 5.2 What TurboOCR gets right

[TurboOCR](https://github.com/aiptimizer/TurboOCR) serves the same PP-OCRv6 models on TensorRT in
production (200–559 img/s on one GPU), and it settles three questions this spec had got wrong:

1. **FP16, not INT8.** TurboOCR is "TensorRT FP16" throughout; INT8 appears nowhere. That matches
   the measurement here — INT8 weights cost more than half the recognised text — and it separates
   the two axes cleanly: *weight precision* (what the ONNX stores) and *execution precision* (what
   the TensorRT builder picks). FP16 is the latter, chosen by the builder, weights untouched, and
   it reportedly gives 3–5× on detection with no accuracy cost.
2. **Long engine builds are normal.** "~90 s on a 5090 and up to an hour on older ones." A 45-minute
   build on a laptop 4050 was not pathological, and the fix is `TRT_OPT_LEVEL=3`, which they report
   cutting build time 3–5× for a small speed regression (`ocr.trt_opt_level`).
3. **Many narrow profiles beat one wide one.** Their recognition model carries **nine** discrete
   width profiles (320/480/800/1200/1600/2000/2500/3200/4000) and they attribute ~8× to that.
   ORT's TRT EP exposes only a single profile, so the equivalent here is keeping the range tight
   around the width buckets actually fed to it — the same principle that fixed the detector's
   32×32→960×960 span.

Two further ideas not adopted, recorded so the choice is deliberate: a fused
`resize+normalize+transpose` CUDA kernel, and a custom CUDA connected-components kernel for DB
post-processing (this pipeline does both on CPU via OpenCV). Both matter at 500 img/s; neither
matters for batch ingest of a personal archive.

**Why detection and recognition are not treated alike.** Two separate axes get conflated here:
*weight precision* (what the ONNX file stores) and *engine backend* (who executes it). Recognition
is fp32 because INT8 weights cost more than half the text (below) — that is a quantisation
decision. Whether fp32 recognition should still run **through** TensorRT is a different question,
and §5.2 measures it rather than assuming the INT8 build failures carried over.

TensorRT builds an **INT8** engine directly from the QDQ graph — a QDQ model carries its own
scales, so no separate calibration table is needed. Explicit shape profiles are mandatory:
PP-OCR's own `inference.yml` advertises a 4000×4000 detection profile, which produces an enormous
engine and a very long build for shapes we never feed it. `TRT_PROFILES` narrows detection to
`ocr.det_long_edge` and recognition to the largest width bucket.

**[M2] cuDNN needs help being found.** The CUDA EP `dlopen`s the unversioned `libcudnn.so`, but
torch ships it as `nvidia/cudnn/lib/libcudnn.so.9`, and glibc matches an already-loaded library by
SONAME so `ctypes` preloading does not help. `server/models/_native.py` symlinks the unversioned
names into `$VIRTUAL_ENV/lib/zetadb-native`, puts it on `LD_LIBRARY_PATH` and re-execs once.
It must link cuDNN's *siblings* too (`libcudnn_graph`, `libcudnn_engines_*`): linking only the
dispatcher yields `Invalid handle. Cannot load symbol cudnnCreate`.

TensorRT engines are built once at first run from exported ONNX, cached in `data/models/trt/`, and keyed by `{model_hash}_{trt_version}_{gpu_name}_{max_shape}`. Engine build takes minutes; do it in the background with a visible progress state, never lazily inside an ingest batch.

Dynamic shapes: detection at 960px long edge; recognition at fixed height 48, width bucketed to `[64, 128, 256, 512]` to keep the engine profile count small.

Boxes with recognition score < 0.5 are dropped before layout.

---

## 6. Layout ordering

Adapted from LiteParse's grid projection algorithm. LiteParse solves "PDF glyphs have coordinates but no reading order" — OCR output has exactly the same shape, so the algorithm transfers with minor changes.

Input: `[{quad: [(x,y)×4], text: str, score: float}]`.

### 6.1 Algorithm

**Step 0 — Deskew.** Reduce each quad to an axis-aligned bbox, then estimate global skew from
**baseline drift within provisional line groups**, take the median slope, and rotate by `-θ` when
`0.5° < |θ| < 20°`.

*[M2] The obvious implementation is wrong.* Measuring the angle between consecutive boxes in `y`
order mixes cross-line pairs with same-line pairs, and same-line pairs can have negative `dx`, so
`atan2` returns values near ±π. The median of that mixture rotated real pages by ~180°, reversing
both line order and reading order — and a synthetic test passed anyway because its boxes had no
backward pairs. Grouping into lines first and only measuring left-to-right within a group keeps
every `dx` positive. `test_deskew_does_not_flip_an_unskewed_page` pins it.

**Step 1 — Line grouping.** `H = median(y1-y0)`. `tol = max(0.5*H, 5.0)` px. Sort by y-centre, greedily group; start a new line when `|y_centre - line_mean_y| > tol`. Sort within a line by `x0`. Emit a blank line where the inter-line gap exceeds `1.4*H`.

**Step 2 — Character width.** `cw_i = (x1-x0) / max(1, len(text_i))`; `W = max(1.0, median(cw_i))`. This is the OCR substitute for LiteParse's font metrics.

**Step 3 — Anchor extraction.** For every box emit three candidates in grid units (`x0/W`, `x1/W`,
`cx/W`), each rounded to 0.25. Count votes **per distinct line, not per box**; keep an anchor at
`≥ max(3, ceil(0.08 * n_lines))` votes; drop any within 1.0 grid unit of a stronger one.

*[M2] Per-box voting invents anchors.* An anchor means "text aligns here down the page"; two words
sharing an x within one line is not alignment. Counting per box let prose accumulate seven phantom
anchors so it never qualified for the bypass.

**Step 4 — Snap classification.** For each box, match its left/right/centre against kept anchors within 0.5 grid units. Priority `left > right > centre`. Ties resolve to the anchor shared by the most boxes. Unmatched boxes are *floating*.

**Step 5 — Flowing-text bypass.** A *block* is a run of lines uninterrupted by a blank line. It is
prose when its **median widest internal gap ≤ 3 columns** and median line width > 50% of page
width; then join its box texts with single spaces and skip projection.

*[M2] Anchor count alone does not work.* LiteParse's `≤4 anchors AND >50% width` test misfires on a
two-column right-aligned table, which spans the width and has only two anchors — it was rendered
as prose, destroying the alignment. Prose is characterised by *word-sized* internal gaps; a table
always has at least one large one. Gap width is the reliable signal, so it replaced the anchor
count.

**Step 6 — Grid projection.** Priority (floating → left → right → centre) decides whose *desired*
column wins a contested cell, but rendering then walks the line in **x0 order with monotonically
increasing columns**.

*[M2] Collision resolution must never reorder text.* Resolving overlaps by sliding a box rightwards
let a late-priority box jump past words that belonged before it, scrambling prose into word salad.
Reading order within a line is x0 order and is inviolable; a collision may only push a box further
right *within* that order.

```
# desired column, assigned in priority order
left/floating : target = round(x0 / W)
centre        : target = round(cx / W) - len(text)//2
right         : end    = max(round(x1 / W), forward_right[anchor])
                target = end - len(text)
desired = max(0, target, forward_left[anchor])

# render in x0 order, monotonically
col = max(desired, cursor + 1);  cursor = col + len(text)
if anchored: (forward_right if right else forward_left)[anchor] = col (+len for right)
```

Forward anchors hold table columns aligned across rows, and they are tracked **separately per
alignment**: a left column propagates its start, a right column propagates its **end**.

*[M2] One shared forward map breaks right alignment.* Propagating a right-aligned column's *left*
edge walks it rightward every row — a 3-row money column came out at right edges 48, 48, 51.

**Step 7 — Post-process.** In blocks that are > 80% whitespace, collapse runs of ≥ 4 spaces to 2. Strip the common leading indent. Trim trailing whitespace. Remove nulls.

### 6.2 Two outputs, always

| Output | Content | Consumers |
|---|---|---|
| `layout_text` | The grid render, whitespace preserved | VLM prompt, Needle 3 input, UI display |
| `flow_text` | Lines joined in reading order, single spaces, zero padding | FTS5 index, both embedding models |
| `boxes` (JSON) | Post-deskew bboxes + text + score | Search-hit highlighting on the image |

Indexing `layout_text` in FTS5 would pollute the tokeniser with runs of spaces and destroy BM25 length normalisation. Keep them separate.

### 6.3 Parameters

All of the above live in `config.layout.*` so they can be tuned against real screenshots without a code change.

### 6.4 PDF path (v1.5, designed for now)

The layout stage takes `list[TextBox]` and knows nothing about where the boxes came from. A PDF ingest source that uses **LiteParse proper** — exact text layer, no OCR, far faster and more accurate — produces the same `TextBox` list and reuses stages 2–7 unchanged. Do not couple the layout module to PP-OCR types.

---

## 7. Tagging and the vocabulary system

### 7.1 How JoyTag is actually used

> **[M8] Superseded:** JoyTag is replaced by Desert Ant Labs' Eye, which keeps this section's
> design (a scorer against our vocabulary, not a generator) with a calibrated curve per tag and a
> type head that names screenshots, whiteboards and receipts — §20.3. Kept as the record of why.

JoyTag returns a **sigmoid score for every one of its ~5000 Danbooru tags, independent per tag**:

```python
scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
```

So it is not used as a tag *generator* with a global 0.4 cutoff. It is used as a **scorer against our vocabulary**: for each of our terms that has a Danbooru alias, we read off that term's applicability confidence. That gives us per-term, independently-calibrated thresholds.

The constraint this creates: a vocab term is only JoyTag-scorable if it is expressible in Danbooru
tag space. **[M4] Verified against the real `top_tags.txt` (5813 tags):** `tree`, `night`,
`building`, `map`, `chart`, `forest`, `grass`, `sky` are present; `screenshot` and `text` are not.
Of the 54 seed terms, 38 have a Danbooru alias and 16 are CLIP-only. Terms without an alias fall
through to the CLIP path, which is why both signals are needed for a mixed corpus.

Two practical notes. First, JoyTag is Danbooru-schema and heavily NSFW-weighted; `config.tagging.suppressed_tags` holds a blocklist applied before fusion. Second, its reported F1 is 0.578 across its whole set including photos — treat scores as a strong signal, not ground truth.

### 7.2 Three tagging signals

| Signal | Applies to | Produces |
|---|---|---|
| **JoyTag** | vocab terms with a Danbooru alias | independent sigmoid 0–1 |
| **EmbeddingGemma 2 zero-shot** | *any* vocab term | calibrated cosine z-score |
| **VLM + Needle 3** | any vocab term | membership + Needle confidence |

The zero-shot path is what makes arbitrary user vocabulary work at all, and it costs nothing
extra: the image vector is computed once in Stage 3 and **persisted there** as `kind='visual'`.
It is still called "the CLIP path" throughout the config for continuity, but there is no CLIP and
no separate text tower — a term string and an image are directly comparable because they share one
space (§9.3).

**[M3, superseded] jina-clip-v2 had to be split into two single-tower graphs.** EmbeddingGemma 2
needs no such surgery: pooling is inside the graph and `sentence_embedding` is the finished vector.
Kept because the failure mode generalises to any dual-encoder ONNX export. The published export declares
`input_ids` *and* `pixel_values` as required inputs. Requesting only the text output does **not**
prune the vision branch — ORT still runs it over whatever dummy pixels you pass, which both wasted
the compute and OOMed the card on a batch of 512×512 dummies (`Failed to allocate ... 113306112`,
inside `vision_model`). `onnx.utils.extract_model` splits it once on first use into a 312 MB image
tower (GPU, pool A) and a 563 MB text tower (CPU). Image-tower VRAM fell from 616 MB to 122 MB and
text embedding got ~10× faster, with identical cosines.

**[M3] One shared instance, not one per stage.** Tagging and embedding each constructed their own
`JinaClip`; the second CUDA session failed to allocate and the CLIP path silently produced nothing
(visible only as the term-score count dropping from 162 to 114 — exactly the 38 JoyTag-scorable
terms). Model wrappers are process-wide singletons behind `engine()`.

### 7.3 CLIP score calibration — do not skip this

Raw CLIP cosines are compressed (typically 0.10–0.35) and the usable range differs per term, so a single global threshold is meaningless. For each vocab term, compute its cosine distribution over a random sample of N = 500 existing images, store `μ_v` and `σ_v` on the term, and score with the z-score `(cos - μ_v) / σ_v` squashed through a logistic. Recalibrate when the corpus grows 2×.

Term prompt template: `"a photo of {term}"` for concrete nouns, `"{term}"` for abstract/style terms
— a per-term `prompt_template` column, defaulting to the former.

**[M6] Measured cosine bands, which is why this section is not optional — and every number moved
when the model did.** In EmbeddingGemma 2's space an exact query→term match is 1.000, a plural
0.890, a synonym 0.706–0.924, and an unrelated pair 0.517–0.689, so `tag_seed_min_cos` is **0.70**
and that line is thin. Vector floors are **0.55** for both kinds, against the 0.30/0.18 they
replace — EG2 has a high floor and carrying the old numbers across would have left them filtering
nothing at all.

**[!] The absolute cosine is the weak part of this path.** Over 10 real images the entire band
across six unrelated terms is 0.470–0.671: "invoice" and "tree" score the same images almost
identically. EG2 is retrieval-trained rather than contrastively trained, so its text↔image cosines
compress into a narrow high band and `clip_abs_floor` only catches true garbage. The per-term
z-score is carrying essentially all of the signal. If the review queue looks noisy, cut
`tagging.w_clip` rather than adjusting the floor.

### 7.4 Fusion

```
j = joytag_score(v, i)                     if v.joytag_alias is not null
c = logistic((cos(E_txt(v), E_img(i)) - μ_v) / σ_v)
m = needle_confidence                      if v ∈ CaptionTags(i) else 0

fused = Σ(w_s · s) / Σ(w_s)                over present sources only
defaults: w_j = 0.45, w_c = 0.30, w_m = 0.25
```

- `fused ≥ θ_v` → tag applied, `source='machine'`, `state='accepted'`
- `0.25 ≤ fused < θ_v` → `state='pending'`, shown in the review queue
- below → discarded (score still retained in `tag_scores` for later recalibration)

`θ_v` starts at 0.40 and adapts from user feedback: `θ_v ← clamp(θ_v ± 0.02, 0.15, 0.85)` on each accept/reject. Simple, transparent, and inspectable in the UI.

### 7.5 Vocabulary lifecycle (seeded + gated proposals + background merge)

**Seed.** `vocab/seed.yaml`, hand-authored, plus `vocab/joytag_aliases.yaml` mapping terms to Danbooru tags.

```yaml
- term: tree
  aliases: [trees]
  joytag: tree
  kind: object
- term: wireframe
  aliases: [wireframes, lo-fi mockup]
  joytag: null          # CLIP-only
  kind: artifact
```

**Write-time gate.** When the VLM proposes a term not in the vocabulary, embed it with EmbeddingGemma 2 against all existing term embeddings:

| max cosine | action |
|---|---|
| ≥ 0.86 | auto-alias to the existing term, no new entry |
| 0.72 – 0.86 | queue as *possible duplicate*, showing the near-match |
| < 0.72 | queue as *new term proposal* |

This is what stops `tree` / `trees` / `foliage` / `greenery` fragmenting the index.

**Rest-time merge job.** Nightly (or on demand): embed all terms, run agglomerative clustering at distance 0.18, and propose merges. Additionally propose a merge when two terms co-occur on > 60% of each other's items — a co-occurrence signal that catches synonyms embeddings miss.

**Merging** rewrites `item_tags` to the surviving term, appends the losing term to its alias list, and is reversible from an `audit_log` entry for 30 days.

### 7.6 Human authority

`item_tags.source ∈ {'human','machine'}`. A human-sourced row is never modified or deleted by any pipeline stage. Re-ingestion updates machine rows only. In the UI, human tags render as solid pills, machine tags as dotted, pending as dashed with the confidence shown in mono.

---

## 8. Structured extraction — Needle 3

Needle 3: 121M params, CQ2 2-bit, 8–29 MB, Apache-2.0, Pydantic schema → byte-level grammar → guaranteed-parseable output, with a calibrated confidence head. CPU-only, so it never contends for VRAM.

Four schemas run per item:

### 8.3 [M5] What extraction actually ships

Needle 3 turned out to be the wrong tool for this job, and the measurements are unambiguous:

| Input | Result |
|---|---|
| Clean 2-line invoice string (M0) | correct, identical across 3 runs |
| 779-char caption + `layout_text` | `None`, or fabricated: dates not in the source, twelve invented `$1.00` amounts, `example.com/price/1.00` URLs |
| 196-char caption, wording A | dates/amounts/URL right, one invented total, `emails`/`orgs`/`people` all filled with the same junk pair |
| 196-char caption, wording B | `None`, 3 times out of 3 |
| 114-char note with two explicit intentions | `None`, 3 times out of 3 |

**So the primary extractor is a regex pass over the full text.** Dates, amounts, URLs and emails
are *defined by their shape*: a regex has perfect precision, runs over everything instead of a
320-character window, needs no model, costs nothing, and cannot fabricate. Using a 2-bit 121M LLM
for it was the wrong instinct.

Needle still runs, but only as a supplement that may add spans the regex missed, behind three
guards: a shape test per field, a **verbatim check** that the span appears literally in the source
(normalising commas and spacing), and de-duplication against the regex results. Fields with no
checkable shape — `orgs`, `people` — were removed, because the verbatim check alone cannot catch
cross-contamination when the junk value really is somewhere in the text.

One genuine subtlety: bare decimals are money only in context. `86.2` on a benchmark row is a
score, so a bare decimal counts as an amount only on a line naming one (`total`, `tax`, `due`…).
Measured on the test corpus: 7/7 fields found on the invoice, 0 false positives on the dashboard.

`ActionItems` is **not shipped**. Needle returned `None` on every attempt, and the VLM route would
require notes to pass through a text-only captioning call. The `todos` table stays; nothing writes
to it.

```python
class CaptionTags(BaseModel):          # from VLM caption + layout_text
    tags: list[str]                     # constrained to vocab ∪ proposals
    subject: str
    setting: str | None
    is_chart: bool
    is_screenshot: bool

class OcrEntities(BaseModel):          # Needle supplement only; see 8.3
    dates: list[str]                    # ISO-8601
    amounts: list[str]
    urls: list[str]
    emails: list[str]                   # orgs/people removed: unverifiable

class ProjectRouting(BaseModel):       # from caption + flow_text + note body
    project: str | None                 # constrained to existing project names
    is_new_project: bool
    proposed_name: str | None

# ActionItems: specified, measured, not shipped (see 8.3)
```

**Confidence policy**, uniform across all four: `≥ 0.70` auto-apply · `0.40–0.70` apply as pending, surfaced for review · `< 0.40` discard and log. The confidence is stored per field group, not per item, so a good `OcrEntities` result isn't thrown away because `ProjectRouting` was uncertain.

### 8.1 [M0] What was actually measured

Validated on this machine against `cactus-needle` 3.0.5:

- **`linux-x86_64` is supported.** The platform tag resolves to `manylinux2014_x86_64`; the docs only
  showed arm64 as an example. The wheel is pure Python — the 1.3 MB engine `.so` and the 34 MB
  `needle3.cact` weights are fetched at runtime into `~/.cache/cactus-needle/`, with automatic
  glibc/musl retry. **The top risk in §17 is closed.**
- **The base weights do carry a confidence head** (`_confidence_head_present` → `True`). The warning
  about fine-tune-only confidence applies to weights that lack it, not to `needle3` base.
- **Confidence is on `complete()`, not `extract()`.** `needle.extract()` returns the bare typed
  object. To get confidence the pipeline must call `Needle.complete()` / `_complete()` and read
  `response["confidence"]` alongside `function_calls`. The dict also carries `prefill_tps`,
  `decode_tps` and `peak_ram_mb`, which are worth logging per stage.
- **`extract()` returns `None` when the model emits no function call**, which happens on short or
  abstract inputs. This is a normal outcome, not an error: every call site must handle `None`, and
  the pipeline should pass the fullest text it has (caption **+** `layout_text`) rather than a
  terse fragment.
- **Deterministic and accurate on extractive spans**, ~0.5 s/call on CPU: the invoice probe returned
  identical dates/amounts/URLs across three runs.
- **Poor at generative fields.** Asked for `CaptionTags` it returned `tags=['Tag Fusion','Tag Fusion']`
  — duplicated, and it missed the obvious `dashboard`, `chart`, `dark`, `screenshot`. It also missed
  `orgs=['Acme Corp']` on the invoice.

**Consequent design change.** Needle 3 is a grammar-constrained span extractor, so give it only
extractive work. `CaptionTags` moved **off** Needle and onto the VLM under a llama.cpp **GBNF
grammar** — same guaranteed-parse property, from a model that can actually see the image. Needle
keeps `OcrEntities` and `ActionItems`, and is given the fullest text available (caption +
`layout_text`) with a 40-character floor, below which it reliably returns `None`.

`ProjectRouting` is **not implemented**: it is closed-set classification that the grammar would
make structurally safe, but M0 gave no evidence Needle does it *well*, and nothing in M1–M5 needs
it. It stays specified and unbuilt rather than shipped untested.

### 8.2 [M5] Captioning with Qwen3.5-4B

- **Thinking must be switched off.** Qwen3.5 is a hybrid reasoner. Left on, it spends the entire
  token budget in `reasoning_content` and returns an **empty** `content` — a silent failure that
  looks like a broken vision pipeline. `chat_template_kwargs: {"enable_thinking": false}` fixes it.
  (The reasoning trace did prove the vision path worked: it read the dashboard's title, model names
  and columns correctly.)
- **GBNF is stricter than the docs suggest.** llama.cpp rejected both `{m,n}` repetition and `\xNN`
  escapes inside character classes with a bare `400 ... failed to parse grammar`. The working
  grammar uses only alternation, `*`, and plain character classes.
- **Surface the response body on HTTP errors.** `raise_for_status()` alone turns llama.cpp's
  specific complaint into an undiagnosable `400 Bad Request`.
- **Never sleep only in the `except` branch when polling `/health`.** A 503 means "still loading",
  not "unreachable"; the request *succeeds*, so the loop never slept and hot-spun thousands of
  requests per second until the log filled and the server died.
- Measured: **3585 MB** resident at 8K ctx, **2.4–5.8 s** per image including the grammar pass.

`Needle.embed()` works and returns normalised 3072-d vectors, but §9 stands: these are not used for
search.

`ProjectRouting.project` is grammar-constrained to the set of existing project names plus a null option, which makes hallucinated projects structurally impossible rather than merely unlikely.

**Build target risk:** Cactus docs show `needle build --platform linux-arm64`. Confirm a `linux-x64` target exists in M0. If it does not, the fallback is the C API via `ctypes`, or WASI through `wasmtime`. This is the single least-certain dependency in the stack — validate it first.

---

## 9. Data model

SQLite in WAL mode. **The ingest worker is the only writer**; API writes are funnelled through a single-threaded write queue. Anything else produces `SQLITE_BUSY` under concurrent enrichment. `busy_timeout=5000`, `synchronous=NORMAL`, `foreign_keys=ON`.

Blobs and thumbnails live on disk, not in the database — the vec0 shadow tables make the file large enough already.

### 9.1 Core tables

```sql
CREATE TABLE items (
  id            INTEGER PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('note','image')),
  status        TEXT NOT NULL DEFAULT 'processing'
                  CHECK (status IN ('processing','ready','error')),
  title         TEXT,
  body          TEXT,                      -- notes: the text. images: null
  blob_path     TEXT,                      -- images: relative to data/blobs
  thumb_path    TEXT,
  mime          TEXT,
  width         INTEGER, height INTEGER, bytes INTEGER,
  content_hash  TEXT UNIQUE,               -- BLAKE3
  dhash         BLOB,                      -- 8 bytes, 64-bit perceptual
  near_dup_of   INTEGER REFERENCES items(id) ON DELETE SET NULL,
  project_id    INTEGER REFERENCES projects(id) ON DELETE SET NULL,
  caption       TEXT,
  caption_model TEXT,                      -- 'qwen3.5-4b' | 'gemma4-12b'
  layout_text   TEXT,
  flow_text     TEXT,
  boxes_json    TEXT,
  source        TEXT NOT NULL,             -- 'watch' | 'upload' | 'paste' | 'compose'
  source_path   TEXT,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  seen_at       INTEGER
);
CREATE INDEX idx_items_created ON items(created_at DESC);
CREATE INDEX idx_items_project ON items(project_id);
CREATE INDEX idx_items_status  ON items(status);

CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  color TEXT,
  created_at INTEGER NOT NULL
);

CREATE TABLE vocab_terms (
  id INTEGER PRIMARY KEY,
  term TEXT NOT NULL UNIQUE,
  kind TEXT,                               -- object | style | artifact | abstract
  joytag_alias TEXT,                       -- null = CLIP-only term
  prompt_template TEXT DEFAULT 'a photo of {term}',
  threshold REAL NOT NULL DEFAULT 0.40,    -- θ_v, adapts from feedback
  clip_mu REAL, clip_sigma REAL,           -- per-term calibration
  state TEXT NOT NULL DEFAULT 'active'
          CHECK (state IN ('active','pending','merged','retired')),
  merged_into INTEGER REFERENCES vocab_terms(id),
  accept_count INTEGER DEFAULT 0,
  reject_count INTEGER DEFAULT 0,
  created_at INTEGER NOT NULL
);

CREATE TABLE vocab_aliases (
  term_id INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  alias TEXT NOT NULL UNIQUE
);

CREATE TABLE item_tags (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  term_id INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  confidence REAL NOT NULL,                -- fused score; 1.0 for human
  source TEXT NOT NULL CHECK (source IN ('human','machine')),
  state  TEXT NOT NULL CHECK (state IN ('accepted','pending','rejected')),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (item_id, term_id)
);
CREATE INDEX idx_item_tags_term ON item_tags(term_id, state);

-- raw per-source scores, retained so recalibration never needs re-inference
CREATE TABLE tag_scores (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  term_id INTEGER NOT NULL REFERENCES vocab_terms(id) ON DELETE CASCADE,
  joytag REAL, clip REAL, vlm REAL,
  PRIMARY KEY (item_id, term_id)
);

CREATE TABLE links (
  id INTEGER PRIMARY KEY,
  src INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  dst INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('manual','semantic')),
  weight REAL NOT NULL DEFAULT 1.0,
  label TEXT,
  created_at INTEGER NOT NULL,
  UNIQUE (src, dst, kind)
);
CREATE INDEX idx_links_src ON links(src);
CREATE INDEX idx_links_dst ON links(dst);

CREATE TABLE entities (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,                      -- date|amount|url|email|org|person
  value TEXT NOT NULL,
  confidence REAL NOT NULL,
  PRIMARY KEY (item_id, kind, value)
);

CREATE TABLE todos (
  id INTEGER PRIMARY KEY,
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  due_date TEXT,
  done INTEGER NOT NULL DEFAULT 0,
  confidence REAL NOT NULL
);

CREATE TABLE ingest_queue (
  id INTEGER PRIMARY KEY,
  item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  escalate INTEGER NOT NULL DEFAULT 0,     -- flagged for Gemma 4
  enqueued_at INTEGER NOT NULL
);

CREATE TABLE item_errors (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  stage TEXT NOT NULL, error TEXT NOT NULL, ts INTEGER NOT NULL
);

CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY, ts INTEGER NOT NULL,
  action TEXT NOT NULL, payload TEXT NOT NULL   -- JSON, for undo
);
```

### 9.2 FTS5

```sql
CREATE VIRTUAL TABLE fts_items USING fts5(
  title, body, flow_text, caption, tags,
  content='',                       -- contentless; we own the sync
  tokenize='unicode61 remove_diacritics 2'
);
```

Contentless and written explicitly in Stage 7 rather than by triggers — triggers would fire mid-enrichment when `flow_text` and `caption` are still null. `tags` is the space-joined accepted term names, which is what lets a tag name score through BM25 as well as through the tag path.

BM25 column weights at query time: `bm25(fts_items, 3.0, 1.0, 1.0, 1.5, 2.0)`.

### 9.3 Vector tables

**One space, 768-d, for everything.** EmbeddingGemma 2 embeds notes, images and vocabulary terms
into the same geometry, so any of them can be compared with any other. This replaced a two-space
arrangement — Qwen3-Embedding-0.6B for text, jina-clip-v2 for images — which could not compare a
note with an image at all, and whose workaround is described and retired in §9.4.

```sql
-- EmbeddingGemma 2, every modality. Two kinds per image, and the
-- distinction matters:
--   kind='item'    the searchable vector. A note's text, or an image
--                  INTERLEAVED with its own caption and OCR text.
--   kind='visual'  the image with NO text in it. Zero-shot term scoring
--                  reads this one, so that scoring "chart" against a
--                  screenshot cannot succeed merely because the screenshot
--                  says "chart" (that leak is what vocab_terms.requires_text
--                  exists to prevent).
CREATE VIRTUAL TABLE vec_items USING vec0(
  rowid INTEGER PRIMARY KEY,
  item_id INTEGER,          -- metadata column
  kind TEXT,                -- metadata column: 'item' | 'visual'
  embedding FLOAT[768]
);

-- Vocabulary term embeddings, in that same space - which is what lets a
-- term string be scored against an image directly, with no CLIP text tower.
CREATE VIRTUAL TABLE vec_vocab USING vec0(
  term_id INTEGER PRIMARY KEY,
  embedding FLOAT[768]
);

-- Perceptual hashes for near-duplicate detection, Hamming distance.
CREATE VIRTUAL TABLE vec_phash USING vec0(
  item_id INTEGER PRIMARY KEY,
  hash BIT[64]
);
```

**Normalisation is not optional.** Every stored vector is L2-normalised, and `pack()` is the only
place that happens. Matryoshka truncation (`embeddings.truncate_dim`, 0 = keep all 768) slices a
unit vector, and a slice of a unit vector is not unit length — skipping the re-normalise degrades
ranking silently, with plausible scores in the wrong order.

**vec0 constraints to design within:** max 16 metadata columns, max 4 partition keys, max 16 auxiliary columns. Metadata supports only `=, !=, <, <=, >, >=` — no `LIKE`, no `IS NULL`. Flat exact KNN is single-digit milliseconds to a few hundred thousand vectors, far beyond this corpus, so no ANN index is needed.

**[M3] sqlite-vec's default metric is L2, not cosine — and it is easy to get this wrong silently.**
Every stored vector is L2-normalised, so `d² = 2(1 − cos)`, i.e. `cos = 1 − d²/2`. Reading
`1 − distance` as a cosine (which looks right and never errors) reported a true 0.671 as 0.189.
That single mistake disabled *every* similarity threshold at once: tag-seed expansion never fired,
semantic graph edges were never seeded, and vector floors never triggered. `cosine_from_l2()` is
the only place this conversion is allowed to happen, and `test_cosine_conversion_against_sqlite_vec`
asserts both that the identity holds against the real extension and that the naive reading is wrong.

**[M0 verified]** against SQLite 3.53.1 + sqlite-vec v0.1.9: `bit[64]` is the correct declaration
(`binary[64]` is rejected), and both inserts and match queries must wrap the blob in `vec_bit(?)`
or sqlite-vec reports a float32/bit type mismatch. Distance is Hamming via `vec_distance_hamming`.
`int8[N]` works the same way through `vec_int8()`. Metadata `LIKE` is rejected exactly as
documented. FTS5 and `float[1024]` both confirmed present.

### 9.4 ~~One deviation from the brief, flagged~~ — retired [M6]

The brief specified Qwen3-Embedding-0.6B for text-only items and jina-clip-v2 for images, and this
section recommended *additionally* embedding every image's `caption + flow_text` a second time
through Qwen, under `embeddings.qwen_covers_image_text`. The reason was that two spaces cannot be
compared, so without it nothing contained both notes and images and cross-modal ranking had no
geometry to work in.

**EmbeddingGemma 2 puts them in one space natively, so the workaround is gone** — along with the
config flag, the duplicate embedding per image, the `vec_text` and `vec_jina` tables, and ~2.3 GB
of weights. Kept here as the record of why a single space was worth migrating to.

---

## 10. Search

### 10.1 Query plan

A query string runs four retrievers in parallel, fused by Reciprocal Rank Fusion.

1. **Tag path** — resolve the query to vocabulary terms: exact, then alias, then KNN over `vec_vocab` above `search.tag_seed_min_cos` ([M6] 0.70; see §7.3 for why that line is thin) ("tag seeds"). Items carrying those terms rank by `tag_confidence × term_similarity`. *This is the mechanism behind the original brief's example: searching `tree` matches items tagged `tree`, and also items tagged with terms semantically near `tree`.*
2. **Lexical** — FTS5 BM25 with the column weights above.
3. **Item semantic** — EmbeddingGemma 2 over the query with its `SearchQuery` prefix, KNN over `vec_items` where `kind='item'`, k = 50.
4. **Visual semantic** — the *same* query vector, KNN over `vec_items` where `kind='visual'`, k = 50. One embedding call serves both: since §9.3 collapsed the spaces there is no second tower to run, and this is what makes a plain-language query find an image directly.

### 10.2 Fusion

```
RRF(item) = Σ_retrievers  w_r / (60 + rank_r(item))

w_tag = 1.30   w_lexical = 1.00   w_text_vec = 1.00   w_img_vec = 0.90
```

Weights live in config. `k = 60` is the standard RRF constant and should not be tuned before the weights are.

### 10.3 Explaining the match

Every result carries the set of retrievers that surfaced it and its best per-retriever rank. The card renders a small monospace badge — `tagged` · `similar` · `text` · `ocr` · `visual` — and the detail view shows the contributing retrievers with scores.

This is a trust feature, not a debug feature. When a fuzzy semantic hit outranks an exact tag match and the user cannot see why, they stop believing the search. A "group by match reason" toggle switches the canvas from one ranked masonry flow into labelled sections.

### 10.4 Filters

Structured filters compose as SQL predicates *before* fusion: project, date range, `kind`, has-OCR, entity presence, tag include/exclude, `caption_model`. Tag filters use `item_tags`, not vec0 metadata, which sidesteps the 16-column ceiling entirely.

### 10.5 Dedupe

- **Exact** — BLAKE3 collision at Stage 0. No new item; `source_path` appended to the existing row and the UI says so.
- **Near-duplicate images** — `vec_phash` Hamming ≤ 6. The item *is* created but `near_dup_of` is set; the canvas collapses near-dup clusters into a single card with a count badge, expandable.
- **Near-duplicate notes** — cosine > `dedupe.note_cosine_max` (0.97) in `vec_items` where
  `kind='item'`. Same treatment. Not re-measured in the 768-d space: near-duplicate notes
  should still sit very close to 1.0, but confirm before relying on it to suppress anything.

---

## 11. Graph and links

**Neighbourhood only. The global graph is never rendered.**

### 11.1 Edges

- **Manual** (`kind='manual'`) — created by the user, `weight = 1.0`, optionally labelled. Rendered as solid lines.
- **Semantic** (`kind='semantic'`) — seeded in Stage 7: for each new item, take its top-8 neighbours above threshold in **both** questions the one space still answers separately — "means something similar" (`kind='item'`) and "looks similar" (`kind='visual'`). [M6] both thresholds are **0.85**, not the 0.70/0.62 they replace: measured as an all-pairs distribution over a real corpus the *median* pair already scores 0.765, so 0.70 called 63% of all pairs related, which is the same as calling none of them related. 0.85 sits at roughly p99. Rendered as dotted lines with thickness ∝ weight.

Semantic edges are recomputed for an item on re-enrichment. Manual edges are never touched by any pipeline stage.

### 11.2 Rendering

`GET /api/items/{id}/graph?hops=2&limit=60` returns the focused node, its neighbours to *n* hops, and the edges among them. Force-directed layout (`d3-force`) on ≤ 60 nodes, which settles in well under a frame budget.

**Nodes are thumbnails, not circles.** Images show their thumb; notes show a typographic tile with their first line. This one choice is most of why the graph looks like nothing else.

Clicking a node re-focuses the graph on it. Shift-drag from one node to another creates a manual link. Dotted (semantic) edges have a "make permanent" affordance that converts them to manual.

---

## 12. HTTP API

REST, JSON, no auth (`127.0.0.1` bind only).

```
GET    /api/items?cursor=&limit=60&project=&kind=&status=
POST   /api/items                      multipart (files) or JSON (note)
GET    /api/items/{id}
PATCH  /api/items/{id}                 title, body, project_id, tags
DELETE /api/items/{id}
POST   /api/items/{id}/reenrich        body: {from_stage, escalate: bool}

GET    /api/search?q=&group=&limit=&<filters>
GET    /api/items/{id}/related?limit=12
GET    /api/items/{id}/graph?hops=2&limit=60

POST   /api/links                      {src, dst, label?}
DELETE /api/links/{id}

GET    /api/vocab?state=
POST   /api/vocab                      {term, joytag_alias?, kind?}
PATCH  /api/vocab/{id}
POST   /api/vocab/{id}/merge           {into}
GET    /api/vocab/merge-suggestions

GET    /api/review?limit=50            pending tags + pending vocab proposals
POST   /api/review/{kind}/{id}/accept
POST   /api/review/{kind}/{id}/reject

GET    /api/projects
POST   /api/projects

GET    /api/queue                      depth per stage, current lease, ETA
GET    /api/health                     model/backend status, VRAM, DB size
GET    /api/events                     SSE: item.created, item.updated,
                                            item.stage, queue.changed
GET    /api/thumb/{id}
GET    /api/file/{id}
```

SSE is what makes optimistic ingest work: a card appears at `item.created` in a skeleton state and fills in as `item.stage` events land.

---

## 13. Frontend

Svelte 5 (runes) + Vite + TypeScript. Built to `web/dist`, served by FastAPI as static files.

### 13.1 Information architecture

Three surfaces, no navigation sidebar:

- **Canvas** — the uniform masonry grid. Default view, reverse-chronological, infinite scroll.
- **Search** — the same canvas, filtered and reordered, with match-reason badges.
- **Focus** — one item: full media, tags with confidences, OCR, entities, related strip, and the neighbourhood graph. Opens as an overlay over the canvas; the URL updates so it is linkable and back-button works.

Plus two utility overlays: **Review** (pending tags and vocab proposals, keyboard-driven for fast bulk triage) and **Settings**.

`⌘K` / `Ctrl-K` is the only navigation: search, new note, jump to project, add tag, open review, toggle theme.

### 13.2 The canvas

One card species. A text note and an image are the same object in the grid — the note renders as a typographic card, large first line, generous leading, no chrome; the image bleeds edge-to-edge. Equal visual weight is the whole point: browsing reads as a moodboard, not a file list.

- **[M6] Columns are uniform and the pitch divides the viewport exactly**: `columns = max(2, round(vw / (TARGET + GAP)))`, `pitch = vw / columns`, `width = pitch - GAP`, with `TARGET = 172 px` and `GAP = 12 px`. At 992 px that is 5 columns of 186. Varying the width per column (the first cut used `[268, 212, 300, 236, 260]` at a 6 px gutter) looked livelier standing still and broke the vertical rhythm the moment you panned sideways.
- Tiles carry a 3 px radius and nothing else — no border, no shadow, no hover lift.
- JS-computed absolute positioning, not CSS columns — CSS columns lay out top-to-bottom per column, which destroys keyboard and reading order.
- Virtualise beyond 500 items: render the viewport ± 1.5 screens.
- Processing cards show a skeleton with a thin indeterminate accent bar; fields pop in as SSE events land.
- Near-dup clusters collapse to one card with a mono `×3` badge.

### 13.3 Design tokens

> **[M7] Superseded.** The visual system was re-pointed at the Apple
> product-page reference: Obsidian ground, SF Pro, 28px geometry, no
> shadows, blue confined to one filled action. The shipped tokens are in
> `web/src/lib/design/tokens.css` and the system that governs them is
> [DESIGN.md](DESIGN.md). What follows is the superseded light-first
> system, kept because §17.6–17.8 reason about it.

```css
:root {
  --bg:        #FAF9F7;   --surface:   #FFFFFF;
  --border:    #E7E4DE;   --text:      #1A1A1C;
  --muted:     #6E6E76;   --accent:    #FF5A1F;
  --radius: 6px;  --gutter: 12px;
  --font-content: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono:    "JetBrains Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0B0B0C;  --surface: #141416;
    --border: #232327; --text: #EDEDEF; --muted: #8A8A93;
  }
}
:root[data-theme="dark"] {
  --bg: #0B0B0C;  --surface: #141416;
  --border: #232327; --text: #EDEDEF; --muted: #8A8A93;
}
```

**[M6] The light ground is now plain `#FFFFFF`** and the warm off-white moved to the floating chrome (`--chrome: #F2F0EE`). A tinted page ground fights photographs: every image picked up a cast from the paper behind it. White recedes; the chrome is the only warm surface left, which is what makes it read as an object sitting on the canvas.

Near-black, never `#000`, for text. One accent, `#FF5A1F`, used **only** for focus rings, active state, the selected tag, and the highlighted graph edge — never for body text, because it clears 4.5:1 on the dark ground but not on the light one. No gradients anywhere.

Type scale 12 / 13 / 15 / 18 / 24 / 32. Content in the humanist sans; **all metadata in monospace** — tags, dates, confidences, match badges, counts. That single contrast does most of the aesthetic work and costs nothing.

### 13.4 Tag pills

| State | Treatment |
|---|---|
| human | 1px solid border, full opacity |
| machine, accepted | 1px dotted border, 0.75 opacity |
| machine, pending | 1px dashed border + confidence in mono, e.g. `wireframe 0.62` |
| hover, any | confidence and source in a mono tooltip; `×` to remove |

Provenance is legible at a glance with no legend.

### 13.5 Motion budget: two animations

1. **Card settle** — on load/append: `translateY(6px) → 0`, opacity `0 → 1`, 180 ms `cubic-bezier(.2,.8,.2,1)`, 20 ms stagger capped at 12 cards.
2. **Focus transition** — FLIP from card rect to overlay rect, 240 ms, same easing. Reversed on close.

Everything else is instant: no hover fades, no skeleton shimmer, no page transitions. Restraint is what makes it feel fast and expensive. Both animations are disabled under `prefers-reduced-motion: reduce`.

### 13.6 Keyboard

`/` focus search · `j`/`k` move · `Enter` focus item · `Esc` close · `t` tag · `e` edit · `l` link mode · `g` graph · `x` select · `⌘K` palette. Review overlay: `a` accept, `r` reject, `→` next — bulk triage without touching the mouse.

### 13.7 Responsive

Works to 400 px: columns collapse to 1, the focus overlay becomes full-screen, the graph drops to 1 hop. Minimum 16 px side gutter at every width, set once on the app wrapper.

---

## 14. Configuration

`config.toml`, hot-reloaded where safe.

```toml
[server]
host = "127.0.0.1"
port = 8777

[paths]
data = "./data"
watch_dirs = ["~/dump"]

[gpu]
budget_mb = 7200
pool_a_ttl_s = 300
pool_b_ttl_s = 120

[ocr]
backend = "auto"            # auto | tensorrt | onnx-cuda | onnx-cpu
model = "PP-OCRv6_medium"
det_long_edge = 960
min_rec_score = 0.5

[layout]
line_tol_factor = 0.5
line_tol_min_px = 5.0
blank_line_factor = 1.4
anchor_round = 0.25
anchor_min_count_frac = 0.08
flowing_max_anchors = 4
flowing_min_width_frac = 0.5
sparse_whitespace_frac = 0.8

[vlm]
default = "qwen3.5-4b"
default_quant = "Q4_K_M"
escalate = "gemma4-12b"
escalate_quant = "IQ4_XS"
escalate_min_batch = 8
escalate_max_wait_s = 600
escalate_on_box_count = 40
escalate_on_text_len = 600
ctx = 8192

[embeddings]
# One model, one space, one width (§9.3). truncate_dim = 0 keeps all 768.
dim = 768

[tagging]
w_joytag = 0.45
w_clip = 0.30
w_vlm = 0.25
default_threshold = 0.40
review_floor = 0.25
threshold_step = 0.02
clip_calibration_sample = 500
suppressed_tags_file = "vocab/suppressed.txt"

[vocab]
merge_auto = 0.86
merge_review = 0.72
cluster_distance = 0.18
cooccurrence_merge = 0.60

[search]
rrf_k = 60
w_tag = 1.30
w_lexical = 1.00
w_text_vec = 1.00
w_img_vec = 0.90
tag_seed_min_cos = 0.75

[graph]
seed_top_k = 8
seed_min_cos_item   = 0.85
seed_min_cos_visual = 0.85
default_hops = 2
max_nodes = 60

[dedupe]
phash_hamming_max = 6
note_cosine_max = 0.97
```

---

## 15. Project layout

```
ZetaDB/
├── spec.md
├── config.toml
├── pyproject.toml
├── server/
│   ├── app.py                  FastAPI, static mount, SSE hub
│   ├── api/                    items, search, graph, vocab, review, admin
│   ├── db/
│   │   ├── schema.sql
│   │   ├── migrations/
│   │   ├── writer.py           single-writer queue
│   │   └── queries.py
│   ├── ingest/
│   │   ├── watcher.py
│   │   ├── queue.py
│   │   ├── scheduler.py        stage batching
│   │   ├── arbiter.py          GPU leases, VRAM accounting
│   │   └── stages/             intake ocr layout tag_visual caption
│   │                           extract embed finalize
│   ├── models/
│   │   ├── ppocr.py            TRT + ORT fallback, engine cache
│   │   ├── joytag.py
│   │   ├── eg2.py              vision encoder (GPU) + backbone (CPU)
│   │   ├── embeddings.py       pack/normalise, lazy embedder access
│   │   ├── llama_server.py     llama-server lifecycle
│   │   └── needle.py           4 schemas
│   ├── layout/
│   │   ├── grid.py             the projection algorithm
│   │   └── types.py            TextBox — source-agnostic, PDF-ready
│   └── search/
│       ├── hybrid.py           RRF
│       ├── vocab.py            gating, merging, calibration
│       └── dedupe.py
├── web/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/     Card Masonry Palette TagPill Graph Focus
│   │   │   ├── stores/         items.svelte.ts  sse.svelte.ts
│   │   │   └── design/tokens.css
│   │   └── routes/
│   └── vite.config.ts
├── vocab/
│   ├── seed.yaml
│   ├── joytag_aliases.yaml
│   └── suppressed.txt
└── data/                       gitignored
    ├── zetadb.db
    ├── blobs/  thumbs/  models/
```

---

## 16. Build order

| M | Deliverable | Done when |
|---|---|---|
| **M0** ✅ *partial* | Skeleton. FastAPI + SQLite schema + Svelte shell. Validate the risky dependencies. | **Done:** schema + sqlite-vec verified, `GpuArbiter` + `vram` harness, `scripts/probe.py`, `/api/health`, Svelte shell, Needle 3 validated, GPU measured. **Remaining:** download weights and fill the weight-dependent VRAM rows in §3 |
| **M1** ✅ | Capture + canvas. Watcher, drag/paste, intake, thumbnails, masonry, design tokens. | Done: watch folder + upload + paste, exact and perceptual dedupe, SSE, masonry canvas |
| **M2** ✅ | OCR + layout + lexical search. PP-OCRv6, grid projection, FTS5. | Done: real screenshots OCR'd on the CUDA EP, table columns preserved in `layout_text`, text searchable |
| **M3** ✅ | Embeddings + hybrid search. Both embedders, three vec spaces, RRF, match-reason badges. | Done: four retrievers fuse; plain-language query finds images by content |
| **M4** ✅ | Tagging + vocabulary. JoyTag, CLIP zero-shot, fusion, gating, review API. | Done: brief's example works — tag `tree`, search `tree`, get tag + semantic matches, each labelled with why |
| **M5** ✅ | VLM + extraction + scheduler. llama-server lifecycle, arbiter, Needle. | Done: stage-batched ingest, one lease per stage, pools never co-resident, peak 3.8 GB of 5.4 GB budget |
| **M6** | Graph + links. Neighbourhood view, thumbnail nodes, manual link creation UI. | API + Svelte component exist; manual link creation and edge-threshold calibration outstanding |
| **M7** | Polish. Merge clustering UI, review overlay, keyboard map, calibration at scale. | Two weeks of daily use without reaching for the mouse |

**M1–M3 are the product.** Use it daily for two weeks before starting M4. If the dump habit doesn't stick at M3, tagging and graphs will not save it — and you'll have found that out for the cost of three milestones instead of seven.

---

## 17. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| ~~Needle 3 has no `linux-x64` build target~~ | **CLOSED [M0]** | `manylinux2014_x86_64` is supported and a real extraction runs in 0.5 s. See §8.1 |
| Needle 3 is weak at generative fields | **Medium [M0, new]** | Measured: duplicated/garbage tags on `CaptionTags`. Mitigated by moving `CaptionTags` to the VLM + GBNF grammar (§8.1). `ProjectRouting` quality still unvalidated — gate it in M5 |
| **Gemma 4 12B does not fit in 6 GB** | **High [M0, confirmed blocking]** | UD-Q3_K_XL ~5.4 GB + mmproj F16 ~0.9 GB = ~6.3 GB against 5853 MiB usable. Disabled in config. Three options in §17.1 |
| Needle 3 telemetry on by default | Low [M0, new] | `NEEDLE_TELEMETRY=0` + `DO_NOT_TRACK=1` set in `server/app.py` before import |
| ~~jina-clip-v2 is CC-BY-NC~~ | **CLOSED [M6]** | Replaced by EmbeddingGemma 2, Apache-2.0, which also retired the two-space workaround in §9.4 |
| JoyTag ships no licence | Medium | Personal use only pending clarification. The CLIP path already covers the full vocabulary, so JoyTag is removable without redesign |
| TensorRT engine build fragility | Low | ORT is the default. TRT is an optimisation, gated behind successful build + validation |
| Danbooru vocabulary surfaces NSFW tags | Low | `vocab/suppressed.txt` blocklist applied before fusion |
| CLIP cosines uncalibrated → useless thresholds | Medium | §7.3 per-term μ/σ calibration is mandatory, not optional |
| `SQLITE_BUSY` under concurrent enrichment | Medium | Single-writer queue, WAL, `busy_timeout=5000`. Never let an API handler write directly |
| Tag vocabulary rots anyway | Medium | Write-time gate plus rest-time clustering plus co-occurrence merges — three independent defences |
| Model wrappers constructed per call site | **CLOSED [M5]** | Each `QwenEmbedder` owns a llama-server subprocess and each `JinaClip` an 874 MB session. Per-call construction spawned ~40 servers in one batch and OOMed the card. All wrappers are `engine()` singletons, enforced by a test that greps for direct construction |
| CLIP scores uncalibrated in practice | **Open** | `calibrate_clip()` needs ≥8 image vectors and is a no-op on a small library, so the uncalibrated fallback is live until the corpus grows. Re-run `POST /api/vocab/calibrate` at ~500 images |

### 17.2 What is deliberately not built

- **`ProjectRouting`** — specified in §8, unbuilt. M0 gave no evidence Needle classifies well, and
  nothing in M1–M5 depends on it.
- **Manual link creation UI** — the `links` table, the `/api/links` contract and the neighbourhood
  renderer exist; the shift-drag gesture does not.
- **Review overlay** — the full review/merge API is implemented and tested by hand; the keyboard
  triage surface is not built.
- ~~**TensorRT** — unreachable on this machine (§5); the CUDA EP is fast enough for batch
  ingest.~~ **Wrong, corrected [M7c].** TensorRT runs here and is the default. Both
  detection and recognition load on `TensorrtExecutionProvider` from the fp16 engines in
  `data/models/trt`, at **0.16 s mean per image against 1.72 s on the CPU EP** — ~11x, which
  is not "fast enough either way". What was missing was never TensorRT: it was `libcudnn.so`,
  which ships inside a wheel, so a venv without `[torch]` fails at the first Conv node and
  looks like a TensorRT problem.

### 17.1 The escalation tier — decision needed

Gemma 4 12B was the escalation captioner for text-dense screenshots. At 6 GB it cannot load, and
quantising further (Q2_K_XL, ~4.4 GB + mmproj) puts a 12B model at a quality level where it likely
loses to Qwen3.5-4B at Q3_K_XL. Options, in my order of preference:

1. **Drop the second model; escalate by effort instead.** Re-run Qwen3.5-4B on flagged items at
   higher image resolution (more vision tokens) with a longer, structured two-pass prompt. Zero extra
   VRAM, zero extra download, no scheduler complexity, and it keeps the two-tier *policy* while
   collapsing it to one model. Escalation becomes a parameter, not a process.
2. **Partial CPU offload.** Keep Gemma 4 12B with `-ngl` splitting layers across GPU and RAM. It
   would run, but system RAM is 15 GB with ~7 GB free, so a ~6 GB split is tight, and throughput
   drops to a few tokens/sec. Defensible only for a small, batched escalation queue.
3. **A different escalation model in the 6–8B class** — one that fits at UD-Q3_K_XL alongside its
   mmproj in ~4 GB. Needs a candidate that actually beats Qwen3.5-4B on screen/chart understanding,
   which is unproven.

Option 1 is assumed in the code until decided: `models.vlm_escalate.enabled = false`.

---

## 17.3 [M6] What the first real corpus found

29 items (21 Unsplash photos, 25 of them over 4 MP, up to 51.8 MP). Two bugs that only a real
library could surface, both of which synthetic test images had hidden.

**Full-resolution images were being sent to the VLM.** A 4000x6000 photo killed the llama-server
connection outright (`Server disconnected without sending a response`), and the rest were merely
wasteful: captioning was 65% of total ingest time. The model downsamples internally, so nothing was
gained. Downscaling to a 1024 px long edge (1536 for escalated text-dense items) turned a 14.9 MB
original into a 0.23 MB payload, cut captioning from **9.6 s to 4.5 s per image**, and eliminated
the failure.

**Calibrated CLIP scores were centred on the mean, so an average term passed.** `logistic(0) = 0.5`
clears a 0.40 threshold, which meant that once calibration was enabled, **53 of 54 vocabulary terms
were accepted for a single image** and the corpus went from 118 to 469 accepted tags. A tag should
apply when the image is an *outlier* for that term: the squash is now centred on
`tagging.clip_z_center` sigmas (default 2.0). Result: 93 accepted, ~3 per image, and they are
right — the bird photo gets `bird(0.91), animal(0.57)` and nothing else.

Calibration also showed why per-term mu/sigma is not optional. On a nature-heavy corpus:

| term | mu | | term | mu |
|---|---|---|---|---|
| photo | 0.234 | | dashboard | 0.140 |
| mountain | 0.233 | | chart | 0.141 |
| sunset | 0.232 | | spreadsheet | 0.143 |

`mountain` scores high against *everything* here. A single global threshold cannot work.

**Measured throughput**, 23 images cold, stage-batched on the 6 GB budget:

| stage | time | per image |
|---|---|---|
| ocr (TensorRT FP16) | 45.8 s | 2.0 s |
| tag_visual (JoyTag + CLIP) | 50.9 s | 2.2 s |
| caption (Qwen3.5-4B) | 129.4 s | 4.5 s |
| extract + embed | 20.3 s | 0.9 s |

Zero TensorRT fallbacks, zero errors, VRAM back to 191 MiB after drain. The graph seeded **90
semantic edges** across 29 items, which is the "populated on day one" design working on real data.

### 17.4 Relative calibration needs an absolute anchor

Per-term calibration fixes one problem and creates another. The z-score says *"this image is
unusual for this term, relative to the corpus"* — not *"this image contains this thing"*. On a
homogeneous library every term finds its own outliers regardless of whether the concept is present
anywhere. On 29 photographs, 7 of 66 accepted tags were UI/document terms: `whiteboard`,
`handwriting`, `diagram`, `site-plan` on landscape photos, simply because *something* had to be the
most whiteboard-like image in the set.

Raw cosines do not separate the two cleanly — true positives ran 0.232–0.338, these false
positives 0.203–0.267 — so a floor alone cannot fix it. Two gates together can:

1. **`tagging.clip_abs_floor`** (0.22): no relative score can lift a term whose raw similarity is
   low in absolute terms.
2. **`vocab_terms.requires_text`**: a term that only makes sense on a text-bearing image is
   skipped entirely when OCR found none. The pipeline already knows this; it was just not being
   used. 12 of 54 seed terms are flagged.

Result on the photo corpus: **7 → 3** UI false positives, with the text-bearing synthetic documents
keeping all their correct tags. The three that remain are genuinely ambiguous — the `handwriting`
image really does contain text, and the `site-plan` one is a high-angle architectural interior.
Those are review-queue material, not bugs.

**The general lesson:** a purely relative score will always tag *something*. Pair it with an
absolute threshold and, wherever the pipeline already knows something categorical about the item,
use that as a hard gate.

### 17.5 [M6] What only a browser could find

The SPA was written blind and passed every contract test before anyone looked at it. Driving it
found four things no API test could:

- **Retriever names leaked into the interface.** Every search result carried a bright accent chip
  reading `text_vec` / `img_vec` / `lexical`. They are internal identifiers, and because every
  result matches *something* the badge was on every card — pure noise over the photographs. The
  label is now human (`similar wording`, `looks similar`, `text match`, `tagged`) and lives in the
  hover veil, not permanently on the card. The summary line carries the same wording.
- **The overlay did not lock the page.** Scrolling over the focus panel scrolled the canvas behind
  it while the panel sat still. `document.body` is not the scroller here — `documentElement` is,
  and both need locking.
- **The hero image filled the panel.** A 2:3 photo at full width pushed title, tags and caption
  below the fold on open. Capped at `54vh` with `object-fit: contain`.
- **The related strip read as debug output** — six thumbnails each captioned `semantic 0.715`. The
  edge kind and weight are plumbing; they moved to the `title` attribute.

Verified across dark and light, 430 px to 1400 px (2 to 5 columns), all three sorts, hover, focus
and search. No console errors.

**Worth stating plainly:** every one of these passed the API contract tests, and the first
screenshot was of a page whose images had not finished loading — which looked exactly like a
catastrophic rendering bug. A UI is not tested until it has been looked at.

## 17.6 [M6] The canvas

The home surface is a canvas that pans forever in both directions over a finite library
(`web/src/lib/components/Canvas.svelte`), with the visual language taken from
[cosmos.so](https://www.cosmos.so/public-work): edge-to-edge, ~6px gutters, no card chrome at all.

Worth recording: **cosmos itself does not scroll horizontally.** Inspected live, it reports
`overflowX: hidden`, `scrollX: 0`, `bodyWidth === innerWidth` — it is vertical masonry with varied
column widths. The two-axis scroll is this project's own addition.

**How it tiles.** Columns are indexed by any integer, so there is no left or right edge; column `c`
draws from the item list at a different rotation, so panning sideways keeps turning up new things
rather than repeating a block. Vertically each column wraps modulo *its own* stack height, so there
is no global tile boundary to line up and therefore no seam. Column widths come from a short
repeating pattern — uniform columns read as a spreadsheet.

**Three bugs, each of which froze or broke it:**

1. **Keying tiles by pixel position.** `{#each ... (`${c}:${id}:${top}`)}` changes key on every
   scrolled pixel, so Svelte destroyed and recreated every tile *and every `<img>`* on every wheel
   event. The renderer locked up hard enough that CDP screenshots timed out. Key on identity
   (column, repetition, slot), never on position.
2. **No tile budget.** 24 columns x every repetition intersecting a tall viewport generated ~1000
   absolutely-positioned nodes. Cull to the viewport, then cap (`MAX_TILES`).
3. **`setPointerCapture` on the canvas.** Capture retargets pointer events to the canvas, so the
   tile underneath never receives its `click` — clicking did nothing, while a programmatic
   `.click()` worked, which is what isolated it. Track the drag on `window` instead. Also measure
   drag distance from where the pointer went *down*, not as a sum of increments, or jitter makes a
   stationary click register as a drag.

**Clicking an item enters a related view** rather than a modal: the canvas repopulates with its
neighbours and a bar names the source, the count and the relation. `/api/items/{id}/related` takes
`mode=similar|tag` — `similar` is a live KNN in whichever embedding spaces the item has, `tag` is
overlap on accepted tags ranked by how many are shared. They answer different questions and the
toggle sits next to the thing it filters. Measured on the 29-item library: `similar` returns 28,
`tag` returns 8 and is visibly tighter.

### 17.7 Motion, and the orbit

**Smooth scrolling** is a target/actual split: `tx/ty` is where the canvas is heading, `ox/oy` is
what is drawn, and a rAF loop closes the gap with `k = 1 - exp(-dt/tau)` — framerate-independent,
so it behaves the same at 60 and 144 Hz. A wheel tick moves the target; a drag moves both together
so the canvas stays welded to the cursor, then hands its velocity over as decaying momentum.
Measured: a 400px wheel delta spreads over ~30 frames on a decaying curve (peak 27px/frame, easing
to 13) instead of jumping 400px in one. `prefers-reduced-motion` collapses all of it to instant.

**Clicking grows an orbit** rather than replacing the view: the clicked item stays where it is and
its neighbours arrive around it in rings, scaling up from nothing, staggered by rank so the ring
blooms outward. Switching relation (similar <-> same tag) keys the tiles on the mode, so the old
ring leaves and the new one arrives with the focus untouched.

Ring geometry is **derived from the actual tiles**, not hardcoded. Each tile is placed at an angle
proportional to its own footprint, and a ring's radius is the larger of (total footprint / 2*pi)
and whatever clears the ring inside it. Sizing by width alone was not enough — a portrait tile is
up to 1.55x taller than wide, and those collided around the circle.

**Three things this cost:**

- **Transitions must not apply to library tiles.** The keyed list churns constantly as tiles cross
  the cull window; animating each one left a permanent backlog of entering and (via `out:`) still-
  dying nodes. Transitions are scoped to the orbit, which is also better UX — a tile "arriving"
  while you pan is noise.
- **A click is decided from the click event's own coordinates** against where the pointer went
  down, not from a flag accumulated during the drag. One stray move between down and up flipped
  that flag and swallowed every click.
- **`visibility: "hidden"` looks exactly like a frozen renderer.** A backgrounded tab never fires
  rAF, so transitions stall mid-flight and `Page.captureScreenshot` times out. Several "freezes"
  during development were this, not the code — check `document.visibilityState` before believing a
  hang.

An item with no accepted tags (only pending ones) legitimately has zero tag-neighbours. The orbit
then shows the focus alone with "nothing shares its tags", rather than the library's empty state.

### 17.8 [M6] Matching the reference, and one line that broke everything

The target was cosmos.so/public-work. Reading the live page rather than guessing at it is what made
the difference — the numbers below are measured off it, not estimated:

| | reference | ours |
|---|---|---|
| column pitch | uniform, `vw / n` | uniform, `vw / n` |
| gutter | 12 px both axes | 12 px both axes |
| tile radius | 3 px | 3 px |
| tile border / shadow | none | none |
| page ground | `#FFFFFF` | `#FFFFFF` |
| chrome fill | `lab(95.2 0.21 0.60)` ≈ `#F2F0EE` | `#F2F0EE` |
| chrome height | 57.6 px capsule | 52 px capsule |
| chrome border | `0.8px solid rgba(0,0,0,.12)` | same |
| chrome shadow | `inset 0 1px 1.5px #fff b3`, `inset 0 -1px 1.5px #0000 08`, `0 1px 8px #0000 0d` | same |
| search type | 14 px, `letter-spacing: -0.28px` | 13 px, `-0.02em` |
| canvas | `fixed inset-0`, `cursor: grab` / `grabbing` | same |

Two structural things follow from that table. The chrome **floats on** the canvas instead of sitting
above it in a bar, so the canvas is `fixed inset-0` and nothing reserves vertical space; and every
control — the mark, the search capsule, the round buttons, the related bar, the search summary, the
details card — is the *same* recipe (full rounding, `--chrome` fill, 0.8 px hairline, inner top
highlight, soft drop shadow). One recipe applied six times is what makes a set of controls read as
a family rather than as a toolbar.

Where we deliberately differ: the reference *replaces* the canvas with a fresh grid when you ask for
similar items; ours grows an orbit around the item you clicked (§17.7), which was an explicit
requirement. The subject of an orbit is marked by lift — a hairline and a drop shadow — not by an
accent outline, because an orange stripe across a photograph is the one thing the reference would
never do.

**The bug.** `extent` (how far an orbit reaches, used to bound panning) was a `$state` assigned at
the end of the layout function — which runs inside `$derived.by`. Svelte throws
`state_unsafe_mutation` there, and the throw took the whole layout down with it: no tile click did
anything at all, in silence. The fix is that the extent travels **out** of the layout as part of its
result (`{ tiles, extent }`) instead of being written sideways into state.

Two things about finding it are worth keeping:

- The console exception named the rule exactly, and it had been sitting there for several browser
  passes before anyone read it. Check the console *before* re-deriving behaviour from screenshots.
- The bundle name in the stack trace was the *old* hash. The page was being served from cache, so
  the first two fixes were never actually loaded. A cache-busting query string on the reload is the
  cheapest way to be sure you are looking at what you just built.

## 17.9 [M7] The captioner bake-off — Gemma 3 4B, Moondream, and two 2B models

Qwen3.5-4B was chosen in M0 on reputation, not on this corpus. `scripts/vlm_bakeoff.py` re-asks the
question properly: every candidate runs through the production path — same prompt, same GBNF
grammar, same 1024 px downscale, same `LlamaServer` — over all 29 images, and is scored on what a
caption is actually FOR here.

The scoring metric matters more than the leaderboard. A caption feeds EmbeddingGemma 2 and becomes
part of how an image is found, so each caption is embedded as a `SearchQuery` and KNN'd against the
`kind='visual'` vectors — the image with **no text in it**. Scoring against `kind='item'` would
reward a caption for echoing OCR text already baked into that vector, which flatters text-dense
screenshots and measures nothing.

| model | VRAM | s/cap | json | r@1 | margin | ocr | chars | tags |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-4B** UD-Q3_K_XL | 3747 | 3.47 | **29/29** | **1.00** | **0.1408** | **0.711** | 177 | 8.1 |
| Qwen3-VL 4B UD-Q3_K_XL | 4457 | 3.30 | 29/29 | 0.97 | 0.1384 | 0.667 | 212 | 8.9 |
| LFM2.5-VL 3B Q6 + Q8 mmproj | 3079 | 3.02 | 29/29 | 0.97 | 0.1351 | 0.555 | 148 | **10.0** |
| Gemma 3 4B IT UD-Q3_K_XL | 3591 | 3.33 | 29/29 | 0.97 | 0.1217 | 0.622 | 216 | 6.9 |
| InternVL3 2B Q4_K_M | **2033** | **1.92** | 25/29 | 0.96 | 0.1279 | 0.422 | 119 | 9.6 |
| SmolVLM2 2.2B Q4_K_M | 3811 | 1.73 | 18/29 | 0.94 | 0.0840 | — | 52 | 5.3 |
| Moondream2 f16 | — | — | 0/29 | — | — | — | — | — |

A mean margin over 29 items is a thin thing to rank models on, so the comparison is **paired** —
same item, same prompt, incumbent minus candidate — which is what separates a real gap from a
reshuffle:

| vs incumbent | n | incumbent better on | mean Δmargin | t |
|---|---|---|---|---|
| Qwen3-VL 4B | 29 | 17/29 | +0.0024 | **+0.73** |
| LFM2.5-VL 3B | 29 | 16/29 | +0.0057 | **+1.18** |
| Gemma 3 4B | 29 | 24/29 | +0.0191 | **+4.72** |
| InternVL3 2B | 25 | 17/25 | +0.0150 | +2.48 |
| SmolVLM2 2.2B | 18 | 15/18 | +0.0402 | +5.20 |

**Gemma 3 4B's deficit is real** — it is behind on 24 of 29 individual items, not on an average.
**Qwen3-VL 4B's is not**: t = 0.73 over 29 paired items is a tie on retrieval margin. It is a
vision specialist and much stronger on document QA benchmarks, but captioning-for-retrieval is not
document QA, and here it buys nothing while costing 710 MB more VRAM and a little OCR grounding.

**The incumbent stays**, against all five challengers. Gemma 3 4B is 156 MB lighter and 4% faster
and loses on every quality axis that matters: worse retrieval margin, worse OCR grounding, fewer
tags. On item 13 it was simply wrong, calling a sculpture of stacked Mercedes-Benz cars "a robotic
insect-like vehicle" where every other model saw cars.

**The preamble is not the explanation, which was worth checking before blaming the model.** Gemma
opens nearly every caption with "The image shows" / "The image depicts" — constant text in a field
whose only job is to be embedded. `--no-preamble` asks for it to stop. Gemma's margin moved
0.1217 → 0.1251, still far behind, and its captions got *longer* (216 → 230 chars) rather than
shorter: it did not really comply. The same flag cost the incumbent OCR grounding (0.711 → 0.600)
for no margin gain, so the production prompt stays exactly as it is. The deficit is the model.

**`ocr_recall` is only comparable at 29/29.** It averages over captions that parsed, so a model that
fails on exactly the hard text-dense items scores itself on the easy remainder. InternVL3 first
measured 0.967 and then 0.422 on a re-run, purely because different items failed; that figure is
not evidence of anything on its own. Raising `--max-tokens` from 400 to 900 did **not** fix the
failures — these models ramble inside a grammar-valid string until the cap, rather than being cut
off just short.

**Moondream2 cannot be served here at all**, and this is a serving fact, not a quality one. Its
GGUF carries a vicuna chat template that emits no media marker, so `llama-server`'s OAI endpoint
renders a prompt with zero markers against one bitmap and returns `Failed to tokenize prompt` on
every image. Injecting `<__media__>` into the message text does not help — the template escapes it.
The legacy `/completion` + `image_data` path accepts the request and *silently ignores the image*:
17 tokens evaluated, and a confident description of a river and a boat for a picture of a red
rectangle and a blue circle. The model itself is fine — `llama-mtmd-cli` describes that same test
image correctly — but it is unreachable from pool B, which is HTTP + grammar by construction. It
also trains at `n_ctx_train=2048` against our 16384, and the build warns `missing pre-tokenizer
type ... GENERATION QUALITY WILL BE DEGRADED`.

**LFM2.5-VL 3B is the real alternative, and the most interesting result here.** It was already
built on this machine, so it cost nothing to try. It ties the incumbent on retrieval margin
(t = 1.18 over 29 paired items, ahead on 13 of them) while holding **668 MB less VRAM**, loading in
2.1 s against 9.6 s, captioning 13% faster, returning a complete object on all 29 images, and
proposing more vocabulary tags than anything else measured. On a 6 GB card where §17.1 had to drop
a whole escalation tier for want of room, 668 MB is not a rounding error. It needs `--jinja`:
without it llama.cpp does not render LFM2.5-VL's chat template.

Its weakness is specific and shows up in the captions, not just the metric. On the five text-dense
items it describes the *structure* of an interface where the incumbent reads its *content* — the
dashboard became "three rows of data, each containing a tag name, a numerical value, and a
percentage" where the incumbent named PP-OCRv6_tiny, _small and _medium outright, and its invoice
caption drops the issue date, due date, portal link and contact email that the incumbent keeps.
Per-item OCR grounding over those five: incumbent 0.427, Qwen3-VL 0.400, Gemma 3 0.373, LFM2.5-VL
0.333.

That matters more than this corpus shows. 24 of these 29 images are photographs, which is what lets
the margins tie; a real idea-dump skews to screenshots, which is the exact content LFM2.5-VL reads
least well. **The incumbent therefore stays as the default, and LFM2.5-VL 3B becomes the first
candidate if pool B ever has to shrink** — ahead of InternVL3 2B, which is lighter still but cannot
be trusted to return an object at all on the text-dense items.

**InternVL3 2B is the one to remember for size.** It holds 1714 MB less VRAM than the incumbent and captions
in just over half the time, at a retrieval margin between Gemma's and Qwen's. What disqualifies it
today is that 4 of 29 items produce no usable object, and they are the text-dense ones — the invoice
is among them. It also transcribes rather than describes, reading a dashboard back as a list of
bare numbers, which duplicates work PP-OCRv6 already does better (§1.1). If pool B ever has to
shrink to let pool A coexist, this is the first candidate to re-measure.

**[M7b] LFM2.5-VL re-measured against its own documentation.** Liquid's docs claim detection and
grounding, screen and document understanding, OCR and function calling, which sits badly against the
0.555 OCR grounding measured above. Checked directly, and the claim is right: asked
*"Read every line of text in this interface verbatim"*, LFM reads the dashboard back as
`PP-OCR6b_tiny 1.5M 806 735 / PP-OCR6b_small 7.7M 841 81.3 / PP-OCR6b_medium 345M 86.2 83.2` —
near-perfect, off only by the digit grouping. **The model can read screens. It just does not do it
inside a captioning instruction**, where it reports an interface's *structure* ("three entries, each
showing a tag name, a numerical value, and two highlighted percentages") instead of its *content*.
That is instruction-following under a multi-field grammar, not a vision limit.

Three levers were tried against it, and none moved it far:

| condition | json | margin | ocr | s/cap |
|---|---|---|---|---|
| default (llama.cpp sampling, 256 image tokens) | 29/29 | 0.1351 | 0.555 | 3.02 |
| Liquid's documented sampling + `--image-max-tokens 1024` | 29/29 | 0.1315 | 0.489 | 2.56 |
| the same, on the §4.3 escalated path (1536 px, stricter prompt) | 29/29 | 0.1319 | 0.555 | 3.09 |

`--image-max-tokens` is worth knowing about: this build gives an image **256 tokens by default** and
`1024` lifts it to ~512, but `2048` produces an identical 531-token prompt — 512 is the ceiling the
image's own tiling allows, so the knob saturates immediately. Liquid's recommended sampling
(`--top-k 50 --repeat-penalty 1.0`) made it ~15% faster and no more accurate. The escalated prompt,
which explicitly says *"read the interface carefully"*, lifts the **incumbent** to 0.889 OCR
grounding and leaves LFM at 0.555: told plainly to read the screen, Qwen does and LFM still
describes.

**`ocr_recall` is the wrong stick to beat this model with, and §1.1 says so.** PP-OCRv6 is the OCR
authority and the VLM never does OCR — the text is already in `flow_text`, already in FTS5, and
already interleaved into the `kind='item'` vector. A caption that re-transcribes it is duplicating
a job the design gave to a better model. On the metric that actually decides this, LFM2.5-VL is
**indistinguishable from the incumbent**: re-baselined, it is ahead on 15 of 29 items, t = 0.49.

| vs a fresh incumbent baseline | n | incumbent better on | mean Δmargin | t |
|---|---|---|---|---|
| LFM2.5-VL 3B (default) | 29 | 14/29 | +0.0024 | **+0.49** |
| LFM2.5-VL 3B (tuned) | 29 | 15/29 | +0.0059 | +1.13 |
| Qwen3-VL 4B | 29 | 15/29 | −0.0009 | **−0.28** |
| Gemma 3 4B | 29 | 20/29 | +0.0157 | +3.27 |

**Re-running the incumbent against itself is the control this needed.** Its own margin moved
0.1408 → 0.1374 between two identical runs at temperature 0.2 — ±0.003, the same size as every gap
being argued over except Gemma's. Gemma 3 4B is the only candidate that stays significantly behind
across both baselines (t = 4.72 then 3.27); every "tie" above is a genuine tie, and the single-run
margins in the first table should not be read to three decimal places.

**What LFM2.5-VL would actually buy is capability this project does not use yet.** Detection and
grounding (RefCOCO 87.9), ScreenSpot-v2 at 78–82, and function calling are not captioning upgrades —
they are new surfaces. If ZetaDB ever wants "point at the chart in this screenshot" or a
tool-calling path, that is the reason to load it, not a better caption.

**Not tested, and why.** Gemma 4 E4B is the obvious "newer Gemma" candidate and does **not** fit:
UD-Q3_K_XL is 4588 MB and its mmproj another 990 MB, so 5578 MB of weights alone against a 5400 MB
budget, before any KV cache — the same arithmetic that killed Gemma 4 12B in §17.1. "E4B" is
effective parameters, not resident ones; the BF16 file is 15 GB. Gemma 4 **E2B** does fit
(2924 + 986 = 3910 MB) and is the next candidate worth an hour. Florence-2 and JoyCaption are
excluded on serving grounds rather than quality: neither is reachable through llama.cpp, so neither
can live in pool B without a second runtime.

## 17.10 [M7c] Could LFM2.5-VL replace PP-OCRv6? No — but PP-OCR has a hole

§1.1 asserts that PP-OCRv6_medium (34.5M) beats billion-scale VLMs on recognition, and the whole
two-jobs-two-models split rests on it. That was inherited, never measured here.
`scripts/ocr_bakeoff.py` measures it: five synthetic images whose ground truth is exact *because
they are rendered from a known string*, plus the library's own two text-dense items transcribed by
eye. PP-OCR is scored on what the pipeline actually emits — detect, recognise, then the §6 grid
projection — because `flow_text` is what reaches FTS5, embeddings and extraction.

| case | PP-OCRv6 + layout | LFM2.5-VL @1024 |
|---|---|---|
| invoice (synthetic) | **0.005** | 0.015 |
| metrics table | **0.014** | 0.021 |
| code block | 0.237 | **0.028** |
| prose paragraph | 0.908 | **0.028** |
| small UI chrome (14px) | 0.327 | **0.048** |
| item 1, the real dashboard | **0.009** | 0.036 |
| item 4, the real invoice | **0.006** | 0.011 |
| **median CER** | **0.014** | 0.028 |
| **mean CER** | 0.215 | **0.027** |
| **worst CER** | 0.908 | **0.048** |
| **critical tokens** | **18/19** | 15/19 |

**The mean is a trap and the median is the answer.** PP-OCR is twice as accurate as LFM in the
typical case and near-perfect on exactly the documents this corpus holds — 0.005 on an invoice,
0.009 on a real dashboard. Its mean is wrecked by one catastrophic case, not by general weakness.

**PP-OCR keeps the argument that matters.** Critical tokens — emails, URLs, ISO dates, decimal
amounts — are what `extract.py` is built on, and §5.1 already established that a subtly wrong one is
worse than a missing one. PP-OCR returned 18 of 19 verbatim. LFM dropped
`finance@acme.example`, `https://billing.acme.example/inv/8813`, `2026-03-14` and
`help@iwashere.example`. A VLM paraphrases under pressure; an OCR engine does not. **So the
captioner does not become the OCR authority, and §1.1 stands.**

Three further reasons replacement is not close, independent of accuracy: LFM emits **no boxes**, so
`layout/grid.py`, `layout_text`, `boxes_json`, the `escalate_on_box_count` policy and Focus's `ocr`
tab all die with it; it is ~10x slower per image; and it costs 3079 MB in pool B against PP-OCR's
~150 MB in pool A, on a card that had to drop an escalation tier for want of room.

**But PP-OCRv6 detection has a real hole, and it is worth fixing.** On a flowing prose paragraph it
detects **4–5 of 8 lines** and returns garbage for the ones it does find — `'hy-UHDPUyd'`,
`',PUU'` — for CER 0.908. This is not box-merging: the boxes it returns are single-line height, it
simply misses the rest. It is not a font-size artifact either; swept from 14 px to 34 px the line
count stays at 4–5 and CER stays between 0.844 and 0.904. The same engine on the same backend reads
the invoice at 11 boxes for 11 lines, every score 0.98–1.00, so neither the model nor the provider
is broken in general — it is specific to dense uniform prose. Code (0.237) and 14 px UI chrome
(0.327) are weak for the same reason.

That matters more as the library grows than the current corpus suggests: 24 of 29 items today are
photographs, but an idea-dump accumulates article screenshots and documentation, which is exactly
the shape that fails.

**The design this points at is a fallback, not a replacement.** When PP-OCR's box count is
implausibly low for an image's text density, re-read that image with the VLM and keep its text for
`flow_text` only, leaving `boxes_json` empty and letting §6.1's prose bypass handle the ordering.
That rescues the one case PP-OCR loses while keeping boxes, speed and verbatim fidelity everywhere
else.

**[M7c] The caveat is discharged: re-run on TensorRT fp16, the hole is identical.** The first pass
used `onnx-cpu` at fp32 for want of cuDNN, which left open whether the prose failure was an
artefact of the provider. It is not. The project's own venv already carries TensorRT 10.16.1.11 for
cu13 alongside cuDNN 9, and with `ocr.det_backend = rec_backend = "tensorrt"` both sessions load on
`TensorrtExecutionProvider`:

| case | onnx-cpu fp32 | **TensorRT fp16** |
|---|---|---|
| invoice | 0.005 | 0.005 |
| metrics | 0.014 | 0.014 |
| code | 0.237 | 0.237 |
| **prose** | **0.908** | **0.908** |
| ui_small | 0.327 | 0.327 |
| item 1 | 0.009 | 0.009 |
| item 4 | 0.006 | 0.011 |
| mean | 0.215 | 0.216 |
| critical tokens | 18/19 | 18/19 |

Character-for-character the same everywhere but item 4, which moves by 0.005 — the only visible
fp16-against-fp32 difference in the set, and in PP-OCR's favour on the metric that matters since
both still return 7/7 critical tokens. **The prose hole is a property of the detector, not of the
execution provider, so the §17.10 fallback is justified on the production path.**

What fp16 does change is the clock: **0.16 s mean per image against 1.72 s**, roughly 11x, which
also retires §17.2's claim that TensorRT is unreachable here and that the CUDA EP is fast enough
either way.

One remaining device caveat: LFM's numbers above were taken on CPU, because another workload held
3292 MB of the card. Its *text* is device-independent, but the seconds-per-image column must not be
read across devices.

## 18. Open questions

1. What should the seed vocabulary actually contain? It needs ~100–200 terms to be useful at M4. Bootstrapping option: run JoyTag over the first 200 ingested images and promote its most frequent above-threshold tags as seed candidates for manual curation.
2. Should notes support Markdown rendering, or stay plain text? Plain text is assumed throughout; Markdown affects the card renderer and the FTS tokenisation.
3. Retention: does anything ever get deleted, or is the dump append-only with archive?
4. Should `todos` extracted by Needle 3 surface anywhere beyond the item detail view — a dedicated surface, or deliberately kept passive?

---

## 19. References

- Eye (Desert Ant Labs) — https://huggingface.co/desert-ant-labs/eye · https://desertant.com/models/eye/ (used with permission)
- ~~JoyTag~~ — https://github.com/fpgaminer/joytag (replaced in [M8], §20.3)
- Needle 3 — https://github.com/cactus-compute/needle · https://cactuscompute.com/needle
- PP-OCRv6 — https://huggingface.co/blog/PaddlePaddle/pp-ocrv6 · https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html
- LiteParse grid projection — https://www.llamaindex.ai/blog/how-liteparse-turns-pdfs-into-text-a-deep-dive-into-the-grid-projection-algorithm · https://github.com/run-llama/liteparse
- Qwen3.5 — https://qwen.ai/blog?id=qwen3.5 · https://huggingface.co/Qwen/Qwen3.5-4B
- Gemma 4 12B — https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- EmbeddingGemma 2 — https://huggingface.co/onnx-community/embeddinggemma-2-ONNX
- LFM2.5-VL 3B — https://www.liquid.ai/blog/lfm2-5-vl-3b · https://docs.liquid.ai/lfm/models/lfm25-vl-3b
- ~~Qwen3-Embedding-0.6B~~, ~~jina-clip-v2~~ — both retired in [M6], see §9.4
- sqlite-vec — https://github.com/asg017/sqlite-vec · https://alexgarcia.xyz/sqlite-vec/
- Hybrid FTS5 + vector + RRF — https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html

---

## 20. [M8] ZetaDB — one engine, Eye, and audio/video

iwashere is renamed **ZetaDB** and takes over that project's repository. Three changes come with
the name: every model now runs in one engine, JoyTag is replaced by Desert Ant Labs' **Eye**, and
**audio and video** become items that can be searched down to the moment. Everything in this
section marked *measured* was measured; everything else is design, and §20.6 lists what still has
to be measured on the GPU.

### 20.1 The rename

User-visible strings, the package, the FastAPI title and logger, and the native-library guard
(`ZETADB_NATIVE_READY`) say ZetaDB. The database moves from `data/iwashere.db` to
`data/zetadb.db`; `server.db.conn._adopt_legacy_db` renames the old file (with its `-wal` and
`-shm`) on first start, so an existing library carries over. `tests/test_zetadb.py` pins that.

### 20.2 One engine (`server/engine/`)

The ingest arbiter and the unload list `app.py` used to maintain by hand are replaced by one
registry:

| Backend | Models |
|---|---|
| TensorRT FP16 → CUDA → CPU | PP-OCRv6 det + rec, Eye, and the EG2 encoders when their graph is TRT-runnable |
| ONNX Runtime CPU | EG2 backbone, by design (§3, a query never waits for the GPU) |
| llama.cpp | the captioning VLM, as a `llama-server` subprocess |

Every model registers a `ModelSpec` (pool, backend, unload, provider). A lease on one GPU pool
unloads *every* model registered in the other, so a new model cannot be forgotten and left holding
VRAM — lesson 2 of §3 now enforced by construction rather than by remembering. All ONNX sessions
come from `server.engine.onnx.session()`: one provider chain, one TensorRT engine cache, per-model
optimisation profiles, and per-session degradation (a TRT *build* failure raises instead of
falling back, §5.1).

**TensorRT cannot run a dynamically quantised graph** (`MatMulInteger` / `DynamicQuantizeLinear`):
it shatters into tiny partitions for no gain. `backend_for()` therefore sends such graphs to CUDA.
That covers EG2's published `*_quantized` encoders, so they stay on the CUDA EP exactly as before.
Putting the vision encoder on TensorRT means selecting its fp16 graph
(`models.eg2.vision_dtype = "fp16"`) — §3's note that onnx-community's fp16 export is mixed
precision and safe still applies — but that swap is **not measured yet** (§20.6).

### 20.3 Eye replaces JoyTag

JoyTag scored a Danbooru schema: no word for `screenshot`, `whiteboard`, `receipt` or `text`
(§7.1), heavily NSFW-weighted, and a sigmoid whose meaning differed per tag. Eye
(`desert-ant-labs/eye`, PE-Core S/16-384 encoder plus heads, used **with permission** from Desert
Ant Labs) gives:

- `type` — 16-way single-label: photo, selfie, screenshot, photo_of_screen, receipt, document,
  id_card, whiteboard, map, chart_or_slide, meme_or_graphic, illustration_or_art, … (+ the summed
  `reference` tag)
- `scenes` — 558 Open Images labels, multi-label
- a **calibration curve per tag** (`eye_meta.json`): raw head output → confidence where 0.5 is the
  publisher's recommended cut and 0.9 the strict one. A score means the same for every tag.

Like JoyTag it is a *scorer against our vocabulary*, not a generator: `vocab_terms.eye_alias`
names the Eye tag a term reads. 43 seed terms have one; terms Eye has no word for (wireframe,
terminal, night, a data `table` — Eye's `table` is furniture) stay zero-shot.

**Runtime.** Eye ships as LiteRT (`eye.tflite`) and Core ML only. `scripts/convert_eye.py`
converts the `photo` signature to ONNX once, so Eye runs in the engine like everything else.
tf2onnx needed five fixes for this file, each one a real incompatibility rather than a workaround:

1. **External buffers.** 601 of 927 weight buffers use the TFLite large-model layout (data at a
   file offset, `Buffer.data` empty). tf2onnx read only `data`, so those weights vanished.
2. **Opcode width.** Newer writers leave `deprecated_builtin_code` at 0 — which decodes as ADD —
   and put the real opcode in `builtin_code`. tf2onnx trusted the old byte, so every DEQUANTIZE
   arrived as a one-input int8 "ADD". TFLite's own reader takes the larger of the two.
3. **Name clash.** The file already names 75 tensors `…_dequant`; tf2onnx's inserted dequant nodes
   use the same suffix, and the graph closed into a cycle.
4. **Missing options.** Element-wise ops with no `BuiltinOptions` table have no
   `fused_activation_function`; a missing one means NONE.
5. **GELU.** No tf2onnx handler; each of the 22 (all exact, `approximate = False`) becomes
   `0.5·x·(1 + erf(x/√2))` — Erf runs on every provider, TensorRT included.

**[M8] measured** against the LiteRT reference on 24 real library images: max |Δ| ≤ 1.3e-4 on
every head (1e-5 on `type` and `scenes`), embedding cosine 1.000000, and every head already emits
probabilities, so the curves apply to raw outputs directly.

**Licence.** Eye's licence and third-party notices ship beside the weights; the UI carries the
required "Powered by Desert Ant Labs" credit where machine tags are explained.

### 20.4 Audio and video, searched by moment

**Why windows.** EmbeddingGemma 2's audio front end truncates at 480 000 samples — **30 s at
16 kHz**. Measured on the sentence-transformers reference (in the previous ZetaDB): the embedding
of a 60 s clip has cosine **1.0000** with its first 30 s and 0.979 with its first 20 s; the second
half simply is not heard. A video is vision-only frames. So one vector per file would describe the
first half-minute of a podcast and nothing else.

| Kind | Window | Embedded as |
|---|---|---|
| `sound` | 30 s every 25 s; < −50 dBFS skipped | `<boa> <audio>×n <eoa>` via the EG2 audio encoder |
| `frames` | 10 s, 5 frames (1 every 2 s), ≤ 512 px | per frame `<boi> <video>×140 <eoi>`, no timestamps |

The token layout is the one `EmbeddingGemma2Processor` builds; the audio placeholder count is the
encoder's own output length, so the two cannot disagree. Videos also get their soundtrack as
`sound` windows. Past 100 min of audio / 20 min of video the windows spread out to cover the file.
Decoding is ffprobe/ffmpeg — a runtime dependency for media only.

Schema: `segments(item_id, kind, start_s, end_s, frame_path)` and `vec_segments` (768-d, same
space). The item's own `kind='item'` vector is the normalised mean of its segments, so related
items, semantic edges and `text_vec` treat media like any other item.

**Search** gains a fifth retriever, `moment`: KNN over `vec_segments`, one best window per item,
gated like `_knn` (absolute floor `search.min_cos_moment`, then the relative margin). A result it
surfaces carries `moment = {kind, start, end, frame_url, score}` and the UI opens the player there.

**Tagging.** A video is tagged from up to three keyframes (the middle frame of each window): a
term's score is its best over those frames, and the middle keyframe's vector becomes the video's
`kind='visual'` row.

**[M8] measured on CPU** (GPU busy at the time), through the real int8 ONNX encoders and the
ffmpeg windowing, on synthetic media — a sanity check, not a calibration. Every text query ranked
its own file first:

| query | tone.wav | pink noise | test-pattern video | red video |
|---|---|---|---|---|
| "a steady high pitched beep" | **0.693** | 0.653 | 0.592 | 0.591 |
| "static hiss like heavy rain" | 0.573 | **0.680** | 0.590 | 0.561 |
| "a solid red screen" | 0.604 | 0.545 | 0.612 | **0.759** |
| "colourful television test pattern with bars" | 0.522 | 0.519 | **0.704** | 0.547 |

End to end through the running server, a beep query landed on the test-pattern video's
*soundtrack* (a 440 Hz sine) and on the tone file's 25–40 s window; "test pattern" landed on the
video's 10–20 s scene. A bug only a real run could find: transformers 5.17's audio feature
extractor transposes an un-batched 1-D waveform the wrong way, so `embed_audio` passes a
one-element batch, as `EmbeddingGemma2Processor` itself does.

**EG2's int8 graphs are not dynamically quantised** — they use ORT's weight-only block ops
(`MatMulNBits`, `GatherBlockQuantized`). TensorRT cannot run those either, so the engine's check
is "contains ORT-only quantisation ops", which sends them to CUDA and sends Eye to TensorRT.

### 20.5 Schema v4

`joytag_alias → eye_alias`, `tag_scores.joytag → eye` (values dropped — Danbooru names and JoyTag
sigmoids mean nothing to Eye), `items.kind` admits audio/video and gains `duration` (rebuilt the
documented SQLite way, ids preserved), every image requeued for `tag_visual`, and the vocabulary
re-seeded once at startup. **[M8] measured** on the real 32-item library: 32/32 ids preserved,
32/32 FTS rows intact, 0 foreign-key violations, 16/16 human tags kept, 29 images queued, 43 terms
re-aliased.

### 20.6 Measured on the GPU (RTX 4050 Laptop, 6 GB)

`scripts/m8_gpu_check.py` plus a full server run over the 29-image corpus and four synthetic
clips. Every number below is measured, not assumed.

| Model | Provider | Load | Per item | VRAM |
| --- | --- | --- | --- | --- |
| Eye (photo) | TensorRT FP16 | 141 s first build, then cached | 34.8 ms/image | 560 MB |
| EG2 vision + audio encoders (int8) | CUDA | — | 1.03 s/image incl. CPU backbone | 1070 MB |
| EG2 backbone, ingest copy | CUDA | — | 30 s audio window 0.13 s (CPU 0.79 s); 32 passages 0.19 s (CPU 2.3 s) | 686 MB |
| PP-OCR det / rec | TensorRT / TensorRT | 2.0 s | 0.248 s/image | 414 MB |
| LFM3-VL 2B Q6 (captioner) | llama.cpp CUDA | 2.1 s | ~2.0 s/image incl. tag proposals | 2020 MB |

The 4.8 s/window first measured for audio was cold-start (session builds), not steady state.

**Pool switching, end to end.** tag_visual (A, 36.7 s for 29) → A evicted → caption (B, 59.0 s)
→ B evicted → extract → embed (A, 13.3 s) → finalize. Peak 2.2 GB on the card; with Qwen3.5-4B
as the captioner the same run peaked at 4.0 GB. 25/25 API smoke checks pass on the result.

**Two leaks the run found, both fixed.**
- The embed stage ran with no pool, yet an image's item vector needs a vision pass, so the
  vision encoder loaded onto CUDA outside any lease and stayed (~0.7 GB) while pool B loaded
  llama.cpp. embed now takes pool A. On the CPU it would cost 2.15 s/image.
- `Engine.holds(pool)` is scoped to the lease-holding thread, so the ingest-time CUDA backbone
  is used only by the stage that holds pool A; a search query running beside it stays on the
  CPU copy and never touches VRAM it does not own. The CUDA copy is dropped with the encoders.

**Eye against JoyTag** (per-image diff, judged by eye from a contact sheet). Gains are mostly
right: *photo, sky, cloud, building*, *table* on the dashboard screenshot, *bridge/road* on the
car-on-a-bridge shot. Doubtful: *night* on a sunset silhouette, *tree* on a skyline. Losses are
on abstract renders (noise, a 3D spiral, a maze texture, a neon sign) where JoyTag's tags were
guesses anyway. The diff also exposed two older bugs, fixed: a re-run of tag_visual re-accepted
tags a person had rejected, and changing a term's prompt kept the old prompt's CLIP calibration
(it inflated *indoors / night / portrait*); a reprompt now clears it and the reseed recalibrates.

**Captioner.** LFM2.5-VL 3B's weights are no longer on this machine; LFM3-VL 2B (local, `~/models`)
replaces it. It is a hybrid reasoner, and under the GBNF grammar thinking ON returns an empty
`content` (llama.cpp files the constrained output under `reasoning_content`), so it runs with
`enable_thinking = false`: 1.2 s/image on 8 images with accurate captions, against Qwen3.5-4B's
2.6 s. It has not been through the §17.9 caption-as-query bake-off; `vlm_bakeoff.py` has it as
`lfm3_vl_2b`.

**A fresh library ([M8], the demo library in `docs/DEMO.md`).** Building one from nothing found
what an upgraded library never exercises:
- The vocabulary was never seeded on a new database, so nothing was ever tagged. Startup now
  seeds when there are no active terms.
- Nothing calibrated CLIP for a new library. tag_visual calibrates once 8+ image vectors exist
  and re-scores everything tagged on the fallback until then, once.
- A re-score only ever added tags; one it no longer supports is now retracted (never a
  rejection, a human tag, or a captioner proposal).
- Eye's scene tags are photo-trained and misfire on graphics (a bar chart got *food*, *person*
  and *chair* at 0.67–0.92 against EG2's 0.01–0.03). They count only when Eye's own type head
  says the image is a photograph.
- The CUDA soname links went dangling when the checkout moved, so cuDNN failed to load after the
  folder rename; they are re-pointed on start.
- Canvas: a short result list wrapped into a wall of repeats, the best hit sat under the search
  bar, and column strides sharing a factor with the list length drew diagonals of the same
  tiles. Audio tiles are now sheets with a theme-coloured waveform; posters are the best of 7
  frames.

On that library (16 photos, 10 documents, 12 notes, 12 recordings, 11 clips, ~50 min of media):
ingest ~7 min with a 3.2 GB foreign process on the card (the audio encoder fell back to CPU);
*"why we go to the moon"* → the Kennedy speech at 8:20, *"a dragon"* → *Sintel* at 5:20,
*"what did I spend at the cafe"* → the receipt, *"rocket launch"*, *"northern lights"*,
*"waterfall"*, *"cat"* each top-1.

**Still owed.**
- `min_cos_moment` (0.50) and `dim_range` (0.15) on *real* recordings. The synthetic clips put
  the best query cosine at 0.62–0.69 and the median at 0.58–0.66, a band too narrow to tune on.
- EG2 vision encoder at fp16 on TensorRT against int8 on CUDA.

### 20.7 Dimming by relevance

RRF ranks by position, which says nothing about whether the 30th result is nearly as good as the
first. A query about a sound let `img_vec` admit a flat tail of 29 photographs, all ~0.1 below the
answer, and drew them at full strength beside it. Each result now carries `match` (0–1):
vector evidence is `1 − (best cosine of this query − its cosine) / search.dim_range` (0.15),
keyword evidence never drops below 0.6, tag evidence is relative to the best tag hit, and the
strongest piece wins. The canvas draws a result at `0.2 + 0.8 × match` opacity, full again on
hover or keyboard focus, and says "weak match" to a screen reader below 0.34. On the beep query
the three real answers scored 0.80–1.00 and 26 of the 29 photographs 0.4 or less.
`dim_range` is a placeholder until it is tuned on a real library.
