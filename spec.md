# iwashere — Technical Specification

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

PDFs (the layout engine is built so they slot in later — see §6.4). Audio, video. Multi-user, auth, sync. Mobile. Browser extension. Global hotkey capture.

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
├──────────────────────────────┬─────────────────────────────────┤
│  GPU pools (mutually exclusive)  │  CPU-resident (always)      │
│                                  │                             │
│  Pool A  "vision-small"          │   Needle 3      (own rt)    │
│    PP-OCRv6 det+rec  TensorRT    │   Qwen3-Emb-0.6B (llama.cpp)│
│    JoyTag            PyTorch     │   jina-clip-v2 text tower   │
│    jina-clip-v2 image tower ONNX │     (ONNX int8)             │
│                                  │                             │
│  Pool B  "vlm"  (exactly one)    │                             │
│    Qwen3.5-4B   Q4_K_M  default  │                             │
│    Gemma 4 12B  IQ4_XS  escalate │                             │
│    via llama-server subprocess   │                             │
├──────────────────────────────────┴─────────────────────────────┤
│  SQLite (WAL)  +  sqlite-vec  +  FTS5      data/iwashere.db    │
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
| ONNX (PP-OCR, JoyTag) | **static QDQ INT8** | calibrated locally, see §5.1 |
| ONNX (jina-clip-v2) | **INT8** | published `model_int8`; q4 variants exist but 4-bit measurably degrades cosine geometry, and this model *is* the similarity metric |
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

| Model | Role | Runtime | Precision | Est. VRAM | Pool |
|---|---|---|---|---|---|
| PP-OCRv6_small det+rec | OCR | ONNX Runtime TRT/CUDA | **INT8** | see §5.1 | A |
| JoyTag | vocab tag scoring | ONNX Runtime CUDA | **INT8** | see §5.1 | A |
| jina-clip-v2 image tower | image embedding | ONNX Runtime CUDA | int8 | **122 MB** load, **1140 MB** peak | A |
| jina-clip-v2 text tower | text embedding | ONNX Runtime CPU | int8 | **0** (CPU) | — |
| **Pool A peak** | | | | **~1.2 GB** measured, one stage at a time | |
| Qwen3.5-4B | captioner | llama.cpp CUDA | UD-Q3_K_XL + mmproj F16, q8_0 KV | 16K ctx, see §5.1 | B |
| ~~Gemma 4 12B~~ | ~~escalation~~ | — | — | **~6.3 GB — dropped, see §17.1** | — |
| Qwen3-Embedding-0.6B | text embedding | llama.cpp CPU | Q8_0 | **0** — needs `CUDA_VISIBLE_DEVICES=""` | — |
| Needle 3 | structured extraction | Cactus runtime CPU | CQ2 | **0** (34 MB weights) | — |

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
- **Qwen3-Embedding-0.6B** — Apache-2.0. Clear.
- **Needle 3** — Apache-2.0. Clear. **[M0]** Note: the package sends anonymous telemetry to a
  Supabase endpoint unless `NEEDLE_TELEMETRY=0` or `DO_NOT_TRACK=1` is set. `server/app.py` sets
  both before import.
- **PP-OCRv6** — Apache-2.0 (PaddleOCR). Confirm the model weights carry the same.
- **jina-clip-v2** — ⚠️ Jina has shipped models under CC-BY-NC-4.0. Fine for personal local use, blocks any commercial path. **Verify before building on it.** Fallback if it's a problem: SigLIP-2 SO400M (Apache-2.0).
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
| 3 | tag_visual | GPU A | JoyTag → full 5000-tag score vector; persist only the vocab-relevant slice. jina-clip-v2 image tower → 1024-d image vector. |
| 4 | caption | GPU B | VLM description. Qwen3.5-4B by default; escalation candidates deferred to a second Gemma 4 sub-batch. |
| 5 | extract | CPU | Needle 3 × 4 schemas (§8). |
| 6 | embed | CPU | Qwen3-Embedding-0.6B over note body / image searchable text. jina-clip-v2 text tower over caption + `flow_text`. |
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
names into `$VIRTUAL_ENV/lib/iwashere-native`, puts it on `LD_LIBRARY_PATH` and re-execs once.
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
| **jina-clip-v2 zero-shot** | *any* vocab term | calibrated cosine z-score |
| **VLM + Needle 3** | any vocab term | membership + Needle confidence |

The CLIP path is what makes arbitrary user vocabulary work at all, and it costs nothing extra: the
image vector is computed once in Stage 3 and **persisted there** for the embed stage to reuse.

**[M3] jina-clip-v2 must be split into two single-tower graphs.** The published export declares
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

**[M3] Measured cosine bands, which is why this section is not optional.** jina-clip-v2 text↔image
cosines land in **0.17–0.37** even for a confident match (query "a dark dashboard screenshot with a
table" scored 0.369 against the right image and 0.196 against the wrong one — correct ranking,
useless absolute scale). Qwen3-Embedding-0.6B with its query instruction prefix puts an exact term
match at **0.67–0.71** and a true synonym at **0.55–0.61**. The spec's original
`tag_seed_min_cos = 0.75` could therefore never fire; it is now 0.55, with vector floors of 0.30
(text) and 0.18 (jina) so KNN does not return the whole corpus as "matches" on a small library.

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

**Write-time gate.** When the VLM proposes a term not in the vocabulary, embed it with Qwen3-Embedding-0.6B against all existing term embeddings:

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

Three spaces, never mixed. Both models output 1024-d.

```sql
-- Qwen3-Embedding-0.6B. Text-native items.
CREATE VIRTUAL TABLE vec_text USING vec0(
  item_id INTEGER PRIMARY KEY,
  embedding FLOAT[1024]
);

-- jina-clip-v2 shared space. kind='image' (image tower)
--                            kind='imgtext' (text tower over caption + flow_text)
CREATE VIRTUAL TABLE vec_jina USING vec0(
  rowid INTEGER PRIMARY KEY,
  item_id INTEGER,          -- metadata column
  kind TEXT,                -- metadata column
  embedding FLOAT[1024]
);

-- vocabulary term embeddings, for the merge gate and tag-seed expansion
CREATE VIRTUAL TABLE vec_vocab USING vec0(
  term_id INTEGER PRIMARY KEY,
  embedding FLOAT[1024]     -- Qwen3-Embedding-0.6B over the term string
);

-- perceptual hashes for near-duplicate detection, Hamming distance
CREATE VIRTUAL TABLE vec_phash USING vec0(
  item_id INTEGER PRIMARY KEY,
  hash BIT[64]
);
```

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

### 9.4 One deviation from the brief, flagged

You specified Qwen3-Embedding-0.6B for text-only items and jina-clip-v2 for images and their text. That is implemented exactly as stated. **I recommend additionally** writing a `vec_text` row for images, embedding their `caption + flow_text` with Qwen3-Embedding-0.6B.

Reason: without it there is no single space containing both notes and images, so note↔image semantic edges and cross-modal ranking have nothing to compare in. The cost is one extra CPU embedding per image and ~4 KB of storage.

Controlled by `embeddings.qwen_covers_image_text`, default `true`. Set it to `false` for the literal specified behaviour; the graph then carries image↔image and note↔note edges only.

---

## 10. Search

### 10.1 Query plan

A query string runs four retrievers in parallel, fused by Reciprocal Rank Fusion.

1. **Tag path** — resolve the query to vocabulary terms: exact, then alias, then KNN over `vec_vocab` above cosine 0.75 ("tag seeds"). Items carrying those terms rank by `tag_confidence × term_similarity`. *This is the mechanism behind the original brief's example: searching `tree` matches items tagged `tree`, and also items tagged with terms semantically near `tree`.*
2. **Lexical** — FTS5 BM25 with the column weights above.
3. **Text semantic** — Qwen3-Embedding-0.6B (with its retrieval instruction prefix) KNN over `vec_text`, k = 50.
4. **Visual semantic** — jina-clip-v2 text tower KNN over `vec_jina`, k = 50, both kinds. This is what makes a plain-language query find an image directly.

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
- **Near-duplicate notes** — cosine > 0.97 in `vec_text`. Same treatment.

---

## 11. Graph and links

**Neighbourhood only. The global graph is never rendered.**

### 11.1 Edges

- **Manual** (`kind='manual'`) — created by the user, `weight = 1.0`, optionally labelled. Rendered as solid lines.
- **Semantic** (`kind='semantic'`) — seeded in Stage 7: for each new item, take its top-8 neighbours in its native space and keep those above threshold (`0.62` in jina space, `0.70` in Qwen space — both config, both requiring calibration against real data). Rendered as dotted lines with thickness ∝ weight.

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
text = "Qwen3-Embedding-0.6B"
text_dim = 1024
multimodal = "jina-clip-v2"
multimodal_dim = 1024
qwen_covers_image_text = true      # see §9.4

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
seed_min_cos_jina = 0.62
seed_min_cos_qwen = 0.70
default_hops = 2
max_nodes = 60

[dedupe]
phash_hamming_max = 6
note_cosine_max = 0.97
```

---

## 15. Project layout

```
iwashere/
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
│   │   ├── jina_clip.py        image tower (GPU) + text tower (CPU)
│   │   ├── qwen_embed.py       llama.cpp CPU embeddings
│   │   ├── vlm.py              llama-server lifecycle
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
    ├── iwashere.db
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
| jina-clip-v2 is CC-BY-NC | Medium | Fine for personal use; confirm before any commercial path. Swap to SigLIP-2 SO400M (Apache-2.0) if needed — same role, same dimensionality after projection |
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
- **TensorRT** — unreachable on this machine (§5); the CUDA EP is fast enough for batch ingest.

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
| **Qwen3.5-4B** UD-Q3_K_XL | 3747 | 3.47 | **29/29** | **1.00** | **0.1408** | 0.711 | 177 | 8.1 |
| Gemma 3 4B IT UD-Q3_K_XL | 3591 | 3.33 | 29/29 | 0.97 | 0.1217 | 0.622 | 216 | 6.9 |
| InternVL3 2B Q4_K_M | **2033** | **1.92** | 25/29 | 0.96 | 0.1279 | 0.422 | 119 | 9.6 |
| SmolVLM2 2.2B Q4_K_M | 3811 | 1.73 | 18/29 | 0.94 | 0.0840 | — | 52 | 5.3 |
| Moondream2 f16 | — | — | 0/29 | — | — | — | — | — |

**The incumbent stays.** Gemma 3 4B is 156 MB lighter and 4% faster and loses on every quality axis
that matters: a worse retrieval margin (0.1217 against 0.1408 — the margin, not recall@1, is the
number that still separates models once all of them "pass"), worse OCR grounding, and fewer tags.
It also prefixes nearly every caption with "The image shows" / "The image depicts", which is why its
captions are *longer* while saying less — that boilerplate is dead weight in an embedding. On item
13 it was simply wrong, calling a sculpture of stacked Mercedes-Benz cars "a robotic insect-like
vehicle" where both other models saw cars.

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

**InternVL3 2B is the one to remember.** It holds 1714 MB less VRAM than the incumbent and captions
in just over half the time, at a retrieval margin between Gemma's and Qwen's. What disqualifies it
today is that 4 of 29 items produce no usable object, and they are the text-dense ones — the invoice
is among them. It also transcribes rather than describes, reading a dashboard back as a list of
bare numbers, which duplicates work PP-OCRv6 already does better (§1.1). If pool B ever has to
shrink to let pool A coexist, this is the first candidate to re-measure.

## 18. Open questions

1. What should the seed vocabulary actually contain? It needs ~100–200 terms to be useful at M4. Bootstrapping option: run JoyTag over the first 200 ingested images and promote its most frequent above-threshold tags as seed candidates for manual curation.
2. Should notes support Markdown rendering, or stay plain text? Plain text is assumed throughout; Markdown affects the card renderer and the FTS tokenisation.
3. Retention: does anything ever get deleted, or is the dump append-only with archive?
4. Should `todos` extracted by Needle 3 surface anywhere beyond the item detail view — a dedicated surface, or deliberately kept passive?

---

## 19. References

- JoyTag — https://github.com/fpgaminer/joytag
- Needle 3 — https://github.com/cactus-compute/needle · https://cactuscompute.com/needle
- PP-OCRv6 — https://huggingface.co/blog/PaddlePaddle/pp-ocrv6 · https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html
- LiteParse grid projection — https://www.llamaindex.ai/blog/how-liteparse-turns-pdfs-into-text-a-deep-dive-into-the-grid-projection-algorithm · https://github.com/run-llama/liteparse
- Qwen3.5 — https://qwen.ai/blog?id=qwen3.5 · https://huggingface.co/Qwen/Qwen3.5-4B
- Gemma 4 12B — https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- Qwen3-Embedding-0.6B — https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- jina-clip-v2 — https://huggingface.co/jinaai/jina-clip-v2
- sqlite-vec — https://github.com/asg017/sqlite-vec · https://alexgarcia.xyz/sqlite-vec/
- Hybrid FTS5 + vector + RRF — https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html
