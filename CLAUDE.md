# ZetaDB

Local multimodal archive — screenshots, photos, diagrams, typed notes,
recordings and video, auto-tagged and searchable by keyword, by meaning and by
moment. Single-user, local-only. (Formerly *iwashere*.)

- [spec.md](spec.md) — the technical specification, and the record of what was
  measured rather than assumed. Section 17 is the post-mortem of what the first
  real corpus and the first browser pass found; section 20 is the current state
  (the model engine, Eye, audio/video). Read them before re-deriving a number.
- [docs/SETUP.md](docs/SETUP.md) — install, model weights, run, test.
- [DESIGN.md](DESIGN.md) — the visual system. Every colour pair in it is
  measured against WCAG; do not introduce one that is not.

## Working agreements

- **Branch from `master`. Never commit to it directly.** `master` is the only
  long-lived branch; feature branches are deleted once merged.
- Paths in `config.toml` resolve relative to the repo root, so a git worktree
  gets an empty `data/`. Symlink it (`ln -s ../../../data data`) or the server
  comes up with no library and no model weights. `.gitignore` uses `/data`,
  which covers the symlink as well as the directory.
- **Every model registers with the engine** (`server/engine`): a `ModelSpec`
  with its pool and unload hook, and ONNX sessions only via
  `server.engine.session()`. A model that does not register is never unloaded
  when the other pool needs the card.
- `web/dist` is gitignored. After pulling anything that touches `web/src`,
  rebuild, or you are looking at the old UI. A stale cached bundle has cost real
  debugging time before — see spec §17.8.

## Run and test

```sh
uv run -m server.app                    # backend + built SPA on :8777
cd web && npm run dev                   # hot-reload UI on :5173, proxies /api

uv run --with pytest pytest tests/ -q   # 68 unit + regression tests, ~1s, no GPU
uv run scripts/smoke.py                 # 25 API contract checks, needs a server
cd web && npm run build && npm run check
```

`scripts/smoke.py` is the one that catches a mismatch between
`web/src/lib/api.ts` and `server/api/*`, which breaks the UI silently with no
server-side error.

**None of the above looks at a pixel.** A UI is not tested until it has been
looked at — every bug in spec §17.5 passed the contract tests first. Chromium
is at `~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`; driving it
over CDP with `uv run --no-project --with websockets python <script>` works
well. Instrument `requestAnimationFrame` inside the page rather than timing
screenshots — CDP round-trips are longer than most transitions, which will tell
you an animation is missing when it is not.

## Remaining work

**[M8] Measured on the GPU** (spec §20.6): engine providers, latencies and
VRAM, pool A↔B switching end to end, Eye against JoyTag, and LFM3-VL 2B as
the captioner. Still owed, on real recordings rather than synthetic clips:

```sh
uv run scripts/m8_gpu_check.py --skip-models --media ~/some-recordings-and-clips
```

- `search.min_cos_moment` (0.50) and `search.dim_range` (0.15): text → audio
  and text → video cosine bands, as §10 did for text → image.
- EG2 vision encoder fp16-on-TensorRT against int8-on-CUDA.
- LFM3-VL 2B through the §17.9 caption-as-query bake-off
  (`scripts/vlm_bakeoff.py`, candidate `lfm3_vl_2b`).
- The audio/video UI in a browser: cards, moment badges, player seeking.

**Documented but not built**

- **Processing tile.** DESIGN.md specifies a Carbon tile with a 2px
  indeterminate Signal Blue bar whose fields fill in as SSE stages land.
  `Canvas.svelte` and `App.svelte` never read `item.status`; the only in-flight
  signal is the counter in the chrome.
- **Settings overlay.** Named in spec §13.1, not built.

**Built since** — `Review.svelte` (spec §13.1/§13.6): both queues, `a`/`r`/`→`,
reachable from the Copper "N pending" control in the chrome. Verified in
Chromium in both themes and at 560px. It is the surface DESIGN.md's
dashed-Copper tag outline was reserved for, so that token is now spent.

**Not yet verified in a browser** (both themes reachable via
`Emulation.setEmulatedMedia`)

- Narrow widths below 660px for the CANVAS and focus panel — the review
  overlay is done, the rest is not: one column, full-width focus panel, graph
  at one hop.
- Search results and the result strip; the command palette; Focus's `ocr` and
  `graph` tabs.
- The empty-canvas hero at 80px — it cannot render while the library has items,
  so seeing it needs an empty database.
- `prefers-reduced-motion`, which should collapse the orbit and both panel
  transitions to instant.

**Loose ends**

- `--r-nav`, `--r-button` and `--fs-3xl` are dead tokens since the chrome moved
  to capsules. Either spend them or delete them; DESIGN.md still lists a 170px
  button radius and a 48px display size that nothing uses.
- Svelte transitions use `cubicOut` where CSS uses the `--ease` token. Visually
  indistinguishable at 240ms, but it is two curves for one idea.
- A theme-contrast trap worth not re-learning: the light theme redefines
  `--text` but leaves `--steel` at its dark `#333336`, so any
  `color: var(--text); background: var(--steel)` pair is ~1.3:1 the moment the
  theme flips. Palette and Review both had it; both now fill with `--border`.
  Pending/Copper must likewise be reached through `--warn`, never `--copper`.
- spec §17.6–17.8 still describe the superseded light-first system (3px tiles,
  12px gutter, 172px columns, the chrome-shadow capsule). §13.3 is marked
  superseded; those sections are kept as history and were deliberately not
  rewritten.

## Traps worth not re-learning

- **An out-transition must never depend on the state whose removal triggers
  it.** Its parameters are read when the element starts leaving, by which point
  that flag has already flipped and the duration collapses to zero. Ask the
  leaving element what it is.
- **A CSS `@keyframes` cannot animate a node that is being unmounted.** Enter
  and exit both belong on Svelte `transition:` directives, which also hold the
  unmount until the out-leg finishes.
- **Mid-grey text dies on glass.** Ash measures 2.27:1 over a bright photograph
  at any usable opacity. The chrome has its own on-glass pair; use it.
- **Chrome controls must not size themselves from their content.** Svelte
  appends a scope class to component CSS, so a component's own `button.round`
  out-specifies the host's `.chrome .round`.
- Key canvas tiles on identity, never on pixel position, and never
  `setPointerCapture` on the canvas — both are explained in spec §17.6.
