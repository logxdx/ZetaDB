# iwashere — Style Reference
> Infinite paper canvas. Treat the viewport as one uninterrupted field of edge-to-edge tiles, with every control floating above it as a rounded capsule object rather than sitting in a bar.

**Theme:** light-first, with full dark parity — both grounds are first-class and every value below is given for both.

Values are measured from what ships: `web/src/lib/design/tokens.css`, the component `<style>` blocks under `web/src/lib/components/`, and spec.md §13 / §17.6–17.8. The visual reference is [cosmos.so/public-work](https://www.cosmos.so/public-work), read live rather than estimated; where this project departs from it, the departure is stated.

iwashere is a local idea dump — screenshots, photographs, diagrams and typed notes, auto-tagged and searchable by keyword and meaning. The interface is a moodboard, not a file list: a text note and an image are the same object in the grid and carry equal visual weight. The page ground is plain white so photographs keep their own colour; the only warm surface in the product is the floating chrome, which is what makes it read as an object resting *on* the canvas. Tiles have a 3px radius and nothing else — no border, no shadow, no hover lift. A single orange accent is rationed to focus, active state and the live queue. Everything the machine knows — tags, confidences, dates, counts, match reasons — is set in monospace, and that one typographic contrast does most of the aesthetic work.

## Colors

| Name | Light | Dark | Role |
|------|-------|------|------|
| Paper | `#FFFFFF` | `#0B0B0C` | Canvas ground and the surface behind every tile |
| Shelf | `#FFFFFF` | `#141416` | Overlay surfaces: focus panel, command palette |
| Raised | `#F2F0EE` | `#1C1C1F` | Note-tile fill, tab rests, code blocks, graph ground |
| Chrome | `#F2F0EE` | `#191919` | Every floating capsule — search bar, round buttons, related bar, summary line, info disc |
| Hairline | `rgba(0,0,0,0.12)` | `rgba(255,255,255,0.14)` | The 0.8px chrome keyline, on capsules only |
| Border | `rgba(0,0,0,0.12)` | `#232327` | 1px dividers inside overlays, tag-pill outlines, kbd marks |
| Ink | `#0A0A0A` | `#EDEDEF` | Primary text — near-black and near-white, never `#000` or `#FFF` |
| Muted | `#6E6E76` | `#8A8A93` | Metadata, inactive capsule labels, secondary copy, placeholder text |
| Ember | `#FF5A1F` | `#FF5A1F` | Focus ring, active tab, selected tag, highlighted graph edge, live ingest queue, dropzone |
| Verified | `#2F7A4F` | `#4ADE80` | A probe or stage that succeeded; human-confirmed state |
| Pending | `#A8721A` | `#FBBF24` | Machine tags awaiting review, degraded stages |
| Absent | `#8A8A93` | `#6E6E76` | A component that is not installed — absence, not failure |
| Failed | `#C0392B` | `#F87171` | Stage errors and surfaced exceptions |

**The dark column is the `:root[data-theme="dark"]` block.** `tokens.css` currently defines dark twice — once under `prefers-color-scheme` and once under the explicit toggle — and the two disagree by a hair on six values (`#0A0A0B`/`#0B0B0C`, `#141415`/`#141416`, `#1A1A1C`/`#1C1C1F`, `#242426`/`#232327`, `#ECECEE`/`#EDEDEF`, `#8C8C94`/`#8A8A93`). The drift is invisible in use but means system-dark and toggled-dark are not the same theme; the values above are the canonical set, and the media-query block should be reconciled to them.

Ember is the only chromatic colour in the interface and it is identical in both themes. It clears 4.5:1 on the dark ground but **not** on white, so it is never used for body text, labels, or anything a reader has to parse — only for rings, fills, strokes and a count.

## Typography

### Inter — Content. Note bodies, titles, search input, hover hints, prose, captions, the new-note composer, and glyph-bearing buttons (`×`, `+`). Carries everything a *person* wrote.
- **Substitute:** `ui-sans-serif, system-ui, -apple-system, sans-serif`
- **Weights:** 400, 500, 600
- **Sizes:** 13px, 15px, 18px, 19px, 24px, 32px
- **Line height:** 1.0, 1.45, 1.5, 1.6
- **Letter spacing:** -0.02em on the search field and hover hints, -0.01em on note bodies and headings, 0 on prose

### JetBrains Mono — Metadata. Tags, confidences, dates, counts, match reasons, capsule control labels, relation toggles, layout dumps, graph glyphs, keyboard hints. Carries everything the *machine* inferred.
- **Substitute:** `ui-monospace, "SF Mono", Menlo, Consolas, monospace`
- **Weights:** 400
- **Sizes:** 9px (graph glyphs), 11px (info disc), 12px (everything else)
- **Line height:** 1.45, 1.6
- **Letter spacing:** +0.02em on capsule labels and toggles, -0.01em on tag rows

The split is the rule, not a preference: if a human typed it, it is Inter; if the pipeline produced it, it is mono. Provenance then reads at a glance with no legend anywhere in the product.

### Type Scale

| Role | Family | Weight | Size | Line Height | Letter Spacing |
|------|--------|--------|------|-------------|----------------|
| meta | JetBrains Mono | 400 | 12px | 1.6 | 0.02em |
| tag-pill | JetBrains Mono | 400 | 12px | 1.6 | — |
| capsule-label | JetBrains Mono | 400 | 12px | 1 | 0.02em |
| search-input | Inter | 400 | 13px | 1 | -0.02em |
| note-tile | Inter | 400 | 13px | 1.45 | -0.01em |
| body | Inter | 400 | 15px | 1.5 | — |
| prose | Inter | 400 | 15px | 1.6 | — |
| focus-title | Inter | 600 | 18px | — | -0.01em |
| composer | Inter | 400 | 18px | 1.5 | — |
| display | Inter | 500 | 24px | — | -0.01em |
| display-xl | Inter | 500 | 32px | — | -0.01em |

24px and 32px exist in the scale and are deliberately unspent in the shipped surfaces — this interface has no headline. If a marketing or onboarding surface is ever added, they are its ceiling.

## Spacing & Layout

**Density:** compact

- **Canvas gutter:** 12px, both axes
- **Chrome inset:** 10px from every viewport edge
- **Card padding:** 14px (note tile), 16px (focus body and sections)
- **Element gap:** 4px (tag rows), 8px (chrome children), 12px (focus sections)
- **Column target:** 172px — `columns = max(2, round(vw / (172 + 12)))`, `pitch = vw / columns`, `width = pitch − 12`
- **Chrome height:** 52px (`--chrome-h`); related bar 54px; summary line 42px
- **Minimum side gutter:** 16px at every width, set once on the app wrapper

### Border Radius

- **tiles:** 3px
- **capsules:** 999px — search bar, round buttons, toggles, pills, related bar, summary line, info disc, avatars
- **focusPanel:** 20px
- **surfaces:** 4px — command palette, graph, code and layout dumps, `kbd`
- **tagPills:** 3px
- **dropzone:** 8px

Two radii carry the product: 3px for anything holding content, 999px for anything you can click. The 20px focus panel is the single intermediate value, because a fully-rounded 620px panel reads as a pill rather than a document.

## Components

### Floating Chrome Capsule
**Role:** The one recipe every control shares

Full rounding (999px), `--chrome` fill, a 0.8px `--chrome-border` hairline, an inner top highlight, an inner bottom shade, and a soft drop shadow:

```css
box-shadow:
  inset 0 1px 1.5px rgba(255,255,255,0.7),
  inset 0 -1px 1.5px rgba(0,0,0,0.03),
  0 1px 8px rgba(0,0,0,0.05);          /* dark: 0.07 / 0.3 / 0.45 */
```

Those four layers together are what make a capsule read as a physical control rather than a div. One recipe applied six times — mark, search bar, round buttons, related bar, summary line, details card — is what makes the set read as a family instead of a toolbar.

### Search Capsule
**Role:** Primary input, floating top-left

A 52px capsule, `flex: 1 1 auto`, 8px padding, containing a borderless transparent input at Inter 13px with -0.02em tracking and a `--muted` placeholder. The live ingest count sits inside it in Ember mono. No submit control; search is live.

### Round Utility Button
**Role:** Secondary chrome action

Same capsule recipe, `min-width: 52px`, 0–16px horizontal padding, mono 12px at +0.02em in `--muted`, resolving to `--text` on hover over 140ms. Houses the theme cycle, the palette trigger and view switches.

### Inline Disc Button
**Role:** An action that belongs *inside* a capsule

A 34px disc filled with `--bg` and outlined with the chrome hairline, centred glyph in Inter 19px. Hover lifts colour to `--text` and scales to 1.06 over 160ms. Used for the new-note `+`, so it sits among the field's inhabitants rather than beside the capsule.

### Canvas Tile
**Role:** One item in the grid — image or note, identical weight

3px radius, `--raised` fill, `overflow: hidden`, no border and no shadow. Images are `object-fit: cover`, edge to edge, non-draggable and non-selectable. Absolutely positioned by JS, never CSS columns — CSS columns lay out top-to-bottom per column, which destroys keyboard and reading order. Key tiles on identity (column, repetition, slot), **never** on pixel position.

### Note Tile
**Role:** A typed note in the grid

The one filled surface on the canvas: `--raised` background, 14px padding, Inter 13px at 1.45 line-height and -0.01em, left-aligned, clipped at the tile bound. A note has no image to carry it, so the fill is what lets it hold its own against a photograph in the next column.

### Hover Veil
**Role:** Deferred metadata over a tile

An inset panel at `color-mix(in srgb, var(--bg) 86%, transparent)`, centred column, 5px gaps, fading in over 140ms on hover only. Carries the human-readable match reason (`similar wording`, `looks similar`, `text match`, `tagged`) and a muted 12px tag line. Nothing in the veil is ever shown permanently on the card.

### Info Disc
**Role:** Open details without leaving the canvas

A 20px chrome-recipe disc at the tile's top-right, mono 11px italic `i`, `opacity: 0` until the tile is hovered or it takes focus. Fills with Ember and white on its own hover — the one place the accent becomes a background.

### Tag Pill
**Role:** A tag plus its provenance

3px radius, 1px 6px padding, mono, 1.6 line-height, 4px internal gap.

| State | Treatment |
|---|---|
| human | 1px solid `--border`, `--text`, full opacity |
| machine, accepted | 1px dotted `--border`, `--muted`, 0.75 opacity |
| machine, pending | 1px dashed `--warn`, `--warn` text, confidence in mono — `wireframe 0.62` |
| hover, any | confidence and source in a mono tooltip; `×` removes |

Border style alone encodes provenance. No colour-coding, no legend, no badge containers.

### Relation / Sort Toggle
**Role:** A small set of mutually exclusive modes

A 2px-gap row of transparent 999px capsules with a 0.8px transparent border, mono 12px `--muted`, 6–7px by 12–13px padding. The selected one — and only it — takes a `--bg` fill and the chrome hairline, so the row stays monochrome and the active state is a surface rather than a colour.

### Summary Line
**Role:** What the current result set is

A 42px chrome capsule 18px below the chrome, left-aligned, `width: max-content`, mono `--muted` with the count in `--text` at weight 500. Retriever names never appear here — the label is the same human wording the veil uses. Seed counts take Ember.

### Related Bar
**Role:** Names the subject of an orbit

A 54px chrome capsule in the same slot as the summary line: a 40px circular thumbnail (or a `--bg` disc for a note), the title in Inter 13px ellipsised at 34vw, a muted relation count, the similar/tag toggle, and a 30px round close. Never a second header. Edge kind and weight are plumbing — they live in `title` attributes, not on the surface.

### Focus Overlay
**Role:** Everything known about one item

Inset 10px from top, right and bottom, `min(620px, 100vw − 20px)` wide, `--surface` fill, 20px radius, chrome hairline, `--chrome-shadow` plus `0 18px 50px rgba(0,0,0,0.16)`. Enters with a 240ms 16px slide. A `--bg`-tinted scrim at 78% sits behind it so the canvas stays visible. Hero media is capped at `52vh` with `object-fit: contain` — a 2:3 photo at full width otherwise pushes title, tags and caption below the fold on open. Below 660px it spans both edges. Lock `documentElement`, not `body`, or the canvas scrolls behind a stationary panel.

### Command Palette
**Role:** The only navigation

Centred at 18vh, `min(640px, 100vw − 32px)`, `--surface`, 1px `--border`, 4px radius. A mono tab row (search ↔ new note) over a borderless Inter 18px input. The active tab is marked by an Ember border, the only accent in the overlay. `⌘K` / `Ctrl-K` opens, `Tab` switches, `⌘↵` saves, `Esc` closes.

### Graph Panel
**Role:** The neighbourhood of one item

Full-width SVG on `--raised` with a 1px `--border` and a 4px radius. Node labels in mono 9px `--muted`; a node's rect strokes Ember on hover. Drops to one hop under 400px.

### Dropzone
**Role:** Drag-and-drop target

Appears on dragover only: `inset: 10px`, a 1.5px dashed Ember border, 8px radius, `--bg` at 72% opacity, centred Ember label. The whole window is the target.

### Orbit Focus Lift
**Role:** Marks the item an orbit grew from

`0 0 0 0.8px var(--chrome-border), 0 14px 38px rgba(0,0,0,0.22)`. Lift, never colour — an orange stripe across a photograph is the one thing the reference would never do.

## Do's and Don'ts

### Do
- Keep the page ground plain `#FFFFFF` and let the chrome be the only warm surface.
- Give every clickable control the one chrome recipe — 999px, `--chrome`, 0.8px hairline, inner highlight, soft shadow.
- Hold tiles at a 3px radius with no border, no shadow and no hover lift.
- Set all machine-produced text in JetBrains Mono and all human text in Inter.
- Keep the gutter at 12px on both axes and the column pitch exactly `vw / n`.
- Encode tag provenance with border style — solid, dotted, dashed — not colour.
- Spend Ember only on focus rings, active state, the selected tag, the highlighted edge, the ingest queue and the dropzone.
- Mark the subject of an orbit with a hairline and a drop shadow.
- Name a match in human words — `similar wording`, `looks similar`, `text match`, `tagged` — and put it in the hover veil.
- Hold the focus hero at `max-height: 52vh` with `object-fit: contain`.
- Keep a 16px side gutter at every width down to 400px, set once on the wrapper.

### Don't
- Do not tint the page ground: every photograph picks up a cast from warm paper behind it.
- Do not use Ember for body text or any label — it fails 4.5:1 on white.
- Do not give tiles borders, shadows, hover lifts or radii above 3px.
- Do not vary column widths to make the grid look livelier — it breaks the vertical rhythm the moment you pan sideways.
- Do not surface retriever identifiers (`text_vec`, `img_vec`, `lexical`) or edge weights (`semantic 0.715`) anywhere on a card.
- Do not put a badge on every card: a permanent chip over every photograph is pure noise, because every result matches *something*.
- Do not place a second bar under the chrome — floating capsules in the same slot, never a header stack.
- Do not use pure `#000` or `#FFF` for text.
- Do not introduce gradients, skeleton shimmer, hover fades or page transitions.
- Do not key tiles on pixel position, or Svelte destroys and recreates every `<img>` on every wheel event.
- Do not call `setPointerCapture` on the canvas — the tile underneath then never receives its click.

## Elevation

Elevation is a single recipe reused at three intensities, never a scale. The chrome shadow (`--chrome-shadow`: two inset lines plus `0 1px 8px` at 5% light / 45% dark) separates every floating control from the canvas. The focus panel adds `0 18px 50px rgba(0,0,0,0.16)` on top of it because it is the only surface that must sit above *everything*. The orbit subject takes `0 14px 38px rgba(0,0,0,0.22)` with a 0.8px ring. Nothing else in the product casts a shadow — tiles, pills and the grid are flat by rule, and layers are separated by surface shifts and 0.8px hairlines instead.

## Surfaces

- **Paper Canvas** (`#FFFFFF` / `#0B0B0C`) — `position: fixed; inset: 0`, `cursor: grab`. The page itself; nothing reserves vertical space above it.
- **Chrome Capsule** (`#F2F0EE` / `#191919`) — every floating control, resting on the canvas rather than docked to an edge.
- **Raised Fill** (`#F2F0EE` / `#1C1C1F`) — note tiles, tab rests, code and layout dumps, the graph ground.
- **Overlay Shelf** (`#FFFFFF` / `#141416`) — focus panel and command palette, each over a `--bg`-tinted scrim at 70–78%.

## Motion

Two budgets, both collapsing to instant under `prefers-reduced-motion: reduce`.

**Canvas motion** is a target/actual split: `tx/ty` is where the canvas is heading, `ox/oy` is what is drawn, and a rAF loop closes the gap with `k = 1 − exp(−dt/tau)` — framerate-independent, identical at 60 and 144 Hz. `TAU_FOLLOW = 95` for the wheel, `TAU_GLIDE = 230` for released momentum. A 400px wheel delta spreads over ~30 frames on a decaying curve rather than jumping. A drag moves target and actual together so the canvas stays welded to the cursor, then hands its velocity over.

**The orbit** grows rather than replaces: the clicked tile stays put and its neighbours scale up around it in rings, staggered by rank so the ring blooms outward. Ring radius is derived from the actual tiles — each is placed at an angle proportional to its own footprint, and a ring clears the one inside it. Sizing by width alone collides, because a portrait tile is up to 1.55× taller than wide.

Everything else: 140ms colour and opacity on chrome and veils, 160ms transform on the inline disc, 240ms on the focus slide, all on `cubic-bezier(0.2, 0.8, 0.2, 1)`. Transitions are scoped to the orbit and must never apply to library tiles — the keyed list churns constantly as tiles cross the cull window, and animating each one leaves a permanent backlog of entering and dying nodes.

## Imagery

Imagery is whatever the user dumped in, which means it cannot be art-directed — screenshots, phone photos, diagrams, scans and UI crops all land in the same grid. The design's job is to stay out of its way: no tint behind it, no frame around it, no chrome on top of it. Tiles crop to `cover` so the grid rhythm holds regardless of source aspect ratio; the focus hero switches to `contain` because that is the one place the whole image matters. Processing cards show a skeleton with a thin indeterminate Ember bar and fill in as SSE stages land. Near-duplicate clusters collapse to one tile with a mono `×3` badge. Icons are 1.3px-stroke monochrome line glyphs at 15px, `currentColor`, with no fills except where a theme icon deliberately inverts.

## Layout

The canvas *is* the page: `fixed inset-0`, panning forever in both directions over a finite library, with the chrome floating on top so nothing reserves vertical space. Columns are indexed by any integer, so there is no left or right edge; column `c` draws from the item list at a different rotation, so panning sideways keeps turning up new things rather than repeating a block. Each column wraps modulo its own stack height, so there is no global tile boundary to line up and therefore no seam. Cull to the viewport, then cap on a tile budget — 140 in the shipped canvas — because 24 columns × every repetition intersecting a tall viewport generates around a thousand absolutely-positioned nodes otherwise.

Chrome sits 10px from the top and both sides: the search capsule flexed wide with round buttons beside it, and one capsule 18px beneath it — summary line or related bar, never both. Clicking a tile grows an orbit of its neighbours around it in place; the info disc opens the focus panel down the right edge. Below 660px the focus panel spans both edges; at 400px the grid is two columns and the graph drops to one hop.

## Reference Points

- **cosmos.so** — The direct reference: uniform `vw / n` pitch, 12px gutters, 3px tiles, no card chrome, white ground, and the floating capsule recipe. It is vertical masonry with `overflowX: hidden`; the two-axis pan and the orbit are this project's own additions.
- **Are.na** — Shares the premise that a saved image and a typed note are the same kind of object, and that provenance matters more than decoration.
- **Raycast** — The same conviction that a single keyboard-first palette can replace navigation entirely, and the same monospace-for-metadata discipline.
- **Apple Photos** — Edge-to-edge uniform-pitch media grids where the interface disappears behind the content and controls float as translucent capsules.
