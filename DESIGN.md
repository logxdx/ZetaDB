# ZetaDB — Style Reference
> Midnight library. Treat the canvas as a black exhibition floor and each tile as one precisely lit object on it, with typography and blue controls operating as restrained labels around the work.

**Theme:** dark

The visual system here is adapted from the Apple product-page reference: its palette, type system, 28px geometry, shadowless elevation and rationed blue are taken as given. What changes is the context — those values are mapped onto ZetaDB's surfaces (infinite canvas, floating chrome, tiles, tags, focus panel, orbit) rather than onto a storefront. Colour values and tracking are the reference's measurements; role assignments are this project's.

ZetaDB (formerly iwashere) is a local multimodal archive — screenshots, photographs, diagrams, typed notes, recordings and video, auto-tagged and searchable by keyword, by meaning and by moment. Under this system it stops looking like a tool and starts looking like a gallery: an uninterrupted Obsidian field that pans forever, oversized SF Pro Display type where the page is empty or a note leads, and dense SF Pro Text metadata sitting quietly in Ash beside the imagery. Broad 28px corners on every tile and panel, near-invisible elevation, compact floating chrome, and a single Signal Blue reserved for the one action in view make browsing read as a controlled exhibition rather than a card-based file list.

## Colors

| Name | Value | Role |
|------|-------|------|
| Obsidian | `#000000` | The canvas ground — the infinite field every tile sits on — plus image tiles, media wells and the graph stage |
| Graphite | `#0e0e0e` | Floating chrome: search bar, round controls, related bar, result strip |
| Carbon | `#1d1d1f` | Note tiles, focus panel, command palette, and any surface that must read as a sheet rather than a hole |
| Carbon Fade | `linear-gradient(#1d1d1f, #000000 288px)` | The only permitted gradient: the top of the focus panel dissolving into the canvas behind it |
| Steel | `#333336` | Hairline keylines on panels, translucent control fills, the hover veil, the near-dup count marker |
| Slate | `#6e6e73` | Input outlines, dividers, and the 1px ring that marks the subject of an orbit |
| Ash | `#86868b` | All machine metadata — tags, dates, counts, confidences, match reasons — and inactive control labels |
| Porcelain | `#f5f5f7` | Primary text on every dark surface: note bodies, titles, search input, headings |
| White | `#ffffff` | Light-mode tile and panel fills, and icon detail on blue |
| Signal Blue | `#0071e3` | Filled primary controls only — save a note, accept a tag, confirm in the palette |
| Electric Link | `#2997ff` | Interactive text on Obsidian and Carbon: the selected tag, the highlighted graph edge, inline links, the focus ring |
| Cobalt Link | `#0066cc` | The same on White light-mode surfaces |
| Copper | `#b64400` / `#ef6c1a` | Pending machine tags, unreviewed vocabulary proposals, "new since last visit" |
| Optical Yellow | `#ffd500` / `#7a5c00` | Confidence readouts and technical values inside the focus panel |

**Why those two are pairs.** The reference uses Copper and Optical Yellow on *white*, and neither survives a black ground. Measured: Copper `#b64400` is 3.82:1 on Obsidian and 3.06:1 on Carbon — it fails 4.5:1 on both, and it carries a pending tag, which is text a person has to read. `#ef6c1a` is the nearest lift that passes (6.83 / 5.47). Optical Yellow is the mirror case — 11.84:1 on Carbon, 1.42:1 on White. Every status role therefore resolves per theme: the first value is the light-surface one, the second the dark.

**Two additions the reference has no equivalent for.** A marketing page has no failing pipeline; ZetaDB does, and four ingest stages can each degrade independently. Extending the palette minimally rather than overloading Copper:

| Name | Light | Dark | Role |
|------|-------|------|------|
| Verified | `#1d7a3e` | `#30d158` | A stage that completed, a human-confirmed tag — the reference's lone green technical cue, generalised |
| Alert | `#c0392b` | `#ff453a` | Stage failure and surfaced exceptions, nothing else |

Both are used as small text and 1px marks only, never as fills. Copper must not absorb this work: it annotates newness, not failure.

Everything in this section is measured, not assumed. Porcelain on Obsidian is 19.29:1, Ash on Carbon 4.65:1, Electric Link on Carbon 5.58:1, White on Signal Blue 4.70:1. Slate sits at 4.14:1 on Obsidian, which is why it is an outline colour and never text — 1px marks need 3:1, not 4.5:1.

## Typography

### SF Pro Text — Note bodies, search input, tag labels, confidences, dates, counts, match reasons, control labels, keyboard hints, captions, prose. The compact negative tracking keeps dense interface language visually quiet beside the imagery.
- **Substitute:** Inter
- **Weights:** 400, 500, 600
- **Sizes:** 12px, 14px, 17px, 20px
- **Line height:** 1.24, 1.29, 1.33, 1.43, 1.47
- **Letter spacing:** -0.264px at 12px, -0.224px at 14px, -0.374px at 17px, -0.38px at 20px
- **OpenType features:** `"numr"` on confidence values

### SF Pro Display — Note ledes, item titles, section labels, counts presented as statements, and the empty-canvas statement. At 80px with -1.2px tracking the weight-600 face compresses into a single sculptural block rather than a loud banner.
- **Substitute:** Inter
- **Weights:** 500, 600
- **Sizes:** 19px, 21px, 28px, 32px, 48px, 80px
- **Line height:** 1.00, 1.05, 1.07, 1.08, 1.13, 1.19, 1.21
- **Letter spacing:** -1.2px at 80px, -0.24px at 48px, 0.128px at 32px, 0.196px at 28px, 0.231px at 21px, 0.228px at 19px

**The one deviation, and what it costs.** The previous system set every machine-produced value in JetBrains Mono, and that contrast carried provenance with no legend. This reference has no monospace anywhere, so metadata moves to SF Pro Text 12px/400 in Ash and provenance is carried by **colour and weight instead of typeface**: Porcelain for what a human wrote, Ash for what the machine accepted, Copper for what is still pending. Monospace survives in exactly one place — literal readouts that are code, not labels: the layout dump and the raw-JSON block inside focus. Everywhere else, mono is gone.

### Type Scale

| Role | Family | Weight | Size | Line Height | Letter Spacing |
|------|--------|--------|------|-------------|----------------|
| meta | SF Pro Text | 400 | 12px | 1.33 | -0.264px |
| tag | SF Pro Text | 400 | 12px | 1.33 | -0.264px |
| tag-human | SF Pro Text | 600 | 12px | 1.33 | -0.264px |
| control-label | SF Pro Text | 500 | 12px | 1.33 | -0.12px |
| note-body | SF Pro Text | 400 | 14px | 1.43 | -0.224px |
| search-input | SF Pro Text | 400 | 17px | 1.47 | -0.374px |
| body | SF Pro Text | 400 | 17px | 1.47 | -0.374px |
| section-label | SF Pro Display | 500 | 19px | 1.21 | 0.228px |
| item-title | SF Pro Display | 600 | 21px | 1.19 | 0.231px |
| focus-title | SF Pro Display | 600 | 28px | 1.07 | 0.196px |
| stat | SF Pro Display | 600 | 28px | 1.00 | 0.196px |
| note-lede | SF Pro Display | 600 | 32px | 1.13 | 0.128px |
| display | SF Pro Display | 600 | 48px | 1.08 | -0.24px |
| display-xl | SF Pro Display | 600 | 80px | 1.05 | -1.2px |

The reference spends its display sizes on a hero product reveal. ZetaDB has two honest homes for them: **the empty canvas**, which becomes the hero stage, and **the first line of a note tile**, set as a lede so a typed thought holds the same visual weight as a photograph beside it.

## Spacing & Layout

**Density:** comfortable

- **Section gap:** 24px
- **Card padding:** 16px
- **Element gap:** 4px
- **Canvas gutter:** 24px, both axes
- **Chrome inset:** 24px from every viewport edge
- **Chrome height:** 44px for the global bar, 52px for the result strip and related bar
- **Column target:** 300px — `columns = max(2, round(vw / (300 + 24)))`, `pitch = vw / columns`, `width = pitch − 24`, fluid at every width with no max

**This widens the grid, and it has to.** The previous canvas ran a 172px column target at a 12px gutter — compact density, which is what a 3px tile radius wants. A 28px corner on a 172px tile eats a third of its width and reads as a lozenge; 16px of internal padding leaves a note nowhere to breathe. Comfortable density is not a garnish on this system, it is load-bearing. At 1440px the grid becomes 4 columns of 336px; at 992px, 3 of 307px; at 430px it collapses to 1.

### Border Radius

- **tiles:** 28px
- **panels:** 28px — focus overlay, command palette, graph well
- **navigation:** 9999px — the chrome bar, result strip and related bar are capsules, not rounded rectangles. A glass material reads as an object when it is fully rounded and as a panel when it is not; 20px was what made the old top bar look assembled.
- **pills:** 9999px — tags where they take a container, toggles, the near-dup marker
- **buttons:** 170px — the filled primary action
- **inputs:** 980px — the search field and the palette field
- **compactButtons:** 36px — the round close, the info marker, the theme control
- **links:** 10px — the focus ring's own corner on inline interactive text

## Components

### Liquid Glass Chrome
**Role:** Global controls, floating on the canvas

The chrome is a **material**, not a panel: it lets the library show through, blurs it, and carries a specular edge and a soft lift. A 44px-tall fully-rounded capsule, inset 24px from the top and both sides, holding the search field, the sort row, the add control and the ingest count. The canvas is `fixed inset-0` beneath it and nothing reserves vertical space.

```css
background: rgba(29, 29, 31, 0.78);          /* light: rgba(255,255,255,0.82) */
backdrop-filter: saturate(180%) blur(28px);
border: 1px solid rgba(255, 255, 255, 0.14);
box-shadow:
  inset 0 1px 0 rgba(255, 255, 255, 0.16),   /* specular top edge */
  inset 0 -1px 0 rgba(0, 0, 0, 0.30),        /* shaded bottom edge */
  0 10px 34px rgba(0, 0, 0, 0.42);           /* the lift */
```

**The opacity is measured, not chosen.** Over a white photograph, Carbon glass at 0.55 leaves Porcelain at 3.49:1; 0.78 leaves it at 7.54:1. And it is why **secondary text on glass can never be Ash** — mid-grey on a mid-grey composite tops out at 2.27:1 over a bright backdrop, no matter the opacity. The chrome carries its own secondary, `#c7c7cc` on dark and `#3a3a3c` on light, which hold 4.88:1 and 7.44:1 in the worst case. Electric Link fails here too, at 2.8:1, so live counters on the chrome take the on-glass colour and earn emphasis from weight.

Without `backdrop-filter` support the glass would be a flat wash over the photographs, so an `@supports` fallback drops to the opaque `--chrome` fill.

The result strip and the related bar take the same material — they are the same family, and leaving them opaque beside a glass bar is what made the old chrome look assembled rather than designed.

### Search Field
**Role:** Primary input

`rgba(255,255,255,0.04)` fill, a 1px Slate `#6e6e73` border, a 980px radius, Porcelain text at SF Pro Text 17px/400 with -0.374px tracking, Ash placeholder, 24px left and 45px right padding. Focus applies a 1px Signal Blue `#0071e3` outline at a 1px offset. No submit control; search is live.

### Round Utility Control
**Role:** Secondary chrome action

A **true circle at the bar's own height** — 44×44, fully rounded, in the same glass as the bar beside it, so the two read as one set of objects rather than a bar with something stuck to it. Hover tints the material with an inset spread rather than replacing the background, so the blur and specular edge survive. Houses the theme cycle.

A control in the chrome must never size itself from its content. This one did, and because Svelte appends a scope class to component CSS, its own `button.round` rule out-specified the chrome's `.chrome .round` — so it collapsed to a 43×19 slab beside a 44px bar. The chrome owns the box; the component owns only its glyph.

### Filled Action Pill
**Role:** The one primary action in view

Signal Blue `#0071e3` fill, White SF Pro Text 12px/400, 16px line-height, -0.12px tracking, a 170px radius, 7px vertical and 14px horizontal padding when compact. There is **at most one** on screen: save the note, accept the tag, confirm the palette. Every other control is outlined or text-only, which is what keeps this one legible as the thing to do.

### Canvas Tile
**Role:** One item in the grid — image or note, identical weight

28px radius, Obsidian `#000000` fill, `object-fit: cover`, no border, no shadow, no hover lift. Absolutely positioned by JS, never CSS columns — CSS columns lay out top-to-bottom per column, which destroys keyboard and reading order. Key tiles on identity (column, repetition, slot), **never** on pixel position.

### Note Tile
**Role:** A typed note, holding its own against a photograph

Carbon `#1d1d1f` fill at a 28px radius with 16px padding. The first line is a lede in SF Pro Display at 1.13 line-height and 0.128px tracking; the remainder is SF Pro Text 14px/400 in Ash at 1.43. The lede is 32px at the 300px column target and **scales with the tile** — `clamp` it to the width, bottoming out at 15px, or a 190px orbit tile gets a headline in a stamp. Split at the first line break, else the first sentence, else a word boundary near 64 characters. A photograph is carried by its own content, so a note is given the one filled sheet and the one piece of display type on the canvas, and the grid stays balanced.

### Tile Hover Veil
**Role:** Deferred metadata over a tile

An inset panel at `color-mix(in srgb, #000000 72%, transparent)`, 28px radius to match its tile, fading in over 140ms on hover only. Carries the human-readable match reason — `similar wording`, `looks similar`, `text match`, `tagged` — in Porcelain SF Pro Text 12px, over an Ash tag line. Nothing in the veil is ever shown permanently on the tile.

### Tag Label
**Role:** A tag plus its provenance

**Text-only, no container**, following the reference's treatment of status annotations. SF Pro Text 12px, 16px line-height, -0.264px tracking.

| State | Treatment |
|---|---|
| human | Porcelain `#f5f5f7`, weight 600 |
| machine, accepted | Ash `#86868b`, weight 400 |
| machine, pending | Copper `#b64400`, weight 400, with the confidence following in Optical Yellow — `wireframe 0.62` |
| selected | Electric Link `#2997ff` |
| hover, any | source and confidence in a translucent Steel `#333336` tooltip at a 9999px radius; `×` removes |

Weight and colour carry provenance where border style used to. The Review overlay is the one exception: there a pending tag takes a 1px dashed Copper outline at a 9999px radius, because that surface is about acting on tags and needs the hit target to be visible.

### Relation / Sort Toggle
**Role:** A small set of mutually exclusive modes

A row of transparent 9999px pills, each with a 1px Slate `#6e6e73` outline and an Ash SF Pro Text 12px/500 label. The selected one — and only it — takes a Carbon `#1d1d1f` fill and a Porcelain label. The row stays monochrome; Signal Blue is not spent here, because a sort order is not a primary action.

### Result Strip
**Role:** What the current result set is

Adapted from the reference's announcement strip: Carbon `#1d1d1f`, SF Pro Text 14px/400 in Porcelain with 20px line-height and -0.224px tracking, the count in weight 600. Embedded controls — clear search, change sort — render as Electric Link `#2997ff`. Retriever identifiers never appear; the label is the same human wording the veil uses.

The reference's strip is a full-bleed 72px band. This one is **contained** at a 20px radius and 52px tall, in the slot 12px below the chrome, because the canvas is `fixed inset-0` and nothing is allowed to reserve vertical space above it. A floating 72px block reads as a slab; the band only works when it spans the viewport.

### Related Bar
**Role:** Names the subject of an orbit

A contained Graphite `#0e0e0e` bar at a 20px radius with a 1px `#282828` keyline, in the chrome family and never a second header. A 40px circular thumbnail, the title in SF Pro Display 19px/600 Porcelain, the relation count in Ash, the similar/tag toggle, and a 36px-radius round close. Edge kind and weight are plumbing — they live in `title` attributes, not on the surface.

### Focus Overlay
**Role:** Everything known about one item

A Carbon `#1d1d1f` panel at a 28px radius with **no shadow**, inset 24px from top, right and bottom, `min(680px, 100vw − 48px)` wide. The title is SF Pro Display 28px/600 at 0.196px tracking; support text is Ash. Hero media sits in an Obsidian well at a 28px radius, capped at `52vh` with `object-fit: contain` — a 2:3 photo at full width otherwise pushes title, tags and caption below the fold on open.

It enters **and leaves** on a 240ms 16px slide, with the scrim crossfading over 160ms. Both directions must be driven by Svelte `transition:` directives, not a CSS `@keyframes`: the panel is unmounted the instant focus clears, and CSS cannot animate a node that is already out of the DOM. A transition also holds the unmount until the out-leg finishes, which is the part CSS has no way to do. Tabs — OCR, entities, graph — are outlined pills. Lock `documentElement`, not `body`, or the canvas scrolls behind a stationary panel.

### Command Palette
**Role:** The only navigation

A Carbon `#1d1d1f` panel at a 28px radius, centred at 18vh, `min(680px, 100vw − 48px)` wide, over an Obsidian scrim at 70%. A field in the Search Field recipe sits under a tab row; the active tab is marked by a Porcelain label over a Steel fill, never by a coloured outline — blue is for the action, not for which pane you are looking at. The palette confirms from the keyboard and carries no filled control of its own, which is what keeps the product's single blue pill single. `⌘K` / `Ctrl-K` opens, `Tab` switches search ↔ new note, `⌘↵` saves, `Esc` closes.

The multi-line composer is the one place the two input radii are not interchangeable: a 980px pill cannot hold five rows, so the textarea takes the 28px panel corner instead.

### Graph Well
**Role:** The neighbourhood of one item

An Obsidian `#000000` well at a 28px radius inside the focus panel. Edges are Steel `#333336` 1px; the highlighted edge and a hovered node take Electric Link `#2997ff`. Node labels are SF Pro Text 12px in Ash. Drops to one hop below 430px.

### Processing Tile
**Role:** An item still moving through the pipeline

A Carbon `#1d1d1f` tile at a 28px radius with a 2px indeterminate Signal Blue `#0071e3` bar along its top edge. Fields pop in as SSE stages land. A stage that fails marks its row in the focus panel with Alert `#ff453a` text; a stage that is simply not installed takes Ash, because absence is not failure.

### Orbit Subject Ring
**Role:** Marks the item an orbit grew from

A 1px Slate `#6e6e73` ring — `box-shadow: 0 0 0 1px #6e6e73` used as an outline, not as elevation. No drop shadow and no accent stripe: a blue line across a photograph is the one thing this reference would never do.

### Empty Canvas Stage
**Role:** First run, and an empty search

The hero of the product. A full-bleed uninterrupted Obsidian field with a centred SF Pro Display statement at 80px/600, 84px line-height and -1.2px tracking in Porcelain, one Ash line of SF Pro Text 17px beneath it, and a single Filled Action Pill. This is the only place display-xl appears, and it is what makes an empty library feel like a stage rather than a blank page.

## Do's and Don'ts

### Do
- Use Obsidian `#000000` as the canvas ground and as the fill behind every image tile and media well.
- Set text on dark surfaces in Porcelain `#f5f5f7` and all metadata in Ash `#86868b`.
- Use 28px corners for tiles, the focus panel, the palette and the graph well.
- Keep the floating chrome and the related bar at a 20px radius with a 1px `#282828` keyline.
- Reserve Signal Blue `#0071e3` for the single filled primary action in view.
- Use Electric Link `#2997ff` for interactive text on Obsidian and Carbon, and Cobalt `#0066cc` on White.
- Carry tag provenance with weight and colour — Porcelain 600 human, Ash 400 accepted, Copper pending.
- Use SF Pro Display 80px/600 at -1.2px tracking for the empty-canvas statement, and 32px/600 for a note lede.
- Hold the grid at comfortable density: a 300px column target and a 24px gutter, fluid at every width.
- Mark the subject of an orbit with a 1px Slate ring.
- Hold the focus hero at `max-height: 52vh` with `object-fit: contain`.
- Name a match in human words and put it in the hover veil.

### Don't
- Do not put shadows or translucency on **content** — tiles, the focus panel, the palette and the graph well separate with 1px keylines and the Obsidian-to-Carbon surface shift. The glass chrome is the one exception, and it is the only one.
- Do not put Ash, Slate or Electric Link on the glass: they measure 2.27:1, 3.32:1 and 2.8:1 over a bright photograph. The chrome has its own on-glass pair.
- Do not let a control in the chrome size itself from its content — the chrome owns the box.
- Do not use Signal Blue `#0071e3` as a tile fill, a section background, or an active-sort state.
- Do not put more than one filled blue control on screen at a time.
- Do not round tiles below 28px or replace control pills with rectangular buttons.
- Do not use White `#ffffff` for dark-surface text when Porcelain `#f5f5f7` is available, and never pure `#000` for text.
- Do not use Copper `#b64400` for errors, primary controls or headings — it annotates newness.
- Do not introduce gradients beyond the Carbon-to-Obsidian focus-panel fade.
- Do not reintroduce monospace for labels; it survives only for literal code and layout dumps.
- Do not keep the old compact grid under this geometry — a 28px corner on a 172px tile reads as a lozenge.
- Do not surface retriever identifiers (`text_vec`, `img_vec`, `lexical`) or edge weights (`semantic 0.715`) on a tile.
- Do not key tiles on pixel position, or Svelte destroys and recreates every `<img>` on every wheel event.
- Do not call `setPointerCapture` on the canvas — the tile underneath then never receives its click.

## Elevation

**Content surfaces have none.** Tiles, the focus panel, the palette and the graph well separate through the black-to-charcoal surface shift (Obsidian canvas → Carbon panel), 1px keylines and 28px geometry. The only shadow-like treatment on them is an outline: `0 0 0 1px #6e6e73` on the orbit subject.

**The chrome is the single exception**, and it is deliberate. A liquid-glass material is defined by its light: a specular top edge, a shaded bottom edge, and a drop shadow that puts it above the canvas rather than cut into it. Take those away and translucency alone reads as a washed-out rectangle. So the glass carries `0 10px 34px` plus two inset edges, and nothing else in the product is allowed either.

The rule is therefore not "no shadows" but **"no shadows on content"**. Elevation here distinguishes the layer you operate the product with from the layer the product is about.

## Surfaces

- **Obsidian Canvas** (`#000000`) — `position: fixed; inset: 0`, `cursor: grab`. The infinite field, image tiles, media wells, the graph stage.
- **Liquid Glass** (`rgba(29,29,31,0.78)` / `rgba(255,255,255,0.82)`) — the floating bar, result strip and related bar: translucent, blurred, specular-edged capsules. The only material in the product you can see through, and the only one that casts a shadow. Graphite `#0e0e0e` remains as its no-`backdrop-filter` fallback.
- **Carbon Panel** (`#1d1d1f`) — note tiles, focus overlay, command palette, result strip.
- **White Light Mode** (`#ffffff`) — the light-theme counterpart: White tiles and panels, Carbon `#1d1d1f` text, Cobalt `#0066cc` links, the same 28px geometry and the same shadowless rule.

## Motion

Two budgets, both collapsing to instant under `prefers-reduced-motion: reduce`. The reference is a page that scrolls; this is a canvas that pans, so motion is the one system ZetaDB brings of its own.

**Canvas motion** is a target/actual split: `tx/ty` is where the canvas is heading, `ox/oy` is what is drawn, and a rAF loop closes the gap with `k = 1 − exp(−dt/tau)` — framerate-independent, identical at 60 and 144 Hz. `TAU_FOLLOW = 95` for the wheel, `TAU_GLIDE = 230` for released momentum. A drag moves target and actual together so the canvas stays welded to the cursor, then hands its velocity over.

**The orbit** grows rather than replaces: the clicked tile stays put and its neighbours scale up around it in rings, staggered by rank. Ring radius is derived from the actual tiles — each is placed at an angle proportional to its own footprint, and a ring clears the one inside it. Sizing by width alone collides, because a portrait tile is up to 1.55× taller than wide. Ring tiles start at 300px and step down by 26px per ring to a **floor of 190px**: every tile carries a 28px corner, and below about 190px that corner stops reading as a frame and starts reading as a lozenge. The subject itself is 360px.

Everything else is near-instant: 140ms on the hover veil, 240ms on the focus panel in both directions, 160ms on its scrim. CSS transitions take the `--ease` token directly; Svelte's JS transitions take `cubicOut`, which is the same curve for practical purposes — a transition needs an easing *function*, not a CSS bezier. No hover fades on tiles, no skeleton shimmer, no page transitions.

**An animation that only plays one way is a bug, not a style.** This system got it wrong twice, in two different ways, and both measured as a one-frame cut:

- The focus panel had a CSS enter keyframe and no exit. CSS cannot animate a node that is already out of the DOM, and the panel is unmounted the moment focus clears.
- The orbit's out-transition was gated on `orbit` being true — but leaving the orbit is precisely what removes those tiles, so by the time the transition was read the flag was already false and the duration collapsed to zero. **An out-transition must never depend on the state whose removal triggers it.** Ask the leaving element what it is; do not ask the component what mode it is now in.

The second one is the subtler trap, and it is invisible in code review: the expression looks symmetrical with the enter case and reads as correct. Transitions must never apply to library tiles — the keyed list churns constantly as tiles cross the cull window, and animating each one leaves a permanent backlog of entering and dying nodes.

## Imagery

Imagery is whatever the user dumped in, which means it cannot be art-directed — screenshots, phone photos, diagrams, scans and UI crops all land in the same grid. The Obsidian ground is the strongest argument for this system: black is the one ground that casts nothing back onto arbitrary photography, where the previous warm-white paper tinted every image placed on it. Tiles crop to `cover` inside 28px frames so the grid rhythm holds regardless of source aspect ratio; the focus hero switches to `contain` in an Obsidian well, because that is the one place the whole image matters. Icons are thin monochrome utility glyphs in Ash, resolving to Porcelain on hover. Near-duplicate clusters collapse to one tile with a Steel `×3` marker at a 9999px radius. Optical Yellow and Verified green operate as technical cues on confidence and stage state — not as a broad illustration palette.

## Layout

The canvas *is* the page: `fixed inset-0`, panning forever in both directions over a finite library, fluid at every width with no max and no horizontal page scroll. Columns are indexed by any integer, so there is no left or right edge; column `c` draws from the item list at a different rotation, so panning sideways keeps turning up new things rather than repeating a block. Each column wraps modulo its own stack height, so there is no global tile boundary and therefore no seam. Cull to the viewport, then cap on a tile budget — 140 in the shipped canvas — because 24 columns × every repetition intersecting a tall viewport generates around a thousand absolutely-positioned nodes otherwise.

The floating chrome sits 24px from the top and both sides, with one strip 24px beneath it — result strip or related bar, never both. Clicking a tile grows an orbit of its neighbours in place; the info marker opens the focus panel down the right edge. An empty canvas becomes the display-xl hero stage. Below 660px the focus panel spans both edges; at 430px the grid is one column and the graph drops to one hop. Minimum 24px side gutter at every width, set once on the app wrapper.

## The port

This system is implemented, not proposed. What changed:

| | before | now |
|---|---|---|
| ground | `#FFFFFF` light-first | Obsidian `#000000`, dark-first |
| accent | Ember `#FF5A1F`, one token | Signal Blue `#0071e3` fills + Electric Link `#2997ff` text |
| chrome | `--chrome-shadow` capsule recipe | Graphite fill + 1px `#282828` keyline, no shadow |
| tile radius | 3px | 28px |
| column target / gutter | 172px / 12px | 300px / 24px |
| orbit tile floor | 120px | 190px |
| metadata face | JetBrains Mono | SF Pro Text 12px in Ash |
| tag provenance | border style (solid/dotted/dashed) | weight and colour, no container |
| empty state | a muted paragraph | the 80px display hero |
| status colours | one value each | a light and a dark value each, measured |

`--chrome-shadow` is gone and took the visual logic of six controls with it — the single largest edit. The `.mono` utility class was **renamed** `.meta` throughout rather than quietly redefined, because a class called `mono` that is not monospace is a lie that costs the next reader an hour; `.code` carries the monospace face for the layout dump.

Two second sources of truth were removed on the way. `tokens.css` had authored the off-theme palette twice in full and the two copies had drifted on six values, so system-dark and toggled-dark were quietly different themes; the light values now live once in a `--light-*` block and are only *assigned* twice. And `--tile-gap` / `--col` were deleted rather than carried over — the grid is laid out in JS, and a copy of its numbers in CSS is free to drift from the ones the layout actually uses. They live in `Canvas.svelte` as `GAP` and `TARGET`.

Verified: `vite build` clean, `svelte-check` 115 files / 0 errors / 0 warnings, 45 Python tests passing, every colour pair in this document measured against WCAG, and the 25-check API smoke test green against the real library.

Then driven in headless Chrome over CDP, in both themes, which is where the close-transition bug surfaced — exactly as §17.5 predicts. Confirmed on the live page: body `#000000`, tile radius 28px, chrome `#0e0e0e` at a 20px radius with `box-shadow: none`, panel `#1d1d1f` with `box-shadow: none`, the add control at `#0071e3`, four columns at 1400px, and no horizontal scroll. Still unverified: narrow widths below 660px, the orbit, and search.

## Similar Brands

- **Apple** — the source of this system: black product stages, oversized SF Pro Display, 28px media geometry, shadowless elevation, and blue confined to the action path.
- **Cosmos** — the structural reference for the canvas itself: uniform `vw / n` pitch, edge-to-edge tiles, zero card chrome, controls floating on the field rather than docked above it.
- **Are.na** — shares the premise that a saved image and a typed note are the same kind of object, and that provenance matters more than decoration.
- **Linear** — a dark monochrome product surface with one rationed accent, dense quiet metadata, and a keyboard palette standing in for navigation.
