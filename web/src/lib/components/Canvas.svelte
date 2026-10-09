<script lang="ts">
  import { scale } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import type { Item } from "../api";

  let { items, onopen, oninfo, focusItem = null, relKey = "" }:
    { items: Item[]; onopen: (id: number) => void; oninfo: (id: number) => void;
      focusItem?: Item | null; relKey?: string } = $props();

  /*
    One canvas, two layouts.

    LIBRARY pans forever in both directions over a finite set. Columns are
    indexed by any integer so there is no left or right edge; each draws
    from the list at a different rotation, so panning sideways keeps
    turning up different things. Vertically each column wraps modulo its
    own stack height, so there is no global tile boundary and no seam.

    ORBIT holds one item still and grows its neighbours around it in rings,
    nearest first. Entering it does not replace the view — the focus stays
    put and the relatives arrive around it — and swapping the relation
    (similar <-> same tag) lets the old ring leave and the new one arrive,
    each tile scaling up from nothing.
  */
  /* Columns are UNIFORM and the pitch divides the viewport exactly, so the
     grid lines up edge to edge however wide the window is — the reference
     canvas measures a constant 12px gutter and a column width that is
     simply (viewport / column count) minus that gutter. Varying the widths
     per column looked livelier but broke the vertical rhythm the moment
     you panned sideways. */
  /* Comfortable density is load-bearing under this system, not a garnish.
     The grid used to run a 172px target at a 12px gutter, which is what a
     3px tile radius wants; a 28px corner on a 172px tile eats a third of
     its width and reads as a lozenge, and 16px of internal padding leaves
     a note nowhere to breathe. At 1440px this is 4 columns of 336. */
  const GAP = 24;
  const TARGET = 300;                       // the width a column wants to be
  const mod = (n: number, m: number) => ((n % m) + m) % m;

  /* A note's first line is set as a lede so a typed thought carries the
     same weight as a photograph in the next column — the display face has
     no other home on the canvas. Split at the first line break, else the
     first sentence, else a word boundary. */
  const LEDE_MAX = 64;
  function splitNote(it: Item): { lede: string; rest: string } {
    const text = (it.preview || it.title || "…").trim();
    let cut = text.indexOf("\n");
    if (cut < 0 || cut > LEDE_MAX) {
      const dot = text.search(/[.!?](\s|$)/);
      cut = dot >= 0 && dot <= LEDE_MAX ? dot + 1 : -1;
    }
    if (cut < 0) {
      if (text.length <= LEDE_MAX) cut = text.length;
      else {
        const sp = text.lastIndexOf(" ", LEDE_MAX);
        cut = sp > 16 ? sp : LEDE_MAX;
      }
    }
    return { lede: text.slice(0, cut).trim(), rest: text.slice(cut).trim() };
  }
  /* Scaled to the tile: 32px inside a 190px orbit tile is a headline in a
     stamp, so the lede tracks the column width and bottoms out at 15px. */
  const ledeSize = (w: number) => Math.max(15, Math.min(32, Math.round(w * 0.105)));

  const orbit = $derived(!!focusItem);

  let vw = $state(1200);
  let vh = $state(800);
  const columns = $derived(Math.max(2, Math.round(vw / (TARGET + GAP))));
  const pitch = $derived(vw / columns);
  const colW = $derived(pitch - GAP);
  const colX = (c: number) => c * pitch + GAP / 2;
  let el = $state<HTMLDivElement | null>(null);
  let dragging = $state(false);
  let moved = $state(false);

  // `ox/oy` is drawn; `tx/ty` is where it is heading. A wheel tick moves
  // the target and the frame loop chases it, turning a mouse wheel's ~100px
  // steps into a glide. A drag moves both so the canvas stays welded to the
  // cursor, then hands its velocity over as momentum.
  let ox = $state(0);
  let oy = $state(0);
  let tx = 0, ty = 0, vx = 0, vy = 0, raf = 0, last = 0;

  const reduced = typeof matchMedia === "function"
    && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TAU_FOLLOW = 95;
  const TAU_GLIDE = 230;
  const EPS = 0.06;

  function frame(now: number) {
    raf = 0;
    const dt = Math.min(48, Math.max(1, now - last));
    last = now;
    if (!dragging && (Math.abs(vx) > EPS || Math.abs(vy) > EPS)) {
      tx += vx * dt; ty += vy * dt; clampTargets();
      const decay = Math.exp(-dt / TAU_GLIDE);
      vx *= decay; vy *= decay;
    }
    const k = 1 - Math.exp(-dt / TAU_FOLLOW);       // framerate-independent
    const nx = ox + (tx - ox) * k;
    const ny = oy + (ty - oy) * k;
    const done = Math.abs(tx - nx) < EPS && Math.abs(ty - ny) < EPS
      && Math.abs(vx) < EPS && Math.abs(vy) < EPS;
    ox = done ? tx : nx;
    oy = done ? ty : ny;
    if (!done) raf = requestAnimationFrame(frame);
  }
  function run() {
    if (reduced) { ox = tx; oy = ty; vx = vy = 0; return; }
    if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
  }

  type Tile = {
    key: string; item: Item; x: number; y: number;
    w: number; h: number; rank: number; isFocus: boolean;
  };

  /* How far the orbit reaches from its centre — used to bound panning so
     the subject stays reachable. It travels OUT of the layout as part of
     the result, never as a state write: assigning to $state from inside a
     $derived is an unsafe mutation, and Svelte throwing there took the
     whole layout (and with it every tile click) down. The library is
     endless, so its extent is 0 and clamping is skipped. */
  type Layout = { tiles: Tile[]; extent: number };

  const ratioOf = (it: Item) =>
    it.width && it.height ? Math.min(Math.max(it.height / it.width, 0.62), 1.55) : 0.72;

  // ---------------------------------------------------------------- library
  function libraryTiles(): Layout {
    const out: Tile[] = [];
    if (!items.length || vw <= 0 || vh <= 0) return { tiles: out, extent: 0 };
    let budget = 140;                 // cull to the viewport, then cap
    for (let c = Math.floor(ox / pitch) - 2; c <= Math.ceil((ox + vw) / pitch) + 2; c++) {
      if (budget <= 0) break;
      const x = colX(c) - ox;
      const w = colW;
      if (x > vw + 120 || x + w < -120) continue;

      const cells: { item: Item; y: number; h: number }[] = [];
      let cy = 0;
      const start = mod(c * 7, items.length);
      for (let i = 0; i < Math.min(items.length, 10); i++) {
        const item = items[mod(start + i * 3, items.length)];
        const h = Math.round(item.kind === "image"
          ? w * ratioOf(item) : Math.max(150, w * 0.72));
        cells.push({ item, y: cy, h });
        cy += h + GAP;
      }
      if (!cy) continue;

      for (let k = Math.floor((oy - cy) / cy); k <= Math.ceil((oy + vh) / cy); k++) {
        if (budget <= 0) break;
        for (let i = 0; i < cells.length; i++) {
          const top = k * cy + cells[i].y - oy;
          if (top > vh + 80 || top + cells[i].h < -80) continue;
          out.push({ key: `L${c}:${k}:${i}`, item: cells[i].item, x, y: top,
                     w, h: cells[i].h, rank: 0, isFocus: false });
          budget--;
        }
      }
    }
    return { tiles: out, extent: 0 };
  }

  // ------------------------------------------------------------------ orbit
  function orbitTiles(): Layout {
    const out: Tile[] = [];
    if (!focusItem || vw <= 0) return { tiles: out, extent: 0 };
    const cx = vw / 2 - ox;
    const cy = vh / 2 - oy;

    const FW = 360;
    const fh = Math.round(FW * ratioOf(focusItem));
    out.push({ key: `F:${focusItem.id}`, item: focusItem,
               x: cx - FW / 2, y: cy - fh / 2, w: FW, h: fh,
               rank: 0, isFocus: true });

    /* Ring geometry is derived from the ACTUAL tiles, not guessed.
       Each tile is placed at an angle proportional to its own footprint,
       and the ring's radius is the larger of (total footprint / 2*pi) and
       whatever clears the ring inside it. Sizing the gap by width alone
       was not enough: a portrait tile is up to 1.55x taller than it is
       wide, and those collided with their neighbours around the circle. */
    const others = items.filter((i) => i.id !== focusItem.id);
    const GAPR = 36;
    let rank = 0, ring = 1, inner = Math.max(FW, fh) / 2;

    while (rank < others.length && ring < 9) {
      /* The floor is 190px, not 120px: every tile carries a 28px corner,
         and below about 190 that corner stops reading as a frame and
         starts reading as a lozenge. */
      const size = Math.max(190, 300 - (ring - 1) * 26);
      const count = Math.min(others.length - rank, 7 + (ring - 1) * 4);
      const slice = others.slice(rank, rank + count).map((it) => {
        const h = Math.round(size * ratioOf(it));
        return { it, w: size, h, foot: Math.max(size, h) + GAPR };
      });
      const total = slice.reduce((a, c) => a + c.foot, 0);
      const radius = Math.max(total / (2 * Math.PI),
                              inner + size / 2 + GAPR);
      let cursor = 0;
      let outer = 0;
      for (const c of slice) {
        // Angle proportional to footprint, so wide and tall tiles each get
        // the room they actually need.
        const a = (2 * Math.PI * (cursor + c.foot / 2)) / total + ring * 0.9;
        cursor += c.foot;
        out.push({
          key: `O${relKey}:${c.it.id}`, item: c.it,
          x: cx + Math.cos(a) * radius - c.w / 2,
          y: cy + Math.sin(a) * radius - c.h / 2,
          w: c.w, h: c.h, rank: rank + 1, isFocus: false,
        });
        rank++;
        outer = Math.max(outer, Math.max(c.w, c.h) / 2);
      }
      inner = radius + outer;
      ring++;
    }
    return { tiles: out, extent: inner };
  }

  // The library is endless, so it has no extent to clamp against.
  const layout = $derived.by(() => (orbit ? orbitTiles() : libraryTiles()));
  const tiles = $derived(layout.tiles);

  $effect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => { vw = e.contentRect.width; vh = e.contentRect.height; });
    ro.observe(el);
    return () => ro.disconnect();
  });
  $effect(() => () => { if (raf) cancelAnimationFrame(raf); });

  // Recentre when the library set changes, but NOT when entering orbit —
  // the whole point is that the focus does not jump.
  $effect(() => {
    items;
    if (!focusItem) { ox = oy = tx = ty = 0; vx = vy = 0; }
  });

  /** In orbit, keep the subject reachable: never pan past the outermost
      ring plus a screen's worth of slack. The library is unbounded. */
  function clampTargets() {
    if (!orbit) return;
    const lx = Math.max(200, layout.extent - vw / 2 + 140);
    const ly = Math.max(200, layout.extent - vh / 2 + 140);
    tx = Math.max(-lx, Math.min(lx, tx));
    ty = Math.max(-ly, Math.min(ly, ty));
  }

  /** Recentre — `h` for home. In orbit that puts the subject back in the
      middle; in the library it returns to the origin. */
  export function recentre() { tx = ty = 0; vx = vy = 0; run(); }

  function wheel(e: WheelEvent) {
    e.preventDefault();
    tx += e.shiftKey ? e.deltaY : e.deltaX;
    ty += e.shiftKey ? 0 : e.deltaY;
    vx = vy = 0;
    clampTargets();
    run();
  }

  // Drag vs click: measure from where the pointer went DOWN (summing
  // increments makes jitter read as a drag), and never setPointerCapture on
  // the canvas — capture retargets events and the tile never gets its click.
  const SLOP = 6;
  let px = 0, py = 0, sx = 0, sy = 0, pt = 0, hadDown = false;

  /* Whether a click counts is decided from the CLICK event's own
     coordinates against where the pointer went down — not from a flag
     accumulated during the drag. A single stray move between down and up
     (which synthetic clicks emit) was flipping that flag and swallowing
     every click. If no pointerdown was seen at all, trust the click. */
  function tileClick(e: MouseEvent, id: number) {
    const wasDrag = hadDown && Math.hypot(e.clientX - sx, e.clientY - sy) > SLOP;
    hadDown = false;
    if (!wasDrag) onopen(id);
  }

  function down(e: PointerEvent) {
    if (e.button !== 0) return;
    dragging = true; moved = false; hadDown = true;
    px = sx = e.clientX; py = sy = e.clientY; pt = e.timeStamp;
    vx = vy = 0; tx = ox; ty = oy;
  }
  function move(e: PointerEvent) {
    if (!dragging) return;
    const dx = e.clientX - px, dy = e.clientY - py;
    const dt = Math.max(1, e.timeStamp - pt);
    vx = vx * 0.7 + (-dx / dt) * 0.3;
    vy = vy * 0.7 + (-dy / dt) * 0.3;
    tx -= dx; ty -= dy; clampTargets(); ox = tx; oy = ty;   // 1:1 while held
    px = e.clientX; py = e.clientY; pt = e.timeStamp;
    if (Math.hypot(e.clientX - sx, e.clientY - sy) > SLOP) moved = true;
  }
  function up(e?: PointerEvent) {
    if (!dragging) return;
    dragging = false;
    if (e && e.timeStamp - pt > 90) { vx = vy = 0; }   // a pause is not a flick
    const cap = 4;
    vx = Math.max(-cap, Math.min(cap, vx));
    vy = Math.max(-cap, Math.min(cap, vy));
    run();
  }
  function key(e: KeyboardEvent) {
    // The canvas listens on the window, so it must stay out of the way of
    // anything being typed — otherwise "h" in the search box recentres the
    // view and arrow keys pan it instead of moving the caret.
    const el2 = e.target as HTMLElement | null;
    if (el2 && (/^(INPUT|TEXTAREA|SELECT)$/.test(el2.tagName) || el2.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const step = e.shiftKey ? 600 : 220;
    const map: Record<string, [number, number]> = {
      ArrowRight: [step, 0], ArrowLeft: [-step, 0],
      ArrowDown: [0, step], ArrowUp: [0, -step],
    };
    if (e.key === "h" || e.key === "H") { recentre(); e.preventDefault(); return; }
    const d = map[e.key];
    if (!d) return;
    tx += d[0]; ty += d[1]; clampTargets(); run(); e.preventDefault();
  }

  /* Transitions belong to the ORBIT only.
     In the library the keyed list churns constantly as tiles scroll in and
     out of the cull window; animating every one of those meant a permanent
     backlog of entering and (via `out:`) still-dying nodes, which pinned
     the main thread. Library tiles appear and disappear instantly — which
     is also what you want, since a tile "arriving" while you pan is noise.
     Staggered by rank so the ring blooms outward instead of popping. */
  const enter = (rank: number) =>
    !orbit || reduced
      ? { duration: 0 }
      : { duration: 420, delay: Math.min(rank * 26, 520),
          start: 0.3, opacity: 0, easing: cubicOut };
  const leave = () =>
    !orbit || reduced
      ? { duration: 0 }
      : { duration: 220, start: 0.4, opacity: 0, easing: cubicOut };
</script>

<svelte:window onkeydown={key} onpointermove={move} onpointerup={up} onpointercancel={up} />

<div class="canvas" bind:this={el} class:dragging class:orbit
     onwheel={wheel} onpointerdown={down}
     role="application" aria-label="library canvas" tabindex="-1">
  {#each tiles as t (t.key)}
    <button class="tile" class:isfocus={t.isFocus}
            style:left="{t.x}px" style:top="{t.y}px"
            style:width="{t.w}px" style:height="{t.h}px"
            style:z-index={t.isFocus ? 2 : 1}
            in:scale={enter(t.rank)}
            out:scale={leave()}
            onclick={(e) => tileClick(e, t.item.id)}
            aria-label={t.item.title ?? `item ${t.item.id}`}>
      {#if t.item.kind === "image" && t.item.thumb_url}
        <img src={t.item.thumb_url} alt={t.item.title ?? ""}
             loading="eager" decoding="async" draggable="false" />
      {:else}
        {@const n = splitNote(t.item)}
        <span class="note" style:--lede="{ledeSize(t.w)}px">
          <span class="lede">{n.lede}</span>
          {#if n.rest}<span class="rest">{n.rest}</span>{/if}
        </span>
      {/if}
      <!-- The reference lightens the tile and states the one thing a plain
           click does. Tags ride underneath it, small, because they are the
           reason two things end up next to each other. -->
      <span class="veil">
        <span class="hint meta">
          {orbit ? "click to re-centre" : "click to view similar"}
        </span>
        {#if t.item.tags.length}
          <span class="tags meta">{t.item.tags.slice(0, 3).map((x) => x.term).join(" · ")}</span>
        {/if}
      </span>
      <!-- Details are a deliberate second action: a plain click explores
           outward, the "i" opens the record. -->
      <span class="info" role="button" tabindex="-1" title="details"
            onclick={(e) => { e.stopPropagation(); oninfo(t.item.id); }}
            onkeydown={(e) => { if (e.key === "Enter") { e.stopPropagation(); oninfo(t.item.id); } }}
            aria-label="details">i</span>
    </button>
  {/each}
</div>

<style>
  /* The canvas IS the page: it fills the viewport and the chrome floats on
     top of it, the way the reference does. */
  .canvas {
    position: fixed; inset: 0;
    overflow: hidden; background: var(--bg);
    cursor: grab; touch-action: none; overscroll-behavior: none;
  }
  .canvas.dragging { cursor: grabbing; }

  /* 28px corners, no border, no shadow, no hover lift. The ground behind
     an image is Obsidian, which is the one ground that casts nothing back
     onto an arbitrary photograph. */
  .tile {
    position: absolute; padding: 0; border: 0; background: var(--well);
    overflow: hidden; cursor: pointer; border-radius: var(--r-tile);
    transition: opacity 160ms var(--ease);
  }
  .tile img {
    width: 100%; height: 100%; object-fit: cover; display: block;
    user-select: none; -webkit-user-drag: none;
  }
  /* A note has no image to carry it, so it takes the one filled sheet on
     the canvas and the one piece of display type. */
  .tile .note {
    display: flex; flex-direction: column; gap: 10px;
    padding: 16px; height: 100%; overflow: hidden;
    text-align: left; color: var(--text); background: var(--raised);
  }
  .note .lede {
    font-family: var(--font-display); font-weight: 600;
    font-size: var(--lede, var(--fs-2xl)); line-height: 1.13;
    letter-spacing: var(--ls-32);
  }
  .note .rest {
    font-size: var(--fs-sm); line-height: 1.43; letter-spacing: var(--ls-14);
    color: var(--muted); overflow: hidden;
  }
  /* The subject of an orbit is marked by a 1px ring — used as an outline,
     not as elevation. Nothing on this canvas casts a shadow, and a blue
     stripe across a photograph is the one thing the reference would never
     do. */
  .tile.isfocus { box-shadow: 0 0 0 1px var(--outline); }

  .info {
    position: absolute; top: 10px; right: 10px; z-index: 3;
    width: 24px; height: 24px; border-radius: var(--r-compact);
    display: grid; place-items: center;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-style: italic;
    background: color-mix(in srgb, var(--steel) 72%, transparent);
    color: var(--text); border: 1px solid var(--outline);
    cursor: pointer;
    opacity: 0; transition: opacity 140ms var(--ease), background 140ms var(--ease);
  }
  .tile:hover .info, .info:focus-visible { opacity: 1; }
  .info:hover { background: var(--blue); border-color: var(--blue); color: #fff; }

  .veil {
    position: absolute; inset: 0; padding: 16px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 8px; text-align: center;
    background: color-mix(in srgb, var(--bg) 72%, transparent);
    opacity: 0; transition: opacity 140ms var(--ease);
  }
  .tile:hover .veil { opacity: 1; }
  .hint {
    color: var(--text); font-size: var(--fs-xs); letter-spacing: var(--ls-12);
  }
  .tags {
    color: var(--muted); font-size: var(--fs-xs); letter-spacing: var(--ls-12);
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }

  @media (prefers-reduced-motion: reduce) {
    .tile, .veil { transition: none; }
  }
</style>
