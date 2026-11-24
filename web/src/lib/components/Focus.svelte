<script lang="ts">
  import { fade, fly } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import { api, clock, isTimed, type ItemDetail, type Item, type Moment,
           type Segment } from "../api";
  import TagPill from "./TagPill.svelte";
  import Graph from "./Graph.svelte";

  /* The panel used to enter on a CSS @keyframes and leave on nothing at
     all: App unmounts it the instant focusId goes null, and a CSS
     animation cannot play on a node that is already out of the DOM. So
     opening glided and closing was a hard cut — measured at one frame,
     full opacity to absent, with the scrim going with it.

     Svelte transitions run in both directions and, crucially, hold the
     unmount until the out-transition finishes, which is the part CSS
     could not do. cubicOut stands in for the --ease token because a
     transition needs an easing FUNCTION, not a CSS curve; the two are
     indistinguishable over 240ms. */
  const reduced = typeof matchMedia === "function"
    && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PANEL_MS = reduced ? 0 : 240;
  const SCRIM_MS = reduced ? 0 : 160;

  /* `moment` is where a search matched inside this item, when the panel was
     opened from that search result; App clears it on any other way in. */
  let { id, moment = null, onclose, onopen }:
    { id: number; moment?: Moment | null; onclose: () => void;
      onopen: (id: number) => void } = $props();

  let item = $state<ItemDetail | null>(null);
  let related = $state<Item[]>([]);
  let tab = $state<"content" | "ocr" | "graph">("content");

  // ---- audio / video ------------------------------------------------
  let player = $state<HTMLMediaElement | null>(null);
  let now = $state(0);                       // playhead, for the timeline
  let seeded = false;

  const timed = $derived(!!item && isTimed(item.kind));
  const frames = $derived((item?.segments ?? []).filter((x) => x.kind === "frames"));
  const sounds = $derived((item?.segments ?? []).filter((x) => x.kind === "sound"));
  const matchId = $derived(moment?.segment_id ?? null);
  /* Eye scores images and video keyframes; its licence asks for a visible
     credit wherever its tags are shown. */
  const eyeTagged = $derived(!!item && (item.kind === "image" || item.kind === "video")
    && item.tags.some((t) => t.source === "machine"));

  /* Sound windows overlap (30s every 25s), so "the one playing" is the
     latest that has started and not yet ended. */
  function under(list: Segment[], t: number): number | null {
    let cur: number | null = null;
    for (const x of list) if (x.start <= t + 0.05 && t < x.end) cur = x.id;
    return cur;
  }
  const nowFrame = $derived(under(frames, now));
  const nowSound = $derived(under(sounds, now));

  /* Land on the matched moment once the player knows its length — setting
     currentTime before metadata is a no-op in some engines. It stays
     paused: the person chose to look, not to listen. */
  function ready() {
    if (seeded || !player || !moment) return;
    seeded = true;
    player.currentTime = moment.start;
    now = moment.start;
  }
  function seek(x: Segment) {
    if (!player) return;
    player.currentTime = x.start;
    now = x.start;
    player.play().catch(() => { /* autoplay refused: the seek still landed */ });
  }
  /* Centre the matched window in its strip without scrolling the panel —
     scrollIntoView would also move the panel itself. Instant, so there is
     nothing for prefers-reduced-motion to collapse. */
  function centre(node: HTMLElement, on: boolean) {
    const go = (yes: boolean) => {
      const p = node.parentElement;
      if (yes && p) p.scrollLeft = node.offsetLeft - (p.clientWidth - node.offsetWidth) / 2;
    };
    go(on);
    return { update: go };
  }

  $effect(() => {
    let alive = true;
    item = null; related = []; tab = "content"; now = 0; seeded = false;
    api.item(id).then((d) => alive && (item = d)).catch(() => {});
    api.related(id).then((r) => alive && (related = r.related)).catch(() => {});
    return () => { alive = false; };
  });

  // Without this the wheel scrolls the canvas behind the panel, which
  // looks broken: the overlay sits still while the page moves under it.
  $effect(() => {
    // The scroller is the document element, not body — locking body alone
    // left the canvas scrolling underneath a stationary panel.
    const html = document.documentElement;
    const prevHtml = html.style.overflow;
    const prevBody = document.body.style.overflow;
    html.style.overflow = "hidden";
    document.body.style.overflow = "hidden";
    return () => { html.style.overflow = prevHtml; document.body.style.overflow = prevBody; };
  });
</script>

<div class="scrim" onclick={onclose} role="presentation"
     transition:fade={{ duration: SCRIM_MS }}></div>
<div class="focus" role="dialog" aria-modal="true" aria-label="item detail"
     transition:fly={{ x: 16, opacity: 0, duration: PANEL_MS, easing: cubicOut }}>
  {#if !item}
    <p class="meta muted pad">loading…</p>
  {:else}
    <header>
      <span class="meta muted">#{item.id} · {item.kind}{#if timed && item.duration} · {clock(item.duration)}{/if} · {item.status}</span>
      <div class="spacer"></div>
      <button class="close" onclick={onclose} aria-label="close">×</button>
    </header>

    {#if item.kind === "image" && item.file_url}
      <img class="hero" src={item.file_url} alt={item.title ?? ""} />
    {:else if item.kind === "video" && item.file_url}
      <!-- svelte-ignore a11y_media_has_caption -->
      <!-- No caption track exists: nothing in the pipeline transcribes. -->
      <video class="hero" bind:this={player} src={item.file_url} poster={item.thumb_url}
             controls preload="metadata" playsinline
             onloadedmetadata={ready} ontimeupdate={() => (now = player?.currentTime ?? 0)}
             onseeked={() => (now = player?.currentTime ?? 0)}></video>
    {:else if item.kind === "audio" && item.file_url}
      <div class="well">
        {#if item.thumb_url}<img class="wave" src={item.thumb_url} alt="" />{/if}
        <!-- svelte-ignore a11y_media_has_caption -->
        <audio bind:this={player} src={item.file_url} controls preload="metadata"
               onloadedmetadata={ready} ontimeupdate={() => (now = player?.currentTime ?? 0)}
               onseeked={() => (now = player?.currentTime ?? 0)}></audio>
      </div>
    {/if}

    <div class="body">
      <h2>{item.title ?? "untitled"}</h2>

      {#if item.tags.length}
        <div class="tags">
          {#each item.tags as t (t.term)}<TagPill tag={t} />{/each}
        </div>
        {#if eyeTagged}
          <!-- Required attribution for the image tagger, kept as quiet
               metadata beside the tags it produced. -->
          <p class="credit meta">image tags ·
            <a href="https://desertant.com" target="_blank" rel="noopener noreferrer">Powered by Desert Ant Labs</a></p>
        {/if}
      {/if}

      {#if timed && (frames.length || sounds.length)}
        <!-- Where the file was cut up to be searched. A click jumps there;
             the matched window carries a ring and the word "match". -->
        {#if frames.length}
          <h3 class="meta">scenes</h3>
          <div class="strip">
            {#each frames as x (x.id)}
              <button class="frame" class:match={x.id === matchId} class:now={x.id === nowFrame}
                      use:centre={x.id === matchId} onclick={() => seek(x)}
                      aria-label="play from {clock(x.start)}{x.id === matchId ? ', search match' : ''}">
                {#if x.frame_url}
                  <img src={x.frame_url} alt="" loading="lazy" decoding="async" />
                {:else}
                  <span class="noframe"></span>
                {/if}
                <span class="t meta">{clock(x.start)}{#if x.id === matchId} · match{/if}</span>
              </button>
            {/each}
          </div>
        {/if}
        {#if sounds.length}
          <h3 class="meta">sound</h3>
          <div class="strip chips">
            {#each sounds as x (x.id)}
              <button class="chip" class:match={x.id === matchId} class:on={x.id === nowSound}
                      use:centre={x.id === matchId} onclick={() => seek(x)}
                      aria-label="play from {clock(x.start)}{x.id === matchId ? ', search match' : ''}"
                      >{clock(x.start)}{#if x.id === matchId} · match{/if}</button>
            {/each}
          </div>
        {/if}
      {/if}

      <nav class="tabs meta">
        <button class:on={tab === "content"} onclick={() => (tab = "content")}>content</button>
        {#if item.layout_text && !timed}
          <button class:on={tab === "ocr"} onclick={() => (tab = "ocr")}>ocr</button>
        {/if}
        <button class:on={tab === "graph"} onclick={() => (tab = "graph")}>graph</button>
      </nav>

      {#if tab === "content"}
        {#if item.body}<p class="prose">{item.body}</p>{/if}
        {#if item.caption}
          <p class="prose caption">{item.caption}</p>
          <p class="meta muted">caption · {item.caption_model ?? "?"}</p>
        {/if}
        {#if !item.body && !item.caption}
          <p class="meta muted">no description yet</p>
        {/if}
        {#if item.entities.length}
          <h3 class="meta">entities</h3>
          <ul class="ents meta">
            {#each item.entities as e}
              <li><span class="k">{e.kind}</span> {e.value}</li>
            {/each}
          </ul>
        {/if}
        {#if item.todos.length}
          <h3 class="meta">todos</h3>
          <ul class="ents meta">
            {#each item.todos as t}<li>{t.text}{#if t.due_date} · {t.due_date}{/if}</li>{/each}
          </ul>
        {/if}
        {#if item.errors.length}
          <h3 class="meta err">stage errors</h3>
          <ul class="ents meta err">
            {#each item.errors as e}<li>{e.stage}: {e.error}</li>{/each}
          </ul>
        {/if}
      {:else if tab === "ocr"}
        <!-- layout_text: grid-projected, columns preserved (spec 6.2) -->
        <pre class="layout">{item.layout_text}</pre>
      {:else}
        <Graph {id} {onopen} />
      {/if}

      {#if related.length}
        <h3 class="meta">related</h3>
        <div class="related">
          {#each related as r (r.id)}
            <!-- The edge kind and weight are plumbing; a row of thumbnails
                 each captioned "semantic 0.71" reads as debug output. -->
            <button class="rel" onclick={() => onopen(r.id)}
                    title="{r.title ?? ''}{r.weight ? ` · ${r.link} ${r.weight}` : ''}">
              {#if r.thumb_url}
                <img src={r.thumb_url} alt={r.title ?? ""} />
              {:else}
                <span class="reltext">{r.preview.slice(0, 70) || r.kind}</span>
              {/if}
            </button>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .scrim { position: fixed; inset: 0; background: color-mix(in srgb, var(--bg) 78%, transparent); z-index: 40; }
  /* Details are an OVERLAY, never a page: a floating card in the same
     family as the rest of the chrome — inset from every edge, fully
     rounded, hairline and soft shadow — so the canvas stays visible
     behind it and nothing about the view is lost. */
  .focus {
    position: fixed;
    top: var(--inset); right: var(--inset); bottom: var(--inset);
    width: min(680px, calc(100vw - var(--inset) * 2));
    background: var(--surface);
    border: 1px solid var(--keyline);
    border-radius: var(--r-panel); z-index: 41; overflow-y: auto;
  }
  /* Enter and exit both live on the `transition:` directives in the
     markup — a CSS animation here would double up with them on open and
     still do nothing on close. */
  @media (max-width: 660px) {
    .focus { left: var(--inset); width: auto; }
  }

  header {
    display: flex; align-items: center; gap: 10px; padding: 16px 20px;
    border-bottom: 1px solid var(--border); position: sticky; top: 0;
    background: var(--surface); z-index: 1;
  }
  .spacer { flex: 1; }
  /* Outlined explore pills, the same recipe the sort and relation toggles
     use — one family of controls, no shadow on any of them. */
  button {
    background: none; color: var(--muted);
    border: 1px solid var(--outline); border-radius: var(--r-pill);
    padding: 5px 12px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; letter-spacing: var(--ls-ctl);
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  button:hover { color: var(--text); background: var(--raised); }
  /* The glyph carries the meaning, so it needs a square footprint rather
     than the label's padding. */
  .close {
    font-size: var(--fs-md); line-height: 1;
    width: 32px; height: 32px; padding: 0;
    display: grid; place-items: center;
    border-radius: var(--r-compact);
  }

  /* Cap the hero so the title, tags and caption are visible without
     scrolling. A 2:3 photo otherwise fills the whole panel on open. */
  /* The hero sits in an Obsidian well, because black is the one ground
     that casts nothing back onto an arbitrary photograph. */
  .hero {
    width: 100%; max-height: 52vh; object-fit: contain;
    display: block; background: var(--well);
  }
  /* Audio has no picture of its own beyond the drawn waveform, so the well
     holds that and the native player under it. */
  .well {
    background: var(--well); padding: 16px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .wave {
    width: 100%; max-height: 160px; object-fit: contain; display: block;
    border-radius: var(--r-link);
  }
  audio { width: 100%; display: block; }

  .credit { margin: -8px 0 0; color: var(--muted); }
  .credit a { color: inherit; text-underline-offset: 2px; }
  .credit a:hover { color: var(--text); }

  /* Timeline strips scroll inside themselves — never the page — so a
     two-hour file at 400px wide is still one row and no horizontal page
     scroll. position: relative makes offsetLeft measure from the strip. */
  .strip {
    position: relative; display: flex; gap: 10px;
    overflow-x: auto; padding: 4px 2px 6px; margin-top: -8px;
    scrollbar-width: thin;
  }
  .frame {
    flex: 0 0 120px; padding: 0; border: 0; background: none;
    display: flex; flex-direction: column; gap: 6px; align-items: flex-start;
    border-radius: var(--r-link);
  }
  .frame:hover { background: none; }
  .frame img, .frame .noframe {
    width: 120px; height: 86px; object-fit: cover; display: block;
    border-radius: var(--r-tile); background: var(--well);
    transition: opacity 140ms var(--ease);
  }
  .frame:hover img { opacity: 0.78; }
  /* The match is marked the way the orbit marks its subject: a 1px Slate
     ring used as an outline — no blue across a photograph. */
  .frame.match img, .frame.match .noframe {
    outline: 1px solid var(--outline); outline-offset: 2px;
  }
  .frame .t { color: var(--muted); font-variant-numeric: tabular-nums; }
  .frame.now .t, .frame.match .t { color: var(--text); }
  .frame.match .t { font-weight: 600; }

  .chip { flex: 0 0 auto; font-variant-numeric: tabular-nums; }
  /* The playhead's window fills like an active tab; the matched one keeps
     its outline in the text colour and bolds, so both read at once. */
  .chip.on { color: var(--text); background: var(--border); }
  .chip.match { color: var(--text); border-color: var(--text); font-weight: 600; }

  .body { padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .pad { padding: 20px; }
  h2 {
    margin: 0; font-family: var(--font-display);
    font-size: var(--fs-xl); font-weight: 600;
    line-height: 1.07; letter-spacing: var(--ls-28);
  }
  h3 {
    margin: 10px 0 0; font-size: var(--fs-xs); font-weight: 400;
    letter-spacing: var(--ls-12);
    color: var(--muted); text-transform: lowercase;
  }
  /* Tags are text now, with no container to separate them, so the gap has
     to do that work on its own. */
  .tags { display: flex; flex-wrap: wrap; gap: 4px 14px; }
  .tabs { display: flex; gap: 6px; }
  /* --border, not --steel: the light theme flips --text but not --steel,
     and that pair falls to ~1.3:1 (see Palette/Review). */
  .tabs .on { color: var(--text); background: var(--border); }
  .prose {
    margin: 0; white-space: pre-wrap;
    font-size: var(--fs-md); line-height: 1.6; letter-spacing: var(--ls-17);
  }
  .caption { color: var(--muted); }
  /* A literal readout, so it keeps the monospace face — the one exception
     to metadata being set in SF Pro Text. */
  .layout {
    margin: 0; padding: 16px;
    background: var(--well); border: 1px solid var(--border);
    border-radius: var(--r-panel);
    font-family: var(--font-mono); font-size: var(--fs-xs); line-height: 1.45;
    overflow-x: auto; white-space: pre;
  }
  .ents { margin: 0; padding: 0; list-style: none; display: flex;
          flex-direction: column; gap: 4px; color: var(--text); }
  /* The entity kind is a label, not a link — blue would promise a
     destination this has no way to honour. */
  .ents .k { color: var(--muted); margin-right: 8px; }
  .err { color: var(--err); }
  .related { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .rel { flex: 0 0 120px; padding: 0; border: 0; background: none;
         border-radius: var(--r-tile); overflow: hidden; display: block;
         cursor: pointer; }
  .rel img { width: 120px; height: 86px; object-fit: cover; display: block;
             border-radius: var(--r-tile); transition: opacity 140ms var(--ease); }
  .rel:hover img { opacity: 0.78; }
  .reltext { display: block; padding: 12px; font-size: var(--fs-xs);
             line-height: 1.33; letter-spacing: var(--ls-12);
             text-align: left; height: 86px; overflow: hidden; color: var(--text);
             background: var(--raised); border-radius: var(--r-tile); }
</style>
