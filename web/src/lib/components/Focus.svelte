<script lang="ts">
  import { api, type ItemDetail, type Item } from "../api";
  import TagPill from "./TagPill.svelte";
  import Graph from "./Graph.svelte";

  let { id, onclose, onopen }:
    { id: number; onclose: () => void; onopen: (id: number) => void } = $props();

  let item = $state<ItemDetail | null>(null);
  let related = $state<Item[]>([]);
  let tab = $state<"content" | "ocr" | "graph">("content");

  $effect(() => {
    let alive = true;
    item = null; related = []; tab = "content";
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

<div class="scrim" onclick={onclose} role="presentation"></div>
<div class="focus" role="dialog" aria-modal="true" aria-label="item detail">
  {#if !item}
    <p class="meta muted pad">loading…</p>
  {:else}
    <header>
      <span class="meta muted">#{item.id} · {item.kind} · {item.status}</span>
      <div class="spacer"></div>
      <button class="close" onclick={onclose} aria-label="close">×</button>
    </header>

    {#if item.kind === "image" && item.file_url}
      <img class="hero" src={item.file_url} alt={item.title ?? ""} />
    {/if}

    <div class="body">
      <h2>{item.title ?? "untitled"}</h2>

      {#if item.tags.length}
        <div class="tags">
          {#each item.tags as t (t.term)}<TagPill tag={t} />{/each}
        </div>
      {/if}

      <nav class="tabs meta">
        <button class:on={tab === "content"} onclick={() => (tab = "content")}>content</button>
        {#if item.layout_text}
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
                <span class="reltext">{r.preview.slice(0, 70)}</span>
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
    animation: slide 240ms var(--ease);
  }
  @keyframes slide { from { transform: translateX(16px); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .focus { animation: none; } }
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
  .tabs .on { color: var(--text); background: var(--steel); }
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
