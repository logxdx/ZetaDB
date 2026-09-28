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
    <p class="mono muted pad">loading…</p>
  {:else}
    <header>
      <span class="mono muted">#{item.id} · {item.kind} · {item.status}</span>
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

      <nav class="tabs mono">
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
          <p class="mono muted">caption · {item.caption_model ?? "?"}</p>
        {/if}
        {#if !item.body && !item.caption}
          <p class="mono muted">no description yet</p>
        {/if}
        {#if item.entities.length}
          <h3 class="mono">entities</h3>
          <ul class="ents mono">
            {#each item.entities as e}
              <li><span class="k">{e.kind}</span> {e.value}</li>
            {/each}
          </ul>
        {/if}
        {#if item.todos.length}
          <h3 class="mono">todos</h3>
          <ul class="ents mono">
            {#each item.todos as t}<li>{t.text}{#if t.due_date} · {t.due_date}{/if}</li>{/each}
          </ul>
        {/if}
        {#if item.errors.length}
          <h3 class="mono err">stage errors</h3>
          <ul class="ents mono err">
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
        <h3 class="mono">related</h3>
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
    position: fixed; top: 10px; right: 10px; bottom: 10px;
    width: min(620px, calc(100vw - 20px));
    background: var(--surface); border: 0.8px solid var(--chrome-border);
    border-radius: 20px; z-index: 41; overflow-y: auto;
    box-shadow: var(--chrome-shadow), 0 18px 50px rgba(0, 0, 0, 0.16);
    animation: slide 240ms var(--ease);
  }
  @keyframes slide { from { transform: translateX(16px); opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .focus { animation: none; } }
  @media (max-width: 660px) { .focus { left: 10px; width: auto; } }

  header {
    display: flex; align-items: center; gap: 10px; padding: 12px 16px;
    border-bottom: 1px solid var(--border); position: sticky; top: 0;
    background: var(--surface); z-index: 1;
  }
  .spacer { flex: 1; }
  button { background: none; border: 0.8px solid var(--chrome-border);
           border-radius: 999px; color: var(--muted); padding: 5px 11px;
           cursor: pointer; font-family: var(--font-mono); font-size: var(--fs-xs);
           transition: color 140ms var(--ease), background 140ms var(--ease); }
  button:hover { color: var(--text); background: var(--raised); }
  /* The glyph carries the meaning now, so it needs the content face and a
     square footprint rather than the mono label's padding. */
  .close { font-family: var(--font-content); font-size: var(--fs-md);
           line-height: 1; padding: 4px 9px; }

  /* Cap the hero so the title, tags and caption are visible without
     scrolling. A 2:3 photo otherwise fills the whole panel on open. */
  .hero {
    width: 100%; max-height: 52vh; object-fit: contain;
    display: block; background: var(--raised);
  }
  .body { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .pad { padding: 16px; }
  h2 { margin: 0; font-size: var(--fs-lg); font-weight: 600; letter-spacing: -0.01em; }
  h3 { margin: 8px 0 0; font-size: var(--fs-xs); font-weight: 400;
       color: var(--muted); text-transform: lowercase; }
  .tags { display: flex; flex-wrap: wrap; gap: 4px; }
  .tabs { display: flex; gap: 6px; }
  .tabs .on { color: var(--text); background: var(--raised); }
  .prose { margin: 0; white-space: pre-wrap; line-height: 1.6; }
  .caption { color: var(--muted); }
  .layout {
    margin: 0; padding: 12px; background: var(--raised); border-radius: var(--radius);
    font-family: var(--font-mono); font-size: var(--fs-xs); line-height: 1.45;
    overflow-x: auto; white-space: pre;
  }
  .ents { margin: 0; padding: 0; list-style: none; display: flex;
          flex-direction: column; gap: 3px; color: var(--muted); }
  .ents .k { color: var(--accent); margin-right: 6px; }
  .err { color: var(--err); }
  .related { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; }
  .rel { flex: 0 0 96px; padding: 0; border: 0; background: none;
         border-radius: var(--radius); overflow: hidden; display: block;
         cursor: pointer; }
  .rel img { width: 96px; height: 68px; object-fit: cover; display: block;
             border-radius: var(--radius); transition: opacity 140ms var(--ease); }
  .rel:hover img { opacity: 0.78; }
  .reltext { display: block; padding: 8px; font-size: var(--fs-xs);
             text-align: left; height: 68px; overflow: hidden; color: var(--text);
             background: var(--raised); border-radius: var(--radius); }
</style>
