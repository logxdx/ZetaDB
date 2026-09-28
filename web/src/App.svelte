<script lang="ts">
  import { store, connectEvents } from "./lib/stores/items.svelte";
  import { api, whyLabel } from "./lib/api";
  import Canvas from "./lib/components/Canvas.svelte";
  import RelatedBar from "./lib/components/RelatedBar.svelte";
  import Focus from "./lib/components/Focus.svelte";
  import Palette from "./lib/components/Palette.svelte";
  import SortBar from "./lib/components/SortBar.svelte";
  import ThemeToggle from "./lib/components/ThemeToggle.svelte";

  let paletteOpen = $state(false);
  let paletteMode = $state<"search" | "note">("search");
  let focusId = $state<number | null>(null);
  let dragging = $state(false);
  let fileInput = $state<HTMLInputElement | null>(null);
  let uploadNote = $state<string | null>(null);

  async function pickFiles(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = "";                     // so the same file can be re-picked
    if (!files.length) return;
    uploadNote = `uploading ${files.length}…`;
    try {
      const r = await api.upload(files);
      const bad = r.created.filter((c: any) => c.error);
      const dup = r.created.filter((c: any) => c.created === false && !c.error);
      uploadNote = bad.length
        ? `${bad.length} rejected — ${bad[0].error}`
        : `added ${r.created.length - dup.length}${dup.length ? `, ${dup.length} already here` : ""}`;
      await store.loadStream(true);
    } catch (err) {
      uploadNote = err instanceof Error ? err.message : String(err);
    }
    setTimeout(() => (uploadNote = null), 4000);
  }
  let q = $state("");

  store.loadStream(true);
  $effect(() => connectEvents());

  function compose() { paletteMode = "note"; paletteOpen = true; }

  // Escape has ONE owner. It used to be handled in three places — on the
  // palette's input element (dead the moment focus left the field), behind
  // this handler's "is the user typing" early return, and again on a window
  // listener inside Focus — so which layer it closed depended on where the
  // caret happened to be. Now every layer is dismissed from here, topmost
  // first.
  function key(e: KeyboardEvent) {
    const el = e.target as HTMLElement | null;
    const typing = /^(INPUT|TEXTAREA)$/.test(el?.tagName ?? "");

    if (e.key === "Escape") {
      if (paletteOpen) paletteOpen = false;
      else if (focusId !== null) focusId = null;
      else if (store.mode === "related") store.exitRelated();
      else if (typing) el?.blur();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); paletteMode = "search"; paletteOpen = true; return;
    }
    if (typing) return;
    if (e.key === "/") { e.preventDefault(); paletteMode = "search"; paletteOpen = true; }
  }

  async function drop(e: DragEvent) {
    e.preventDefault(); dragging = false;
    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length) { await api.upload(files); }
  }

  async function paste(e: ClipboardEvent) {
    const files = Array.from(e.clipboardData?.items ?? [])
      .filter((i) => i.kind === "file").map((i) => i.getAsFile()!).filter(Boolean);
    if (files.length) { e.preventDefault(); await api.upload(files); return; }
    const text = e.clipboardData?.getData("text")?.trim();
    if (text && text.length > 12 && !/^(INPUT|TEXTAREA)$/.test((e.target as HTMLElement)?.tagName ?? "")) {
      e.preventDefault(); await api.createNote(text); store.loadStream(true);
    }
  }

  function runSearch(v: string) { q = v; store.search(v); }

  function openItem(id: number) {
    const it = store.items.find((x) => x.id === id);
    if (it) store.showRelated(it);
  }
  function clearSearch() { q = ""; store.loadStream(true); }
</script>

<svelte:window onkeydown={key} onpaste={paste}
  ondragover={(e) => { e.preventDefault(); dragging = true; }}
  ondragleave={() => (dragging = false)} ondrop={drop} />

<!-- The chrome floats ON the canvas instead of sitting above it in a bar:
     one capsule holding the search field and the "+" that adds things, then
     the theme toggle. Nothing has a hard edge against the grid. -->
<div class="chrome">
  <form class="bar" onsubmit={(e) => { e.preventDefault(); runSearch(q); }}>
    <input bind:value={q} placeholder="Search…" aria-label="search"
           oninput={() => { if (!q) clearSearch(); }} />
    {#if store.queueDepth > 0}
      <span class="mono queue">{store.queueDepth} processing</span>
    {/if}
    {#if uploadNote}<span class="mono note">{uploadNote}</span>{/if}
    {#if store.mode === "stream"}
      <SortBar value={store.sort} onchange={(s) => store.setSort(s)} />
    {/if}
    <!-- The one way in: writing and uploading are the same gesture, so they
         live behind the same button rather than as two pieces of chrome. -->
    <button class="plus" type="button" onclick={compose}
            title="write a note or upload files" aria-label="add">+</button>
  </form>

  <input bind:this={fileInput} type="file" multiple hidden onchange={pickFiles}
         accept="image/*,.txt,.md,.markdown,.text,.rst,.org" />
  <ThemeToggle />
</div>

<main class:related={store.mode === "related"} class:searching={store.mode === "search"}>
  {#if store.mode === "search"}
    <div class="sumline mono">
      <strong>{store.items.length}</strong> for “{store.query}”
      {#each Object.entries(store.retrievers) as [k, n]}
        <span class="ret">{whyLabel(k)} {n}</span>
      {/each}
      {#if store.seeds.length}
        <span class="seeds">tag seeds: {store.seeds.map((s) => s.term).join(", ")}</span>
      {/if}
      <button class="mono" onclick={clearSearch}>clear</button>
    </div>
  {/if}

  {#if store.error}
    <p class="mono err">{store.error}</p>
  {:else if !store.items.length && !store.loading && store.mode !== "related"}
    <div class="empty">
      <p class="mono muted">nothing here yet</p>
      <p class="muted">Drop files anywhere, paste an image or text, or press
        <kbd class="mono">+</kbd> to dump an idea. The watched folder is
        <code class="mono">~/dump</code>.</p>
    </div>
  {:else}
    <Canvas items={store.items} onopen={openItem} oninfo={(id) => (focusId = id)}
            focusItem={store.mode === "related" ? store.relatedTo : null}
            relKey={store.mode === "related" ? store.relMode : ""} />
  {/if}
</main>

{#if store.mode === "related" && store.relatedTo}
  <RelatedBar item={store.relatedTo} mode={store.relMode}
              count={store.items.length}
              onmode={(m) => store.setRelMode(m)}
              onclose={() => store.exitRelated()} />
{/if}

{#if focusId !== null}
  <Focus id={focusId} onclose={() => (focusId = null)} onopen={(id) => (focusId = id)} />
{/if}
<Palette bind:open={paletteOpen} bind:mode={paletteMode} onsearch={runSearch}
         onupload={() => fileInput?.click()}
         oncreated={() => store.loadStream(true)} />

{#if dragging}<div class="dropzone"><span class="mono">drop to ingest</span></div>{/if}

<style>
  /* Floating chrome, after the reference: a capsule on the canvas rather
     than a bar above it. Every surface here shares one recipe — full
     rounding, an off-white fill, a hairline, an inner top highlight and a
     soft drop shadow — so the controls read as one family of objects. */
  .chrome {
    position: fixed; top: 10px; left: 10px; right: 10px; z-index: 20;
    display: flex; align-items: center; gap: 8px;
    pointer-events: none;
  }
  .chrome > :global(*) { pointer-events: auto; }

  .bar, :global(.chrome .round) {
    height: var(--chrome-h);
    background: var(--chrome);
    border: 0.8px solid var(--chrome-border);
    box-shadow: var(--chrome-shadow);
    border-radius: 999px;
  }

  .bar {
    flex: 1 1 auto; min-width: 0;
    display: flex; align-items: center; gap: 8px; padding: 8px;
  }
  .bar input {
    flex: 1 1 auto; min-width: 0;
    background: none; border: 0; color: var(--text); padding: 0 6px 0 10px;
    font-family: var(--font-content); font-size: var(--fs-sm);
    letter-spacing: -0.02em; outline: none;
  }
  .bar input::placeholder { color: var(--muted); }

  /* Sized and skinned like the field's other inhabitants: a 34px disc that
     sits inside the capsule rather than beside it. */
  .plus {
    flex: 0 0 auto; width: 34px; height: 34px; padding: 0;
    display: grid; place-items: center; cursor: pointer;
    background: var(--bg); color: var(--muted);
    border: 0.8px solid var(--chrome-border); border-radius: 999px;
    font-family: var(--font-content); font-size: 19px; line-height: 1;
    transition: color 140ms var(--ease), transform 160ms var(--ease);
  }
  .plus:hover { color: var(--text); transform: scale(1.06); }

  .queue, .note { color: var(--accent); letter-spacing: 0.02em; white-space: nowrap; }

  :global(.chrome .round) {
    flex: 0 0 auto; padding: 0 16px; min-width: var(--chrome-h);
    color: var(--muted); cursor: pointer; letter-spacing: 0.02em;
    display: inline-grid; place-items: center;
    transition: color 140ms var(--ease);
  }
  :global(.chrome .round:hover) { color: var(--text); }

  /* The canvas is fixed at inset 0 and the chrome floats over it, so the
     only thing that needs a top offset is flow content. */
  .sumline {
    position: fixed; top: calc(var(--chrome-h) + 18px); left: 10px; z-index: 15;
    width: max-content; max-width: calc(100vw - 20px);
    display: flex; align-items: center; gap: 12px;
    color: var(--muted); padding: 0 8px 0 16px; height: 42px;
    background: var(--chrome); border: 0.8px solid var(--chrome-border);
    box-shadow: var(--chrome-shadow); border-radius: 999px;
  }
  .sumline strong { color: var(--text); font-weight: 500; }
  .ret { color: var(--muted); }
  .ret::before { content: "· "; opacity: 0.5; }
  .seeds { color: var(--accent); }
  .sumline button {
    background: none; border: 0; color: var(--muted);
    padding: 1px 6px; cursor: pointer; border-radius: 999px;
  }
  .sumline button:hover { color: var(--text); }

  .empty { padding: 110px 26px; max-width: 460px; }
  .empty p { margin: 0 0 10px; }
  .muted { color: var(--muted); }
  .err { color: var(--err); }
  kbd { border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; }
  code { color: var(--text); }
  .dropzone {
    position: fixed; inset: 10px; border: 1.5px dashed var(--accent);
    border-radius: 8px; z-index: 60; pointer-events: none;
    display: grid; place-items: center;
    background: color-mix(in srgb, var(--bg) 72%, transparent);
  }
  .dropzone span { color: var(--accent); font-size: var(--fs-md); letter-spacing: 0.02em; }
</style>
