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
     one contained Graphite bar holding the search field and the "+" that
     adds things, then the theme control. No hard edge against the grid,
     and no shadow — a 1px keyline is the whole separation. -->
<div class="chrome">
  <form class="bar" onsubmit={(e) => { e.preventDefault(); runSearch(q); }}>
    <input bind:value={q} placeholder="Search…" aria-label="search"
           oninput={() => { if (!q) clearSearch(); }} />
    {#if store.queueDepth > 0}
      <span class="meta queue">{store.queueDepth} processing</span>
    {/if}
    {#if uploadNote}<span class="meta note">{uploadNote}</span>{/if}
    {#if store.mode === "stream"}
      <SortBar value={store.sort} onchange={(s) => store.setSort(s)} />
    {/if}
    <!-- The one way in: writing and uploading are the same gesture, so they
         live behind the same button rather than as two pieces of chrome —
         and so this is the single filled blue control in the product. -->
    <button class="plus" type="button" onclick={compose}
            title="write a note or upload files" aria-label="add">+</button>
  </form>

  <input bind:this={fileInput} type="file" multiple hidden onchange={pickFiles}
         accept="image/*,.txt,.md,.markdown,.text,.rst,.org" />
  <ThemeToggle />
</div>

<main class:related={store.mode === "related"} class:searching={store.mode === "search"}>
  {#if store.mode === "search"}
    <div class="sumline">
      <strong>{store.items.length}</strong> for “{store.query}”
      {#each Object.entries(store.retrievers) as [k, n]}
        <span class="ret">{whyLabel(k)} {n}</span>
      {/each}
      {#if store.seeds.length}
        <span class="seeds">tag seeds: {store.seeds.map((s) => s.term).join(", ")}</span>
      {/if}
      <button onclick={clearSearch}>clear</button>
    </div>
  {/if}

  {#if store.error}
    <p class="err">{store.error}</p>
  {:else if !store.items.length && !store.loading && store.mode !== "related"}
    <!-- An empty library is the hero stage, not an error state: one
         display-scale statement on an uninterrupted Obsidian field. -->
    <div class="empty">
      <h1 class="hero">Nothing here yet.</h1>
      <p class="lede">Drop files anywhere, paste an image or a block of text,
        or press <kbd>+</kbd> to dump an idea.</p>
      <p class="where meta">watching <code>~/dump</code></p>
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

{#if dragging}<div class="dropzone"><span>drop to ingest</span></div>{/if}

<style>
  /* The chrome floats on the canvas, but it is no longer a capsule with
     its own light: under this system an object is its fill plus a 1px
     keyline, and nothing casts a shadow. A contained Graphite bar at a
     20px radius, after the reference's local product navigation. */
  .chrome {
    position: fixed; top: var(--inset); left: var(--inset); right: var(--inset);
    z-index: 20; display: flex; align-items: center; gap: 8px;
    pointer-events: none;
  }
  .chrome > :global(*) { pointer-events: auto; }

  /* Liquid glass. The bar is a material, not a panel: it lets the library
     show through, blurs it, and carries a specular top edge plus a soft
     lift. This is the ONE place the product is allowed translucency or a
     shadow — every content surface stays flat and opaque. */
  .bar, :global(.chrome .round) {
    background: var(--glass-bg);
    -webkit-backdrop-filter: var(--glass-blur);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-sheen);
    color: var(--on-glass);
  }
  /* Without backdrop-filter the glass would be a flat wash over the
     photographs, so fall back to the opaque chrome instead. */
  @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
    .bar, :global(.chrome .round) { background: var(--chrome); }
  }

  .bar {
    flex: 1 1 auto; min-width: 0;
    height: var(--chrome-h);
    display: flex; align-items: center; gap: 10px; padding: 6px 6px 6px 0;
    border-radius: var(--r-pill);
  }
  /* The bar IS the field, so the focus ring lands on the container rather
     than nesting a second outlined pill inside a 44px bar. */
  .bar:focus-within { outline: 1px solid var(--blue); outline-offset: 2px; }

  .bar input {
    flex: 1 1 auto; min-width: 0;
    background: none; border: 0; outline: none;
    color: var(--on-glass); padding: 0 6px 0 20px;
    font-family: var(--font-content);
    font-size: var(--fs-md); letter-spacing: var(--ls-17);
  }
  /* Ash tops out near 2.3:1 on glass over a bright photograph, so the
     chrome's secondary text is dimmed-but-light rather than mid-grey. */
  .bar input::placeholder { color: var(--on-glass-dim); }

  /* The one filled blue control in the product. The reference spends its
     concentrated blue on exactly this kind of control and nothing else. */
  .plus {
    flex: 0 0 auto; width: 32px; height: 32px; padding: 0;
    display: grid; place-items: center; cursor: pointer;
    background: var(--blue); color: #fff;
    border: 0; border-radius: var(--r-pill);
    font-family: var(--font-content); font-size: 19px; line-height: 1;
    transition: filter 140ms var(--ease);
  }
  .plus:hover { filter: brightness(1.12); }

  /* Electric Link lands at 2.8:1 on glass over a bright photograph, so the
     live counters take the on-glass colour and earn their emphasis from
     weight instead. */
  .queue, .note {
    color: var(--on-glass); font-weight: 500; white-space: nowrap;
  }

  /* A true circle at the bar's own height, so the two read as one set of
     glass objects. It was previously sized by its content and came out a
     squat 43x19 slab beside a 44px bar. */
  :global(.chrome .round) {
    flex: 0 0 auto;
    width: var(--chrome-h); height: var(--chrome-h);
    padding: 0; border-radius: var(--r-pill);
    cursor: pointer;
    font-size: var(--fs-xs); letter-spacing: var(--ls-ctl);
    display: inline-grid; place-items: center;
    transition: background 140ms var(--ease);
  }
  /* An inset spread tints the glass without replacing it, so the blur and
     the specular edge survive the hover. */
  :global(.chrome .round:hover) {
    box-shadow: var(--glass-sheen), inset 0 0 0 999px var(--glass-fill);
  }

  /* Result strip — the reference's announcement band, contained rather
     than full-bleed, because the canvas is fixed at inset 0 and nothing
     is allowed to reserve vertical space above it. */
  .sumline {
    position: fixed;
    top: calc(var(--inset) + var(--chrome-h) + 12px); left: var(--inset);
    z-index: 15;
    width: max-content; max-width: calc(100vw - var(--inset) * 2);
    display: flex; align-items: center; gap: 14px;
    height: var(--strip-h); padding: 0 10px 0 20px;
    color: var(--on-glass);
    font-size: var(--fs-sm); line-height: 20px; letter-spacing: var(--ls-14);
    background: var(--glass-bg);
    -webkit-backdrop-filter: var(--glass-blur);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-sheen);
    border-radius: var(--r-pill);
  }
  @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
    .sumline { background: var(--chrome); }
  }
  .sumline strong { font-weight: 600; }
  .ret { color: var(--on-glass-dim); }
  .ret::before { content: "· "; opacity: 0.5; }
  .seeds { color: var(--on-glass-dim); }
  .sumline button {
    background: none; border: 0; color: var(--on-glass);
    font-family: var(--font-content); font-size: var(--fs-sm);
    font-weight: 500; letter-spacing: var(--ls-14);
    padding: 2px 8px; cursor: pointer; border-radius: var(--r-link);
  }
  .sumline button:hover { text-decoration: underline; }

  /* An empty library is this product's hero stage. It is the only place
     80px display type appears. */
  .empty {
    position: fixed; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 18px; padding: 0 var(--inset); text-align: center;
  }
  .hero {
    margin: 0; max-width: 14ch;
    font-family: var(--font-display); font-weight: 600;
    font-size: clamp(40px, 9vw, var(--fs-hero));
    line-height: 1.05; letter-spacing: var(--ls-80);
    color: var(--text);
  }
  .lede {
    margin: 0; max-width: 46ch; color: var(--muted);
    font-size: var(--fs-md); line-height: 1.47; letter-spacing: var(--ls-17);
  }
  .where { color: var(--muted); margin: 0; }
  .err { color: var(--err); padding: calc(var(--inset) * 4) var(--inset); }
  kbd {
    border: 1px solid var(--outline); border-radius: var(--r-link);
    padding: 1px 6px; color: var(--text);
  }
  code { font-family: var(--font-mono); color: var(--text); }

  .dropzone {
    position: fixed; inset: var(--inset);
    border: 1px solid var(--link); border-radius: var(--r-panel);
    z-index: 60; pointer-events: none;
    display: grid; place-items: center;
    background: color-mix(in srgb, var(--bg) 72%, transparent);
  }
  .dropzone span { color: var(--link); font-size: var(--fs-md); }
</style>
