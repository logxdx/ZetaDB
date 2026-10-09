<script lang="ts">
  import { api } from "../api";
  // The one place things get added: write text, or hand off to the file
  // picker. Search rides along because ⌘K and / still land here.
  let { open = $bindable(), mode = $bindable(), onsearch, onupload, oncreated }:
    { open: boolean; mode: "search" | "note"; onsearch: (q: string) => void;
      onupload: () => void; oncreated: () => void } = $props();

  let value = $state("");
  let input = $state<HTMLTextAreaElement | HTMLInputElement | null>(null);
  let busy = $state(false);

  $effect(() => { if (open) queueMicrotask(() => input?.focus()); });

  async function submit() {
    const v = value.trim();
    if (!v || busy) return;
    if (mode === "search") { onsearch(v); open = false; value = ""; return; }
    busy = true;
    try { await api.createNote(v); value = ""; open = false; oncreated(); }
    finally { busy = false; }
  }

  function upload() { open = false; onupload(); }

  // Escape is deliberately absent: App owns it on the window, so it works
  // whether or not the caret is still in this field.
  function key(e: KeyboardEvent) {
    // Ctrl+Enter saves a note. Plain Enter must stay free so a note can
    // contain line breaks.
    if (e.key === "Enter" && (mode === "search" || e.ctrlKey)) {
      e.preventDefault(); submit();
    }
    if (e.key === "Tab") { e.preventDefault(); mode = mode === "search" ? "note" : "search"; }
  }
</script>

{#if open}
  <div class="scrim" onclick={() => (open = false)} role="presentation"></div>
  <div class="palette" role="dialog" aria-modal="true" aria-label="add or search">
    <div class="tabs meta">
      <button class:on={mode === "note"} onclick={() => (mode = "note")}>write text</button>
      <button onclick={upload}>upload files</button>
      <button class:on={mode === "search"} onclick={() => (mode = "search")}>search</button>
      <span class="spacer"></span>
      <span class="hint">tab switches · {mode === "note" ? "ctrl↵ saves" : "↵ searches"}</span>
      <button class="x" onclick={() => (open = false)} aria-label="close">×</button>
    </div>
    {#if mode === "search"}
      <input bind:this={input} bind:value placeholder="search everything…" onkeydown={key} />
    {:else}
      <textarea bind:this={input} bind:value rows="5"
                placeholder="dump an idea…" onkeydown={key}></textarea>
    {/if}
  </div>
{/if}

<style>
  .scrim {
    position: fixed; inset: 0; z-index: 50;
    background: color-mix(in srgb, var(--bg) 70%, transparent);
  }
  /* A Carbon panel at the panel radius, with no shadow — it is separated
     from the canvas by the scrim and a keyline, not by elevation. */
  .palette {
    position: fixed; top: 18vh; left: 50%; transform: translateX(-50%);
    width: min(680px, calc(100vw - var(--inset) * 2)); z-index: 51;
    background: var(--surface);
    border: 1px solid var(--keyline);
    border-radius: var(--r-panel); overflow: hidden;
  }
  .tabs {
    display: flex; align-items: center; gap: 6px;
    padding: 12px 14px; border-bottom: 1px solid var(--border);
  }
  /* The active tab is a filled surface, not a coloured outline: blue is
     reserved for the action, never for which pane you are looking at. */
  .tabs button {
    background: none; border: 0; border-radius: var(--r-pill);
    color: var(--muted); padding: 5px 12px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; letter-spacing: var(--ls-ctl);
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  .tabs button:hover { color: var(--text); }
  .tabs .on { color: var(--text); background: var(--steel); }
  .spacer { flex: 1; }
  .hint {
    color: var(--muted); font-size: var(--fs-xs);
    letter-spacing: var(--ls-12);
  }
  .x { font-size: var(--fs-md); line-height: 1; padding: 2px 8px; }

  /* The field takes the reference's device-selector recipe: a barely-lit
     fill, a Slate hairline, and a blue focus outline at a 1px offset. */
  input, textarea {
    display: block; width: calc(100% - var(--gutter) * 2);
    margin: var(--gutter); resize: none;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--outline);
    color: var(--text);
    font-family: var(--font-content);
    font-size: var(--fs-md); line-height: 1.47;
    letter-spacing: var(--ls-17);
  }
  input {
    border-radius: var(--r-input); padding: 12px 24px;
  }
  /* A multi-line field cannot take the 980px pill, so it takes the panel
     corner instead — the one place the two radii are not interchangeable. */
  textarea {
    border-radius: var(--r-panel); padding: 16px 20px;
  }
  input:focus-visible, textarea:focus-visible {
    outline: 1px solid var(--blue); outline-offset: 1px;
  }
  input::placeholder, textarea::placeholder { color: var(--muted); }
</style>
