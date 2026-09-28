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
    <div class="tabs mono">
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
  .scrim { position: fixed; inset: 0; background: color-mix(in srgb, var(--bg) 70%, transparent); z-index: 50; }
  .palette {
    position: fixed; top: 18vh; left: 50%; transform: translateX(-50%);
    width: min(640px, calc(100vw - 32px)); z-index: 51;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }
  .tabs { display: flex; align-items: center; gap: 6px; padding: 8px 10px;
          border-bottom: 1px solid var(--border); }
  .tabs button { background: none; border: 1px solid transparent; border-radius: 3px;
                 color: var(--muted); padding: 2px 8px; cursor: pointer;
                 font-family: var(--font-mono); font-size: var(--fs-xs); }
  .tabs button:hover { color: var(--text); }
  .tabs .on { color: var(--text); border-color: var(--accent); }
  .spacer { flex: 1; }
  .hint { color: var(--muted); font-family: var(--font-mono); font-size: var(--fs-xs); }
  .x { font-family: var(--font-content); font-size: var(--fs-md);
       line-height: 1; padding: 1px 6px; }
  input, textarea {
    width: 100%; border: 0; outline: none; resize: none;
    background: var(--surface); color: var(--text);
    font-family: var(--font-content); font-size: var(--fs-lg);
    padding: 14px 16px; line-height: 1.5;
  }
  textarea { font-size: var(--fs-md); }
</style>
