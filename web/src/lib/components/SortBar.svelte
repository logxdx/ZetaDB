<script lang="ts">
  import type { Sort } from "../api";
  let { value, onchange }: { value: Sort; onchange: (s: Sort) => void } = $props();

  const OPTIONS: { key: Sort; label: string; title: string }[] = [
    { key: "date",  label: "recent", title: "newest first" },
    { key: "title", label: "name",   title: "by filename, A–Z" },
    { key: "tags",  label: "tags",   title: "grouped by strongest tag" },
  ];
</script>

<div class="sortbar mono" role="group" aria-label="sort">
  {#each OPTIONS as o (o.key)}
    <button class:on={value === o.key} title={o.title}
            aria-pressed={value === o.key}
            onclick={() => onchange(o.key)}>{o.label}</button>
  {/each}
</div>

<style>
  .sortbar { display: flex; gap: 2px; flex: 0 0 auto; }
  /* Capsules, like every other control on the canvas — the selected one is
     the only filled surface, so the row stays monochrome. */
  button {
    background: none; border: 0.8px solid transparent; color: var(--muted);
    padding: 6px 12px; cursor: pointer; border-radius: 999px;
    font-family: var(--font-mono); font-size: var(--fs-xs);
    letter-spacing: 0.02em;
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  button:hover { color: var(--text); }
  button.on {
    color: var(--text); background: var(--bg); border-color: var(--chrome-border);
  }
</style>
