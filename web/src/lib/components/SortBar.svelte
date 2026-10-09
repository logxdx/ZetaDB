<script lang="ts">
  import type { Sort } from "../api";
  let { value, onchange }: { value: Sort; onchange: (s: Sort) => void } = $props();

  const OPTIONS: { key: Sort; label: string; title: string }[] = [
    { key: "date",  label: "recent", title: "newest first" },
    { key: "title", label: "name",   title: "by filename, A–Z" },
    { key: "tags",  label: "tags",   title: "grouped by strongest tag" },
  ];
</script>

<div class="sortbar meta" role="group" aria-label="sort">
  {#each OPTIONS as o (o.key)}
    <button class:on={value === o.key} title={o.title}
            aria-pressed={value === o.key}
            onclick={() => onchange(o.key)}>{o.label}</button>
  {/each}
</div>

<style>
  .sortbar { display: flex; gap: 6px; flex: 0 0 auto; }
  /* Outlined explore pills. The selected one is the only filled surface,
     so the row stays monochrome — a sort order is not a primary action,
     and blue is not spent here. */
  button {
    background: none; color: var(--muted);
    border: 1px solid var(--outline); border-radius: var(--r-pill);
    padding: 5px 12px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; line-height: 16px; letter-spacing: var(--ls-ctl);
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  button:hover { color: var(--text); }
  button.on {
    color: var(--text); background: var(--surface);
    border-color: var(--outline);
  }
</style>
