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
  .sortbar { display: flex; gap: 2px; flex: 0 0 auto; }
  /* This row lives INSIDE the glass bar, so it drops the outlines: an
     outlined pill inside a translucent capsule reads as a box in a box.
     The selected one is the only filled surface, and the fill is a tint
     of the glass rather than an opaque panel — a sort order is not a
     primary action, and blue is not spent here.

     Labels take the on-glass colours: Ash would fall to ~2.3:1 when a
     bright photograph pans under the bar. */
  button {
    background: none; color: var(--on-glass-dim);
    border: 0; border-radius: var(--r-pill);
    padding: 6px 13px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; line-height: 16px; letter-spacing: var(--ls-ctl);
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  button:hover { color: var(--on-glass); }
  button.on { color: var(--on-glass); background: var(--glass-fill); }
</style>
