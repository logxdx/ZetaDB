<script lang="ts">
  import type { Tag } from "../api";
  let { tag }: { tag: Tag } = $props();
  // Provenance is legible without a legend: human solid, machine dotted,
  // pending dashed with its confidence shown in mono (spec 13.4).
  const cls = $derived(
    tag.source === "human" ? "human" : tag.state === "pending" ? "pending" : "machine"
  );
</script>

<span class="pill mono {cls}" title="{tag.source} · {tag.state} · {tag.confidence}">
  {tag.term}{#if cls === "pending"}<em>{tag.confidence.toFixed(2)}</em>{/if}
</span>

<style>
  .pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 1px 6px; border-radius: 3px; white-space: nowrap;
    color: var(--muted); line-height: 1.6;
  }
  .human   { border: 1px solid var(--border); color: var(--text); }
  .machine { border: 1px dotted var(--border); opacity: 0.75; }
  .pending { border: 1px dashed var(--warn); color: var(--warn); }
  em { font-style: normal; opacity: 0.7; }
</style>
