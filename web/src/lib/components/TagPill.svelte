<script lang="ts">
  import type { Tag } from "../api";
  let { tag }: { tag: Tag } = $props();
  /* Provenance is carried by weight and colour, not by a container: the
     reference renders status annotations as plain text and explicitly
     never puts them in a badge. Porcelain 600 for what a human wrote, Ash
     400 for what the machine accepted, Copper for what is still pending,
     with the confidence following in Optical Yellow. The dashed outline
     survives only in the Review overlay, where the tag is a hit target. */
  const cls = $derived(
    tag.source === "human" ? "human" : tag.state === "pending" ? "pending" : "machine"
  );
</script>

<span class="tag meta {cls}" title="{tag.source} · {tag.state} · {tag.confidence}">
  {tag.term}{#if cls === "pending"}<em>{tag.confidence.toFixed(2)}</em>{/if}
</span>

<style>
  .tag { white-space: nowrap; }
  .human   { color: var(--text); font-weight: 600; }
  .machine { color: var(--muted); }
  .pending { color: var(--warn); }
  em {
    font-style: normal; margin-left: 4px;
    color: var(--conf); font-variant-numeric: tabular-nums;
  }
</style>
