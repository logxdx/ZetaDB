<script lang="ts">
  import { api } from "../api";
  // Neighbourhood only - the global graph is never rendered (spec 11).
  // Nodes are thumbnails, not circles; manual edges solid, semantic dotted.
  let { id, onopen }: { id: number; onopen: (id: number) => void } = $props();

  type N = { id: number; kind: string; title: string; thumb_url: string | null;
             preview: string; x: number; y: number };
  let nodes = $state<N[]>([]);
  let edges = $state<{ src: number; dst: number; kind: string; weight: number }[]>([]);
  let msg = $state("loading…");

  const W = 560, H = 360;

  $effect(() => {
    let alive = true;
    api.graph(id, 2).then((g) => {
      if (!alive) return;
      if (g.nodes.length <= 1) { msg = "no links yet — semantic edges appear once embeddings run"; nodes = []; return; }
      // Deterministic radial layout: focus at centre, neighbours by hop.
      const others = g.nodes.filter((n: any) => n.id !== id);
      nodes = [
        { ...g.nodes.find((n: any) => n.id === id)!, x: W / 2, y: H / 2 },
        ...others.map((n: any, i: number) => {
          const a = (i / others.length) * Math.PI * 2 - Math.PI / 2;
          const r = Math.min(W, H) * 0.36;
          return { ...n, x: W / 2 + Math.cos(a) * r, y: H / 2 + Math.sin(a) * r };
        }),
      ];
      edges = g.edges;
      msg = "";
    }).catch((e) => (msg = String(e)));
    return () => { alive = false; };
  });

  const at = (nid: number) => nodes.find((n) => n.id === nid);
</script>

{#if msg}
  <p class="mono muted">{msg}</p>
{:else}
  <svg viewBox="0 0 {W} {H}" class="graph" role="img" aria-label="neighbourhood graph">
    {#each edges as e}
      {@const a = at(e.src)}{@const b = at(e.dst)}
      {#if a && b}
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke="var(--border)"
              stroke-width={0.8 + e.weight * 2.4}
              stroke-dasharray={e.kind === "semantic" ? "3 3" : "none"} />
      {/if}
    {/each}
    {#each nodes as n (n.id)}
      <g class="node" onclick={() => onopen(n.id)} role="button" tabindex="0"
         onkeydown={(ev) => ev.key === "Enter" && onopen(n.id)}>
        <clipPath id="c{n.id}"><rect x={n.x - 22} y={n.y - 16} width="44" height="32" rx="3" /></clipPath>
        {#if n.thumb_url}
          <image href={n.thumb_url} x={n.x - 22} y={n.y - 16} width="44" height="32"
                 preserveAspectRatio="xMidYMid slice" clip-path="url(#c{n.id})" />
        {:else}
          <rect x={n.x - 22} y={n.y - 16} width="44" height="32" rx="3"
                fill="var(--raised)" />
          <text x={n.x} y={n.y + 3} text-anchor="middle" class="glyph">note</text>
        {/if}
        <rect x={n.x - 22} y={n.y - 16} width="44" height="32" rx="3"
              fill="none" stroke={n.id === id ? "var(--accent)" : "var(--border)"}
              stroke-width={n.id === id ? 2 : 1} />
        <title>{n.preview}</title>
      </g>
    {/each}
  </svg>
  <p class="mono muted legend">
    <span class="k">solid</span> manual · <span class="k">dotted</span> semantic ·
    thickness = similarity
  </p>
{/if}

<style>
  .graph { width: 100%; height: auto; background: var(--raised);
           border-radius: var(--radius); border: 1px solid var(--border); }
  .node { cursor: pointer; }
  .node:hover rect { stroke: var(--accent); }
  .glyph { font-family: var(--font-mono); font-size: 9px; fill: var(--muted); }
  .legend { margin: 6px 0 0; }
  .k { color: var(--text); }
</style>
