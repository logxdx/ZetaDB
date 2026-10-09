<script lang="ts">
  import type { Item, RelMode } from "../api";
  let { item, mode, onmode, onclose, count }:
    { item: Item; mode: RelMode; onmode: (m: RelMode) => void;
      onclose: () => void; count: number } = $props();
</script>

<!-- Anchors the related view: what you clicked, how the set was chosen,
     and the way back. The toggle is the point of the view, so it sits
     next to the thing it is filtering. -->
<div class="bar">
  {#if item.thumb_url}
    <img src={item.thumb_url} alt="" />
  {:else}
    <span class="notechip meta">note</span>
  {/if}
  <div class="what">
    <span class="title">{item.title ?? `item ${item.id}`}</span>
    <span class="meta" class:muted={count > 0} class:none={count === 0}>
      {#if count === 0}
        {mode === "tag" ? "nothing shares its tags" : "nothing looks like it"}
      {:else}
        {count} {mode === "tag" ? "sharing tags" : "that look alike"}
      {/if}
    </span>
  </div>
  <div class="toggle meta" role="group" aria-label="relation">
    <button class:on={mode === "similar"} aria-pressed={mode === "similar"}
            onclick={() => onmode("similar")}>similar</button>
    <button class:on={mode === "tag"} aria-pressed={mode === "tag"}
            onclick={() => onmode("tag")}>same tag</button>
  </div>
  <button class="close" onclick={onclose} aria-label="back to library">×</button>
</div>

<style>
  /* A contained Graphite bar in the chrome family, in the same slot the
     result strip uses — never a second header, and never a shadow. */
  .bar {
    position: fixed;
    top: calc(var(--inset) + var(--chrome-h) + 12px); left: var(--inset);
    z-index: 15;
    max-width: calc(100vw - var(--inset) * 2);
    display: flex; align-items: center; gap: 14px;
    height: var(--strip-h); padding: 0 8px 0 6px;
    color: var(--on-glass);
    background: var(--glass-bg);
    -webkit-backdrop-filter: var(--glass-blur);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-sheen);
    border-radius: var(--r-pill);
  }
  @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
    .bar { background: var(--chrome); }
  }
  img {
    width: 40px; height: 40px; object-fit: cover;
    border-radius: var(--r-pill); display: block;
  }
  .notechip {
    width: 40px; height: 40px; display: grid; place-items: center;
    background: var(--glass-fill); border-radius: var(--r-pill);
    color: var(--on-glass-dim);
  }
  .what { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .title {
    font-family: var(--font-display); font-size: 19px; font-weight: 600;
    line-height: 1.21; letter-spacing: var(--ls-19);
    color: var(--on-glass);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 34vw;
  }
  /* On-glass secondary, not Ash — see tokens.css. */
  .muted { color: var(--on-glass-dim); }
  .none { color: var(--warn); }

  /* Borderless inside the glass, like the sort row: the filled state is
     a tint of the material rather than a panel laid on top of it. */
  .toggle { display: flex; gap: 2px; margin-left: 8px; }
  .toggle button {
    background: none; color: var(--on-glass-dim);
    border: 0; border-radius: var(--r-pill);
    padding: 6px 13px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; line-height: 16px; letter-spacing: var(--ls-ctl);
    transition: background 140ms var(--ease), color 140ms var(--ease);
  }
  .toggle button:hover { color: var(--on-glass); }
  .toggle button.on { color: var(--on-glass); background: var(--glass-fill); }

  .close {
    background: none; color: var(--on-glass-dim);
    border: 0; border-radius: var(--r-pill);
    padding: 0; width: 36px; height: 36px; cursor: pointer;
    display: grid; place-items: center;
    font-family: var(--font-content); font-size: var(--fs-md); line-height: 1;
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  .close:hover { color: var(--on-glass); background: var(--glass-fill); }

  @media (max-width: 620px) { .title { max-width: 30vw; } }
</style>
