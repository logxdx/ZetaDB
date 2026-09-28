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
    <span class="notechip mono">note</span>
  {/if}
  <div class="what">
    <span class="title">{item.title ?? `item ${item.id}`}</span>
    <span class="mono" class:muted={count > 0} class:none={count === 0}>
      {#if count === 0}
        {mode === "tag" ? "nothing shares its tags" : "nothing looks like it"}
      {:else}
        {count} {mode === "tag" ? "sharing tags" : "that look alike"}
      {/if}
    </span>
  </div>
  <div class="toggle mono" role="group" aria-label="relation">
    <button class:on={mode === "similar"} aria-pressed={mode === "similar"}
            onclick={() => onmode("similar")}>similar</button>
    <button class:on={mode === "tag"} aria-pressed={mode === "tag"}
            onclick={() => onmode("tag")}>same tag</button>
  </div>
  <button class="close" onclick={onclose} aria-label="back to library">×</button>
</div>

<style>
  /* A floating capsule under the search bar, in the same family as the rest
     of the chrome — never a second header. */
  .bar {
    position: fixed; top: calc(var(--chrome-h) + 18px); left: 10px; z-index: 15;
    max-width: calc(100vw - 20px);
    display: flex; align-items: center; gap: 12px;
    padding: 7px 8px 7px 7px; height: 54px;
    background: var(--chrome); border: 0.8px solid var(--chrome-border);
    box-shadow: var(--chrome-shadow); border-radius: 999px;
  }
  img { width: 40px; height: 40px; object-fit: cover; border-radius: 999px; display: block; }
  .notechip {
    width: 40px; height: 40px; display: grid; place-items: center;
    background: var(--bg); border-radius: 999px; color: var(--muted);
  }
  .what { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
  .title {
    font-size: var(--fs-sm); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; max-width: 34vw;
  }
  .muted { color: var(--muted); }
  .none { color: var(--warn); }
  .toggle { display: flex; gap: 2px; margin-left: 8px; }
  .toggle button {
    background: none; border: 0.8px solid transparent; color: var(--muted);
    padding: 7px 13px; border-radius: 999px; cursor: pointer;
    font-family: var(--font-mono); font-size: var(--fs-xs); letter-spacing: 0.02em;
    transition: background 140ms var(--ease), color 140ms var(--ease);
  }
  .toggle button:hover { color: var(--text); }
  .toggle button.on {
    color: var(--text); background: var(--bg); border-color: var(--chrome-border);
  }
  .close {
    background: none; border: 0.8px solid var(--chrome-border); color: var(--muted);
    padding: 0; width: 30px; height: 30px; border-radius: 999px; cursor: pointer;
    display: grid; place-items: center;
    font-family: var(--font-content); font-size: var(--fs-md); line-height: 1;
  }
  .close:hover { color: var(--text); background: var(--bg); }
  @media (max-width: 620px) { .title { max-width: 30vw; } }
</style>
