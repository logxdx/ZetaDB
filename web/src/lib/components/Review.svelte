<script lang="ts">
  import { api, type ReviewTag, type ReviewProposal } from "../api";

  // Bulk triage, keyboard-first (spec 13.6): a accept, r reject, → next.
  // The whole tagging design depends on this loop - vocab_terms carries
  // accept_count/reject_count and the per-term threshold moves with them
  // (spec 7.1) - so until this surface existed, nothing could adapt.
  let { open = $bindable(), onreviewed }:
    { open: boolean; onreviewed?: () => void } = $props();

  type Queue = "tags" | "proposals";
  let queue = $state<Queue>("tags");
  let tags = $state<ReviewTag[]>([]);
  let proposals = $state<ReviewProposal[]>([]);
  let at = $state(0);
  let done = $state(0);
  let busy = $state(false);
  let err = $state<string | null>(null);
  let last = $state<string | null>(null);

  // Item context, fetched lazily per tag. Judging "wireframe 0.62" without
  // seeing the picture is not review, it is guessing - but the review
  // endpoint returns no thumbnail, so the detail comes from /api/items.
  let detail = $state<any | null>(null);
  const cache = new Map<number, any>();

  const list = $derived(queue === "tags" ? tags : proposals);
  const current = $derived(list[at] ?? null);
  const remaining = $derived(Math.max(0, list.length - at));

  async function load() {
    busy = true; err = null;
    try {
      const r = await api.review(200);
      tags = r.tags; proposals = r.proposals; at = 0; done = 0;
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally { busy = false; }
  }

  $effect(() => { if (open) load(); });

  // Pull the item behind the current tag, and prefetch the next one so the
  // picture is already there when the key is pressed.
  $effect(() => {
    if (!open || queue !== "tags") { detail = null; return; }
    const row = tags[at];
    if (!row) { detail = null; return; }
    const want = row.item_id;
    if (cache.has(want)) { detail = cache.get(want); }
    else {
      detail = null;
      api.item(want).then((d) => { cache.set(want, d); if (tags[at]?.item_id === want) detail = d; })
        .catch(() => {});
    }
    const nxt = tags[at + 1];
    if (nxt && !cache.has(nxt.item_id)) {
      api.item(nxt.item_id).then((d) => cache.set(nxt.item_id, d)).catch(() => {});
    }
  });

  async function act(action: "accept" | "reject") {
    const row: any = current;
    if (!row || busy) return;
    busy = true;
    try {
      if (queue === "tags") {
        await api.reviewTag(row.item_id, row.term_id, action);
        last = `${action === "accept" ? "accepted" : "rejected"} ${row.term}`;
      } else {
        await api.reviewProposal(row.id, action);
        last = `${action === "accept" ? "accepted" : "rejected"} “${row.term}”`;
      }
      done += 1;
      at += 1;
      onreviewed?.();
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally { busy = false; }
  }

  function skip() { if (at < list.length) { at += 1; last = "skipped"; } }
  function swap() { queue = queue === "tags" ? "proposals" : "tags"; at = 0; }

  // Escape is NOT handled here. App owns it on the window and dismisses the
  // topmost layer, which is how it stays correct regardless of where focus
  // sits (see the comment on App.key).
  function key(e: KeyboardEvent) {
    if (!open) return;
    const el = e.target as HTMLElement | null;
    if (/^(INPUT|TEXTAREA)$/.test(el?.tagName ?? "")) return;
    const k = e.key.toLowerCase();
    if (k === "a") { e.preventDefault(); act("accept"); }
    else if (k === "r") { e.preventDefault(); act("reject"); }
    else if (e.key === "ArrowRight") { e.preventDefault(); skip(); }
    else if (e.key === "Tab") { e.preventDefault(); swap(); }
  }
</script>

<svelte:window onkeydown={key} />

{#if open}
  <div class="scrim" onclick={() => (open = false)} role="presentation"></div>
  <div class="review" role="dialog" aria-modal="true" aria-label="review pending tags">
    <div class="head meta">
      <button class:on={queue === "tags"} onclick={() => { queue = "tags"; at = 0; }}>
        tags {tags.length}
      </button>
      <button class:on={queue === "proposals"} onclick={() => { queue = "proposals"; at = 0; }}>
        proposals {proposals.length}
      </button>
      <span class="spacer"></span>
      {#if done}<span class="did">{done} reviewed</span>{/if}
      <span class="hint">a accept · r reject · → skip · tab switches</span>
      <button class="x" onclick={() => (open = false)} aria-label="close">×</button>
    </div>

    {#if err}
      <p class="err">{err}</p>
    {:else if busy && !current}
      <p class="empty meta">loading…</p>
    {:else if !current}
      <p class="empty meta">
        {list.length ? "Queue done — nothing left to review." : "Nothing pending here."}
      </p>
    {:else if queue === "tags"}
      <div class="body">
        <div class="shot">
          {#if detail?.thumb_url}
            <img src={detail.thumb_url} alt="" />
          {:else}
            <div class="blank"></div>
          {/if}
        </div>
        <div class="side">
          <p class="caption">{detail?.caption ?? detail?.preview ?? ""}</p>
          <!-- The one place a pending tag takes the dashed Copper outline:
               this surface is about acting on tags, so the hit target has
               to be visible (DESIGN.md). -->
          <div class="proposed">
            <span class="pending">{(current as ReviewTag).term}</span>
            <span class="conf">{(current as ReviewTag).confidence.toFixed(2)}</span>
          </div>
          {#if detail?.tags?.length}
            <p class="already meta">already on this item</p>
            <div class="have">
              {#each detail.tags.filter((t: any) => t.state === "accepted") as t}
                <span class="accepted">{t.term}</span>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {:else}
      <div class="body proposal">
        <div class="side">
          <p class="already meta">proposed vocabulary term</p>
          <div class="proposed">
            <span class="pending big">{(current as ReviewProposal).term}</span>
          </div>
          <p class="caption">
            Accepting adds it to the controlled vocabulary and embeds it, so it
            can be matched and searched. Rejecting retires it.
          </p>
        </div>
      </div>
    {/if}

    <div class="foot meta">
      <span>{remaining} left</span>
      {#if last}<span class="last">{last}</span>{/if}
      <span class="spacer"></span>
      <button onclick={() => act("reject")} disabled={!current || busy}>reject</button>
      <button onclick={skip} disabled={!current}>skip</button>
      <button class="go" onclick={() => act("accept")} disabled={!current || busy}>accept</button>
    </div>
  </div>
{/if}

<style>
  .scrim {
    position: fixed; inset: 0; z-index: 60;
    background: color-mix(in srgb, var(--bg) 70%, transparent);
  }
  .review {
    position: fixed; top: 10vh; left: 50%; transform: translateX(-50%);
    width: min(820px, calc(100vw - var(--inset) * 2)); z-index: 61;
    background: var(--surface);
    border: 1px solid var(--keyline);
    border-radius: var(--r-panel); overflow: hidden;
  }
  .head, .foot {
    display: flex; align-items: center; gap: 8px; padding: 12px 14px;
  }
  .head { border-bottom: 1px solid var(--border); }
  .foot { border-top: 1px solid var(--border); }
  .spacer { flex: 1; }
  .hint, .did { color: var(--muted); font-size: var(--fs-xs); letter-spacing: var(--ls-12); }

  .head button, .foot button {
    background: none; border: 0; border-radius: var(--r-pill);
    color: var(--muted); padding: 5px 12px; cursor: pointer;
    font-family: var(--font-content); font-size: var(--fs-xs);
    font-weight: 500; letter-spacing: var(--ls-ctl);
    transition: color 140ms var(--ease), background 140ms var(--ease);
  }
  .head button:hover, .foot button:hover { color: var(--text); }
  /* The active chip fills with --border, NOT --steel. The light theme
     redefines --text but leaves --steel at its dark #333336, so
     `color: var(--text); background: var(--steel)` renders near-black on
     dark grey - about 1.3:1 - the moment the theme flips. --border is
     theme-resolved (#d2d2d7 light, #333336 dark) and pairs with --text at
     roughly 12:1 in both. */
  .head .on { color: var(--text); background: var(--border); }
  .x { font-size: var(--fs-md); line-height: 1; padding: 2px 8px; }
  /* The single filled control, matching the one blue action elsewhere. */
  .foot .go { background: var(--blue); color: #fff; }
  .foot button:disabled { opacity: 0.4; cursor: default; }

  .body {
    display: grid; grid-template-columns: 300px 1fr;
    gap: var(--gutter); padding: var(--gutter);
  }
  .body.proposal { grid-template-columns: 1fr; }
  .shot img, .shot .blank {
    width: 100%; aspect-ratio: 4 / 3; object-fit: cover;
    border-radius: var(--r-tile); border: 1px solid var(--border);
    display: block;
  }
  .shot .blank { background: var(--steel); }
  .side { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  .caption {
    margin: 0; color: var(--text); font-size: var(--fs-sm);
    line-height: 1.47; letter-spacing: var(--ls-17);
  }
  .already { margin: 0; color: var(--muted); font-size: var(--fs-xs); }
  .proposed { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

  /* DESIGN.md: pending tags are Copper; on this surface alone they also
     take a 1px dashed outline at the pill radius, because here a tag is a
     target rather than an annotation.

     --warn, never --copper. Copper #b64400 is the WHITE-surface value and
     measures 3.06:1 on Carbon; --warn resolves to #ef6c1a on dark (5.47:1)
     and back to #b64400 on light. This is read as text, so it has to pass. */
  .pending {
    color: var(--warn);
    border: 1px dashed var(--warn);
    border-radius: var(--r-pill);
    padding: 5px 14px; font-size: var(--fs-sm); font-weight: 400;
    letter-spacing: var(--ls-17);
  }
  .pending.big { font-size: var(--fs-lg); padding: 8px 20px; }
  .conf { color: var(--muted); font-size: var(--fs-xs); }
  .have { display: flex; flex-wrap: wrap; gap: 6px; }
  .accepted {
    color: var(--muted); font-size: var(--fs-xs); font-weight: 400;
    border: 1px solid var(--border); border-radius: var(--r-pill);
    padding: 3px 10px;
  }
  .empty, .err { padding: 28px var(--gutter); margin: 0; color: var(--muted); }
  .err { color: var(--warn); }
  .last { color: var(--muted); font-size: var(--fs-xs); }

  /* Below 660px the canvas drops to one column and so does this: the
     thumbnail stacks above the decision rather than squeezing beside it. */
  @media (max-width: 660px) {
    .body { grid-template-columns: 1fr; }
    .hint { display: none; }
  }
</style>
