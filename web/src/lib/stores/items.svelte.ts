import { api, type Item, type RelMode, type Sort } from "../api";

/** Canvas state. One store drives both the stream and search results:
 *  they are the same masonry surface, differently ordered. */
class ItemStore {
  items = $state<Item[]>([]);
  loading = $state(false);
  done = $state(false);
  mode = $state<"stream" | "search" | "related">("stream");
  relatedTo = $state<Item | null>(null);
  relMode = $state<RelMode>("similar");
  sort = $state<Sort>("date");
  query = $state("");
  seeds = $state<{ term: string; similarity: number }[]>([]);
  retrievers = $state<Record<string, number>>({});
  queueDepth = $state(0);
  error = $state<string | null>(null);
  private cursor: number | null = null;

  async loadStream(reset = false) {
    if (this.loading) return;
    if (reset) { this.cursor = null; this.done = false; this.items = []; this.relatedTo = null; }
    if (this.done) return;
    this.loading = true; this.error = null;
    try {
      const r = await api.items(this.cursor, 60, this.sort);
      this.items = reset ? r.items : [...this.items, ...r.items];
      this.cursor = r.next_cursor;
      this.done = r.next_cursor === null;
      this.mode = "stream";
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally { this.loading = false; }
  }

  async setSort(sort: Sort) {
    if (sort === this.sort) return;
    this.sort = sort;
    await this.loadStream(true);
  }

  /** Enter (or re-query) the related view for one item. */
  async showRelated(item: Item, relMode: RelMode = this.relMode) {
    this.loading = true; this.error = null;
    try {
      const r = await api.related(item.id, relMode);
      this.relatedTo = item;
      this.relMode = relMode;
      this.items = r.related;
      this.mode = "related";
      this.done = true;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally { this.loading = false; }
  }

  async setRelMode(relMode: RelMode) {
    if (this.relatedTo && relMode !== this.relMode) await this.showRelated(this.relatedTo, relMode);
  }

  exitRelated() { this.relatedTo = null; this.loadStream(true); }

  async search(q: string) {
    this.query = q;
    if (!q.trim()) return this.loadStream(true);
    this.relatedTo = null;
    this.loading = true; this.error = null;
    try {
      const r = await api.search(q);
      this.items = r.results;
      this.seeds = r.seeds;
      this.retrievers = r.retrievers;
      this.mode = "search";
      this.done = true;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally { this.loading = false; }
  }

  /** Refresh one card in place when SSE says it changed. */
  async refresh(id: number) {
    const i = this.items.findIndex((x) => x.id === id);
    if (i < 0) return;
    try {
      const fresh = await api.item(id);
      this.items[i] = { ...this.items[i], ...fresh };
    } catch { /* deleted underneath us */ }
  }

  async prepend(id: number) {
    if (this.mode !== "stream" || this.items.some((x) => x.id === id)) return;
    try {
      const fresh = await api.item(id);
      if (fresh.near_dup_of === null) this.items = [fresh as Item, ...this.items];
    } catch { /* ignore */ }
  }

  drop(id: number) { this.items = this.items.filter((x) => x.id !== id); }
}

export const store = new ItemStore();

/** Live updates. Cards appear at item.created and fill in as stages land. */
export function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    let msg: { event: string; data: any };
    try { msg = JSON.parse(m.data); } catch { return; }
    switch (msg.event) {
      case "item.created": store.prepend(msg.data.id); break;
      case "item.updated":
      case "item.stage":   store.refresh(msg.data.id); break;
      case "item.deleted": store.drop(msg.data.id); break;
      case "queue.changed": store.queueDepth = msg.data.total ?? 0; break;
    }
  };
  return () => es.close();
}
