export type Sort = "date" | "title" | "tags";
export type RelMode = "similar" | "tag";

/** Retrievers are named for the machinery inside; nobody reading results
 *  cares that a hit came from `img_vec`. These are what a person calls it. */
export const WHY: Record<string, string> = {
  tag: "tagged",
  lexical: "text match",
  text_vec: "similar wording",
  img_vec: "looks similar",
};
export const whyLabel = (k: string) => WHY[k] ?? k;

export type Tag = { term: string; confidence: number; source: "human" | "machine"; state: string };

export type ReviewTag = {
  item_id: number; term_id: number; term: string;
  confidence: number; kind: "note" | "image"; title: string | null;
};
export type ReviewProposal = { id: number; term: string; created_at: number };

export type Item = {
  id: number; kind: "note" | "image"; status: string;
  title: string | null; preview: string; caption: string | null;
  caption_model: string | null; width: number | null; height: number | null;
  thumb_url?: string; file_url?: string;
  near_dup_of: number | null; project_id: number | null;
  created_at: number; updated_at: number;
  tags: Tag[];
  score?: number; why?: string[]; ranks?: Record<string, number>;
  shared?: string[];
  link?: string; weight?: number;
};

export type ItemDetail = Item & {
  body: string | null; layout_text: string | null; flow_text: string | null;
  boxes: unknown[] | null;
  entities: { kind: string; value: string; confidence: number }[];
  todos: { id: number; text: string; due_date: string | null; done: number }[];
  errors: { stage: string; error: string; ts: number }[];
  near_dups: number[];
};

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
  return r.json() as Promise<T>;
}

export const api = {
  items: (cursor?: number | null, limit = 60, sort: Sort = "date") =>
    j<{ items: Item[]; next_cursor: number | null; sort: Sort }>(
      `/api/items?limit=${limit}&sort=${sort}${cursor ? `&cursor=${cursor}` : ""}`),
  item: (id: number) => j<ItemDetail>(`/api/items/${id}`),
  search: (q: string, group = false) =>
    j<{ query: string; seeds: { term: string; similarity: number }[];
        retrievers: Record<string, number>; results: Item[] }>(
      `/api/search?q=${encodeURIComponent(q)}${group ? "&group=true" : ""}`),
  related: (id: number, mode: RelMode = "similar", limit = 80) =>
    j<{ item_id: number; mode: RelMode; related: Item[] }>(
      `/api/items/${id}/related?mode=${mode}&limit=${limit}`),
  graph: (id: number, hops = 2) =>
    j<{ focus: number; nodes: any[]; edges: any[] }>(`/api/items/${id}/graph?hops=${hops}`),
  createNote: (body: string) => {
    const f = new FormData(); f.append("body", body);
    return j<{ created: any[] }>("/api/items", { method: "POST", body: f });
  },
  upload: (files: File[]) => {
    const f = new FormData();
    for (const file of files) f.append("files", file);
    return j<{ created: any[] }>("/api/items", { method: "POST", body: f });
  },
  remove: (id: number) => j(`/api/items/${id}`, { method: "DELETE" }),
  queue: () => j<{ depths: Record<string, number>; total: number }>("/api/queue"),
  review: (limit = 200) =>
    j<{ tags: ReviewTag[]; proposals: ReviewProposal[] }>(`/api/review?limit=${limit}`),
  reviewTag: (itemId: number, termId: number, action: "accept" | "reject") =>
    j<{ item_id: number; term_id: number; action: string }>(
      `/api/review/tag/${itemId}/${termId}/${action}`, { method: "POST" }),
  reviewProposal: (termId: number, action: "accept" | "reject") =>
    j<{ term_id: number; state: string }>(
      `/api/review/proposal/${termId}/${action}`, { method: "POST" }),
};
