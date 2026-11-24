export type Sort = "date" | "title" | "tags";
export type RelMode = "similar" | "tag";

/** Retrievers are named for the machinery inside; nobody reading results
 *  cares that a hit came from `img_vec`. These are what a person calls it. */
export const WHY: Record<string, string> = {
  tag: "tagged",
  lexical: "text match",
  text_vec: "similar wording",
  img_vec: "looks similar",
  // A match inside an audio or video file. Per item it resolves to what
  // kind of window matched (see whyLabel); this is the aggregate wording.
  moment: "scene / heard",
};
/** `moment` names a matched window, so with the item in hand it says which
 *  kind: a video scene, or something heard on a soundtrack. */
export const whyLabel = (k: string, it?: { moment?: Moment | null }) =>
  k === "moment" && it?.moment
    ? (it.moment.kind === "frames" ? "scene" : "heard")
    : (WHY[k] ?? k);

/** Seconds as m:ss, or h:mm:ss past the hour. */
export function clock(sec: number | null | undefined): string {
  const t = Math.max(0, Math.floor(sec ?? 0));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export type Kind = "note" | "image" | "audio" | "video";
/** Audio and video play over time: a player in focus, a duration on the tile. */
export const isTimed = (k: Kind) => k === "audio" || k === "video";

/** One window an audio/video file was embedded in: 30s sound windows, or a
 *  10s video window whose keyframe is `frame_url`. */
export type Segment = {
  id: number; kind: "sound" | "frames";
  start: number; end: number; frame_url: string | null;
};
/** WHERE in an audio/video file a search matched. */
export type Moment = {
  segment_id: number; kind: "sound" | "frames";
  start: number; end: number; frame_url: string | null; score: number;
};

export type Tag = { term: string; confidence: number; source: "human" | "machine"; state: string };

export type ReviewTag = {
  item_id: number; term_id: number; term: string;
  confidence: number; kind: Kind; title: string | null;
};
export type ReviewProposal = { id: number; term: string; created_at: number };

export type Item = {
  id: number; kind: Kind; status: string;
  title: string | null; preview: string; caption: string | null;
  caption_model: string | null; width: number | null; height: number | null;
  /** Image: the picture. Video: a poster frame. Audio: a drawn waveform. */
  thumb_url?: string;
  /** Image, audio and video; the endpoint honours Range, so media can seek. */
  file_url?: string;
  /** Seconds; audio and video only. */
  duration?: number | null;
  near_dup_of: number | null; project_id: number | null;
  created_at: number; updated_at: number;
  tags: Tag[];
  score?: number; why?: string[]; ranks?: Record<string, number>;
  /** Search only: 0..1, how close this result is to the query's best match
   *  (server: search.hybrid.match_strength). The canvas dims by it. */
  match?: number;
  moment?: Moment | null;
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
  segments: Segment[];
};

export type Health = {
  status: string;
  engine: {
    budget_mb: number; free_mb: number; held: unknown; resident: unknown;
    models: Record<string, { pool: string; backend: string; provider: string; loaded: boolean }>;
  };
  [k: string]: unknown;
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
  health: () => j<Health>("/api/health"),
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
