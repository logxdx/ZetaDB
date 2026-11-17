export type Modality = "image" | "video" | "audio" | "text"

export const MODALITIES: Modality[] = ["image", "video", "audio", "text"]

export interface Item {
  id: string
  modality: Modality
  path: string
  name: string
  text: string
  size: number
  width: number
  height: number
  duration: number
  added_at: string | null
  /** Search only: cosine similarity of the best-matching segment */
  score?: number
  /** Search only: combined ranking score (per-modality z-score + keyword bonus) */
  relevance?: number
  /** Search only: false for "weaker matches" */
  strong?: boolean
  /** Search only: where in the file it matched */
  match?: Match
}

export interface Match {
  kind: "chunk" | "image" | "sound" | "frames"
  start: number
  end: number
  snippet: string
  /** Fraction of query words found in the file */
  words: number
}

export interface Collection {
  name: string
  total: number
  counts: Partial<Record<Modality, number>>
  error?: string
}

export interface Status {
  state: "idle" | "loading" | "ready" | "error"
  error: string | null
  model: string
  device: string
  dimension: number
  modalities: Modality[]
  extensions: Record<Modality, string[]>
  default_collection: string
  default_top_k: number
}

export interface Job {
  id: string
  collection: string
  source: string
  total: number
  done: number
  indexed: number
  skipped: number
  failed: number
  status: "queued" | "running" | "completed" | "failed"
  current: string
  errors: string[]
  started_at: number
  finished_at: number | null
  elapsed: number
  rate: number
  eta: number | null
  rejected?: string[]
}

/** Task prompts from the EmbeddingGemma 2 developer guide (applied to text queries). */
export type Intent = "search" | "question" | "code" | "fact" | "similarity"
export type IntentChoice = "auto" | Intent

export const INTENTS: { value: IntentChoice; label: string; prompt: string; hint: string }[] = [
  { value: "auto", label: "Auto", prompt: "", hint: "Question prompt for questions over notes, search otherwise" },
  { value: "search", label: "Search", prompt: "task: search result | query:", hint: "Find photos, videos, sounds or passages" },
  { value: "question", label: "Question", prompt: "task: question answering | query:", hint: "Find passages that answer a question" },
  { value: "code", label: "Code", prompt: "task: code retrieval | query:", hint: "Find source files from a description" },
  { value: "fact", label: "Fact check", prompt: "task: fact checking | query:", hint: "Find evidence for or against a claim" },
  { value: "similarity", label: "Similar text", prompt: "task: sentence similarity | query:", hint: "Find text that says the same thing" },
]

export function resolveIntent(choice: IntentChoice, text: string, modalities: Modality[]): Intent {
  if (choice !== "auto") return choice
  const textOnly = modalities.length === 1 && modalities[0] === "text"
  return textOnly && text.trim().endsWith("?") ? "question" : "search"
}

export interface SearchQuery {
  collection: string
  topK: number
  modalities: Modality[]
  text?: string
  file?: File | null
  itemId?: string | null
  intent?: Intent
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not json */
    }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
})

export const api = {
  status: () => request<Status>("/api/status"),
  collections: () => request<Collection[]>("/api/collections"),
  createCollection: (name: string) => request<Collection>("/api/collections", json({ name })),
  deleteCollection: (name: string) =>
    request<{ ok: true }>(`/api/collections/${encodeURIComponent(name)}`, { method: "DELETE" }),
  items: (collection: string, modalities: Modality[], signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: "1000" })
    if (modalities.length) params.set("modality", modalities.join(","))
    return request<Item[]>(`/api/collections/${encodeURIComponent(collection)}/items?${params}`, { signal })
  },
  deleteItem: (collection: string, id: string) =>
    request<{ ok: true }>(`/api/collections/${encodeURIComponent(collection)}/items/${id}`, {
      method: "DELETE",
    }),
  indexFolder: (collection: string, path: string) => request<Job>("/api/index/folder", json({ collection, path })),
  upload: (collection: string, files: File[]) => {
    const form = new FormData()
    form.set("collection", collection)
    files.forEach((f) => form.append("files", f, f.name))
    return request<Job>("/api/index/upload", { method: "POST", body: form })
  },
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobs: () => request<Job[]>("/api/jobs"),
  search: (q: SearchQuery, signal?: AbortSignal) => {
    const form = new FormData()
    form.set("collection", q.collection)
    form.set("top_k", String(q.topK))
    if (q.modalities.length) form.set("modalities", q.modalities.join(","))
    if (q.text?.trim()) form.set("text", q.text.trim())
    if (q.file) form.set("file", q.file, q.file.name)
    if (q.itemId) form.set("item_id", q.itemId)
    if (q.intent) form.set("intent", q.intent)
    return request<Item[]>("/api/search", { method: "POST", body: form, signal })
  },
}

export const mediaUrl = (collection: string, id: string) =>
  `/api/media/${encodeURIComponent(collection)}/${encodeURIComponent(id)}`

export function modalityOf(file: File, extensions?: Record<Modality, string[]>): Modality | null {
  const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase()
  if (extensions) {
    for (const m of MODALITIES) if (extensions[m]?.includes(ext)) return m
    return null
  }
  const kind = file.type.split("/")[0]
  if (kind === "image" || kind === "video" || kind === "audio") return kind
  if (kind === "text") return "text"
  return null
}
