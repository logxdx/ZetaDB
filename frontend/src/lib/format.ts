import { FileText, Image, Music2, Video, type LucideIcon } from "lucide-react"
import type { Modality } from "./api"

export const MODALITY_META: Record<
  Modality,
  { label: string; plural: string; icon: LucideIcon; color: string }
> = {
  image: { label: "Photo", plural: "Photos", icon: Image, color: "var(--modality-image)" },
  video: { label: "Video", plural: "Videos", icon: Video, color: "var(--modality-video)" },
  audio: { label: "Audio", plural: "Audio", icon: Music2, color: "var(--modality-audio)" },
  text: { label: "Note", plural: "Notes", icon: FileText, color: "var(--modality-text)" },
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "—"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** i
  return `${value >= 10 || i === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`
}

export function formatDuration(seconds: number): string {
  if (!seconds || !Number.isFinite(seconds)) return "0:00"
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, "0")
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`
}

const dayFormatter = new Intl.DateTimeFormat(undefined, { weekday: "long" })
const dateFormatter = new Intl.DateTimeFormat(undefined, { month: "long", day: "numeric" })
const fullFormatter = new Intl.DateTimeFormat(undefined, { month: "long", day: "numeric", year: "numeric" })
const timeFormatter = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" })

/** Google/Apple Photos style section header for a date. */
export function dayLabel(iso: string | null): string {
  if (!iso) return "Earlier"
  const date = new Date(iso)
  const today = new Date()
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const diff = Math.round((startOf(today) - startOf(date)) / 86_400_000)
  if (diff === 0) return "Today"
  if (diff === 1) return "Yesterday"
  if (diff < 7) return dayFormatter.format(date)
  if (date.getFullYear() === today.getFullYear()) return dateFormatter.format(date)
  return fullFormatter.format(date)
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "—"
  const date = new Date(iso)
  return `${fullFormatter.format(date)} at ${timeFormatter.format(date)}`
}

/** Deterministic hash used for generative artwork. */
export function hash(input: string): number {
  let h = 2166136261
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/** Pseudo-random waveform heights (0..1) seeded by a string. */
export function waveform(seed: string, bars: number): number[] {
  let state = hash(seed) || 1
  const out: number[] = []
  for (let i = 0; i < bars; i++) {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    const r = ((state >>> 0) % 1000) / 1000
    // Smooth envelope so it looks like real audio
    const envelope = 0.45 + 0.55 * Math.sin((Math.PI * (i + 0.5)) / bars)
    out.push(Math.max(0.12, r * envelope))
  }
  return out
}

/** Two harmonious gradient colors seeded by a string (album-art style). */
export function gradientFor(seed: string): [string, string] {
  const h = hash(seed)
  const hue = h % 360
  const hue2 = (hue + 35 + (h % 50)) % 360
  return [`oklch(0.68 0.19 ${hue})`, `oklch(0.5 0.2 ${hue2})`]
}

/** One-line explanation of why a search result matched. */
export function matchReason(match: { kind: string; start: number; end: number; words: number } | undefined, modality: string): string | null {
  if (!match) return null
  const words = match.words >= 0.5 ? " · has your words" : ""
  switch (match.kind) {
    case "frames":
      return `Matches the scene at ${formatDuration(match.start)}–${formatDuration(match.end)}${words}`
    case "sound":
      return `${modality === "video" ? "Matches what’s heard at" : "Matches the moment at"} ${formatDuration(match.start)}${words}`
    case "chunk":
      return `Matches this passage${words}`
    case "image":
      return match.words >= 0.5 ? "Has your words" : "Looks like your search"
    default:
      return null
  }
}

/** Seconds to start playback at for a match (0 for non-temporal matches). */
export function matchStart(match: { kind: string; start: number } | undefined): number {
  return match && (match.kind === "sound" || match.kind === "frames") ? match.start : 0
}

export function stem(name: string): string {
  const i = name.lastIndexOf(".")
  return i > 0 ? name.slice(0, i) : name
}

export function prettyName(name: string): string {
  // Strip the random prefix added to uploaded files
  return stem(name).replace(/^[0-9a-f]{8}-/, "").replace(/[_-]+/g, " ").trim() || name
}
