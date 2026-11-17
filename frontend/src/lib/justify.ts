import type { Item } from "./api"

export interface LaidOutItem {
  item: Item
  width: number
  height: number
}

export interface Row {
  items: LaidOutItem[]
  height: number
}

/** Aspect ratio (w/h) used to lay an item out. */
export function aspectOf(item: Item): number {
  if ((item.modality === "image" || item.modality === "video") && item.width && item.height) {
    // Clamp extreme panoramas / slivers so rows stay balanced
    return Math.min(Math.max(item.width / item.height, 0.5), 2.6)
  }
  switch (item.modality) {
    case "video":
      return 16 / 9
    case "audio":
      return 1
    case "text":
      return 0.82
    default:
      return 4 / 3
  }
}

/**
 * Justified ("Google Photos") layout: fill rows at roughly ``targetHeight``,
 * then scale each row so it exactly spans ``containerWidth``.
 */
export function justify(items: Item[], containerWidth: number, targetHeight: number, gap: number): Row[] {
  if (containerWidth <= 0) return []
  const rows: Row[] = []
  let current: Item[] = []
  let ratioSum = 0

  const flush = (isLast: boolean) => {
    if (!current.length) return
    const available = containerWidth - gap * (current.length - 1)
    let height = available / ratioSum
    // Don't blow up a short last row
    if (isLast && height > targetHeight * 1.15) height = targetHeight
    rows.push({
      height,
      items: current.map((item) => ({ item, height, width: aspectOf(item) * height })),
    })
    current = []
    ratioSum = 0
  }

  for (const item of items) {
    current.push(item)
    ratioSum += aspectOf(item)
    const rowWidth = ratioSum * targetHeight + gap * (current.length - 1)
    if (rowWidth >= containerWidth) flush(false)
  }
  flush(true)
  return rows
}
