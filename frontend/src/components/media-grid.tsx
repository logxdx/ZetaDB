import { useMemo, useRef } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ChevronDown } from "lucide-react"

import type { Item } from "@/lib/api"
import { justify } from "@/lib/justify"
import { useElementWidth } from "@/hooks/use-misc"
import { Tile } from "@/components/tiles"
import { cn } from "@/lib/utils"

export interface Section {
  key: string
  title: string
  subtitle?: string
  items: Item[]
  /** Collapsible sections render a toggle and no tiles while collapsed */
  collapsed?: boolean
  onToggle?: () => void
}

const GAP = 6

function targetRowHeight(width: number) {
  if (width < 520) return 132
  if (width < 900) return 180
  if (width < 1400) return 220
  return 250
}

export function MediaGrid({
  sections,
  collection,
  showScore,
  onOpen,
}: {
  sections: Section[]
  collection: string
  showScore: boolean
  onOpen: (item: Item) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useElementWidth(ref)
  const target = targetRowHeight(width)

  const laidOut = useMemo(
    () =>
      sections.map((s) => {
        const rows = s.collapsed ? [] : justify(s.items, width, target, GAP)
        const tiles: { item: Item; x: number; y: number; w: number; h: number }[] = []
        let y = 0
        for (const row of rows) {
          let x = 0
          for (const { item, width: w, height: h } of row.items) {
            tiles.push({ item, x, y, w, h })
            x += w + GAP
          }
          y += row.height + GAP
        }
        return { ...s, tiles, height: Math.max(0, y - GAP) }
      }),
    [sections, width, target],
  )

  let index = 0
  return (
    <div ref={ref} className="w-full">
      {width > 0 &&
        laidOut.map((section) => (
          <section key={section.key} className="mb-8 last:mb-0">
            <motion.header
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: [0.32, 0.72, 0, 1] }}
              className="mb-2.5 flex items-baseline gap-2 py-1"
            >
              <h2 className="text-[17px] font-semibold">{section.title}</h2>
              {section.subtitle && <span className="text-[13px] text-muted-foreground">{section.subtitle}</span>}
              {section.onToggle && (
                <button
                  onClick={section.onToggle}
                  className="ml-auto flex items-center gap-1 rounded-full px-2.5 py-1 text-[13px] font-medium text-primary transition-colors hover:bg-primary/10"
                >
                  {section.collapsed ? "Show" : "Hide"}
                  <ChevronDown
                    className={cn("size-3.5 transition-transform duration-300", !section.collapsed && "rotate-180")}
                  />
                </button>
              )}
            </motion.header>
            <div className="relative" style={{ height: section.height }}>
              <AnimatePresence mode="popLayout">
                {section.tiles.map(({ item, x, y, w, h }) => (
                  <Tile
                    key={item.id}
                    item={item}
                    collection={collection}
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    index={index++}
                    showScore={showScore}
                    onOpen={onOpen}
                  />
                ))}
              </AnimatePresence>
            </div>
          </section>
        ))}
    </div>
  )
}
