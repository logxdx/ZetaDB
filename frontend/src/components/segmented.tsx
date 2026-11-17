import { motion } from "motion/react"
import { LayoutGrid } from "lucide-react"

import type { Collection, Modality } from "@/lib/api"
import { MODALITIES } from "@/lib/api"
import { MODALITY_META } from "@/lib/format"
import { cn } from "@/lib/utils"

export type Filter = "all" | Modality

/** iOS-style segmented control with a sliding thumb. */
export function Segmented({
  value,
  onChange,
  counts,
}: {
  value: Filter
  onChange: (v: Filter) => void
  counts?: Collection["counts"]
}) {
  const options: { value: Filter; label: string; icon: typeof LayoutGrid; count?: number }[] = [
    { value: "all", label: "All", icon: LayoutGrid },
    ...MODALITIES.map((m) => ({
      value: m as Filter,
      label: MODALITY_META[m].plural,
      icon: MODALITY_META[m].icon,
      count: counts?.[m],
    })),
  ]
  return (
    <div role="radiogroup" aria-label="Filter by type" className="no-scrollbar flex max-w-full overflow-x-auto">
      <div className="inline-flex shrink-0 rounded-[11px] bg-foreground/[0.055] p-[3px]">
        {options.map((o) => {
          const selected = o.value === value
          const Icon = o.icon
          return (
            <button
              key={o.value}
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(o.value)}
              className={cn(
                "relative flex h-8 items-center gap-1.5 rounded-[8px] px-3 text-[13px] font-medium whitespace-nowrap transition-colors duration-200",
                selected ? "text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {selected && (
                <motion.span
                  layoutId="segmented-thumb"
                  className="absolute inset-0 rounded-[8px] bg-background shadow-[0_1px_2px_rgb(0_0_0/0.08),0_3px_8px_rgb(0_0_0/0.06)] dark:bg-white/[0.14]"
                  transition={{ type: "spring", stiffness: 520, damping: 40 }}
                />
              )}
              <Icon
                className="relative size-3.5"
                style={{ color: selected && o.value !== "all" ? MODALITY_META[o.value as Modality].color : undefined }}
              />
              <span className="relative">{o.label}</span>
              {o.count !== undefined && o.count > 0 && (
                <span className="relative text-[11px] tabular-nums text-muted-foreground">{o.count}</span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
