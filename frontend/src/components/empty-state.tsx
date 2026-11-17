import { motion } from "motion/react"
import { FolderPlus, SearchX, Upload } from "lucide-react"

import { MODALITIES } from "@/lib/api"
import { MODALITY_META } from "@/lib/format"
import { Button } from "@/components/ui/button"

const float = (i: number) => ({
  initial: { opacity: 0, y: 20, scale: 0.8 },
  animate: { opacity: 1, y: 0, scale: 1 },
  transition: { type: "spring" as const, stiffness: 260, damping: 20, delay: 0.08 * i },
})

export function EmptyLibrary({ onUpload, onIndexFolder }: { onUpload: () => void; onIndexFolder: () => void }) {
  const tilt = [-10, -3, 4, 11]
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="relative mb-10 flex h-28 items-end">
        {MODALITIES.map((m, i) => {
          const meta = MODALITY_META[m]
          const Icon = meta.icon
          return (
            <motion.div
              key={m}
              {...float(i)}
              whileHover={{ y: -8, rotate: 0, scale: 1.05 }}
              style={{ rotate: tilt[i], background: `color-mix(in srgb, ${meta.color} 14%, var(--card))`, color: meta.color }}
              className="-mx-1.5 grid size-20 place-items-center rounded-[22px] shadow-float ring-1 ring-border"
            >
              <Icon className="size-8" />
            </motion.div>
          )
        })}
      </div>
      <motion.div {...float(4)}>
        <h2 className="text-[28px] font-bold">Bring everything together</h2>
        <p className="mx-auto mt-2 max-w-md text-[15px] text-balance text-muted-foreground">
          Drop photos, videos, audio and notes anywhere on this window. EmbeddingGemma 2 puts them in one space, so you
          can find any of them by describing it.
        </p>
        <div className="mt-7 flex justify-center gap-2.5">
          <Button size="lg" className="h-10 rounded-full px-5" onClick={onUpload}>
            <Upload /> Add files
          </Button>
          <Button size="lg" variant="secondary" className="h-10 rounded-full px-5" onClick={onIndexFolder}>
            <FolderPlus /> Index a folder
          </Button>
        </div>
      </motion.div>
    </div>
  )
}

export function NoResults({ query }: { query: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center py-28 text-center"
    >
      <div className="mb-5 grid size-16 place-items-center rounded-2xl bg-foreground/[0.05] text-muted-foreground">
        <SearchX className="size-7" />
      </div>
      <p className="text-lg font-semibold">No results</p>
      <p className="mt-1 text-sm text-muted-foreground">
        Nothing in this view matches {query ? `“${query}”` : "your search"}. Try another type or collection.
      </p>
    </motion.div>
  )
}

export function GridSkeleton() {
  const widths = [280, 190, 240, 320, 210, 260, 180, 300, 230, 200, 270, 250]
  return (
    <div className="flex flex-wrap gap-1.5">
      {widths.map((w, i) => (
        <div key={i} className="shimmer h-[200px] flex-grow rounded-[10px]" style={{ flexBasis: w }} />
      ))}
    </div>
  )
}
