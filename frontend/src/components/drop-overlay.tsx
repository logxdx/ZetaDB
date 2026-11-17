import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ScanSearch, UploadCloud } from "lucide-react"

import { cn } from "@/lib/utils"

type Zone = "add" | "search"

/**
 * Drag files anywhere onto the window: drop on the left to add them to the
 * collection, or on the right to search with the first file.
 */
export function DropOverlay({
  collection,
  onAdd,
  onSearch,
}: {
  collection: string
  onAdd: (files: File[]) => void
  onSearch: (file: File) => void
}) {
  const [active, setActive] = useState(false)
  const [zone, setZone] = useState<Zone>("add")
  const depth = useRef(0)

  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files")
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth.current++
      setActive(true)
    }
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth.current = Math.max(0, depth.current - 1)
      if (depth.current === 0) setActive(false)
    }
    const onOver = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      setZone(e.clientX > window.innerWidth / 2 ? "search" : "add")
    }
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth.current = 0
      setActive(false)
      const files = Array.from(e.dataTransfer?.files ?? [])
      if (!files.length) return
      if (e.clientX > window.innerWidth / 2) onSearch(files[0])
      else onAdd(files)
    }
    window.addEventListener("dragenter", onEnter)
    window.addEventListener("dragleave", onLeave)
    window.addEventListener("dragover", onOver)
    window.addEventListener("drop", onDrop)
    return () => {
      window.removeEventListener("dragenter", onEnter)
      window.removeEventListener("dragleave", onLeave)
      window.removeEventListener("dragover", onOver)
      window.removeEventListener("drop", onDrop)
    }
  }, [onAdd, onSearch])

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="pointer-events-none fixed inset-0 z-[60] grid grid-cols-2 gap-4 bg-background/70 p-6 backdrop-blur-xl"
        >
          <DropZone
            active={zone === "add"}
            icon={<UploadCloud className="size-9" />}
            title={`Add to ${collection}`}
            subtitle="Photos, videos, audio and text files are embedded with EmbeddingGemma 2"
          />
          <DropZone
            active={zone === "search"}
            icon={<ScanSearch className="size-9" />}
            title="Search with this"
            subtitle="Find anything that looks, sounds or reads like it"
          />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function DropZone({
  active,
  icon,
  title,
  subtitle,
}: {
  active: boolean
  icon: React.ReactNode
  title: string
  subtitle: string
}) {
  return (
    <motion.div
      animate={{ scale: active ? 1 : 0.97, opacity: active ? 1 : 0.55 }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      className={cn(
        "flex flex-col items-center justify-center gap-4 rounded-3xl border-2 border-dashed text-center transition-colors duration-300",
        active ? "border-primary bg-primary/[0.07]" : "border-foreground/15",
      )}
    >
      <motion.div
        animate={{ y: active ? -4 : 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 18, repeat: active ? Infinity : 0, repeatType: "reverse" }}
        className={cn(
          "grid size-20 place-items-center rounded-[22px] transition-colors",
          active ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30" : "bg-foreground/[0.06] text-muted-foreground",
        )}
      >
        {icon}
      </motion.div>
      <div className="px-6">
        <p className="text-xl font-semibold">{title}</p>
        <p className="mt-1 text-sm text-balance text-muted-foreground">{subtitle}</p>
      </div>
    </motion.div>
  )
}
