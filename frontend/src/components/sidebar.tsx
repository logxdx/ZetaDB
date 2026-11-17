import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { FolderPlus, Laptop, Library, Moon, MoreHorizontal, Plus, Sun, Trash2, Upload } from "lucide-react"
import { useTheme } from "next-themes"

import type { Collection, Status } from "@/lib/api"
import { Activity } from "@/components/activity"
import type { Job } from "@/lib/api"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

interface SidebarProps {
  collections: Collection[]
  active: string
  status: Status | null
  jobs: Job[]
  onSelect: (name: string) => void
  onCreate: (name: string) => Promise<boolean>
  onDelete: (name: string) => void
  onUpload: () => void
  onIndexFolder: () => void
}

export function Sidebar({
  collections,
  active,
  status,
  jobs,
  onSelect,
  onCreate,
  onDelete,
  onUpload,
  onIndexFolder,
}: SidebarProps) {
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState("")

  return (
    <div className="flex h-full flex-col">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 pt-5 pb-6">
        <div className="relative grid size-8 place-items-center overflow-hidden rounded-[9px] bg-[conic-gradient(from_210deg,#0a84ff,#bf5af2,#ff375f,#ff9f0a,#0a84ff)] shadow-sm">
          <span className="font-display text-[17px] font-bold text-white drop-shadow">ζ</span>
        </div>
        <div className="leading-tight">
          <p className="text-[15px] font-semibold tracking-tight">Zeta</p>
          <p className="text-[11px] text-muted-foreground">Multimodal search</p>
        </div>
      </div>

      {/* Collections */}
      <div className="flex items-center justify-between px-4 pb-1.5">
        <p className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">Collections</p>
        <button
          onClick={() => setCreating(true)}
          aria-label="New collection"
          className="grid size-5 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
        >
          <Plus className="size-3.5" />
        </button>
      </div>

      <nav className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto px-2.5">
        {collections.map((c) => {
          const selected = c.name === active
          return (
            <div key={c.name} className="group/item relative">
              <button
                onClick={() => onSelect(c.name)}
                className={cn(
                  "relative flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-left text-[13.5px] transition-colors",
                  selected ? "text-foreground" : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60",
                )}
              >
                {selected && (
                  <motion.span
                    layoutId="sidebar-selection"
                    className="absolute inset-0 rounded-lg bg-sidebar-accent"
                    transition={{ type: "spring", stiffness: 500, damping: 38 }}
                  />
                )}
                <Library className={cn("relative size-4 shrink-0", selected ? "text-primary" : "text-muted-foreground")} />
                <span className="relative flex-1 truncate font-medium">{c.name}</span>
                <span className="relative text-[11px] tabular-nums text-muted-foreground transition-opacity group-hover/item:opacity-0">
                  {c.error ? "!" : c.total.toLocaleString()}
                </span>
              </button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    aria-label={`Options for ${c.name}`}
                    className="absolute top-1/2 right-1.5 grid size-6 -translate-y-1/2 place-items-center rounded-md text-muted-foreground opacity-0 transition-opacity group-hover/item:opacity-100 hover:bg-sidebar-accent hover:text-foreground focus-visible:opacity-100 data-[state=open]:opacity-100"
                  >
                    <MoreHorizontal className="size-4" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" side="right" className="w-48">
                  <DropdownMenuItem onSelect={() => { onSelect(c.name); onUpload() }}>
                    <Upload /> Add files…
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => { onSelect(c.name); onIndexFolder() }}>
                    <FolderPlus /> Index a folder…
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={() => onDelete(c.name)}>
                    <Trash2 /> Delete collection
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )
        })}

        <AnimatePresence>
          {creating && (
            <motion.form
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25, ease: [0.32, 0.72, 0, 1] }}
              onSubmit={async (e) => {
                e.preventDefault()
                if (!name.trim()) return setCreating(false)
                if (await onCreate(name.trim())) {
                  setName("")
                  setCreating(false)
                }
              }}
              className="overflow-hidden"
            >
              <div className="flex items-center gap-2.5 rounded-lg bg-sidebar-accent px-2.5 py-[5px]">
                <Library className="size-4 shrink-0 text-primary" />
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onBlur={() => !name && setCreating(false)}
                  onKeyDown={(e) => e.key === "Escape" && (setName(""), setCreating(false))}
                  placeholder="Collection name"
                  className="w-full bg-transparent py-0.5 text-[13.5px] font-medium outline-none placeholder:text-muted-foreground/70"
                />
              </div>
            </motion.form>
          )}
        </AnimatePresence>

        {collections.length === 0 && !creating && (
          <p className="px-2.5 py-2 text-[13px] text-muted-foreground">No collections yet.</p>
        )}
      </nav>

      {/* Footer */}
      <div className="space-y-2 p-2.5">
        <Activity jobs={jobs} />
        <div className="flex items-center gap-1.5 rounded-xl px-1.5 py-1">
          <ModelStatus status={status} />
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}

function ModelStatus({ status }: { status: Status | null }) {
  const state = status?.state ?? "loading"
  const label =
    state === "ready"
      ? `${status!.device.toUpperCase()} · ${status!.dimension}d`
      : state === "error"
        ? "Model failed to load"
        : "Warming up model…"
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2" title={status?.error ?? status?.model}>
      <span className="relative flex size-2 shrink-0">
        {state !== "ready" && state !== "error" && (
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-amber-400 opacity-60" />
        )}
        <span
          className={cn(
            "relative inline-flex size-2 rounded-full",
            state === "ready" ? "bg-emerald-500" : state === "error" ? "bg-destructive" : "bg-amber-400",
          )}
        />
      </span>
      <div className="min-w-0 leading-tight">
        <p className="truncate text-[12px] font-medium">EmbeddingGemma 2</p>
        <p className="truncate text-[10.5px] text-muted-foreground">{label}</p>
      </div>
    </div>
  )
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const next = theme === "light" ? "dark" : theme === "dark" ? "system" : "light"
  const Icon = theme === "light" ? Sun : theme === "dark" ? Moon : Laptop
  return (
    <button
      onClick={() => setTheme(next)}
      aria-label={`Theme: ${theme}. Switch to ${next}`}
      title={`Theme: ${theme}`}
      className="grid size-8 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground"
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          initial={{ rotate: -90, scale: 0.5, opacity: 0 }}
          animate={{ rotate: 0, scale: 1, opacity: 1 }}
          exit={{ rotate: 90, scale: 0.5, opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <Icon className="size-4" />
        </motion.span>
      </AnimatePresence>
    </button>
  )
}
