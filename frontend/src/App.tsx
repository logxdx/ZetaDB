import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { FolderPlus, Menu, Plus, Upload } from "lucide-react"
import { toast } from "sonner"

import {
  api,
  MODALITIES,
  resolveIntent,
  type Collection,
  type IntentChoice,
  type Item,
  type Job,
  type Modality,
  type Status,
} from "@/lib/api"
import { dayLabel, MODALITY_META } from "@/lib/format"
import { useDebounced, useLocalStorage, useScrolled } from "@/hooks/use-misc"
import { Sidebar } from "@/components/sidebar"
import { SearchBar } from "@/components/search-bar"
import { Segmented, type Filter } from "@/components/segmented"
import { MediaGrid, type Section } from "@/components/media-grid"
import { Viewer } from "@/components/viewer"
import { MiniPlayer } from "@/components/audio-player"
import { DropOverlay } from "@/components/drop-overlay"
import { CommandMenu } from "@/components/command-menu"
import { IndexFolderDialog, NewCollectionDialog } from "@/components/dialogs"
import { EmptyLibrary, GridSkeleton, NoResults } from "@/components/empty-state"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { cn } from "@/lib/utils"

const ease = [0.32, 0.72, 0, 1] as const

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [collections, setCollections] = useState<Collection[]>([])
  const [active, setActive] = useLocalStorage("zeta.collection", "")
  const [filter, setFilter] = useState<Filter>("all")
  const [text, setText] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [similar, setSimilar] = useState<Item | null>(null)
  const [topK, setTopK] = useLocalStorage("zeta.topK", 36)
  const [intentChoice, setIntentChoice] = useLocalStorage<IntentChoice>("zeta.intent", "auto")
  const [library, setLibrary] = useState<Item[] | null>(null)
  const [results, setResults] = useState<Item[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [showWeak, setShowWeak] = useState(false)
  const [viewerIndex, setViewerIndex] = useState<number | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [commandOpen, setCommandOpen] = useState(false)
  const [folderOpen, setFolderOpen] = useState(false)
  const [newOpen, setNewOpen] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  const mainRef = useRef<HTMLElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const uploadRef = useRef<HTMLInputElement>(null)
  const scrolled = useScrolled(mainRef, 56)
  const debouncedText = useDebounced(text, 320)

  const modalities: Modality[] = filter === "all" ? [] : [filter]
  const intent = resolveIntent(intentChoice, debouncedText, modalities)
  const current = collections.find((c) => c.name === active)
  const ready = status?.state === "ready"
  const accept = useMemo(
    () => (status ? Object.values(status.extensions).flat().join(",") : "image/*,video/*,audio/*,text/*"),
    [status],
  )

  // ---------------------------------------------------------------- status
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>
    let cancelled = false
    const poll = async () => {
      try {
        const s = await api.status()
        if (cancelled) return
        setStatus(s)
        if (s.state === "ready" || s.state === "error") return
      } catch {
        /* server starting */
      }
      timer = setTimeout(poll, 1500)
    }
    void poll()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  // ----------------------------------------------------------- collections
  const refreshCollections = useCallback(async () => {
    try {
      let list = await api.collections()
      if (!list.length) {
        await api.createCollection("library")
        list = await api.collections()
      }
      setCollections(list)
      return list
    } catch (e) {
      toast.error("Couldn’t reach Zeta", { description: (e as Error).message })
      return []
    }
  }, [])

  useEffect(() => {
    void refreshCollections()
  }, [refreshCollections])

  useEffect(() => {
    if (collections.length && !collections.some((c) => c.name === active)) setActive(collections[0].name)
  }, [collections, active, setActive])

  // --------------------------------------------------------------- library
  const refreshLibrary = useCallback(
    async (signal?: AbortSignal) => {
      if (!active) return
      try {
        const items = await api.items(active, modalities, signal)
        setLibrary(items)
      } catch (e) {
        if ((e as Error).name !== "AbortError") setLibrary([])
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [active, filter],
  )

  useEffect(() => {
    const ctrl = new AbortController()
    setLibrary(null)
    void refreshLibrary(ctrl.signal)
    return () => ctrl.abort()
  }, [refreshLibrary])

  // ---------------------------------------------------------------- search
  const hasQuery = Boolean(debouncedText.trim() || file || similar)
  useEffect(() => {
    if (!active || !hasQuery) {
      setResults(null)
      setSearching(false)
      return
    }
    const ctrl = new AbortController()
    setSearching(true)
    api
      .search(
        { collection: active, topK, modalities, text: debouncedText, file, itemId: similar?.id, intent },
        ctrl.signal,
      )
      .then((r) => {
        setResults(r)
        setSearching(false)
      })
      .catch((e: Error) => {
        if (e.name === "AbortError") return
        setSearching(false)
        toast.error("Search failed", { description: e.message, id: "search-error" })
      })
    return () => ctrl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, debouncedText, file, similar, topK, filter, hasQuery, intent])

  // ------------------------------------------------------------------ jobs
  const seenFinished = useRef(new Set<string>())
  const anyRunning = jobs.some((j) => j.status === "running" || j.status === "queued")
  useEffect(() => {
    if (!anyRunning) return
    const t = setInterval(async () => {
      try {
        const list = await api.jobs()
        setJobs(list)
        let changed = false
        for (const j of list) {
          if ((j.status === "completed" || j.status === "failed") && !seenFinished.current.has(j.id)) {
            seenFinished.current.add(j.id)
            changed = true
            if (j.status === "failed") toast.error(`Indexing ${j.collection} failed`, { description: j.errors.at(-1) })
            else
              toast.success(`${j.indexed.toLocaleString()} item${j.indexed === 1 ? "" : "s"} added to ${j.collection}`, {
                description:
                  [j.skipped && `${j.skipped} already indexed`, j.failed && `${j.failed} couldn’t be read`]
                    .filter(Boolean)
                    .join(" · ") || undefined,
              })
          }
        }
        if (changed) {
          void refreshCollections()
          void refreshLibrary()
        }
      } catch {
        /* ignore transient errors */
      }
    }, 700)
    return () => clearInterval(t)
  }, [anyRunning, refreshCollections, refreshLibrary])

  const trackJob = (job: Job) => setJobs((prev) => [job, ...prev.filter((j) => j.id !== job.id)])

  // --------------------------------------------------------------- actions
  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!files.length || !active) return
      try {
        const job = await api.upload(active, files)
        trackJob(job)
        toast(`Adding ${job.total} file${job.total === 1 ? "" : "s"} to ${active}`, {
          description: job.rejected?.length ? `${job.rejected.length} unsupported file(s) skipped` : undefined,
        })
      } catch (e) {
        toast.error("Upload failed", { description: (e as Error).message })
      }
    },
    [active],
  )

  const searchWithFile = useCallback((f: File) => {
    setSimilar(null)
    setFile(f)
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" })
  }, [])

  const indexFolder = async (path: string) => {
    try {
      const job = await api.indexFolder(active, path)
      trackJob(job)
      toast(`Indexing ${job.total.toLocaleString()} files`, { description: path })
      return true
    } catch (e) {
      toast.error("Couldn’t index folder", { description: (e as Error).message })
      return false
    }
  }

  const createCollection = async (name: string) => {
    try {
      const c = await api.createCollection(name)
      await refreshCollections()
      setActive(c.name)
      return true
    } catch (e) {
      toast.error("Couldn’t create collection", { description: (e as Error).message })
      return false
    }
  }

  const deleteCollection = async (name: string) => {
    try {
      await api.deleteCollection(name)
      toast.success(`Deleted ${name}`)
      const list = await refreshCollections()
      if (name === active) setActive(list[0]?.name ?? "")
    } catch (e) {
      toast.error("Couldn’t delete collection", { description: (e as Error).message })
    }
  }

  const selectCollection = (name: string) => {
    setActive(name)
    setSimilar(null)
    setNavOpen(false)
    mainRef.current?.scrollTo({ top: 0 })
  }

  // ------------------------------------------------------------- sections
  const sections: Section[] = useMemo(() => {
    if (results) {
      const strong = results.filter((r) => r.strong !== false)
      const weak = results.filter((r) => r.strong === false)
      const out: Section[] = []
      if (strong.length)
        out.push({
          key: "results",
          title: similar ? "Similar items" : "Top matches",
          subtitle: `${strong.length} result${strong.length === 1 ? "" : "s"}`,
          items: strong,
        })
      if (weak.length)
        out.push({
          key: "weaker",
          title: "Weaker matches",
          subtitle: `${weak.length} more`,
          items: weak,
          collapsed: !showWeak && strong.length > 0,
          onToggle: strong.length > 0 ? () => setShowWeak((v) => !v) : undefined,
        })
      return out
    }
    if (!library) return []
    const groups = new Map<string, Item[]>()
    for (const item of library) {
      const label = dayLabel(item.added_at)
      if (!groups.has(label)) groups.set(label, [])
      groups.get(label)!.push(item)
    }
    return [...groups].map(([title, items]) => ({
      key: title,
      title,
      subtitle: `${items.length} item${items.length === 1 ? "" : "s"}`,
      items,
    }))
  }, [results, library, similar, showWeak])

  const viewerItems = useMemo(() => sections.filter((s) => !s.collapsed).flatMap((s) => s.items), [sections])

  const openItem = useCallback(
    (item: Item) => setViewerIndex(viewerItems.findIndex((i) => i.id === item.id)),
    [viewerItems],
  )

  const findSimilar = (item: Item) => {
    setViewerIndex(null)
    setText("")
    setFile(null)
    setSimilar(item)
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" })
  }

  const deleteItem = async (item: Item) => {
    try {
      await api.deleteItem(active, item.id)
      const remaining = viewerItems.length - 1
      setLibrary((l) => l?.filter((i) => i.id !== item.id) ?? null)
      setResults((r) => r?.filter((i) => i.id !== item.id) ?? null)
      setViewerIndex((idx) => (idx === null || remaining <= 0 ? null : Math.min(idx, remaining - 1)))
      if (similar?.id === item.id) setSimilar(null)
      void refreshCollections()
      toast.success("Removed from collection")
    } catch (e) {
      toast.error("Couldn’t remove item", { description: (e as Error).message })
    }
  }

  // ------------------------------------------------------------- keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setCommandOpen((o) => !o)
        return
      }
      const target = e.target as HTMLElement
      const typing = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable
      if (!typing && e.key === "/" && viewerIndex === null) {
        e.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [viewerIndex])

  // ---------------------------------------------------------------- render
  const summary = current
    ? MODALITIES.filter((m) => current.counts[m])
        .map((m) => `${current.counts[m]!.toLocaleString()} ${MODALITY_META[m].plural.toLowerCase()}`)
        .join(" · ")
    : ""

  const sidebar = (
    <Sidebar
      collections={collections}
      active={active}
      status={status}
      jobs={jobs}
      onSelect={selectCollection}
      onCreate={createCollection}
      onDelete={setPendingDelete}
      onUpload={() => uploadRef.current?.click()}
      onIndexFolder={() => setFolderOpen(true)}
    />
  )

  const loadingGrid = results === null && library === null
  const isEmptyLibrary = !hasQuery && library !== null && library.length === 0 && filter === "all"

  return (
    <div className="flex h-dvh overflow-hidden">
      <aside className="vibrancy hidden w-[252px] shrink-0 border-r border-sidebar-border md:block">{sidebar}</aside>

      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="vibrancy w-[280px] p-0" showCloseButton={false}>
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          {sidebar}
        </SheetContent>
      </Sheet>

      <main ref={mainRef} className="scrollbar-thin relative min-w-0 flex-1 overflow-y-auto">
        {/* Toolbar */}
        <div
          className={cn(
            "sticky top-0 z-30 border-b border-transparent transition-[background-color,border-color,backdrop-filter] duration-300",
            scrolled && "border-border bg-background/72 backdrop-blur-2xl backdrop-saturate-[1.8]",
          )}
        >
          <div className="flex h-[60px] items-center gap-3 px-4 sm:px-8">
            <button
              onClick={() => setNavOpen(true)}
              className="-ml-1 grid size-9 place-items-center rounded-full hover:bg-accent md:hidden"
              aria-label="Open navigation"
            >
              <Menu className="size-5" />
            </button>
            <div className="hidden w-40 shrink-0 lg:block">
              <AnimatePresence>
                {scrolled && (
                  <motion.p
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.25, ease }}
                    className="truncate text-[15px] font-semibold"
                  >
                    {active}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
            <div className="mx-auto w-full max-w-2xl min-w-0">
              <SearchBar
                ref={searchRef}
                text={text}
                onText={setText}
                file={file}
                onFile={(f) => {
                  setFile(f)
                  if (f) setSimilar(null)
                }}
                similar={similar}
                onClearSimilar={() => setSimilar(null)}
                topK={topK}
                onTopK={setTopK}
                loading={searching}
                accept={accept}
                intent={intentChoice}
                resolved={resolveIntent(intentChoice, text, modalities)}
                onIntent={setIntentChoice}
              />
            </div>
            <div className="flex w-auto shrink-0 justify-end lg:w-40">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="h-9 rounded-full px-3.5 sm:px-4">
                    <Plus className="size-4" />
                    <span className="hidden sm:inline">Add</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-52">
                  <DropdownMenuItem onSelect={() => uploadRef.current?.click()}>
                    <Upload /> Add files…
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setFolderOpen(true)}>
                    <FolderPlus /> Index a folder…
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => setNewOpen(true)}>
                    <Plus /> New collection…
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>

        <div className="px-4 pb-32 sm:px-8">
          {/* Large title */}
          <motion.div
            key={active}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease }}
            className="pt-4 pb-5"
          >
            <h1 className="text-[34px] leading-tight font-bold">{active || "Zeta"}</h1>
            <p className="mt-0.5 h-5 text-[13px] text-muted-foreground">
              {hasQuery
                ? searching
                  ? "Searching…"
                  : `${results?.length ?? 0} matches across ${filter === "all" ? "everything" : MODALITY_META[filter].plural.toLowerCase()}`
                : summary || (current ? "Empty collection" : "")}
              {!ready && status && (
                <span className="ml-2 text-amber-600 dark:text-amber-400">
                  {status.state === "error" ? "· model failed to load" : "· model warming up"}
                </span>
              )}
            </p>
          </motion.div>

          <div className="mb-6">
            <Segmented value={filter} onChange={setFilter} counts={current?.counts} />
          </div>

          {loadingGrid || (hasQuery && searching && results === null) ? (
            <GridSkeleton />
          ) : isEmptyLibrary ? (
            <EmptyLibrary onUpload={() => uploadRef.current?.click()} onIndexFolder={() => setFolderOpen(true)} />
          ) : sections.length === 0 ? (
            <NoResults query={hasQuery ? debouncedText : ""} />
          ) : (
            <motion.div animate={{ opacity: searching ? 0.55 : 1 }} transition={{ duration: 0.2 }}>
              <MediaGrid sections={sections} collection={active} showScore={Boolean(results)} onOpen={openItem} />
            </motion.div>
          )}
        </div>
      </main>

      <input
        ref={uploadRef}
        type="file"
        multiple
        accept={accept}
        className="hidden"
        onChange={(e) => {
          void uploadFiles(Array.from(e.target.files ?? []))
          e.target.value = ""
        }}
      />

      <Viewer
        items={viewerItems}
        index={viewerIndex}
        collection={active}
        onIndexChange={setViewerIndex}
        onFindSimilar={findSimilar}
        onDelete={deleteItem}
      />
      <MiniPlayer />
      <DropOverlay collection={active} onAdd={uploadFiles} onSearch={searchWithFile} />
      <CommandMenu
        open={commandOpen}
        onOpenChange={setCommandOpen}
        collections={collections}
        active={active}
        onSelectCollection={selectCollection}
        onFilter={setFilter}
        onSearch={(q) => {
          setFile(null)
          setSimilar(null)
          setText(q)
        }}
        onUpload={() => uploadRef.current?.click()}
        onIndexFolder={() => setFolderOpen(true)}
        onNewCollection={() => setNewOpen(true)}
      />
      <IndexFolderDialog open={folderOpen} onOpenChange={setFolderOpen} collection={active} onSubmit={indexFolder} />
      <NewCollectionDialog open={newOpen} onOpenChange={setNewOpen} onSubmit={createCollection} />

      <AlertDialog open={pendingDelete !== null} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{pendingDelete}”?</AlertDialogTitle>
            <AlertDialogDescription>
              The collection’s index and any files uploaded into it will be removed. Folders you indexed are left
              untouched on disk.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                if (pendingDelete) void deleteCollection(pendingDelete)
                setPendingDelete(null)
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
