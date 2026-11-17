import { useCallback, useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Info,
  Pause,
  Play,
  Sparkles,
  Trash2,
  X,
} from "lucide-react"

import { mediaUrl, type Item } from "@/lib/api"
import {
  MODALITY_META,
  formatBytes,
  formatDateTime,
  formatDuration,
  matchReason,
  matchStart,
  prettyName,
} from "@/lib/format"
import { aspectOf } from "@/lib/justify"
import { usePlayer, Scrubber } from "@/components/audio-player"
import { Artwork } from "@/components/artwork"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
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

const spring = { type: "spring", stiffness: 340, damping: 34 } as const

interface ViewerProps {
  items: Item[]
  index: number | null
  collection: string
  onIndexChange: (index: number | null) => void
  onFindSimilar: (item: Item) => void
  onDelete: (item: Item) => Promise<void>
}

function useViewport() {
  const [size, setSize] = useState(() => ({ w: window.innerWidth, h: window.innerHeight }))
  useEffect(() => {
    const onResize = () => setSize({ w: window.innerWidth, h: window.innerHeight })
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])
  return size
}

export function Viewer({ items, index, collection, onIndexChange, onFindSimilar, onDelete }: ViewerProps) {
  const item = index !== null ? items[index] : undefined
  const [showInfo, setShowInfo] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [direction, setDirection] = useState(0)
  const viewport = useViewport()
  const player = usePlayer()

  const close = useCallback(() => onIndexChange(null), [onIndexChange])
  const go = useCallback(
    (delta: number) => {
      if (index === null) return
      const next = index + delta
      if (next < 0 || next >= items.length) return
      setDirection(delta)
      onIndexChange(next)
    },
    [index, items.length, onIndexChange],
  )

  useEffect(() => {
    if (!item) return
    const onKey = (e: KeyboardEvent) => {
      if (confirmDelete) return
      const target = e.target as HTMLElement
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return
      if (e.key === "Escape") close()
      else if (e.key === "ArrowRight") go(1)
      else if (e.key === "ArrowLeft") go(-1)
      else if (e.key.toLowerCase() === "i") setShowInfo((v) => !v)
      else if (e.key === " " && item.modality === "audio") {
        e.preventDefault()
        player.play(item, collection, matchStart(item.match))
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [item, close, go, confirmDelete, player, collection])

  // Lock page scroll while open
  useEffect(() => {
    if (!item) return
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = prev
    }
  }, [item])

  const panelWidth = showInfo && viewport.w >= 900 ? 340 : 0
  const stageW = viewport.w - panelWidth - (viewport.w < 640 ? 24 : 160)
  const stageH = viewport.h - (viewport.w < 640 ? 150 : 170)

  return (
    <AnimatePresence>
      {item && (
        <motion.div
          key="viewer"
          className="fixed inset-0 z-50 flex"
          initial={{ opacity: 1 }}
          exit={{ opacity: 1 }}
          role="dialog"
          aria-modal="true"
          aria-label={prettyName(item.name)}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-white/80 backdrop-blur-2xl dark:bg-black/85"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.32, 0.72, 0, 1] }}
            onClick={close}
          />

          <div className="relative flex min-w-0 flex-1 flex-col">
            {/* Toolbar */}
            <motion.header
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
              className="relative z-10 flex h-16 shrink-0 items-center gap-2 px-3 sm:px-5"
            >
              <ToolbarButton label="Close (Esc)" onClick={close}>
                <X />
              </ToolbarButton>
              <div className="min-w-0 flex-1 px-1 text-center">
                <p className="truncate text-sm font-semibold">{prettyName(item.name)}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {matchReason(item.match, item.modality) ?? MODALITY_META[item.modality].label}
                  {` · ${index! + 1} of ${items.length}`}
                </p>
              </div>
              <ToolbarButton label="Find similar" onClick={() => onFindSimilar(item)}>
                <Sparkles />
              </ToolbarButton>
              <ToolbarButton label="Open original" onClick={() => window.open(mediaUrl(collection, item.id), "_blank")}>
                <ExternalLink />
              </ToolbarButton>
              <ToolbarButton label="Info (I)" active={showInfo} onClick={() => setShowInfo((v) => !v)}>
                <Info />
              </ToolbarButton>
              <ToolbarButton label="Delete" onClick={() => setConfirmDelete(true)} destructive>
                <Trash2 />
              </ToolbarButton>
            </motion.header>

            {/* Stage */}
            <div className="relative flex flex-1 items-center justify-center" onClick={close}>
              <AnimatePresence initial={false} custom={direction} mode="popLayout">
                <motion.div
                  key={item.id}
                  custom={direction}
                  variants={{
                    enter: (d: number) => ({ x: d * 80, opacity: 0 }),
                    center: { x: 0, opacity: 1 },
                    exit: (d: number) => ({ x: d * -80, opacity: 0 }),
                  }}
                  initial={direction ? "enter" : false}
                  animate="center"
                  exit={direction ? "exit" : undefined}
                  transition={spring}
                  onClick={(e) => e.stopPropagation()}
                  drag={item.modality === "image" ? "x" : false}
                  dragSnapToOrigin
                  dragElastic={0.4}
                  onDragEnd={(_, info) => {
                    if (info.offset.x < -80) go(1)
                    else if (info.offset.x > 80) go(-1)
                  }}
                >
                  <Stage item={item} collection={collection} maxW={stageW} maxH={stageH} />
                </motion.div>
              </AnimatePresence>

              <NavButton side="left" disabled={index === 0} onClick={() => go(-1)} />
              <NavButton side="right" disabled={index === items.length - 1} onClick={() => go(1)} />
            </div>

            {/* Filmstrip */}
            <Filmstrip items={items} index={index!} collection={collection} onSelect={(i) => {
              setDirection(i > index! ? 1 : -1)
              onIndexChange(i)
            }} />
          </div>

          <AnimatePresence>
            {showInfo && (
              <InfoPanel item={item} onClose={() => setShowInfo(false)} overlay={viewport.w < 900} />
            )}
          </AnimatePresence>

          <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Remove “{prettyName(item.name)}”?</AlertDialogTitle>
                <AlertDialogDescription>
                  It will be removed from this collection’s index. Files you indexed from a folder stay on disk;
                  files you uploaded are deleted.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={async () => {
                    await onDelete(item)
                    setConfirmDelete(false)
                  }}
                >
                  Remove
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function fit(aspect: number, maxW: number, maxH: number) {
  let w = maxW
  let h = w / aspect
  if (h > maxH) {
    h = maxH
    w = h * aspect
  }
  return { width: Math.round(w), height: Math.round(h) }
}

function Stage({ item, collection, maxW, maxH }: { item: Item; collection: string; maxW: number; maxH: number }) {
  const src = mediaUrl(collection, item.id)

  if (item.modality === "image" || item.modality === "video") {
    const aspect = item.width && item.height ? item.width / item.height : aspectOf(item)
    const native = item.width && item.height ? { w: item.width, h: item.height } : null
    let box = fit(aspect, maxW, maxH)
    // Never upscale small images beyond 2x
    if (native && box.width > native.w * 2) box = { width: native.w * 2, height: native.h * 2 }
    return (
      <motion.div
        layoutId={`media-${item.id}`}
        transition={spring}
        className="shadow-float overflow-hidden rounded-xl bg-black"
        style={box}
      >
        {item.modality === "image" ? (
          <img src={src} alt={prettyName(item.name)} className="size-full object-cover" draggable={false} />
        ) : (
          <video
            src={src}
            controls
            autoPlay
            playsInline
            className="size-full bg-black object-contain"
            onLoadedMetadata={(e) => {
              // Start at the moment that matched the search
              const start = matchStart(item.match)
              if (start > 0) e.currentTarget.currentTime = start
            }}
          />
        )}
      </motion.div>
    )
  }

  if (item.modality === "audio") return <AudioStage item={item} collection={collection} maxH={maxH} />
  return <TextStage item={item} src={src} maxW={maxW} maxH={maxH} />
}

function AudioStage({ item, collection, maxH }: { item: Item; collection: string; maxH: number }) {
  const player = usePlayer()
  const current = player.isCurrent(item.id)
  const playing = current && player.playing
  const size = Math.min(360, maxH - 120)
  return (
    <div className="flex flex-col items-center gap-6" style={{ width: Math.max(size, 260) }}>
      <motion.div
        layoutId={`media-${item.id}`}
        transition={spring}
        className="shadow-float overflow-hidden rounded-2xl"
        style={{ width: size, height: size }}
        animate={{ scale: playing ? 1 : 0.94 }}
      >
        <Artwork seed={item.name} playing={playing} className="size-full" bars={32} />
      </motion.div>
      <div className="w-full text-center">
        <p className="truncate text-lg font-semibold">{prettyName(item.name)}</p>
        <p className="text-sm text-muted-foreground">{formatDuration(current ? player.duration : item.duration)}</p>
      </div>
      <div className="w-full">
        {current ? <Scrubber /> : <div className="h-4" />}
        <div className="flex justify-between text-[11px] tabular-nums text-muted-foreground">
          <span>{formatDuration(current ? player.time : 0)}</span>
          <span>-{formatDuration(Math.max(0, (current ? player.duration : item.duration) - (current ? player.time : 0)))}</span>
        </div>
      </div>
      {matchStart(item.match) > 0 && !current && (
        <p className="-mt-3 text-xs font-medium text-primary">
          Plays from the matching moment at {formatDuration(matchStart(item.match))}
        </p>
      )}
      <button
        onClick={() => player.play(item, collection, matchStart(item.match))}
        aria-label={playing ? "Pause" : "Play"}
        className="grid size-16 place-items-center rounded-full bg-foreground text-background shadow-lg transition-transform hover:scale-105 active:scale-95"
      >
        {playing ? <Pause className="size-7 fill-current" /> : <Play className="size-7 translate-x-0.5 fill-current" />}
      </button>
    </div>
  )
}

function TextStage({ item, src, maxW, maxH }: { item: Item; src: string; maxW: number; maxH: number }) {
  const [text, setText] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    setText(null)
    fetch(src)
      .then((r) => r.text())
      .then((t) => !cancelled && setText(t))
      .catch(() => !cancelled && setText(item.text))
    return () => {
      cancelled = true
    }
  }, [src, item.text])
  return (
    <motion.article
      layoutId={`media-${item.id}`}
      transition={spring}
      className="shadow-float scrollbar-thin overflow-y-auto rounded-2xl bg-card px-8 py-9 sm:px-12"
      style={{ width: Math.min(720, maxW), maxHeight: maxH }}
    >
      <div className="mb-4 h-1 w-10 rounded-full" style={{ background: "var(--modality-text)" }} />
      <h1 className="mb-5 text-2xl font-bold">{prettyName(item.name)}</h1>
      {text === null ? (
        <div className="space-y-3">
          {[92, 100, 84, 96, 70].map((w, i) => (
            <div key={i} className="shimmer h-3.5 rounded-full" style={{ width: `${w}%` }} />
          ))}
        </div>
      ) : (
        <HighlightedText text={text} passage={item.match?.kind === "chunk" ? item.match.snippet : ""} />
      )}
    </motion.article>
  )
}

/** Full document with the matched passage highlighted and scrolled into view. */
function HighlightedText({ text, passage }: { text: string; passage: string }) {
  const markRef = useRef<HTMLElement>(null)
  // Snippets are stored trimmed; locate them by their first line
  const probe = passage.trim().split("\n")[0]?.slice(0, 120) ?? ""
  const at = probe ? text.indexOf(probe) : -1
  useEffect(() => {
    if (at >= 0) markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" })
  }, [at])
  if (at < 0) return <p className="text-[15px] leading-7 whitespace-pre-wrap text-foreground/85">{text}</p>
  const end = Math.min(text.length, at + passage.trim().length)
  return (
    <p className="text-[15px] leading-7 whitespace-pre-wrap text-foreground/85">
      {text.slice(0, at)}
      <mark
        ref={markRef}
        className="rounded-md bg-[color-mix(in_srgb,var(--modality-text)_26%,transparent)] px-0.5 text-foreground"
      >
        {text.slice(at, end)}
      </mark>
      {text.slice(end)}
    </p>
  )
}

function InfoPanel({ item, onClose, overlay }: { item: Item; onClose: () => void; overlay: boolean }) {
  const meta = MODALITY_META[item.modality]
  const Icon = meta.icon
  const rows: [string, string][] = [
    ["Kind", meta.label],
    ["Added", formatDateTime(item.added_at)],
    ["Size", formatBytes(item.size)],
  ]
  if (item.width && item.height) rows.push(["Dimensions", `${item.width} × ${item.height}`])
  if (item.duration) rows.push(["Duration", formatDuration(item.duration)])
  if (item.score !== undefined) rows.push(["Similarity", item.score.toFixed(4)])
  if (item.relevance !== undefined) rows.push(["Relevance", item.relevance.toFixed(2)])
  const reason = matchReason(item.match, item.modality)
  return (
    <motion.aside
      initial={{ x: 360, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 360, opacity: 0 }}
      transition={spring}
      className={cn(
        "glass z-20 flex w-[340px] shrink-0 flex-col gap-5 overflow-y-auto p-5",
        overlay ? "shadow-float absolute inset-y-3 right-3 rounded-2xl" : "relative my-3 mr-3 rounded-2xl",
      )}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-[15px] font-semibold">Info</h3>
        <button onClick={onClose} className="grid size-7 place-items-center rounded-full hover:bg-accent" aria-label="Close info">
          <X className="size-4" />
        </button>
      </div>
      <div className="flex items-center gap-3">
        <div
          className="grid size-10 place-items-center rounded-xl"
          style={{ background: `color-mix(in srgb, ${meta.color} 16%, transparent)`, color: meta.color }}
        >
          <Icon className="size-5" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">{prettyName(item.name)}</p>
          <p className="truncate text-xs text-muted-foreground">{item.name}</p>
        </div>
      </div>
      {reason && (
        <div className="rounded-xl bg-primary/[0.08] px-3.5 py-2.5">
          <p className="text-[11px] font-semibold tracking-wide text-primary uppercase">Why it matched</p>
          <p className="mt-0.5 text-[13px] font-medium">{reason}</p>
          {item.match?.kind === "chunk" && item.match.snippet && (
            <p className="mt-1.5 line-clamp-4 text-[12px] leading-relaxed whitespace-pre-line text-muted-foreground">
              {item.match.snippet}
            </p>
          )}
        </div>
      )}
      <dl className="divide-y divide-border rounded-xl bg-accent/60 px-3.5">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-2.5 text-[13px]">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <div>
        <p className="mb-1.5 text-xs font-medium text-muted-foreground">Location</p>
        <p className="rounded-xl bg-accent/60 px-3.5 py-2.5 font-mono text-[11.5px] leading-relaxed break-all">{item.path}</p>
      </div>
    </motion.aside>
  )
}

function Filmstrip({
  items,
  index,
  collection,
  onSelect,
}: {
  items: Item[]
  index: number
  collection: string
  onSelect: (i: number) => void
}) {
  // Show a window of neighbours around the current item
  const start = Math.max(0, Math.min(index - 7, items.length - 15))
  const visible = items.slice(start, start + 15)
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.3, ease: [0.32, 0.72, 0, 1] }}
      className="relative z-10 hidden h-[84px] shrink-0 items-center justify-center gap-1.5 sm:flex"
    >
      {visible.map((it, i) => {
        const real = start + i
        const active = real === index
        return (
          <button
            key={it.id}
            onClick={() => onSelect(real)}
            aria-label={prettyName(it.name)}
            className={cn(
              "relative h-11 overflow-hidden rounded-md bg-tile transition-all duration-300 ease-[var(--ease-spring)]",
              active ? "w-16 opacity-100 ring-2 ring-foreground" : "w-8 opacity-60 hover:opacity-100",
            )}
          >
            <Thumb item={it} collection={collection} />
          </button>
        )
      })}
    </motion.div>
  )
}

function Thumb({ item, collection }: { item: Item; collection: string }) {
  if (item.modality === "image")
    return <img src={mediaUrl(collection, item.id)} alt="" loading="lazy" className="size-full object-cover" />
  if (item.modality === "video")
    return <video src={`${mediaUrl(collection, item.id)}#t=0.1`} preload="metadata" muted className="size-full object-cover" />
  if (item.modality === "audio") return <Artwork seed={item.name} bars={6} className="size-full" />
  return <div className="size-full bg-card" style={{ boxShadow: "inset 0 3px 0 var(--modality-text)" }} />
}

function ToolbarButton({
  label,
  onClick,
  children,
  active,
  destructive,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
  active?: boolean
  destructive?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={onClick}
          aria-label={label}
          className={cn(
            "grid size-9 shrink-0 place-items-center rounded-full transition-colors [&_svg]:size-[18px]",
            active ? "bg-foreground text-background" : "hover:bg-accent",
            destructive && "hover:bg-destructive/10 hover:text-destructive",
          )}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function NavButton({ side, disabled, onClick }: { side: "left" | "right"; disabled: boolean; onClick: () => void }) {
  const Icon = side === "left" ? ChevronLeft : ChevronRight
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      disabled={disabled}
      aria-label={side === "left" ? "Previous" : "Next"}
      className={cn(
        "glass absolute top-1/2 hidden size-11 -translate-y-1/2 place-items-center rounded-full transition-all duration-300 hover:scale-105 active:scale-95 disabled:pointer-events-none disabled:opacity-0 sm:grid",
        side === "left" ? "left-5" : "right-5",
      )}
    >
      <Icon className="size-5" />
    </button>
  )
}
