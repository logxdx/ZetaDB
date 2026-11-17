import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Pause, Play, X } from "lucide-react"

import { mediaUrl, type Item } from "@/lib/api"
import { formatDuration, prettyName } from "@/lib/format"
import { Artwork } from "@/components/artwork"
import { cn } from "@/lib/utils"

interface PlayerState {
  current: { item: Item; collection: string } | null
  playing: boolean
  time: number
  duration: number
  play: (item: Item, collection: string, startAt?: number) => void
  toggle: () => void
  seek: (fraction: number) => void
  stop: () => void
  isCurrent: (id: string) => boolean
}

const PlayerContext = createContext<PlayerState | null>(null)

export function usePlayer() {
  const ctx = useContext(PlayerContext)
  if (!ctx) throw new Error("usePlayer must be used inside <AudioPlayerProvider>")
  return ctx
}

export function AudioPlayerProvider({ children }: { children: ReactNode }) {
  const audio = useRef<HTMLAudioElement | null>(null)
  const [current, setCurrent] = useState<PlayerState["current"]>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const el = new Audio()
    el.preload = "metadata"
    audio.current = el
    const onTime = () => setTime(el.currentTime)
    const onMeta = () => setDuration(el.duration || 0)
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    el.addEventListener("timeupdate", onTime)
    el.addEventListener("loadedmetadata", onMeta)
    el.addEventListener("play", onPlay)
    el.addEventListener("pause", onPause)
    el.addEventListener("ended", onPause)
    return () => {
      el.pause()
      el.src = ""
    }
  }, [])

  const play = useCallback(
    (item: Item, collection: string, startAt = 0) => {
      const el = audio.current
      if (!el) return
      if (current?.item.id === item.id) {
        if (el.paused) void el.play()
        else el.pause()
        return
      }
      el.src = mediaUrl(collection, item.id)
      if (startAt > 0) {
        // Jump to the matched moment once the media can seek
        el.addEventListener("loadedmetadata", () => (el.currentTime = startAt), { once: true })
      }
      setCurrent({ item, collection })
      setTime(startAt)
      setDuration(item.duration || 0)
      void el.play().catch(() => setPlaying(false))
    },
    [current],
  )

  const toggle = useCallback(() => {
    const el = audio.current
    if (!el || !current) return
    if (el.paused) void el.play()
    else el.pause()
  }, [current])

  const seek = useCallback((fraction: number) => {
    const el = audio.current
    if (!el || !el.duration) return
    el.currentTime = Math.min(Math.max(fraction, 0), 1) * el.duration
  }, [])

  const stop = useCallback(() => {
    const el = audio.current
    if (el) {
      el.pause()
      el.removeAttribute("src")
      el.load()
    }
    setCurrent(null)
    setPlaying(false)
  }, [])

  const value = useMemo<PlayerState>(
    () => ({
      current,
      playing,
      time,
      duration,
      play,
      toggle,
      seek,
      stop,
      isCurrent: (id: string) => current?.item.id === id,
    }),
    [current, playing, time, duration, play, toggle, seek, stop],
  )

  return <PlayerContext.Provider value={value}>{children}</PlayerContext.Provider>
}

export function Scrubber({ className }: { className?: string }) {
  const { time, duration, seek } = usePlayer()
  const progress = duration ? time / duration : 0
  return (
    <div
      role="slider"
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(time)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (!duration) return
        if (e.key === "ArrowRight") seek((time + 5) / duration)
        if (e.key === "ArrowLeft") seek((time - 5) / duration)
      }}
      onPointerDown={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        seek((e.clientX - rect.left) / rect.width)
      }}
      className={cn("group/scrub relative h-4 cursor-pointer touch-none", className)}
    >
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 overflow-hidden rounded-full bg-foreground/15 transition-[height] group-hover/scrub:h-1.5">
        <div className="h-full rounded-full bg-foreground/80" style={{ width: `${progress * 100}%` }} />
      </div>
    </div>
  )
}

export function MiniPlayer() {
  const { current, playing, toggle, stop, time, duration } = usePlayer()
  return (
    <AnimatePresence>
      {current && (
        <motion.div
          initial={{ y: 80, opacity: 0, scale: 0.96 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          exit={{ y: 80, opacity: 0, scale: 0.96 }}
          transition={{ type: "spring", stiffness: 380, damping: 32 }}
          className="glass shadow-float fixed bottom-5 left-1/2 z-40 flex w-[min(30rem,calc(100vw-2rem))] -translate-x-1/2 items-center gap-3 rounded-2xl p-2 pr-3"
        >
          <Artwork seed={current.item.name} playing={playing} className="size-11 shrink-0 rounded-[10px]" bars={9} />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-2">
              <p className="truncate text-sm font-medium">{prettyName(current.item.name)}</p>
              <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                {formatDuration(time)} / {formatDuration(duration)}
              </span>
            </div>
            <Scrubber className="mt-0.5" />
          </div>
          <button
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="grid size-9 shrink-0 place-items-center rounded-full bg-foreground text-background transition-transform active:scale-90"
          >
            {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 translate-x-px fill-current" />}
          </button>
          <button
            onClick={stop}
            aria-label="Close player"
            className="grid size-7 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
