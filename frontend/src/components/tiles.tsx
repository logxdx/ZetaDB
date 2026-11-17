import { memo, useRef, useState } from "react"
import { motion } from "motion/react"
import { Crosshair, Pause, Play } from "lucide-react"

import { mediaUrl, type Item } from "@/lib/api"
import { MODALITY_META, formatDuration, matchStart, prettyName } from "@/lib/format"
import { usePlayer } from "@/components/audio-player"
import { Artwork } from "@/components/artwork"
import { cn } from "@/lib/utils"

interface TileProps {
  item: Item
  collection: string
  x: number
  y: number
  width: number
  height: number
  index: number
  showScore: boolean
  onOpen: (item: Item) => void
}

export const Tile = memo(function Tile({
  item,
  collection,
  x,
  y,
  width,
  height,
  index,
  showScore,
  onOpen,
}: TileProps) {
  return (
    <motion.button
      type="button"
      layout
      initial={{ opacity: 0, scale: 0.94, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.94 }}
      transition={{
        type: "spring",
        stiffness: 420,
        damping: 34,
        delay: Math.min(index * 0.018, 0.35),
      }}
      whileTap={{ scale: 0.97 }}
      onClick={() => onOpen(item)}
      style={{ left: x, top: y, width, height }}
      className="group/tile absolute overflow-hidden rounded-[10px] bg-tile text-left outline-none focus-visible:ring-3 focus-visible:ring-ring"
      aria-label={`${MODALITY_META[item.modality].label}: ${prettyName(item.name)}`}
    >
      <motion.div layoutId={`media-${item.id}`} className="absolute inset-0 overflow-hidden rounded-[10px]">
        {item.modality === "image" && <ImageTile item={item} collection={collection} />}
        {item.modality === "video" && <VideoTile item={item} collection={collection} />}
        {item.modality === "audio" && <AudioTile item={item} collection={collection} />}
        {item.modality === "text" && <TextTile item={item} />}
      </motion.div>

      {/* Hover chrome */}
      <div className="pointer-events-none absolute inset-0 rounded-[10px] ring-1 ring-inset ring-black/5 dark:ring-white/5" />
      {item.modality !== "text" && item.modality !== "audio" && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/55 to-transparent opacity-0 transition-opacity duration-300 group-hover/tile:opacity-100" />
      )}
      {item.modality !== "text" && item.modality !== "audio" && (
        <p className="pointer-events-none absolute inset-x-2.5 bottom-2 translate-y-1 truncate text-xs font-medium text-white opacity-0 drop-shadow transition-all duration-300 group-hover/tile:translate-y-0 group-hover/tile:opacity-100">
          {prettyName(item.name)}
        </p>
      )}
      {showScore && item.score !== undefined && <ScoreBadge score={item.score} />}
      {showScore && matchStart(item.match) > 0 && <MomentBadge seconds={matchStart(item.match)} />}
    </motion.button>
  )
})

function ScoreBadge({ score }: { score: number }) {
  return (
    <span className="pointer-events-none absolute top-2 left-2 rounded-full bg-black/45 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-white opacity-0 backdrop-blur-md transition-opacity duration-300 group-hover/tile:opacity-100">
      {score.toFixed(3)}
    </span>
  )
}

function MomentBadge({ seconds }: { seconds: number }) {
  return (
    <span className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-full bg-primary/90 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-white shadow-sm backdrop-blur-md">
      <Crosshair className="size-2.5" />
      {formatDuration(seconds)}
    </span>
  )
}

function ImageTile({ item, collection }: { item: Item; collection: string }) {
  const [loaded, setLoaded] = useState(false)
  return (
    <>
      {!loaded && <div className="shimmer absolute inset-0" />}
      <img
        src={mediaUrl(collection, item.id)}
        alt={prettyName(item.name)}
        loading="lazy"
        decoding="async"
        draggable={false}
        onLoad={() => setLoaded(true)}
        className={cn(
          "size-full object-cover transition-[opacity,filter,transform] duration-700 ease-[var(--ease-spring)] group-hover/tile:scale-[1.03]",
          loaded ? "opacity-100 blur-0" : "opacity-0 blur-md",
        )}
      />
    </>
  )
}

function VideoTile({ item, collection }: { item: Item; collection: string }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [ready, setReady] = useState(false)
  // Preview from the matched moment when this is a search result
  const start = matchStart(item.match)
  const poster = Math.max(start, 0.1)
  return (
    <div
      className="absolute inset-0"
      onPointerEnter={() => {
        const v = ref.current
        if (v) {
          v.currentTime = start
          void v.play().catch(() => {})
        }
      }}
      onPointerLeave={() => {
        const v = ref.current
        if (v) {
          v.pause()
          v.currentTime = poster
        }
      }}
    >
      {!ready && <div className="shimmer absolute inset-0" />}
      <video
        ref={ref}
        src={`${mediaUrl(collection, item.id)}#t=${poster}`}
        muted
        playsInline
        loop
        preload="metadata"
        onLoadedData={() => setReady(true)}
        className={cn(
          "size-full object-cover transition-[opacity,transform] duration-700 ease-[var(--ease-spring)] group-hover/tile:scale-[1.03]",
          ready ? "opacity-100" : "opacity-0",
        )}
      />
      <span className="pointer-events-none absolute top-2 right-2 flex items-center gap-1 rounded-full bg-black/40 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-white backdrop-blur-md">
        <Play className="size-2.5 fill-current" />
        {formatDuration(item.duration)}
      </span>
    </div>
  )
}

function AudioTile({ item, collection }: { item: Item; collection: string }) {
  const player = usePlayer()
  const isCurrent = player.isCurrent(item.id)
  const playing = isCurrent && player.playing
  return (
    <div className="absolute inset-0">
      <Artwork seed={item.name} playing={playing} className="absolute inset-0" />
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/50 to-transparent p-3 pt-10">
        <p className="truncate text-sm font-semibold text-white">{prettyName(item.name)}</p>
        <p className="text-[11px] font-medium tabular-nums text-white/75">{formatDuration(item.duration)}</p>
      </div>
      <span
        role="button"
        tabIndex={-1}
        aria-label={playing ? "Pause" : "Play"}
        onClick={(e) => {
          e.stopPropagation()
          player.play(item, collection, matchStart(item.match))
        }}
        className={cn(
          "absolute top-1/2 left-1/2 grid size-12 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white/25 text-white shadow-lg backdrop-blur-xl transition-all duration-300 hover:scale-110 hover:bg-white/35 active:scale-95",
          playing ? "opacity-100" : "scale-90 opacity-0 group-hover/tile:scale-100 group-hover/tile:opacity-100",
        )}
      >
        {playing ? <Pause className="size-5 fill-current" /> : <Play className="size-5 translate-x-px fill-current" />}
      </span>
    </div>
  )
}

function TextTile({ item }: { item: Item }) {
  return (
    <div className="absolute inset-0 flex flex-col bg-card p-3.5 transition-colors duration-300 group-hover/tile:bg-[color-mix(in_srgb,var(--card),var(--modality-text)_6%)]">
      <div className="mb-2 h-1 w-8 rounded-full" style={{ background: "var(--modality-text)" }} />
      <p className="line-clamp-2 text-[13px] leading-snug font-semibold">{prettyName(item.name)}</p>
      <p className="mask-fade-b mt-1.5 flex-1 overflow-hidden text-[11.5px] leading-relaxed whitespace-pre-line text-muted-foreground">
        {item.match?.snippet || item.text}
      </p>
    </div>
  )
}
