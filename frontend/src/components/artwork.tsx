import { useMemo } from "react"

import { gradientFor, waveform } from "@/lib/format"
import { cn } from "@/lib/utils"

/** Generative album art: a seeded gradient with a waveform. */
export function Artwork({
  seed,
  playing = false,
  bars = 28,
  className,
}: {
  seed: string
  playing?: boolean
  bars?: number
  className?: string
}) {
  const [from, to] = gradientFor(seed)
  const heights = useMemo(() => waveform(seed, bars), [seed, bars])
  return (
    <div
      className={cn("relative overflow-hidden", playing && "playing", className)}
      style={{ background: `radial-gradient(120% 90% at 15% 10%, ${from}, ${to})` }}
    >
      <div className="absolute inset-0 bg-[radial-gradient(80%_60%_at_85%_100%,rgb(0_0_0/0.28),transparent)]" />
      <div className="absolute inset-x-[14%] inset-y-[28%] flex items-center justify-between gap-[3%]">
        {heights.map((h, i) => (
          <span
            key={i}
            className="wave-bar block flex-1 rounded-full bg-white/85"
            style={{ height: `${h * 100}%`, animationDelay: `${(i % 7) * -0.13}s` }}
          />
        ))}
      </div>
    </div>
  )
}
