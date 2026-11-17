import { AnimatePresence, motion } from "motion/react"
import { CheckCircle2, Loader2, XCircle } from "lucide-react"

import type { Job } from "@/lib/api"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

function ProgressRing({ value, size = 18 }: { value: number; size?: number }) {
  const r = (size - 3) / 2
  const c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={2.5} className="stroke-foreground/12" />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        strokeWidth={2.5}
        strokeLinecap="round"
        className="stroke-primary"
        strokeDasharray={c}
        animate={{ strokeDashoffset: c * (1 - value) }}
        transition={{ type: "spring", stiffness: 120, damping: 20 }}
      />
    </svg>
  )
}

export function Activity({ jobs }: { jobs: Job[] }) {
  const running = jobs.filter((j) => j.status === "running" || j.status === "queued")
  const recent = jobs.slice(0, 6)
  const total = running.reduce((s, j) => s + j.total, 0)
  const done = running.reduce((s, j) => s + j.done, 0)
  const progress = total ? done / total : 0

  return (
    <AnimatePresence>
      {recent.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
        >
          <Popover>
            <PopoverTrigger asChild>
              <button className="flex w-full items-center gap-2.5 rounded-xl bg-sidebar-accent/70 px-3 py-2.5 text-left transition-colors hover:bg-sidebar-accent">
                {running.length ? (
                  <ProgressRing value={progress} />
                ) : (
                  <CheckCircle2 className="size-[18px] text-emerald-500" />
                )}
                <div className="min-w-0 flex-1 leading-tight">
                  <p className="truncate text-[12.5px] font-medium">
                    {running.length ? "Indexing…" : "Up to date"}
                  </p>
                  <p className="truncate text-[11px] tabular-nums text-muted-foreground">
                    {running.length
                      ? `${done.toLocaleString()} of ${total.toLocaleString()}${
                          running[0].eta !== null && running[0].eta !== undefined
                            ? ` · ${formatEta(running[0].eta)} left`
                            : ` · ${running[0].current || "starting"}`
                        }`
                      : `${recent[0].indexed.toLocaleString()} added to ${recent[0].collection}`}
                  </p>
                </div>
              </button>
            </PopoverTrigger>
            <PopoverContent side="top" align="start" className="w-80 p-2">
              <p className="px-2 pt-1 pb-2 text-[13px] font-semibold">Activity</p>
              <div className="space-y-1">
                {recent.map((job) => (
                  <JobRow key={job.id} job={job} />
                ))}
              </div>
            </PopoverContent>
          </Popover>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function formatEta(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
}

function JobRow({ job }: { job: Job }) {
  const active = job.status === "running" || job.status === "queued"
  const pct = job.total ? (job.done / job.total) * 100 : 0
  return (
    <div className="rounded-lg px-2 py-2 hover:bg-accent/60">
      <div className="flex items-center gap-2">
        {active ? (
          <Loader2 className="size-3.5 animate-spin text-primary" />
        ) : job.status === "failed" ? (
          <XCircle className="size-3.5 text-destructive" />
        ) : (
          <CheckCircle2 className="size-3.5 text-emerald-500" />
        )}
        <p className="flex-1 truncate text-[12.5px] font-medium">{job.collection}</p>
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {job.done}/{job.total}
        </span>
      </div>
      <p className="mt-0.5 truncate pl-5.5 text-[11px] text-muted-foreground" title={job.source}>
        {job.source}
      </p>
      {active && (
        <>
          <div className="mt-1.5 ml-5.5 h-1 overflow-hidden rounded-full bg-foreground/10">
            <motion.div className="h-full rounded-full bg-primary" animate={{ width: `${pct}%` }} />
          </div>
          {job.rate > 0 && (
            <p className="mt-1 pl-5.5 text-[11px] tabular-nums text-muted-foreground">
              {job.rate.toFixed(1)} items/s{job.eta !== null && ` · ${formatEta(job.eta)} left`}
            </p>
          )}
        </>
      )}
      {!active && (
        <p className={cn("mt-0.5 pl-5.5 text-[11px]", job.failed ? "text-destructive" : "text-muted-foreground")}>
          {job.indexed} added · {job.skipped} skipped{job.failed ? ` · ${job.failed} failed` : ""}
          {job.rate > 0 && ` · ${job.rate.toFixed(1)}/s`}
        </p>
      )}
    </div>
  )
}
