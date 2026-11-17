import { forwardRef, useRef } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Loader2, Paperclip, Search, SlidersHorizontal, Sparkles, X } from "lucide-react"

import type { Item } from "@/lib/api"
import { MODALITY_META, prettyName } from "@/lib/format"
import { useObjectUrl } from "@/hooks/use-misc"
import { Kbd } from "@/components/ui/kbd"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Slider } from "@/components/ui/slider"
import { INTENTS, modalityOf, type Intent, type IntentChoice } from "@/lib/api"
import { cn } from "@/lib/utils"

interface SearchBarProps {
  text: string
  onText: (v: string) => void
  file: File | null
  onFile: (f: File | null) => void
  similar: Item | null
  onClearSimilar: () => void
  topK: number
  onTopK: (k: number) => void
  loading: boolean
  accept: string
  intent: IntentChoice
  resolved: Intent
  onIntent: (i: IntentChoice) => void
  disabled?: boolean
}

export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(function SearchBar(
  {
    text,
    onText,
    file,
    onFile,
    similar,
    onClearSimilar,
    topK,
    onTopK,
    loading,
    accept,
    intent,
    resolved,
    onIntent,
    disabled,
  },
  ref,
) {
  const fileInput = useRef<HTMLInputElement>(null)
  const preview = useObjectUrl(file && file.type.startsWith("image/") ? file : null)
  const fileModality = file ? modalityOf(file) : null
  const FileIcon = fileModality ? MODALITY_META[fileModality].icon : Paperclip

  return (
    <div
      className={cn(
        "group/search flex h-11 w-full items-center gap-1.5 rounded-full border border-transparent bg-foreground/[0.055] pr-1.5 pl-3.5 transition-all duration-300",
        "focus-within:border-ring/60 focus-within:bg-background focus-within:shadow-[0_0_0_4px_color-mix(in_srgb,var(--primary)_14%,transparent)]",
        disabled && "opacity-60",
      )}
    >
      {loading ? (
        <Loader2 className="size-[18px] shrink-0 animate-spin text-muted-foreground" />
      ) : (
        <Search className="size-[18px] shrink-0 text-muted-foreground" />
      )}

      <AnimatePresence initial={false}>
        {text.trim() && resolved !== "search" && (
          <Chip key="intent" onClear={() => onIntent("search")} label="Query type">
            <span className="text-primary">{INTENTS.find((i) => i.value === resolved)?.label}</span>
          </Chip>
        )}
        {similar && (
          <Chip key="similar" onClear={onClearSimilar} label="Similar to">
            <Sparkles className="size-3.5 text-primary" />
            <span className="max-w-32 truncate">{prettyName(similar.name)}</span>
          </Chip>
        )}
        {file && (
          <Chip key="file" onClear={() => onFile(null)} label="Search by file">
            {preview ? (
              <img src={preview} alt="" className="size-5 rounded-[5px] object-cover" />
            ) : (
              <FileIcon className="size-3.5" style={{ color: fileModality ? MODALITY_META[fileModality].color : undefined }} />
            )}
            <span className="max-w-32 truncate">{file.name}</span>
          </Chip>
        )}
      </AnimatePresence>

      <input
        ref={ref}
        value={text}
        onChange={(e) => onText(e.target.value)}
        disabled={disabled}
        placeholder={file || similar ? "Refine with words…" : "Search photos, videos, sounds and notes"}
        className="h-full min-w-0 flex-1 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground/80"
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            if (text) onText("")
            else (e.target as HTMLInputElement).blur()
          }
        }}
      />

      <AnimatePresence>
        {(text || file || similar) && (
          <motion.button
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.6, opacity: 0 }}
            onClick={() => {
              onText("")
              onFile(null)
              onClearSimilar()
            }}
            aria-label="Clear search"
            className="grid size-5 shrink-0 place-items-center rounded-full bg-muted-foreground/30 text-background transition-colors hover:bg-muted-foreground/50"
          >
            <X className="size-3" strokeWidth={3} />
          </motion.button>
        )}
      </AnimatePresence>

      {!text && !file && !similar && (
        <Kbd className="hidden shrink-0 md:inline-flex">⌘K</Kbd>
      )}

      <IconButton label="Search with a file" onClick={() => fileInput.current?.click()} disabled={disabled}>
        <Paperclip />
      </IconButton>
      <input
        ref={fileInput}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ""
        }}
      />

      <Popover>
        <PopoverTrigger asChild>
          <IconButton label="Search options" className={intent !== "auto" ? "text-primary" : undefined}>
            <SlidersHorizontal />
          </IconButton>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-80">
          <p className="text-[13px] font-semibold">Query type</p>
          <p className="mt-0.5 text-[11.5px] text-muted-foreground">
            EmbeddingGemma 2 task prompt applied to your words.
          </p>
          <div role="radiogroup" className="mt-2.5 grid gap-0.5">
            {INTENTS.map((o) => {
              const selected = o.value === intent
              return (
                <button
                  key={o.value}
                  role="radio"
                  aria-checked={selected}
                  onClick={() => onIntent(o.value)}
                  className={cn(
                    "flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors",
                    selected ? "bg-primary/10" : "hover:bg-accent",
                  )}
                >
                  <span
                    className={cn(
                      "mt-1 grid size-3.5 shrink-0 place-items-center rounded-full border transition-colors",
                      selected ? "border-primary bg-primary" : "border-foreground/25",
                    )}
                  >
                    {selected && <span className="size-1.5 rounded-full bg-primary-foreground" />}
                  </span>
                  <span className="min-w-0">
                    <span className="flex items-baseline gap-2">
                      <span className="text-[13px] font-medium">{o.label}</span>
                      {o.value === "auto" && (
                        <span className="text-[11px] text-muted-foreground">→ {INTENTS.find((i) => i.value === resolved)?.label}</span>
                      )}
                    </span>
                    <span className="block text-[11.5px] leading-snug text-muted-foreground">{o.hint}</span>
                    {o.prompt && (
                      <code className="mt-0.5 block truncate font-mono text-[10.5px] text-muted-foreground/80">{o.prompt}</code>
                    )}
                  </span>
                </button>
              )
            })}
          </div>
          <div className="my-3 h-px bg-border" />
          <div className="flex items-baseline justify-between">
            <p className="text-[13px] font-semibold">Results</p>
            <p className="text-[13px] tabular-nums text-muted-foreground">{topK}</p>
          </div>
          <Slider
            className="mt-3"
            min={4}
            max={120}
            step={1}
            value={[topK]}
            onValueChange={([v]) => onTopK(v)}
          />
          <p className="mt-3 text-[11.5px] leading-relaxed text-muted-foreground">
            How many nearest neighbours to fetch from the vector index for each search.
          </p>
        </PopoverContent>
      </Popover>
    </div>
  )
})

const IconButton = forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button"> & { label: string }
>(function IconButton({ label, className, children, ...props }, ref) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "grid size-8 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-foreground/[0.07] hover:text-foreground disabled:pointer-events-none [&_svg]:size-4",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
})

function Chip({ children, onClear, label }: { children: React.ReactNode; onClear: () => void; label: string }) {
  return (
    <motion.span
      layout
      initial={{ opacity: 0, scale: 0.8, width: 0 }}
      animate={{ opacity: 1, scale: 1, width: "auto" }}
      exit={{ opacity: 0, scale: 0.8, width: 0 }}
      transition={{ type: "spring", stiffness: 500, damping: 36 }}
      className="flex shrink-0 items-center gap-1.5 overflow-hidden rounded-full bg-background py-1 pr-1 pl-1.5 text-[12.5px] font-medium whitespace-nowrap shadow-sm ring-1 ring-border"
      title={label}
    >
      {children}
      <button
        onClick={onClear}
        aria-label={`Remove ${label.toLowerCase()}`}
        className="grid size-4 place-items-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground"
      >
        <X className="size-3" />
      </button>
    </motion.span>
  )
}
