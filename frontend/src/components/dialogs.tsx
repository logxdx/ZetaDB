import { useEffect, useState } from "react"
import { FolderOpen, Library } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"

export function IndexFolderDialog({
  open,
  onOpenChange,
  collection,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  collection: string
  onSubmit: (path: string) => Promise<boolean>
}) {
  const [path, setPath] = useState("")
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!open) setBusy(false)
  }, [open])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <form
          onSubmit={async (e) => {
            e.preventDefault()
            if (!path.trim()) return
            setBusy(true)
            if (await onSubmit(path.trim())) onOpenChange(false)
            setBusy(false)
          }}
          className="grid gap-5"
        >
          <DialogHeader>
            <div className="mb-1 grid size-11 place-items-center rounded-xl bg-primary/10 text-primary">
              <FolderOpen className="size-5" />
            </div>
            <DialogTitle>Index a folder</DialogTitle>
            <DialogDescription>
              Every photo, video, audio file and text document inside is embedded into{" "}
              <span className="font-medium text-foreground">{collection}</span>. Files already in the collection are
              skipped.
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/home/you/Pictures"
            className="h-10 font-mono text-[13px]"
            spellCheck={false}
          />
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!path.trim() || busy}>
              {busy ? "Starting…" : "Start indexing"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function NewCollectionDialog({
  open,
  onOpenChange,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (name: string) => Promise<boolean>
}) {
  const [name, setName] = useState("")
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <form
          onSubmit={async (e) => {
            e.preventDefault()
            if (name.trim() && (await onSubmit(name.trim()))) {
              setName("")
              onOpenChange(false)
            }
          }}
          className="grid gap-5"
        >
          <DialogHeader>
            <div className="mb-1 grid size-11 place-items-center rounded-xl bg-primary/10 text-primary">
              <Library className="size-5" />
            </div>
            <DialogTitle>New collection</DialogTitle>
            <DialogDescription>Collections keep separate libraries apart, each with its own index.</DialogDescription>
          </DialogHeader>
          <Input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="travel-2026" className="h-10" />
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim()}>
              Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
