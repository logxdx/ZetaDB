import { FolderPlus, Laptop, Library, Moon, Plus, Search, Sun, Upload } from "lucide-react"
import { useTheme } from "next-themes"
import { useState } from "react"

import type { Collection } from "@/lib/api"
import { MODALITIES } from "@/lib/api"
import { MODALITY_META } from "@/lib/format"
import type { Filter } from "@/components/segmented"
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command"

interface CommandMenuProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  collections: Collection[]
  active: string
  onSelectCollection: (name: string) => void
  onFilter: (f: Filter) => void
  onSearch: (text: string) => void
  onUpload: () => void
  onIndexFolder: () => void
  onNewCollection: () => void
}

export function CommandMenu({
  open,
  onOpenChange,
  collections,
  active,
  onSelectCollection,
  onFilter,
  onSearch,
  onUpload,
  onIndexFolder,
  onNewCollection,
}: CommandMenuProps) {
  const { setTheme } = useTheme()
  const [value, setValue] = useState("")
  const run = (fn: () => void) => () => {
    onOpenChange(false)
    setValue("")
    fn()
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Command menu" description="Search or run a command">
      <Command>
      <CommandInput placeholder="Search everything or type a command…" value={value} onValueChange={setValue} />
      <CommandList className="max-h-[min(420px,60vh)]">
        {!value.trim() && <CommandEmpty>No commands found.</CommandEmpty>}
        {value.trim() && (
          <CommandGroup heading="Search" forceMount>
            <CommandItem value={`search ${value}`} onSelect={run(() => onSearch(value.trim()))} forceMount>
              <Search />
              Search {active} for “{value.trim()}”
              <CommandShortcut>↵</CommandShortcut>
            </CommandItem>
          </CommandGroup>
        )}
        <CommandGroup heading="Show">
          <CommandItem onSelect={run(() => onFilter("all"))}>
            <Library /> Everything
          </CommandItem>
          {MODALITIES.map((m) => {
            const Icon = MODALITY_META[m].icon
            return (
              <CommandItem key={m} onSelect={run(() => onFilter(m))}>
                <Icon style={{ color: MODALITY_META[m].color }} /> {MODALITY_META[m].plural}
              </CommandItem>
            )
          })}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Collections">
          {collections.map((c) => (
            <CommandItem key={c.name} value={`collection ${c.name}`} onSelect={run(() => onSelectCollection(c.name))}>
              <Library className={c.name === active ? "text-primary" : undefined} />
              {c.name}
              <CommandShortcut>{c.total.toLocaleString()}</CommandShortcut>
            </CommandItem>
          ))}
          <CommandItem onSelect={run(onNewCollection)}>
            <Plus /> New collection…
          </CommandItem>
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Actions">
          <CommandItem onSelect={run(onUpload)}>
            <Upload /> Add files to {active}…
          </CommandItem>
          <CommandItem onSelect={run(onIndexFolder)}>
            <FolderPlus /> Index a folder…
          </CommandItem>
          <CommandItem onSelect={run(() => setTheme("light"))}>
            <Sun /> Light appearance
          </CommandItem>
          <CommandItem onSelect={run(() => setTheme("dark"))}>
            <Moon /> Dark appearance
          </CommandItem>
          <CommandItem onSelect={run(() => setTheme("system"))}>
            <Laptop /> Match system appearance
          </CommandItem>
        </CommandGroup>
      </CommandList>
      </Command>
    </CommandDialog>
  )
}
