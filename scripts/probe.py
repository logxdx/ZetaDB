#!/usr/bin/env python
"""M0 probe harness.

    uv run scripts/probe.py                 # run every probe, print a table
    uv run scripts/probe.py needle sqlite   # run a subset
    uv run scripts/probe.py --json          # machine-readable
    uv run scripts/probe.py --download KEY  # fetch weights for one model

Writes data/probe_results.json so /api/health can serve the last run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()

from server.config import load  # noqa: E402
from server.models import probes, vram  # noqa: E402

GLYPH = {"ok": "\033[32m ok \033[0m", "degraded": "\033[33mwarn\033[0m",
         "missing": "\033[90mmiss\033[0m", "error": "\033[31merr \033[0m",
         "skipped": "\033[90mskip\033[0m"}


def download(key: str) -> int:
    cfg = load()
    spec = cfg.models.get(key)
    if not spec:
        print(f"unknown model key: {key}", file=sys.stderr)
        return 2
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("need huggingface-hub:  uv pip install -e '.[hf]'", file=sys.stderr)
        return 2
    dest = cfg.path("paths.models") / key
    dest.mkdir(parents=True, exist_ok=True)
    if spec.get("runtime_fetched"):
        print(f"{key}: fetched at runtime by its own engine, nothing to download")
        return 0
    # file_glob may be a single pattern or a list: a model whose weights
    # are split across several named files (EmbeddingGemma 2 ships a
    # backbone and a per-modality encoder, each with its own .onnx_data)
    # cannot be selected with one glob without dragging in the rest.
    patterns: list[str] = []
    for g in (spec.get("file_glob"), spec.get("mmproj_glob")):
        if isinstance(g, str):
            patterns.append(g)
        elif g:
            patterns.extend(g)
    if patterns:
        # Small sidecars only: tokenizers, configs, label lists. Never
        # "*.onnx_data" - that is external weight data for a model variant
        # we did not ask for and can be gigabytes.
        patterns += ["*.json", "*.txt", "*.yml", "*.model"]
        if spec.get("needs_onnx_data"):
            patterns.append("*.onnx_data")
    print(f"↓ {spec['repo']} → {dest}" + (f"  [{', '.join(patterns)}]" if patterns else ""))
    snapshot_download(repo_id=spec["repo"], local_dir=dest,
                      allow_patterns=patterns or None, max_workers=4)
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  {total/1e9:.2f} GB in {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="*", help="probe names (default: all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--download", metavar="KEY")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("\n".join(probes.names()))
        return 0
    if a.download:
        return download(a.download)

    results = probes.run(a.probes or None)
    payload = {"gpu": None, "results": [r.dict() for r in results]}
    if vram.available():
        g = vram.info()
        payload["gpu"] = {"name": g.name, "total_mb": g.total_mb, "used_mb": g.used_mb,
                          "reserved_mb": g.reserved_mb, "usable_mb": g.usable_mb}

    out = load().path("paths.data") / "probe_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    if a.json:
        print(json.dumps(payload, indent=2))
        return 0

    if payload["gpu"]:
        g = payload["gpu"]
        print(f"\n\033[1m{g['name']}\033[0m  {g['total_mb']} MiB total · "
              f"{g['used_mb']} used · {g['reserved_mb']} driver-reserved · "
              f"\033[1m{g['usable_mb']} usable\033[0m\n")

    w = max(len(r.name) for r in results)
    print(f"  {'':4s}  {'probe':<{w}}  {'vram':>6s}  {'load':>6s}  backend / detail")
    print(f"  {'-'*4}  {'-'*w}  {'-'*6}  {'-'*6}  {'-'*44}")
    for r in results:
        vm = f"{r.vram_mb}MB" if r.vram_mb is not None else "-"
        ls = f"{r.load_s:.2f}s" if r.load_s is not None else "-"
        tail = " · ".join(x for x in (r.backend, r.detail) if x)
        print(f"  {GLYPH[r.status]}  {r.name:<{w}}  {vm:>6s}  {ls:>6s}  {tail}")

    bad = [r for r in results if r.status == "error"]
    print(f"\n  {len([r for r in results if r.status=='ok'])} ok · "
          f"{len([r for r in results if r.status=='degraded'])} degraded · "
          f"{len([r for r in results if r.status=='missing'])} missing · {len(bad)} error")
    print(f"  written → {out}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
