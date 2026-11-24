"""/api/health - what is loadable on this machine, right now."""
from __future__ import annotations

import json
import platform
import sys

from fastapi import APIRouter

from server.config import load
from server.db.conn import reader, versions
from server.db.writer import writer
from server.engine.core import engine as model_engine
from server.models import probes, vram

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(refresh: bool = False) -> dict:
    cfg = load()
    conn = reader()

    gpu = None
    if vram.available():
        g = vram.info()
        gpu = {"name": g.name, "total_mb": g.total_mb, "used_mb": g.used_mb,
               "reserved_mb": g.reserved_mb, "usable_mb": g.usable_mb,
               "budget_mb": cfg.get("gpu.budget_mb")}

    if refresh:
        models = [r.dict() for r in probes.run()]
    else:
        cached = cfg.path("paths.data") / "probe_results.json"
        models = json.loads(cached.read_text())["results"] if cached.exists() else []

    db_path = cfg.path("paths.db")
    counts = {}
    for t in ("items", "vocab_terms", "links", "ingest_queue"):
        try:
            counts[t] = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = None

    return {
        "status": "ok",
        "host": {"python": sys.version.split()[0], "platform": platform.platform(),
                 "machine": platform.machine()},
        "db": {**versions(conn), "path": str(db_path),
               "size_mb": round(db_path.stat().st_size / 1e6, 2) if db_path.exists() else 0,
               "counts": counts, "writer_queue_depth": writer.depth},
        "gpu": gpu,
        "engine": model_engine.status(),
        "models": models,
        "probe_cache": "live" if refresh else "cached",
    }


@router.get("/queue")
def queue() -> dict:
    from server.ingest.scheduler import scheduler
    return scheduler.status()
