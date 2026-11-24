"""iwashere - FastAPI application.

Serves the API and, in production, the built Svelte SPA from web/dist.
Binds to 127.0.0.1 only: this is a single-user local tool with no auth.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

# Cactus Needle phones home to a Supabase endpoint unless told not to.
# iwashere is a local-only tool; opt out before the package is imported.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Put wheel-bundled CUDA libraries (cuDNN) on the loader path before any
# ONNX Runtime import. May re-exec the process once.
from server.models._native import ensure as _ensure_native  # noqa: E402
_ensure_native()
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.api import health, items, search, vocab as vocab_api
from server.config import ROOT, load
from server.db.conn import init_db
from server.db.writer import writer
from server.ingest.events import hub
from server.ingest.scheduler import scheduler
from server.ingest.stages import finalize
from server.ingest.watcher import watcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("iwashere")


def _register_stages() -> None:
    """Register the stage handlers that are actually available.

    Unregistered stages pass through, so the pipeline runs end to end from
    M1 onward and each model plugs in as it lands.
    """
    scheduler.register("finalize", finalize.run)
    try:
        from server.ingest.stages import ocr, layout
        scheduler.register("ocr", ocr.run)
        scheduler.register("layout", layout.run)
    except Exception:
        log.info("OCR stage unavailable; images will skip to captioning")
    try:
        from server.ingest.stages import embed
        scheduler.register("embed", embed.run)
    except Exception:
        log.info("embedding stage unavailable; search runs on BM25 + tags only")
    try:
        from server.ingest.stages import extract
        scheduler.register("extract", extract.run)
    except Exception:
        log.info("extraction stage unavailable")
    try:
        from server.ingest.stages import media
        scheduler.register("media", media.run)
    except Exception:
        log.info("media stage unavailable; audio and video will not be segmented")
    try:
        from server.ingest.stages import tag_visual
        scheduler.register("tag_visual", tag_visual.run)
    except Exception:
        log.info("visual tagging stage unavailable")
    try:
        from server.ingest.stages import caption
        scheduler.register("caption", caption.run)
    except Exception:
        log.info("captioning stage unavailable")


def _reseed_vocab_if_flagged() -> None:
    """Schema v4 renamed JoyTag's aliases to Eye's and cleared them; the
    migration leaves a flag, and the seed is re-applied once here so every
    existing term gets its Eye alias without a manual /api/vocab/seed."""
    from server.db.conn import reader
    from server.search import vocab as V
    row = reader().execute("SELECT value FROM meta WHERE key = 'vocab_reseed'").fetchone()
    if not row or row["value"] != "1":
        return
    try:
        V.seed_from_yaml()
        writer.submit(lambda c: c.execute("DELETE FROM meta WHERE key = 'vocab_reseed'"))
        log.info("vocabulary re-seeded with Eye aliases")
    except Exception:
        log.exception("vocabulary re-seed failed; run POST /api/vocab/seed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load()
    for key in ("paths.blobs", "paths.thumbs", "paths.models"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    init_db().close()
    writer.start()
    _reseed_vocab_if_flagged()
    hub.bind(asyncio.get_running_loop())
    _register_stages()
    scheduler.start()
    watcher.start()
    log.info("iwashere ready on http://%s:%s  stages=%s",
             cfg.get("server.host"), cfg.get("server.port"), scheduler.status()["handlers"])
    yield
    watcher.stop()
    scheduler.stop()
    _shutdown_models()
    writer.stop()


def _shutdown_models() -> None:
    """Unload every registered model. llama-server runs as a subprocess;
    without this it survives the parent and keeps holding VRAM, and
    successive runs stack up orphans."""
    try:
        from server.engine.core import engine as model_engine
        model_engine.unload_all()
    except Exception:
        log.debug("engine shutdown failed", exc_info=True)


app = FastAPI(title="iwashere", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(items.router)
app.include_router(search.router)
app.include_router(vocab_api.router)

_DIST = ROOT / "web" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
else:
    @app.get("/")
    def no_ui():
        return {"detail": "SPA not built. Run: cd web && npm install && npm run build"}


def main() -> None:
    import uvicorn
    cfg = load()
    uvicorn.run("server.app:app", host=cfg.get("server.host", "127.0.0.1"),
                port=cfg.get("server.port", 8777), reload=False)


if __name__ == "__main__":
    main()
