"""
FastAPI application: JSON API + the built shadcn/React frontend (``web/``).
"""

import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (
    DEFAULT_QUERY_INTENT,
    DEFAULT_SEARCH_K,
    QUERY_PROMPTS,
    DEFAULT_TABLE,
    MODALITIES,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    WEB_DIR,
    modality_for,
)
from .engine import Engine
from .index import validate_table_name

logger = logging.getLogger(__name__)

engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model in the background so the UI is available immediately
    engine.load_model_async()
    yield


app = FastAPI(title="Zeta DB", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- helpers
def _serialize(row: Dict) -> Dict:
    out = {k: v for k, v in row.items() if k != "vector"}
    added = out.get("added_at")
    if isinstance(added, datetime):
        out["added_at"] = added.isoformat()
    return out


def _parse_modalities(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    selected = [m.strip() for m in value.split(",") if m.strip() in MODALITIES]
    return selected or None


def _safe_filename(name: str) -> str:
    name = Path(name or "file").name
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).stem).strip() or "file"
    return f"{stem[:80]}{Path(name).suffix.lower()}"


def _http_error(e: Exception) -> HTTPException:
    if isinstance(e, KeyError):
        return HTTPException(404, str(e).strip("'\""))
    if isinstance(e, ValueError):
        return HTTPException(400, str(e))
    if isinstance(e, RuntimeError):
        return HTTPException(503, str(e))
    logger.exception("Unexpected error")
    return HTTPException(500, f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------- status
@app.get("/api/status")
def status():
    return {
        **engine.status(),
        "extensions": SUPPORTED_EXTENSIONS,
        "default_collection": DEFAULT_TABLE,
        "default_top_k": DEFAULT_SEARCH_K,
        "intents": QUERY_PROMPTS,
    }


# ------------------------------------------------------------ collections
class CollectionIn(BaseModel):
    name: str


@app.get("/api/collections")
def list_collections():
    out = []
    for name in engine.index.get_tables():
        try:
            out.append(engine.index.stats(name))
        except ValueError as e:  # e.g. legacy table with a different dimension
            out.append({"name": name, "total": 0, "counts": {}, "error": str(e)})
    return out


@app.post("/api/collections")
def create_collection(body: CollectionIn):
    try:
        engine.index.create_table(body.name)
        return engine.index.stats(validate_table_name(body.name))
    except Exception as e:
        raise _http_error(e)


@app.delete("/api/collections/{name}")
def delete_collection(name: str):
    if not engine.index.has_table(name):
        raise HTTPException(404, f"Collection '{name}' does not exist.")
    engine.index.drop_table(name)
    shutil.rmtree(UPLOAD_DIR / name, ignore_errors=True)
    return {"ok": True}


@app.get("/api/collections/{name}/items")
def list_items(name: str, modality: Optional[str] = None, limit: int = 200, offset: int = 0):
    try:
        rows = engine.index.list_items(name, _parse_modalities(modality), limit=limit, offset=offset)
        return [_serialize(r) for r in rows]
    except Exception as e:
        raise _http_error(e)


@app.delete("/api/collections/{name}/items/{item_id}")
def delete_item(name: str, item_id: str):
    try:
        item = engine.index.get_item(name, item_id)
        if item is None:
            raise KeyError(f"Item '{item_id}' not found.")
        engine.index.delete_item(name, item_id)
        # Only remove the file itself if it was uploaded through the UI
        path = Path(item["path"])
        if UPLOAD_DIR.resolve() in path.resolve().parents:
            path.unlink(missing_ok=True)
        return {"ok": True}
    except Exception as e:
        raise _http_error(e)


# --------------------------------------------------------------- indexing
class FolderIn(BaseModel):
    collection: str
    path: str


@app.post("/api/index/folder")
def index_folder(body: FolderIn):
    try:
        job = engine.index_folder(body.collection, body.path)
        return job.to_dict()
    except Exception as e:
        raise _http_error(e)


@app.post("/api/index/upload")
async def index_upload(collection: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        collection = validate_table_name(collection)
    except ValueError as e:
        raise HTTPException(400, str(e))
    target = UPLOAD_DIR / collection
    target.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    rejected: List[str] = []
    for upload in files:
        if not modality_for(upload.filename or ""):
            rejected.append(upload.filename or "?")
            continue
        dest = target / f"{uuid.uuid4().hex[:8]}-{_safe_filename(upload.filename)}"
        with dest.open("wb") as fh:
            await run_in_threadpool(shutil.copyfileobj, upload.file, fh)
        saved.append(dest)
    if not saved:
        raise HTTPException(400, "None of the uploaded files are a supported type.")
    job = engine.start_index_job(collection, saved, source=f"{len(saved)} uploaded file(s)")
    data = job.to_dict()
    data["rejected"] = rejected
    return data


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict() for j in engine.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = engine.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job.to_dict()


# ----------------------------------------------------------------- search
@app.post("/api/search")
async def search(
    collection: str = Form(...),
    top_k: int = Form(DEFAULT_SEARCH_K),
    modalities: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    item_id: Optional[str] = Form(None),
    intent: str = Form(DEFAULT_QUERY_INTENT),
    file: Optional[UploadFile] = File(None),
):
    tmp_path: Optional[Path] = None
    try:
        if file is not None and file.filename:
            if not modality_for(file.filename):
                raise ValueError(f"Unsupported file type: {file.filename}")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix.lower()
            ) as tmp:
                await run_in_threadpool(shutil.copyfileobj, file.file, tmp)
                tmp_path = Path(tmp.name)
        results = await run_in_threadpool(
            engine.search,
            collection,
            max(1, min(int(top_k), 200)),
            _parse_modalities(modalities),
            text,
            tmp_path,
            item_id,
            intent,
        )
        return [_serialize(r) for r in results]
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error(e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ media
@app.get("/api/media/{collection}/{item_id}")
def media(collection: str, item_id: str):
    """Serve the original file of an indexed item (only items in the index are served)."""
    try:
        item = engine.index.get_item(collection, item_id)
    except Exception as e:
        raise _http_error(e)
    if item is None:
        raise HTTPException(404, "Item not found.")
    path = Path(item["path"])
    if not path.is_file():
        raise HTTPException(410, "The original file is no longer available.")
    return FileResponse(path, filename=item.get("name") or path.name, content_disposition_type="inline")


# --------------------------------------------------------------- frontend
if (WEB_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404)
    candidate = (WEB_DIR / full_path).resolve()
    if full_path and candidate.is_file() and WEB_DIR.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = WEB_DIR / "index.html"
    if index.is_file():
        # The HTML shell must revalidate so new (content-hashed) bundles are picked up
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return HTMLResponse(
        "<h1>Zeta DB</h1><p>The frontend has not been built yet. Run "
        "<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>.</p>",
        status_code=503,
    )
