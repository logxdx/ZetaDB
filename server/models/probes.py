"""M0 dependency probes.

Each probe answers one question: can this component load on THIS machine,
with which backend, and what does it cost in VRAM? Results feed /api/health
and the measured-numbers table in spec section 3.

Probes never raise. A missing dependency is a result, not a crash.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from server.config import load
from server.models import vram

Status = Literal["ok", "missing", "error", "skipped", "degraded"]


@dataclass
class ProbeResult:
    name: str
    role: str
    status: Status
    backend: str | None = None
    vram_mb: int | None = None
    load_s: float | None = None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def dict(self) -> dict:
        return asdict(self)


_REGISTRY: dict[str, tuple[str, Callable[[], ProbeResult]]] = {}


def probe(name: str, role: str):
    def deco(fn: Callable[[], ProbeResult]) -> Callable[[], ProbeResult]:
        def wrapped() -> ProbeResult:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - probes report, never crash
                return ProbeResult(name, role, "error", detail=f"{type(exc).__name__}: {exc}")
        _REGISTRY[name] = (role, wrapped)
        return wrapped
    return deco


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- runtime

@probe("python", "runtime")
def _python() -> ProbeResult:
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 12)
    return ProbeResult(
        "python", "runtime", "ok" if ok else "degraded",
        backend=f"{v.major}.{v.minor}.{v.micro}",
        detail="" if ok else "expected 3.12 (paddlepaddle-gpu and some wheels are cp312-only)",
        extra={"executable": sys.executable},
    )


@probe("ffmpeg", "audio/video decoding")
def _ffmpeg() -> ProbeResult:
    """Audio and video are decoded with ffmpeg/ffprobe (server.ingest.media);
    without them the media stage fails every audio/video item."""
    import shutil
    import subprocess
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ff and fp):
        return ProbeResult("ffmpeg", "audio/video decoding", "missing",
                           detail="install ffmpeg (provides ffprobe) to ingest audio and video")
    first = subprocess.run([ff, "-version"], capture_output=True, text=True,
                           timeout=10).stdout.splitlines()[:1]
    return ProbeResult("ffmpeg", "audio/video decoding", "ok",
                       backend=(first[0].split(" ")[2] if first else None),
                       extra={"ffmpeg": ff, "ffprobe": fp})


@probe("sqlite", "database")
def _sqlite() -> ProbeResult:
    import sqlite3
    from server.db.conn import connect
    c = connect(Path(":memory:"))
    vec = c.execute("SELECT vec_version()").fetchone()[0]
    caps = {}
    c.execute("CREATE VIRTUAL TABLE t USING fts5(a, tokenize='unicode61 remove_diacritics 2')")
    caps["fts5"] = True
    c.execute("CREATE VIRTUAL TABLE vb USING vec0(id INTEGER PRIMARY KEY, h BIT[64])")
    c.execute("INSERT INTO vb(id,h) VALUES (1, vec_bit(?))", (bytes(8),))
    caps["bit64_hamming"] = True
    c.execute("CREATE VIRTUAL TABLE vf USING vec0(id INTEGER PRIMARY KEY, e FLOAT[1024])")
    caps["float1024"] = True
    return ProbeResult("sqlite", "database", "ok", backend=f"sqlite {sqlite3.sqlite_version}",
                       detail=f"sqlite-vec {vec}", extra=caps)


@probe("cuda_torch", "runtime")
def _cuda_torch() -> ProbeResult:
    if not _have("torch"):
        return ProbeResult("cuda_torch", "runtime", "missing",
                           detail="pip install -e '.[torch]'")
    import torch
    if not torch.cuda.is_available():
        return ProbeResult("cuda_torch", "runtime", "error", backend=f"torch {torch.__version__}",
                           detail="torch.cuda.is_available() is False")
    with vram.measure() as m:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
        m.sample()
    ctx = m.delta_mb
    del x
    torch.cuda.empty_cache()
    return ProbeResult(
        "cuda_torch", "runtime", "ok",
        backend=f"torch {torch.__version__} / cuda {torch.version.cuda}",
        vram_mb=ctx, load_s=round(m.load_s, 2),
        detail=f"{torch.cuda.get_device_name(0)}; CUDA context + workspace ~{ctx} MB",
    )


@probe("onnxruntime", "runtime")
def _ort() -> ProbeResult:
    if not _have("onnxruntime"):
        return ProbeResult("onnxruntime", "runtime", "missing",
                           detail="pip install -e '.[onnx]'")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    want = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in providers]
    status: Status = "ok" if want else "degraded"
    return ProbeResult("onnxruntime", "runtime", status,
                       backend=f"onnxruntime {ort.__version__}",
                       detail=(", ".join(want) or "CPU only - no CUDA/TRT provider"),
                       extra={"providers": providers})


@probe("llamacpp", "runtime")
def _llamacpp() -> ProbeResult:
    # Resolve exactly the way the ingest path does. Probing with a bare
    # shutil.which("llama-server") reported MISSING on a machine whose
    # captioner ran fine, because the installed build is named
    # llama-server-cuda - a health check that disagrees with the runtime is
    # worse than no health check.
    from server.models.llama_server import find_binary
    exe = find_binary()
    if not exe:
        return ProbeResult("llamacpp", "runtime", "missing",
                           detail="no llama.cpp server binary on PATH or in tools/; "
                                  "build with -DGGML_CUDA=ON or set llama.server_bin")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
    ver = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "?"
    return ProbeResult("llamacpp", "runtime", "ok", backend=ver, extra={"path": exe})


@probe("needle", "structured-extraction")
def _needle() -> ProbeResult:
    """Cactus Needle 3. The wheel is pure Python; the engine .so and the
    34 MB .cact weights are fetched at runtime into ~/.cache/cactus-needle.

    M0 finding: the PyPI package cactus-needle imports as `needle`, and
    linux-x86_64 IS a supported target despite the docs only showing arm64.
    """
    if not _have("needle"):
        return ProbeResult("needle", "structured-extraction", "missing",
                           detail="uv pip install -e '.[needle]' (package: cactus-needle)")
    import os
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import needle  # type: ignore
    from needle.agent import fetch  # type: ignore

    tag = fetch._platform_tag()
    extra: dict[str, Any] = {"platform_tag": tag, "lib": fetch._lib_name()}

    weights = Path(fetch.cache_dir(3)) / fetch.base_weights(3)
    if weights.exists():
        extra["weights_mb"] = round(weights.stat().st_size / 1e6, 1)
        extra["confidence_head"] = needle._confidence_head_present(str(weights))

    from pydantic import BaseModel

    class _Probe(BaseModel):
        dates: list[str]
        amounts: list[str]
        urls: list[str]

    # M0: Needle returns None (no function call) on short or abstract inputs.
    # Probe with a representative extractive payload and measure hit rate
    # across 3 runs rather than trusting a single call.
    sample = ("Invoice from Acme Corp issued 2026-03-14 due 2026-04-13. "
              "https://billing.acme.example/inv/8813  Total $1,339.20")
    t0 = time.monotonic()
    hits, last = 0, None
    for _ in range(3):
        got = needle.extract(sample, _Probe)
        if got is not None:
            hits += 1
            last = got.model_dump()
    extra["extract_s"] = round((time.monotonic() - t0) / 3, 2)
    extra["hit_rate"] = f"{hits}/3"
    extra["sample"] = last

    if hits == 0:
        return ProbeResult("needle", "structured-extraction", "error",
                           backend=f"needle {tag}",
                           detail="engine loaded but no extraction in 3 attempts",
                           extra=extra)
    status: Status = "ok" if hits == 3 else "degraded"
    return ProbeResult("needle", "structured-extraction", status,
                       backend=f"needle {tag}",
                       detail=f"{hits}/3 extractive, {extra['extract_s']}s/call, "
                              f"confidence head={extra.get('confidence_head')}",
                       extra=extra)


# --------------------------------------------------------------------------- weights

def _model_probe(key: str) -> ProbeResult:
    cfg = load()
    spec = cfg.models.get(key, {})
    role = spec.get("role", key)
    store = cfg.path("paths.models") / key
    files = sorted(p for p in store.rglob("*") if p.is_file()) if store.exists() else []
    total = sum(p.stat().st_size for p in files)
    if not files:
        return ProbeResult(key, role, "missing", backend=spec.get("backend"),
                           detail=f"no weights in {store}; run scripts/probe.py --download {key}",
                           extra={"repo": spec.get("repo")})
    return ProbeResult(key, role, "ok", backend=spec.get("backend"),
                       detail=f"{len(files)} file(s), {total / 1e9:.2f} GB",
                       extra={"repo": spec.get("repo"), "dir": str(store),
                              "files": [p.name for p in files[:8]]})


def _register_model_probes() -> None:
    """Register a weights-presence probe per configured model.

    Never overwrite a hand-written probe of the same name: 'needle' has a
    real runtime probe above, and the config key would otherwise shadow it.
    """
    for key, spec in load().models.items():
        if key in _REGISTRY:
            continue
        role = spec.get("role", key)
        _REGISTRY[key] = (role, lambda k=key: _model_probe(k))


_register_model_probes()


# --------------------------------------------------------------------------- run

def run(names: list[str] | None = None) -> list[ProbeResult]:
    keys = names or list(_REGISTRY)
    out = []
    for k in keys:
        if k not in _REGISTRY:
            out.append(ProbeResult(k, "?", "error", detail="unknown probe"))
            continue
        t0 = time.monotonic()
        r = _REGISTRY[k][1]()
        if r.load_s is None:
            r.load_s = round(time.monotonic() - t0, 2)
        out.append(r)
    return out


def names() -> list[str]:
    return list(_REGISTRY)
