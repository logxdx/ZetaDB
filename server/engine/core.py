"""The model engine: one registry, one VRAM budget, one lease.

Every model ZetaDB runs registers here, whatever executes it:

  backend     models
  tensorrt    PP-OCRv6 det + rec, Eye, EG2 vision/audio encoders (when the
              graph is TRT-runnable - see onnx.backend_for)
  onnx-cuda   anything TensorRT cannot run, as a fall-back
  onnx-cpu    the EG2 backbone, by design: a search query must never wait
              for the GPU
  llama.cpp   the captioning VLM, as a llama-server subprocess

The GPU is shared by lease. Models live in POOLS that are never co-resident:
on this machine pool A (small vision models) and pool B (one VLM) sum to
roughly the whole card, so serialising is mandatory rather than cautious
(spec 3.1). Taking a lease on one pool unloads every model registered in the
other; an idle pool is unloaded after its TTL. There is no preemption - a
lease that cannot be granted blocks.

This replaces the old ingest arbiter plus the per-model unload list that
app.py used to hand-maintain: a model that registers itself is unloaded with
its pool, so a new model can no longer be forgotten and left holding VRAM.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from server.config import load
from server.models import vram

log = logging.getLogger(__name__)

Pool = Literal["A", "B", "CPU"]
GPU_POOLS: tuple[Pool, ...] = ("A", "B")


@dataclass
class ModelSpec:
    """What the engine needs to know about one model."""
    name: str
    pool: Pool
    backend: str                                   # tensorrt | onnx | llama.cpp | ...
    unload: Callable[[], None] | None = None
    provider: Callable[[], str] | None = None      # what is actually running it
    loaded: Callable[[], bool] | None = None

    def describe(self) -> dict:
        prov = None
        if self.provider is not None:
            try:
                prov = self.provider()
            except Exception:
                prov = "unknown"
        is_loaded = None
        if self.loaded is not None:
            try:
                is_loaded = bool(self.loaded())
            except Exception:
                is_loaded = None
        return {"pool": self.pool, "backend": self.backend,
                "provider": prov, "loaded": is_loaded}


@dataclass
class Lease:
    pool: Pool
    model: str | None
    acquired_at: float = field(default_factory=time.monotonic)
    released: bool = False
    thread: int = field(default_factory=threading.get_ident)


class Engine:
    def __init__(self) -> None:
        cfg = load()
        self.budget_mb: int = cfg.get("gpu.budget_mb", 5400)
        self._ttl = {"A": cfg.get("gpu.pool_a_ttl_s", 300),
                     "B": cfg.get("gpu.pool_b_ttl_s", 120)}
        self._lock = threading.RLock()
        self._free = threading.Condition(self._lock)
        self._held: Lease | None = None
        self._resident: dict[Pool, str | None] = {}
        self._last_used: dict[Pool, float] = {}
        self._models: dict[str, ModelSpec] = {}
        self._extra_unloaders: dict[Pool, list[Callable[[], None]]] = {}

    # ------------------------------------------------------------- registry

    def register(self, spec: ModelSpec) -> ModelSpec:
        """Idempotent: re-registering a name replaces the old spec."""
        with self._lock:
            self._models[spec.name] = spec
        return spec

    def on_unload(self, pool: Pool, fn: Callable[[], None]) -> None:
        """Extra work to run when a pool is evicted (rarely needed - models
        that register a ModelSpec with `unload` are covered already)."""
        self._extra_unloaders.setdefault(pool, []).append(fn)

    def models(self) -> dict[str, dict]:
        with self._lock:
            return {n: s.describe() for n, s in sorted(self._models.items())}

    # ---------------------------------------------------------------- lease

    def lease(self, pool: Pool, model: str | None = None, timeout_s: float = 600) -> Lease:
        if pool not in GPU_POOLS:
            return Lease(pool=pool, model=model)          # CPU: nothing to arbitrate
        deadline = time.monotonic() + timeout_s
        with self._free:
            while self._held is not None:
                if not self._free.wait(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"gpu lease timeout waiting for pool {pool}")
            for other in GPU_POOLS:
                if other != pool and other in self._resident:
                    self._evict(other)
            # Same pool, different model (a pool B model switch).
            if pool in self._resident and model is not None and self._resident[pool] != model:
                if pool == "B":
                    self._evict(pool)
            self._held = Lease(pool=pool, model=model)
            self._resident[pool] = model
            self._last_used[pool] = time.monotonic()
            log.info("gpu lease pool=%s model=%s free=%dMB", pool, model, self.free_mb())
            return self._held

    def release(self, lease: Lease) -> None:
        if lease.pool not in GPU_POOLS:
            return
        with self._free:
            if lease.released:
                return
            lease.released = True
            self._last_used[lease.pool] = time.monotonic()
            self._held = None
            self._free.notify_all()

    def holds(self, pool: Pool) -> bool:
        """True when the calling thread holds the lease on `pool`. Lets a
        model use the GPU opportunistically for work done under a lease
        (ingest) while the same call from any other thread (a search query)
        stays on its CPU path and never touches VRAM it does not own."""
        held = self._held
        return (held is not None and not held.released and held.pool == pool
                and held.thread == threading.get_ident())

    @contextmanager
    def use(self, pool: Pool, model: str | None = None, timeout_s: float = 600):
        lease = self.lease(pool, model, timeout_s)
        try:
            yield lease
        finally:
            self.release(lease)

    # ------------------------------------------------------------- eviction

    def _evict(self, pool: Pool) -> None:
        before = vram.used_mb() if vram.available() else 0
        for spec in [s for s in self._models.values() if s.pool == pool]:
            if spec.unload is None:
                continue
            try:
                spec.unload()
            except Exception:
                log.warning("unload failed for %s", spec.name, exc_info=True)
        for fn in self._extra_unloaders.get(pool, []):
            try:
                fn()
            except Exception:
                log.warning("pool %s unload hook failed", pool, exc_info=True)
        self._resident.pop(pool, None)
        if vram.available():
            # Process exit / session drop does not mean the driver has
            # released the memory yet, so wait for nvidia-smi to agree.
            vram.wait_until_free(max(0, before - 100), timeout_s=30)
        log.info("gpu pool %s evicted", pool)

    def sweep_idle(self) -> None:
        """Called periodically: unload pools idle past their TTL."""
        now = time.monotonic()
        with self._lock:
            if self._held is not None:
                return
            for pool, last in list(self._last_used.items()):
                if pool in self._resident and now - last > self._ttl.get(pool, 300):
                    self._evict(pool)

    def unload_all(self) -> None:
        with self._lock:
            for pool in GPU_POOLS:
                self._evict(pool)

    # --------------------------------------------------------------- status

    def free_mb(self) -> int:
        if not vram.available():
            return 0
        return max(0, self.budget_mb - vram.used_mb())

    def status(self) -> dict:
        return {
            "budget_mb": self.budget_mb,
            "free_mb": self.free_mb(),
            "held": None if self._held is None else {"pool": self._held.pool,
                                                     "model": self._held.model},
            "resident": dict(self._resident),
            "models": self.models(),
        }


engine = Engine()
