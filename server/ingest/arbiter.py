"""GPU arbiter.

Hands out leases against a fixed VRAM budget. Pool A (small vision models)
and Pool B (one VLM) are never co-resident: on this machine they sum to
roughly the whole card, so serialising is mandatory rather than cautious
(spec 3.1).

There is no preemption. A lease request that would exceed budget blocks.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from server.config import load
from server.models import vram

log = logging.getLogger(__name__)
Pool = Literal["A", "B"]


@dataclass
class Lease:
    pool: Pool
    model: str | None
    acquired_at: float = field(default_factory=time.monotonic)
    released: bool = False


class GpuArbiter:
    """Single GPU, single lease at a time. Pools are mutually exclusive."""

    def __init__(self) -> None:
        cfg = load()
        self.budget_mb: int = cfg.get("gpu.budget_mb", 5400)
        self._ttl = {"A": cfg.get("gpu.pool_a_ttl_s", 300), "B": cfg.get("gpu.pool_b_ttl_s", 120)}
        self._lock = threading.RLock()
        self._free = threading.Condition(self._lock)
        self._held: Lease | None = None
        self._loaded: dict[Pool, str | None] = {}
        self._unload_hooks: dict[Pool, callable] = {}
        self._last_used: dict[Pool, float] = {}

    def register_unloader(self, pool: Pool, fn) -> None:
        self._unload_hooks[pool] = fn

    def lease(self, pool: Pool, model: str | None = None, timeout_s: float = 600) -> Lease:
        deadline = time.monotonic() + timeout_s
        with self._free:
            while self._held is not None:
                if not self._free.wait(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"gpu lease timeout waiting for pool {pool}")

            # Evict the other pool before taking this one.
            for other in ("A", "B"):
                if other != pool and other in self._loaded:
                    self._evict(other)  # type: ignore[arg-type]

            # Same pool, different model (Pool B model switch).
            if pool in self._loaded and model is not None and self._loaded[pool] != model:
                self._evict(pool)

            self._held = Lease(pool=pool, model=model)
            self._loaded[pool] = model
            self._last_used[pool] = time.monotonic()
            log.info("gpu lease acquired pool=%s model=%s free=%dMB", pool, model, self.free_mb())
            return self._held

    def release(self, lease: Lease) -> None:
        with self._free:
            if lease.released:
                return
            lease.released = True
            self._last_used[lease.pool] = time.monotonic()
            self._held = None
            self._free.notify_all()

    def _evict(self, pool: Pool) -> None:
        hook = self._unload_hooks.get(pool)
        before = vram.used_mb() if vram.available() else 0
        if hook is not None:
            hook()
        self._loaded.pop(pool, None)
        if vram.available():
            vram.wait_until_free(max(0, before - 100), timeout_s=30)
        log.info("gpu pool %s evicted", pool)

    def sweep_idle(self) -> None:
        """Called periodically: unload pools idle past their TTL."""
        now = time.monotonic()
        with self._lock:
            if self._held is not None:
                return
            for pool, last in list(self._last_used.items()):
                if pool in self._loaded and now - last > self._ttl[pool]:
                    self._evict(pool)  # type: ignore[arg-type]

    def free_mb(self) -> int:
        if not vram.available():
            return 0
        return max(0, self.budget_mb - vram.used_mb())

    def status(self) -> dict:
        return {
            "budget_mb": self.budget_mb,
            "free_mb": self.free_mb(),
            "held": None if self._held is None else {"pool": self._held.pool, "model": self._held.model},
            "loaded": dict(self._loaded),
        }


arbiter = GpuArbiter()
