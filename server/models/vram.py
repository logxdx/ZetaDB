"""VRAM accounting via nvidia-smi.

torch.cuda.memory_allocated() undercounts badly: it misses the CUDA
context (~250-400 MB), cuDNN/cuBLAS workspaces, and anything allocated by
a subprocess such as llama-server. nvidia-smi deltas are what the arbiter
budgets against, so they are what we measure.
"""
from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass

_SMI = "nvidia-smi"


def _query(fields: str) -> list[list[str]]:
    out = subprocess.run(
        [_SMI, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip()
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line]


def available() -> bool:
    try:
        _query("name")
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_mb: int
    used_mb: int
    reserved_mb: int

    @property
    def usable_mb(self) -> int:
        return self.total_mb - self.reserved_mb


def info(index: int = 0) -> GpuInfo:
    row = _query("name,memory.total,memory.used,memory.reserved")[index]
    return GpuInfo(row[0], int(row[1]), int(row[2]), int(row[3]))


def used_mb(index: int = 0) -> int:
    return int(_query("memory.used")[index][0])


def wait_until_free(target_mb: int, timeout_s: float = 30.0, index: int = 0) -> bool:
    """Poll until used_mb drops to target. Process exit does not mean the
    driver has released VRAM yet, so never trust returncode alone."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if used_mb(index) <= target_mb:
            return True
        time.sleep(0.25)
    return False


@dataclass
class Measurement:
    baseline_mb: int
    peak_mb: int
    end_mb: int
    load_s: float
    infer_s: float | None = None

    @property
    def delta_mb(self) -> int:
        return self.peak_mb - self.baseline_mb

    @property
    def leaked_mb(self) -> int:
        return self.end_mb - self.baseline_mb


@contextmanager
def measure(index: int = 0, settle_s: float = 1.0):
    """Measure the VRAM footprint of whatever runs in the block.

    Peak is sampled by the caller via m.sample(); at minimum it is sampled
    on exit. settle_s lets async allocations land before the final read.
    """
    baseline = used_mb(index)
    started = time.monotonic()
    m = Measurement(baseline_mb=baseline, peak_mb=baseline, end_mb=baseline, load_s=0.0)

    def sample() -> int:
        cur = used_mb(index)
        m.peak_mb = max(m.peak_mb, cur)
        return cur

    m.sample = sample  # type: ignore[attr-defined]
    try:
        yield m
    finally:
        m.load_s = time.monotonic() - started
        sample()
        time.sleep(settle_s)
        m.end_mb = used_mb(index)
        m.peak_mb = max(m.peak_mb, m.end_mb)
