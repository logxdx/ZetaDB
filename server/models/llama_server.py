"""llama.cpp server lifecycle.

Pool B holds exactly one llama-server at a time. Switching models means
kill, wait for the driver to actually release VRAM, then spawn: process
exit does not mean the memory is back (spec 3.1).
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from server.config import ROOT, load
from server.models import vram

log = logging.getLogger(__name__)


def _candidate_names() -> list[str]:
    """Binary names to try, best first.

    A multi-backend llama.cpp install suffixes the backend onto every
    binary - llama-server-cuda, -vulkan, -cpu - so a plain `llama-server`
    is often absent from PATH on a machine that has llama.cpp built and
    working. Pool B is a GPU pool, so the CUDA build is preferred: falling
    back to a CPU server would not fail, it would quietly caption at a few
    tokens per second and never appear in the VRAM budget this design is
    built around.
    """
    names = []
    configured = load().get("llama.server_bin")
    if configured:
        names.append(str(configured))
    names += ["llama-server-cuda", "llama-server"]
    return list(dict.fromkeys(names))


def find_binary(name: str | None = None) -> str | None:
    """Resolve the llama.cpp server binary: PATH first, then tools/.

    PATH is searched under EVERY candidate name before the vendored build
    in tools/ is considered - a binary the user installed is newer than the
    one checked in beside it.
    """
    names = [name] if name else _candidate_names()
    for n in names:
        onpath = shutil.which(n)
        if onpath:
            return onpath
    for n in names:
        for cand in sorted((ROOT / "tools").rglob(n)):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return None


# Resolved at import time, on purpose. preexec_fn runs in the forked child
# BEFORE exec, where only async-signal-safe work is allowed - and this
# process is multithreaded, so a dlopen there can deadlock on the loader
# lock. Loading libc here means the child only makes one resolved call.
try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # pragma: no cover - non-glibc
    _LIBC = None

_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child when its parent dies.

    atexit and explicit stop() cover a clean shutdown, but neither runs
    when the parent is SIGKILLed or OOM-killed - and llama-server then
    survives, reparented to init, holding GPU and host memory. Repeated
    runs left orphans stacking up. PR_SET_PDEATHSIG is the only thing that
    covers the hard-kill case.
    """
    if _LIBC is not None:
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, model: Path, *, mmproj: Path | None = None, ctx: int = 8192,
                 ngl: int = 99, embedding: bool = False, extra: list[str] | None = None,
                 cpu_only: bool = False, cache_type: str | None = None,
                 flash_attn: bool = False):
        self.model = model
        self.mmproj = mmproj
        self.ctx = ctx
        self.ngl = ngl
        self.embedding = embedding
        self.extra = extra or []
        self.cpu_only = cpu_only
        self.cache_type = cache_type
        self.flash_attn = flash_attn
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self._client: httpx.Client | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float = 240) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        exe = find_binary()
        if exe is None:
            raise RuntimeError(
                f"no llama.cpp server binary found; tried {_candidate_names()} "
                "on PATH and under tools/")
        self.port = _free_port()
        cmd = [exe, "-m", str(self.model), "--port", str(self.port),
               "--host", "127.0.0.1", "-c", str(self.ctx), "-ngl", str(self.ngl),
               "--no-warmup"]
        if self.mmproj:
            cmd += ["--mmproj", str(self.mmproj)]
        if self.flash_attn:
            cmd += ["--flash-attn", "on"]
        if self.cache_type:
            # Only pay for a quantised cache when flash attention can use it.
            if not self.flash_attn:
                log.warning("cache_type=%s without flash attention is much slower",
                            self.cache_type)
            cmd += ["--cache-type-k", self.cache_type, "--cache-type-v", self.cache_type]
        if self.embedding:
            cmd += ["--embedding", "--pooling", "last"]
        cmd += self.extra

        env = dict(os.environ)
        if self.cpu_only:
            # -ngl 0 still initialises a CUDA context worth a few hundred MB.
            # Hiding the device is the only way to make a CPU model cost zero
            # VRAM, which matters on a 6 GB budget (spec 3).
            env["CUDA_VISIBLE_DEVICES"] = ""
        # Resolve the symlink FIRST. An installed llama.cpp is usually a link
        # in ~/.local/bin pointing into the build tree, and the .so files sit
        # beside the real binary, not beside the link. Taking the link's own
        # parent puts ~/.local/bin on LD_LIBRARY_PATH, which finds nothing.
        lib = Path(exe).resolve().parent
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH','')}"
        log.info("starting llama-server: %s", self.model.name)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, text=True, start_new_session=False,
                                     preexec_fn=_die_with_parent)
        atexit.register(self._atexit)
        self._client = httpx.Client(base_url=self.url, timeout=300)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read()[-1500:] if self.proc.stdout else ""
                raise RuntimeError(f"llama-server exited early:\n{out}")
            try:
                # 503 means "loading", not "unreachable". Sleeping only in the
                # except branch turned that into a hot loop that hammered the
                # server thousands of times a second and flooded the log.
                if self._client.get("/health", timeout=5).status_code == 200:
                    log.info("llama-server ready on :%d (%.0f MB vram in use)",
                             self.port, vram.used_mb() if vram.available() else -1)
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server did not become ready in {timeout_s}s")

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self.proc is None:
            return
        before = vram.used_mb() if vram.available() else 0
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None
        if vram.available() and before > 0:
            vram.wait_until_free(max(0, before - 200), timeout_s=30)

    def _atexit(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.start()
        r = self._client.post("/v1/embeddings", json={"input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def chat(self, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.2, grammar: str | None = None,
             thinking: bool | None = False, extra: dict | None = None) -> str:
        self.start()
        payload: dict = {"messages": messages, "max_tokens": max_tokens,
                         "temperature": temperature, "stream": False}
        # Only a hybrid reasoner declares enable_thinking. Qwen3.5 needs it
        # off or it spends the whole budget on reasoning_content and returns
        # an empty content field; a model whose template does not declare it
        # must not be sent it at all, so None means "omit".
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if grammar:
            payload["grammar"] = grammar
        # Publisher-recommended sampling, e.g. LFM2.5-VL's top_k / repeat_penalty.
        if extra:
            payload.update(extra)
        r = self._client.post("/v1/chat/completions", json=payload)
        if r.status_code >= 400:
            # The body carries llama.cpp's actual complaint; raise_for_status
            # alone throws it away and makes a 400 undiagnosable.
            raise RuntimeError(f"llama-server {r.status_code}: {r.text[:400]}")
        return r.json()["choices"][0]["message"]["content"]
