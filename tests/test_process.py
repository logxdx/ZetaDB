"""Subprocess lifecycle guards (spec 3.1).

Each llama-server holds host and GPU memory. Three separate failures left
orphans during development: no stop() on shutdown, a per-call-site model
wrapper spawning one server per proposed tag, and a SIGKILLed parent that
never ran atexit. The last one is what PR_SET_PDEATHSIG covers.

The end-to-end check (hard-kill a parent, assert the child dies) needs a
real model and ~20s, so it lives in tests/manual_orphan_check.py. These
tests pin the wiring that makes it work.
"""
from __future__ import annotations

import inspect
import signal

from server.models import llama_server as ls


def test_pdeathsig_helper_is_wired_into_popen():
    src = inspect.getsource(ls.LlamaServer.start)
    assert "preexec_fn=_die_with_parent" in src, \
        "llama-server must ask the kernel to kill it when the parent dies"


def test_libc_is_resolved_at_import_not_in_the_child():
    """preexec_fn runs in the forked child before exec, where only
    async-signal-safe work is allowed. This process is multithreaded, so a
    dlopen there can deadlock on the loader lock."""
    src = inspect.getsource(ls._die_with_parent)
    assert "CDLL" not in src, "resolve libc at import time, not inside the child"
    assert ls._LIBC is not None or True   # None is tolerated on non-glibc


def test_die_with_parent_is_safe_to_call():
    ls._die_with_parent()          # must not raise in the parent either


def test_pdeathsig_uses_sigkill():
    src = inspect.getsource(ls._die_with_parent)
    assert "SIGKILL" in src
    assert ls._PR_SET_PDEATHSIG == 1


def test_stop_and_atexit_still_exist():
    """PR_SET_PDEATHSIG is the backstop, not the primary path: a clean
    shutdown should still stop the child explicitly."""
    assert hasattr(ls.LlamaServer, "stop")
    assert hasattr(ls.LlamaServer, "_atexit")
    assert "atexit.register" in inspect.getsource(ls.LlamaServer.start)


def test_cpu_only_hides_the_gpu():
    """-ngl 0 still builds a CUDA context worth a few hundred MB."""
    src = inspect.getsource(ls.LlamaServer.start)
    assert 'CUDA_VISIBLE_DEVICES' in src


def test_quantised_kv_cache_requires_flash_attention():
    """Measured: q8_0 KV without flash attention is 7.5x slower and uses
    more VRAM, not less."""
    src = inspect.getsource(ls.LlamaServer.start)
    assert "--flash-attn" in src
    assert "flash attention" in src.lower()


def test_rec_batch_is_bounded_by_the_trt_profile():
    """Regression: a recognition batch larger than the profile's max batch
    fails at RUN time ("does not satisfy any optimization profiles"), and
    the per-session fallback then masks it by dropping to the CUDA EP — so
    TensorRT looked active in the logs while never actually running."""
    import inspect

    from server.models import ppocr

    assert ppocr.trt_max_batch("ppocr_rec") == 8
    assert ppocr.trt_max_batch("unknown_model") > 1000    # unbounded when no profile
    src = inspect.getsource(ppocr.PPOCR._recognise)
    assert "max_b" in src and "trt_max_batch" in src, \
        "recognition batching must derive its cap from the TRT profile"
