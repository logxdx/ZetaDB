"""Make the CUDA native libraries that ship inside Python wheels findable.

torch bundles cuDNN at nvidia/cudnn/lib/libcudnn.so.9, but ONNX Runtime's
CUDA provider dlopen()s the unversioned soname "libcudnn.so". glibc matches
an already-loaded library by SONAME, so preloading with ctypes does not
help - the file itself has to be on the loader path under the name ORT asks
for. We therefore symlink the unversioned names into a cache dir, put it on
LD_LIBRARY_PATH, and re-exec once so the loader picks it up.

Call ensure() before importing onnxruntime.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_GUARD = "IWASHERE_NATIVE_READY"
# Prefixes, not exact names: cuDNN 9 is a dispatcher that loads siblings
# (libcudnn_graph, libcudnn_engines_precompiled, ...) and linking only
# libcudnn.so yields "Invalid handle. Cannot load symbol cudnnCreate".
_WANTED = ("libcudnn", "libcublas", "libcudart", "libcufft", "libcurand",
           "libcusparse", "libnvrtc", "libnvjitlink", "libcupti",
           # TensorRT: the EP needs the parser as well as the runtime, and
           # missing libnvonnxparser fails exactly like a missing libnvinfer.
           "libnvinfer", "libnvonnxparser")


def _wheel_lib_dirs() -> list[Path]:
    """Directories inside installed wheels that hold CUDA/TRT .so files.

    Two layouts: NVIDIA's CUDA wheels use nvidia/<pkg>/lib, while
    tensorrt-cu13-libs drops its libraries straight into tensorrt_libs/.
    """
    out: list[Path] = []
    for site in map(Path, sys.path):
        nv = site / "nvidia"
        if nv.is_dir():
            out.extend(p for p in nv.glob("*/lib") if p.is_dir())
        for flat in ("tensorrt_libs", "tensorrt_bindings"):
            d = site / flat
            if d.is_dir():
                out.append(d)
    return out


def link_dir() -> Path:
    return Path(sys.prefix) / "lib" / "iwashere-native"


def _build_links() -> Path:
    dest = link_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for d in _wheel_lib_dirs():
        for so in d.glob("*.so*"):
            m = re.match(r"^(lib[A-Za-z0-9_]+)\.so(\.\d+)*$", so.name)
            if not m or not any(m.group(1).startswith(w) for w in _WANTED):
                continue
            for name in (so.name, f"{m.group(1)}.so"):
                link = dest / name
                if not link.exists():
                    try:
                        link.symlink_to(so)
                    except OSError:
                        pass
    return dest


def ensure() -> bool:
    """Returns True if the process is ready. Re-execs once if it is not."""
    if os.environ.get(_GUARD):
        return True
    dest = _build_links()
    if not any(dest.iterdir()):
        os.environ[_GUARD] = "1"
        return True
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(dest) in current.split(":"):
        os.environ[_GUARD] = "1"
        return True
    os.environ["LD_LIBRARY_PATH"] = f"{dest}:{current}" if current else str(dest)
    os.environ[_GUARD] = "1"
    # sys.orig_argv keeps the real command line ("-c <code>", "-m mod", ...);
    # sys.argv would drop the payload of -c and break the re-exec.
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    if "-" in argv[1:2] or not sys.stdin.isatty() and argv[1:2] == ["-"]:
        # The script is being piped on stdin; re-exec would lose it. The
        # caller must set LD_LIBRARY_PATH itself, so just warn.
        import warnings
        warnings.warn(f"set LD_LIBRARY_PATH={dest} for GPU providers", stacklevel=2)
        return True
    os.execv(argv[0], argv)


def status() -> dict:
    dest = link_dir()
    names = sorted({p.name for p in dest.glob("*.so")}) if dest.exists() else []
    return {"link_dir": str(dest), "unversioned": names,
             "on_path": str(dest) in os.environ.get("LD_LIBRARY_PATH", "").split(":")}


def pick_onnx(directory, stem_hint: str | None = None, *, quantised: bool = True):
    """Choose an ONNX graph from a directory.

    Policy: never load fp32/fp16 when a quantised graph exists. Preference
    order is int8/uint8 (and fp8 if a build ever ships one) over the plain
    export. Returns None when the directory has no .onnx at all.

    `quantised=False` inverts the preference for models where quantisation
    was measured to cost too much accuracy - see ocr.rec_quantised.
    """
    from pathlib import Path
    d = Path(directory)
    if not d.exists():
        return None
    cands = [p for p in sorted(d.rglob("*.onnx")) if "_prep" not in p.name]
    if stem_hint:
        narrowed = [p for p in cands if stem_hint in p.name]
        cands = narrowed or cands
    if not cands:
        return None
    rank = {"fp8": 0, "int8": 1, "uint8": 1, "qint8": 1, "q4": 2, "bnb4": 2}

    def score(p):
        low = p.name.lower()
        for token, r in rank.items():
            if token in low:
                return (r, len(p.name))
        return (9, len(p.name))     # unquantised: last resort

    def inverted(p):
        r, ln = score(p)
        return (0 if r == 9 else 1, ln)

    return sorted(cands, key=score if quantised else inverted)[0]
