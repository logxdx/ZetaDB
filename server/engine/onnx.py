"""ONNX Runtime sessions for every model, built one way.

TensorRT first, then CUDA, then CPU. Every ONNX model in ZetaDB - PP-OCR,
EmbeddingGemma 2's vision and audio encoders, Eye - asks for its session
here, so precision, engine caching, optimisation profiles and the fall-back
order are decided in exactly one place instead of once per model.

Two rules carried over from PP-OCR (spec 5.1), which is where they were
learned:

  * TensorRT runs at FP16. That is the builder's execution precision, not a
    change to the ONNX weights, and it is the axis that keeps accuracy: INT8
    weight quantisation cost more than half the recognised text.
  * A TensorRT *engine build* failure raises instead of falling back, so a
    session is built one provider at a time, degrading on failure. One
    model's TRT failure must not take the others down with it.

A graph that uses ONNX Runtime-only quantisation ops has no TensorRT kernels;
TRT would split it into many tiny partitions for no gain. That covers dynamic
quantisation (MatMulInteger, DynamicQuantizeLinear) and weight-only block
quantisation (MatMulNBits, GatherBlockQuantized) - the latter is what
onnx-community's EmbeddingGemma 2 "*_quantized" graphs actually use.
`backend_for()` sends such graphs to CUDA unless a model says otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from server.config import load

log = logging.getLogger(__name__)

Backend = str   # "auto" | "tensorrt" | "onnx-cuda" | "onnx-cpu"


@dataclass(frozen=True)
class Profile:
    """A TensorRT optimisation profile, in ORT's 'name:AxBxC' string form.

    RANGE costs more than maximum (spec 5.1): TensorRT must find kernels
    valid across the whole span, so profiles are kept tight around the
    shapes a model is really fed.
    """
    min: str
    opt: str
    max: str

    def max_batch(self) -> int:
        # "x:8x3x48x512" -> 8 ; multi-input profiles are comma separated
        first = self.max.split(",")[0]
        return int(first.split(":")[1].split("x")[0])


# Registered by each model module at import time.
PROFILES: dict[str, Profile] = {}


def register_profile(key: str, profile: Profile) -> None:
    PROFILES[key] = profile


def max_batch(key: str) -> int:
    """Largest batch the model's TensorRT profile was built for."""
    prof = PROFILES.get(key)
    return prof.max_batch() if prof else 1 << 30


def available_providers() -> set[str]:
    try:
        import onnxruntime as ort
        return set(ort.get_available_providers())
    except Exception:
        return set()


def providers(backend: Backend, key: str | None = None, *, int8: bool = False) -> list:
    """Provider chain for one model and one requested backend."""
    cfg = load()
    have = available_providers()
    chain: list = []
    if backend in ("auto", "tensorrt") and "TensorrtExecutionProvider" in have:
        cache = cfg.path("paths.models") / "trt"
        cache.mkdir(parents=True, exist_ok=True)
        opts = {
            "trt_fp16_enable": True,
            "trt_int8_enable": int8,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
            "trt_timing_cache_enable": True,
            # TurboOCR: level 3 cuts build time 3-5x for a small speed
            # regression. Builds run to tens of minutes on a laptop GPU.
            "trt_builder_optimization_level": cfg.get("engine.trt_opt_level",
                                                      cfg.get("ocr.trt_opt_level", 3)),
        }
        prof = PROFILES.get(key or "")
        if prof:
            opts["trt_profile_min_shapes"] = prof.min
            opts["trt_profile_opt_shapes"] = prof.opt
            opts["trt_profile_max_shapes"] = prof.max
        chain.append(("TensorrtExecutionProvider", opts))
    if backend in ("auto", "tensorrt", "onnx-cuda") and "CUDAExecutionProvider" in have:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


# ONNX Runtime contrib / integer ops with no TensorRT implementation.
ORT_ONLY_QUANT_OPS = frozenset({
    "MatMulInteger", "DynamicQuantizeLinear", "ConvInteger",       # dynamic quantisation
    "MatMulNBits", "GatherBlockQuantized", "MatMulBnb4",          # weight-only block quant
})


def is_ort_quantised(path: Path) -> bool:
    """True for graphs that rely on ORT-only quantisation ops, which TRT
    cannot run. Reads only the graph's op types - the weights stay on disk."""
    try:
        import onnx
        m = onnx.load(str(path), load_external_data=False)
        return bool({n.op_type for n in m.graph.node} & ORT_ONLY_QUANT_OPS)
    except Exception:
        return "quantized" in Path(path).name


def backend_for(path: Path, requested: Backend) -> Backend:
    """Resolve 'auto' for one graph: TensorRT when it can run the graph,
    CUDA when it cannot."""
    if requested != "auto":
        return requested
    if "TensorrtExecutionProvider" in available_providers() and not is_ort_quantised(path):
        return "tensorrt"
    return "onnx-cuda"


def session(path: str | Path, key: str, backend: Backend = "auto", *,
            graph_opt: bool = True):
    """Build an InferenceSession, degrading one provider at a time.

    Returns the session; `session.get_providers()[0]` is what actually ran,
    and the engine reports it so a silent fall-back to CPU is visible.
    """
    import onnxruntime as ort
    path = Path(path)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if graph_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    resolved = backend_for(path, backend)
    int8 = "int8" in path.name
    attempts = [(resolved, "requested")]
    if resolved in ("auto", "tensorrt"):
        attempts.append(("onnx-cuda", "CUDA after TensorRT failed"))
    if resolved != "onnx-cpu":
        attempts.append(("onnx-cpu", "CPU last resort"))
    last: Exception | None = None
    for be, why in attempts:
        try:
            sess = ort.InferenceSession(str(path), so, providers=providers(be, key, int8=int8))
            if why != "requested":
                log.warning("%s: %s", key, why)
            log.info("%s: %s on %s", key, path.name, sess.get_providers()[0])
            return sess
        except Exception as exc:
            last = exc
            log.warning("%s: %s backend unavailable (%s)", key, be, str(exc)[:160])
    raise RuntimeError(f"no usable backend for {key}") from last
