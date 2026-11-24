"""Convert Desert Ant Labs' Eye (eye.tflite) to ONNX, so it runs in the
same engine as every other model (TensorRT FP16 -> CUDA -> CPU).

Run in a throwaway tool environment - TensorFlow is a conversion-time
dependency only, never a serving one:

    uv run --no-project --with tensorflow-cpu --with tf2onnx \
        --with ai-edge-litert --with onnx --with onnxruntime --with pillow \
        python scripts/convert_eye.py [image_dir]

Only the `photo` signature is converted: it carries every head ZetaDB uses
(type, scenes, sensitive, reject, rotation, embedding, aesthetic). Converting
the whole file fails - tf2onnx walks every subgraph and trips on the
sharpness models' BatchNorm folding - and none of the other signatures are
needed.

Writes, next to eye.tflite:
    eye_photo.onnx    input `image` (1,3,384,384) float32 in 0..1
    eye_photo.json    output names, per-head activation, and the measured
                      parity against the LiteRT reference on `image_dir`

Licence: Eye is used under permission from Desert Ant Labs. The licence and
notice files ship alongside the weights and must stay with them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # conversion is CPU-only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[1]
EYE_DIR = ROOT / "data" / "models" / "eye"
TFLITE = EYE_DIR / "eye.tflite"
SIGNATURE = "photo"
OPSET = 18

# Heads whose raw values feed the calibration curves in eye_meta.json, and
# the activation they need if the graph emits logits rather than scores.
HEAD_ACTIVATION = {"type": "softmax", "rotation": "softmax",
                   "scenes": "sigmoid", "sensitive": "sigmoid", "reject": "sigmoid"}


def _read_offset_buffers(file_bytes: bytes) -> None:
    """Teach tf2onnx the TFLite large-model buffer layout.

    eye.tflite stores 601 of its 927 weight buffers OUTSIDE the flatbuffer
    table: `Buffer.offset/size` point at raw bytes later in the file, and
    `Buffer.data` is empty. tf2onnx only reads `data`, so those weights came
    back as missing tensors and every op consuming one referenced a tensor
    no node produces. Offsets are absolute positions in the file; 0 and 1
    mean "no external data".
    """
    import numpy as np
    from tf2onnx.tflite.Buffer import Buffer

    orig_is_none, orig_numpy = Buffer.DataIsNone, Buffer.DataAsNumpy

    def is_none(self):
        return orig_is_none(self) and self.Offset() <= 1

    def as_numpy(self):
        if self.Offset() > 1:
            return np.frombuffer(file_bytes, dtype=np.uint8,
                                 count=int(self.Size()), offset=int(self.Offset()))
        return orig_numpy(self)

    Buffer.DataIsNone, Buffer.DataAsNumpy = is_none, as_numpy


def _tolerate_missing_fused_activation() -> None:
    """Newer TFLite writers omit `fused_activation_function` on element-wise
    ops that have none (no BuiltinOptions table at all). tf2onnx indexes the
    attribute unconditionally; a missing one simply means NONE."""
    import importlib
    import pkgutil

    import tf2onnx.tflite_handlers as handlers

    for info in pkgutil.iter_modules(handlers.__path__):
        mod = importlib.import_module(f"{handlers.__name__}.{info.name}")
        orig = getattr(mod, "separate_fused_activation_function", None)
        if orig is None or getattr(orig, "_zeta_safe", False):
            continue

        def safe(ctx, node, *a, _orig=orig, **kw):
            if "fused_activation_function" not in node.attr:
                return None
            return _orig(ctx, node, *a, **kw)
        safe._zeta_safe = True
        setattr(mod, "separate_fused_activation_function", safe)


def _expand_gelu(proto) -> int:
    """tf2onnx has no handler for TFL_GELU and passes it through untouched.
    Eye's GELUs are all the exact form (GeluOptions.approximate = False), so
    each becomes 0.5 * x * (1 + erf(x / sqrt(2))) in plain ONNX ops - Erf is
    supported by every provider in the engine, TensorRT included, which is
    not true of opset 20's fused Gelu everywhere."""
    import numpy as np
    from onnx import helper, numpy_helper

    g = proto.graph
    consts = {
        "zeta_gelu_inv_sqrt2": np.array(1 / np.sqrt(2), dtype=np.float32),
        "zeta_gelu_one": np.array(1.0, dtype=np.float32),
        "zeta_gelu_half": np.array(0.5, dtype=np.float32),
    }
    new_nodes, n = [], 0
    for node in g.node:
        if node.op_type != "TFL_GELU":
            new_nodes.append(node)
            continue
        approx = any(a.name == "approximate" and a.i for a in node.attribute)
        if approx:
            raise SystemExit(f"approximate GELU not handled: {node.name}")
        x, y, p = node.input[0], node.output[0], f"zeta_gelu{n}"
        new_nodes += [
            helper.make_node("Mul", [x, "zeta_gelu_inv_sqrt2"], [f"{p}_s"], name=f"{p}_scale"),
            helper.make_node("Erf", [f"{p}_s"], [f"{p}_e"], name=f"{p}_erf"),
            helper.make_node("Add", [f"{p}_e", "zeta_gelu_one"], [f"{p}_a"], name=f"{p}_add"),
            helper.make_node("Mul", [x, f"{p}_a"], [f"{p}_m"], name=f"{p}_mul"),
            helper.make_node("Mul", [f"{p}_m", "zeta_gelu_half"], [y], name=f"{p}_half"),
        ]
        n += 1
    if n:
        del g.node[:]
        g.node.extend(new_nodes)
        g.initializer.extend(numpy_helper.from_array(v, k) for k, v in consts.items())
    print(f"expanded {n} GELU node(s)")
    return n


def _fix_opcode_lookup() -> None:
    """Read each operator code as max(deprecated_builtin_code, builtin_code).

    The TFLite schema widened opcodes past 127 by adding `builtin_code` and
    keeping the old byte as `deprecated_builtin_code`. Writers are supposed
    to fill both, but newer ones leave the old byte at 0 - which decodes as
    ADD. tf2onnx trusts the old byte first, so eye.tflite's DEQUANTIZE ops
    came through as one-input int8 "ADD"s. TFLite's own reader takes the
    larger of the two, and so does this.
    """
    from tf2onnx import tflite_utils

    orig = tflite_utils.read_tflite_model

    def read(path):
        graphs, opcodes, model, shapes = orig(path)
        for i in range(model.OperatorCodesLength()):
            oc = model.OperatorCodes(i)
            code = max(int(oc.DeprecatedBuiltinCode()), int(oc.BuiltinCode()))
            name = tflite_utils.lookup_enum(code, "BuiltinOperator")
            if name == "CUSTOM":
                name = oc.CustomCode().decode()
            opcodes[i] = name
        return graphs, opcodes, model, shapes
    tflite_utils.read_tflite_model = read


def _avoid_dequant_name_clash() -> None:
    """eye.tflite already names 75 tensors '<weight>_dequant' (the outputs of
    its own DEQUANTIZE ops). With dequantize=True tf2onnx inserts dequant
    nodes named '<tensor>_dequant' too, so its node and the file's tensor
    share a name and the graph closes into a cycle. Give the file's tensors
    a distinct suffix as they are read; the rename is internal to the graph."""
    from tf2onnx.tflite.Tensor import Tensor

    orig = Tensor.Name

    def name(self):
        n = orig(self)
        return n + b"__tfl" if n.endswith(b"_dequant") else n
    Tensor.Name = name


def _signature(model, name: str):
    """(subgraph index, input tensor name, {output name: tensor name})."""
    for i in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(i)
        if sd.SignatureKey().decode() != name:
            continue
        g = model.Subgraphs(sd.SubgraphIndex())
        tensor = lambda idx: g.Tensors(idx).Name().decode()      # noqa: E731
        ins = {sd.Inputs(j).Name().decode(): tensor(sd.Inputs(j).TensorIndex())
               for j in range(sd.InputsLength())}
        outs = {sd.Outputs(j).Name().decode(): tensor(sd.Outputs(j).TensorIndex())
                for j in range(sd.OutputsLength())}
        return sd.SubgraphIndex(), ins, outs
    raise SystemExit(f"signature {name!r} not found")


def convert() -> tuple[Path, dict[str, str]]:
    import onnx
    from tf2onnx import convert as tfconv
    from tf2onnx import tflite_utils
    from tf2onnx.tflite.Model import Model

    buf = bytearray(TFLITE.read_bytes())
    _read_offset_buffers(bytes(buf))
    _tolerate_missing_fused_activation()
    _fix_opcode_lookup()
    _avoid_dequant_name_clash()
    model = Model.GetRootAsModel(buf, 0)
    idx, ins, outs = _signature(model, SIGNATURE)

    # Restrict tf2onnx to the photo subgraph and whatever it calls.
    def only_photo(m):
        need, stack = [], [idx]
        while stack:
            cur = stack.pop()
            if cur in need:
                continue
            need.append(cur)
            stack.extend(tflite_utils.get_subgraph_dependencies(m, cur))
        return [m.Subgraphs(i) for i in reversed(need)]      # main graph last
    tflite_utils.get_model_subgraphs = only_photo

    in_tensor = ins["image"]
    out_tensors = list(outs.values())
    # dequantize=True: Eye stores weights as int8 per output channel and
    # dequantizes them at load. Without it tf2onnx drops the quantised weight
    # tensors and the scene head's conv is left reading a tensor no node
    # produces. LiteRT runs float32 activations anyway, so this is the same
    # arithmetic the reference does - and TensorRT then builds the graph at
    # FP16 like every other model in the engine.
    proto, _ = tfconv._convert_common(
        None, name="eye_photo", opset=OPSET, input_names=[in_tensor],
        output_names=out_tensors, tflite_path=str(TFLITE), dequantize=True)

    _expand_gelu(proto)

    # Rename graph IO to the signature's names.
    rename = {in_tensor: "image", **{t: n for n, t in outs.items()}}
    g = proto.graph
    for node in g.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    for vi in list(g.input) + list(g.output):
        vi.name = rename.get(vi.name, vi.name)
    out = EYE_DIR / f"eye_{SIGNATURE}.onnx"
    onnx.save(proto, str(out))
    onnx.checker.check_model(str(out))
    return out, outs


def _prep(path: Path):
    """The documented preparation: upright, short side to 384 (bicubic),
    centre crop 384, RGB in 0..1, NCHW. Normalisation is inside the model."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        s = 384 / min(w, h)
        im = im.resize((max(384, round(w * s)), max(384, round(h * s))), Image.Resampling.BICUBIC)
        w, h = im.size
        left, top = (w - 384) // 2, (h - 384) // 2
        im = im.crop((left, top, left + 384, top + 384))
        a = np.asarray(im, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None]


def validate(onnx_path: Path, names: list[str], image_dir: Path | None) -> dict:
    import numpy as np
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    images = sorted(p for p in (image_dir or Path()).rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:24] if image_dir else []
    if not images:
        print("no images given: skipping parity check")
        return {}
    lite = Interpreter(model_path=str(TFLITE)).get_signature_runner(SIGNATURE)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = {n: 0.0 for n in names}
    ranges = {n: [np.inf, -np.inf] for n in names}
    sums = {n: [] for n in names}
    emb_cos = []
    for p in images:
        x = _prep(p)
        ref = lite(image=x)
        got = dict(zip([o.name for o in sess.get_outputs()],
                       sess.run(None, {"image": x})))
        for n in names:
            a, b = ref[n].ravel(), got[n].ravel()
            worst[n] = max(worst[n], float(np.max(np.abs(a - b))))
            ranges[n] = [min(ranges[n][0], float(a.min())), max(ranges[n][1], float(a.max()))]
            sums[n].append(float(a.sum()))
        a, b = ref["embedding"].ravel(), got["embedding"].ravel()
        emb_cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"parity over {len(images)} image(s):")
    for n in names:
        print(f"  {n:<10} max|diff| {worst[n]:.2e}   range [{ranges[n][0]:.3f}, {ranges[n][1]:.3f}]")
    print(f"  embedding cosine min {min(emb_cos):.6f}")

    # Decide, per head, whether the graph already emits scores.
    acts = {}
    for head, act in HEAD_ACTIVATION.items():
        lo, hi = ranges[head]
        in_unit = lo >= -1e-4 and hi <= 1 + 1e-4
        if act == "softmax":
            normalised = in_unit and all(abs(s - 1) < 1e-3 for s in sums[head])
            acts[head] = "none" if normalised else "softmax"
        else:
            acts[head] = "none" if in_unit else "sigmoid"
    return {"images": len(images), "max_abs_diff": worst,
            "embedding_cos_min": min(emb_cos), "activations": acts}


def main() -> None:
    image_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT / "data" / "blobs"
    if not TFLITE.exists():
        raise SystemExit(f"{TFLITE} not found - download desert-ant-labs/eye first")
    onnx_path, outs = convert()
    report = validate(onnx_path, list(outs), image_dir)
    meta = {
        "source": {"file": TFLITE.name,
                   "sha256": hashlib.sha256(TFLITE.read_bytes()).hexdigest()},
        "signature": SIGNATURE,
        "input": "image",
        "outputs": list(outs),
        # Fall back to the documented semantics when no parity run happened.
        "activations": report.get("activations", HEAD_ACTIVATION),
        "parity": {k: v for k, v in report.items() if k != "activations"},
    }
    (EYE_DIR / f"eye_{SIGNATURE}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {onnx_path.name} and eye_{SIGNATURE}.json")


if __name__ == "__main__":
    main()
