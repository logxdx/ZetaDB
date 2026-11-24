"""EmbeddingGemma 2 - one model, one space, for text and images.

Replaces jina-clip-v2 (vec_jina) and Qwen3-Embedding-0.6B (vec_text). Every
modality lands in the same 768-d space, which is what retires the spec 9.4
workaround: today an image's caption is embedded a *second* time through
Qwen purely so that one space exists containing both notes and images.

Shape of the ONNX export (onnx-community/embeddinggemma-2-ONNX), which
drives every design choice below:

  vision_encoder.onnx   pixel_values       [B, patches, 768]  -> image_features [tokens, 512]
                        pixel_position_ids [B, patches, 2]
  model.onnx            input_ids, attention_mask,            -> sentence_embedding [B, 768]
                        image_features / video_features / audio_features

Two consequences:

  * Pooling is inside the graph. `sentence_embedding` is the finished
    vector - there is no mean-pool step to get wrong, and no hand-split of
    a combined graph the way jina-clip needed (spec 7.2 [M3]).
  * An image embedding needs BOTH graphs. The vision encoder only emits
    soft tokens; they are spliced into the backbone's sequence at the
    <|image|> placeholders and the backbone produces the vector. There is
    no standalone image tower, so "load the image side only" is not a
    thing here.

Precision: int8 by default, because it is smallest and fastest. [M0]
against the unquantised graphs, cosine agreement with fp32 on identical
input is 0.9954-1.0000 for int8 and 0.9999-1.0000 for fp16, with retrieval
margin 0.212 vs 0.216 - all three are interchangeable on quality, and int8
is 509 MB against fp16's 878 MB.

Google's card warns that fp16 gives NaN or silently degraded embeddings
because the activation range exceeds fp16's dynamic range. That does not
reproduce on THIS export, on CPU or CUDA: onnx-community ships a mixed
precision graph (320 fp16 weight tensors, 7 kept fp32, and ReduceSum / Max
/ Sin / Cos still emitting fp32), so the sensitive ops never run in fp16.
The warning applies to a torch_dtype=float16 load, which this is not.

Placement: vision encoder on the GPU (pool A), backbone on the CPU. The
backbone is the only part search needs, and keeping it off the GPU means a
query never waits for pool A to wake - the same reasoning that put jina's
text tower and Qwen on the CPU (spec 7.2). The soft tokens crossing between
them are [~280, 512] floats, about 570 KB, so the split costs nothing.

Audio and video are deliberately not wired up: intake accepts images and
text only, so the 300M audio encoder is never downloaded.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from server.config import load
from server.engine import ModelSpec, Profile, register_profile, session
from server.engine.core import engine as model_engine
from server.engine.onnx import available_providers
from server.models.embeddings import pack

log = logging.getLogger(__name__)

# Task instruction prefixes, from the model card's prefix table. These apply
# to TEXT ONLY - images are passed with no prefix at all. Omitting them
# still works but measurably costs precision, and query/document asymmetry
# is the whole point: mixing the two sides measurably hurt retrieval with
# the Qwen embedder this replaced, too.
QUERY_PREFIX = "task: search result | query: "
SIMILARITY_PREFIX = "task: sentence similarity | query: "

# Placeholders understood by format_document(); each is expanded to the real
# begin / soft-token run / end at tokenisation time, in the layout
# EmbeddingGemma2Processor uses (transformers, processing_embedding_gemma2):
#   image  <boi> <image>*n <eoi>
#   video  per frame: <boi> <video>*n <eoi>   (no timestamps)
#   audio  <boa> <audio>*n <eoa>
IMAGE_MARK = "<|image|>"
VIDEO_MARK = "<|video|>"
AUDIO_MARK = "<|audio|>"
_MARKS = re.compile(r"(<\|image\|>|<\|video\|>|<\|audio\|>)")

# Fallbacks only; the real values are read from the checkpoint's config.json.
_DEFAULT_IDS = {"bos": 2, "boi": 255999, "image": 258880, "eoi": 258882,
                "boa": 256000, "audio": 258881, "eoa": 258883, "video": 258884}


def format_query(text: str) -> str:
    """Query side of an asymmetric retrieval pair."""
    return QUERY_PREFIX + text


def format_document(content: str, title: str | None = None) -> str:
    """Document side. The card requires the title field to always be
    present, with the literal `none` when an item has no title - and
    `prompt_name="Document"` in sentence-transformers does exactly this."""
    t = title.strip() if title and title.strip() else "none"
    return f"title: {t} | text: {content}"


# TensorRT profile for the vision encoder, used only when a TRT-runnable
# (non-dynamically-quantised) graph is selected: one image or one video
# frame at a time, up to the larger of the two soft-token budgets * 3x3 pool.
def _vision_profile(max_tokens: int) -> Profile:
    hi = max_tokens * 9
    return Profile(
        min="pixel_values:1x9x768,pixel_position_ids:1x9x2",
        opt=f"pixel_values:1x{hi}x768,pixel_position_ids:1x{hi}x2",
        max=f"pixel_values:1x{hi}x768,pixel_position_ids:1x{hi}x2")


class Eg2:
    def __init__(self, key: str = "eg2") -> None:
        """`key` selects the config block, so an alternate build (the fp32
        reference, say) can be instantiated beside the default one to
        measure what quantisation costs. Everything in the serving path
        uses engine(), which only ever builds the default."""
        cfg = load()
        d = cfg.path("paths.models") / key
        self.key = key
        self.dir = d
        dtype = cfg.get(f"models.{key}.dtype", "quantized")
        sfx = f"_{dtype}" if dtype else ""      # "" selects the fp32 graphs
        # The encoders may use a different graph from the backbone: TensorRT
        # cannot run a dynamically quantised graph, so selecting the fp16 or
        # fp32 vision encoder is what puts it on TensorRT (spec 3.0).
        vdtype = cfg.get(f"models.{key}.vision_dtype", dtype)
        vsfx = f"_{vdtype}" if vdtype else ""
        adtype = cfg.get(f"models.{key}.audio_dtype", dtype)
        asfx = f"_{adtype}" if adtype else ""
        self.backbone_path = d / "onnx" / f"model{sfx}.onnx"
        self.vision_path = d / "onnx" / f"vision_encoder{vsfx}.onnx"
        self.audio_path = d / "onnx" / f"audio_encoder{asfx}.onnx"
        self.tok_path = d / "tokenizer.json"
        self.dim = cfg.get(f"models.{key}.dim", 768)
        # Matryoshka: 768 native, truncatable to 512/256/128. 0 = no
        # truncation. Truncating REQUIRES re-normalising, which pack() does.
        self.truncate_dim = cfg.get(f"models.{key}.truncate_dim", 0) or None
        self.soft_tokens = cfg.get(f"models.{key}.image_soft_tokens", 280)
        # Per video frame. The checkpoint's video processor uses 140.
        self.video_soft_tokens = cfg.get(f"models.{key}.video_soft_tokens", 140)
        self.backbone_gpu = cfg.get(f"models.{key}.backbone_device", "cpu") == "cuda"
        # Work done under the pool A lease (ingest: images, frames, audio,
        # passages) borrows a CUDA copy of the backbone, released with the
        # encoders. Search keeps the CPU copy. Measured on the 6 GB card:
        # +686 MB, a 30 s audio window 0.79 s -> 0.13 s, 32 passages 2.3 s -> 0.19 s.
        self.ingest_backbone_gpu = (not self.backbone_gpu and cfg.get(
            f"models.{key}.ingest_backbone_device", "cuda") == "cuda")
        self.vision_backend = cfg.get(f"models.{key}.vision_backend", "auto")
        self.audio_backend = cfg.get(f"models.{key}.audio_backend", "auto")
        self.max_tok = cfg.get(f"models.{key}.max_text_tokens", 8192)
        self.available = bool(self.backbone_path.exists() and self.tok_path.exists())
        self.vision_available = self.available and self.vision_path.exists()
        self.audio_available = self.available and self.audio_path.exists()
        self._bb = None
        self._bb_gpu = None
        self._vis = None
        self._aud = None
        self._tok = None
        self._procs: dict[int, object] = {}
        self._afe = None
        self._ids = dict(_DEFAULT_IDS)
        cfg_json = d / "config.json"
        if cfg_json.exists():
            try:
                c = json.loads(cfg_json.read_text())
                self._ids = {
                    "bos": c.get("text_config", {}).get("bos_token_id", _DEFAULT_IDS["bos"]),
                    "boi": c.get("boi_token_id", _DEFAULT_IDS["boi"]),
                    "image": c.get("image_token_id", _DEFAULT_IDS["image"]),
                    "eoi": c.get("eoi_token_id", _DEFAULT_IDS["eoi"]),
                    "boa": c.get("boa_token_id", _DEFAULT_IDS["boa"]),
                    "audio": c.get("audio_token_id", _DEFAULT_IDS["audio"]),
                    # the checkpoint spells this one eoa_token_index
                    "eoa": c.get("eoa_token_id", c.get("eoa_token_index", _DEFAULT_IDS["eoa"])),
                    "video": c.get("video_token_id", _DEFAULT_IDS["video"]),
                }
            except Exception:
                log.warning("eg2 config.json unreadable; using default token ids", exc_info=True)
        if key == "eg2":
            register_profile("eg2_vision", _vision_profile(max(self.soft_tokens,
                                                               self.video_soft_tokens)))
            model_engine.register(ModelSpec(
                name="eg2-encoders", pool="A", backend="onnx", unload=self.unload,
                provider=self._encoder_providers,
                loaded=lambda: (self._vis is not None or self._aud is not None
                                or self._bb_gpu is not None)))
            model_engine.register(ModelSpec(
                name="eg2-backbone", pool="A" if self.backbone_gpu else "CPU",
                backend="onnx", unload=None,
                provider=lambda: self._bb.get_providers()[0] if self._bb else "not loaded",
                loaded=lambda: self._bb is not None))

    def describe(self) -> str:
        where = "cuda" if self.backbone_gpu else "cpu"
        d = self.truncate_dim or self.dim
        return f"{self.backbone_path.name} ({where}, {d}d)"

    def _encoder_providers(self) -> str:
        parts = []
        if self._vis is not None:
            parts.append(f"vision={self._vis.get_providers()[0]}")
        if self._aud is not None:
            parts.append(f"audio={self._aud.get_providers()[0]}")
        if self._bb_gpu is not None:
            parts.append(f"ingest backbone={self._bb_gpu.get_providers()[0]}")
        return ", ".join(parts) or "not loaded"

    # ------------------------------------------------------------- sessions

    def load_backbone(self) -> None:
        if self.available and self._bb is None:
            backend = "onnx-cuda" if self.backbone_gpu else "onnx-cpu"
            self._bb = session(self.backbone_path, "eg2_backbone", backend)

    def _backbone(self):
        """The session to run: the CUDA copy when this thread holds pool A
        and ingest is allowed to use it, the resident one otherwise."""
        if (self.ingest_backbone_gpu and self.available and model_engine.holds("A")
                and "CUDAExecutionProvider" in available_providers()):
            if self._bb_gpu is None:
                self._bb_gpu = session(self.backbone_path, "eg2_backbone_gpu", "onnx-cuda")
            return self._bb_gpu
        self.load_backbone()
        return self._bb

    def load_vision(self) -> None:
        if self.vision_available and self._vis is None:
            self._vis = session(self.vision_path, "eg2_vision", self.vision_backend)

    def load_audio(self) -> None:
        if self.audio_available and self._aud is None:
            self._aud = session(self.audio_path, "eg2_audio", self.audio_backend)

    def unload(self) -> None:
        """Release the GPU-resident encoders. The backbone is on the CPU and
        search needs it on every query, so it stays; its ingest-time CUDA
        copy goes with the encoders."""
        self._vis = None
        self._aud = None
        self._bb_gpu = None
        if self.backbone_gpu:
            self._bb = None

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self.tok_path))
        return self._tok

    def _processor(self, max_soft_tokens: int | None = None):
        """Reference preprocessor, PIL/numpy backend - no torch in the
        serving path. Gemma 4 resizes aspect-preserving to a multiple of
        patch_size * pooling_kernel_size and emits flattened 16x16 patches
        plus their (row, col) ids, which is not something worth
        reimplementing from the config file by hand. Video frames use the
        same processor at the video budget (140), exactly as the
        checkpoint's video processor does."""
        budget = max_soft_tokens or self.soft_tokens
        if budget not in self._procs:
            from transformers.models.gemma4.image_processing_pil_gemma4 import (
                Gemma4ImageProcessorPil,
            )
            self._procs[budget] = Gemma4ImageProcessorPil(
                patch_size=16,
                pooling_kernel_size=3,
                max_soft_tokens=budget,
                do_convert_rgb=True,
                do_rescale=True,
                rescale_factor=1.0 / 255.0,
                do_normalize=False,   # image_mean 0 / image_std 1 per processor_config
            )
        return self._procs[budget]

    def _audio_features_extractor(self):
        """Gemma 4's mel front end (numpy only), configured from the
        checkpoint's processor_config.json. It truncates at 480 000 samples
        - 30 s at 16 kHz - which is the model's real audio horizon: anything
        longer is silently ignored, so callers window long audio first."""
        if self._afe is None:
            from transformers.models.gemma4.feature_extraction_gemma4 import (
                Gemma4AudioFeatureExtractor,
            )
            kwargs = {}
            pc = self.dir / "processor_config.json"
            if pc.exists():
                try:
                    fe = json.loads(pc.read_text()).get("feature_extractor", {})
                    allowed = {"feature_size", "sampling_rate", "padding_value",
                               "min_frequency", "max_frequency", "preemphasis",
                               "preemphasis_htk_flavor", "fft_overdrive", "dither",
                               "input_scale_factor", "mel_floor"}
                    kwargs = {k: v for k, v in fe.items() if k in allowed and v is not None}
                except Exception:
                    log.warning("processor_config.json unreadable; default audio front end",
                                exc_info=True)
            self._afe = Gemma4AudioFeatureExtractor(**kwargs)
        return self._afe

    # ------------------------------------------------------------- encoding

    def _encode_ids(self, text: str, n_soft: int = 0, *,
                    video_frames: list[int] | None = None,
                    audio_tokens: list[int] | None = None) -> list[int]:
        """Token ids for `text`, expanding each placeholder into its real
        begin / soft-token / end run. Segments are encoded separately because
        the markers are not tokens in the text vocabulary.

        `n_soft` is the soft-token count for every IMAGE_MARK; `video_frames`
        gives, per VIDEO_MARK in order, the soft-token count of each frame;
        `audio_tokens` gives the count for each AUDIO_MARK in order."""
        tok = self._tokenizer()
        ids: list[int] = [self._ids["bos"]]
        vids = list(video_frames or [])
        auds = list(audio_tokens or [])
        for part in _MARKS.split(text):
            if not part:
                continue
            if part == IMAGE_MARK:
                ids += [self._ids["boi"], *[self._ids["image"]] * n_soft, self._ids["eoi"]]
            elif part == VIDEO_MARK:
                for n in (vids.pop(0) if vids else []):
                    ids += [self._ids["boi"], *[self._ids["video"]] * n, self._ids["eoi"]]
            elif part == AUDIO_MARK:
                n = auds.pop(0) if auds else 0
                ids += [self._ids["boa"], *[self._ids["audio"]] * n, self._ids["eoa"]]
            else:
                ids.extend(tok.encode(part, add_special_tokens=False).ids)
        if len(ids) > self.max_tok:
            # Soft tokens must never be cut mid-run: the backbone requires
            # every feature row to have its placeholder. Text after the media
            # is what gets dropped.
            log.warning("eg2 input of %d tokens truncated to %d", len(ids), self.max_tok)
        return ids[:self.max_tok]

    def _run_backbone(self, batch_ids: list[list[int]],
                      image_features: np.ndarray | None = None,
                      video_features: np.ndarray | None = None,
                      audio_features: np.ndarray | None = None) -> np.ndarray:
        """Right-pad to a rectangle and run. Unused modality inputs are
        required by the graph, so they are passed as empty [0, 512]."""
        bb = self._backbone()
        width = max(len(i) for i in batch_ids)
        ids = np.zeros((len(batch_ids), width), dtype=np.int64)
        mask = np.zeros((len(batch_ids), width), dtype=np.int64)
        for r, row in enumerate(batch_ids):
            ids[r, :len(row)] = row
            mask[r, :len(row)] = 1
        empty = np.zeros((0, 512), dtype=np.float32)
        feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "image_features": empty if image_features is None else image_features,
            "video_features": empty if video_features is None else video_features,
            "audio_features": empty if audio_features is None else audio_features,
        }
        out = bb.run(["sentence_embedding"], feeds)[0]
        return out[:, :self.truncate_dim] if self.truncate_dim else out

    # --------------------------------------------------------------- public

    def embed_texts(self, texts: list[str], *, is_query: bool = False,
                    batch: int = 8) -> list[bytes]:
        if not self.available or not texts:
            return []
        out: list[bytes] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            prepped = [format_query(t) if is_query else t for t in chunk]
            vecs = self._run_backbone([self._encode_ids(t) for t in prepped])
            out.extend(pack(v) for v in vecs)
        return out

    def embed_text(self, text: str) -> bytes:
        """Query-side embedding, used by search."""
        out = self.embed_texts([text], is_query=True)
        if not out:
            raise RuntimeError("eg2 embedder unavailable")
        return out[0]

    def _vision_feats(self, im, budget: int) -> np.ndarray:
        """One PIL image -> soft tokens [n, 512] at the given budget."""
        self.load_vision()
        feat = self._processor(budget)(images=[im.convert("RGB")], return_tensors="np")
        px = np.asarray(feat["pixel_values"], dtype=np.float32)
        pos = np.asarray(feat["image_position_ids"], dtype=np.int64)
        if px.ndim == 2:          # [patches, 768] -> [1, patches, 768]
            px, pos = px[None], pos[None]
        return self._vis.run(
            ["image_features"], {"pixel_values": px, "pixel_position_ids": pos})[0]

    def _vision_pass(self, path: Path) -> np.ndarray | None:
        """Image -> soft tokens. The expensive half, and the only half that
        touches the GPU, so it runs once per image even when both the
        interleaved and the bare vector are wanted."""
        if not self.vision_available:
            return None
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                return self._vision_feats(im, self.soft_tokens)
        except Exception:
            log.warning("eg2 image prep failed: %s", path, exc_info=True)
            return None

    def _from_soft_tokens(self, feats: np.ndarray, text: str | None,
                          title: str | None) -> bytes:
        body = IMAGE_MARK if not text else f"{IMAGE_MARK} {text}"
        ids = self._encode_ids(format_document(body, title), int(feats.shape[0]))
        return pack(self._run_backbone([ids], feats)[0])

    def embed_image_pair(self, path: Path, text: str | None = None,
                         title: str | None = None) -> tuple[bytes | None, bytes | None]:
        """(item, visual) for one image, from a single vision pass.

        `item` interleaves the image with its own caption and OCR text at
        the <|image|> placeholder. [M0] on 8 captioned images that is worth
        0.291 of margin against 0.224 for the image alone - the text says
        things the pixels do not, and both end up in one vector instead of
        the three rows spec 9.3 used to write.

        `visual` is the image with no text at all. Zero-shot tag scoring
        must use it: scoring the term "chart" against a vector that already
        contains the image's OCR text would match any screenshot whose text
        merely says "chart" (spec 7.2, and the reason requires_text exists).

        The second vector costs one extra backbone pass on the CPU, not a
        second vision pass - the soft tokens are computed once and reused.
        """
        feats = self._vision_pass(path)
        if feats is None:
            return None, None
        item = self._from_soft_tokens(feats, text, title) if text else None
        visual = self._from_soft_tokens(feats, None, None)
        return (item or visual), visual

    def embed_image(self, path: Path, text: str | None = None,
                    title: str | None = None) -> bytes | None:
        """One vector for an image, optionally interleaved with its text."""
        feats = self._vision_pass(path)
        return None if feats is None else self._from_soft_tokens(feats, text, title)

    # ----------------------------------------------------------- audio/video

    def embed_frames(self, frames: list, title: str | None = None) -> bytes | None:
        """A window of video frames (PIL images, in time order) as ONE vector,
        in the checkpoint's video layout: each frame is its own
        <boi> <video>*n <eoi> run at the video budget, no timestamps. No task
        prefix - like images, media is embedded bare (spec 9.3); the title
        rides in the document format so a well-named clip still says so."""
        if not self.vision_available or not frames:
            return None
        feats = [self._vision_feats(f, self.video_soft_tokens) for f in frames]
        counts = [int(f.shape[0]) for f in feats]
        ids = self._encode_ids(format_document(VIDEO_MARK, title), video_frames=[counts])
        return pack(self._run_backbone([ids], video_features=np.concatenate(feats))[0])

    def embed_audio(self, wave: np.ndarray, sampling_rate: int = 16_000,
                    title: str | None = None) -> bytes | None:
        """One window of mono audio (<= 30 s, float32 in -1..1) as one vector.

        The audio encoder emits exactly as many rows as there are audio soft
        tokens; that count IS the placeholder count, so the two can never
        disagree the way a separately computed count could."""
        if not self.audio_available or wave is None or len(wave) == 0:
            return None
        self.load_audio()
        # Passed as a one-element batch: that is how EmbeddingGemma2Processor
        # calls it, and transformers 5.17's un-batched path transposes a bare
        # 1-D waveform the wrong way and fails in _extract_spectrogram.
        fe = self._audio_features_extractor()(
            [np.asarray(wave, dtype=np.float32)], sampling_rate=sampling_rate,
            return_tensors="np")
        x = np.asarray(fe["input_features"], dtype=np.float32)
        m = np.asarray(fe["input_features_mask"]).astype(bool)
        feats = self._aud.run(["audio_features"],
                              {"input_features": x, "input_features_mask": m})[0]
        if feats.shape[0] == 0:
            return None
        ids = self._encode_ids(format_document(AUDIO_MARK, title),
                               audio_tokens=[int(feats.shape[0])])
        return pack(self._run_backbone([ids], audio_features=feats)[0])


_engine: Eg2 | None = None


def engine() -> Eg2:
    """Process-wide singleton.

    Same reason as jina-clip (spec 7.2 [M3]): the tagging stage and the
    embed stage each used to construct their own model, the second CUDA
    session failed to allocate, and the only visible symptom was a silent
    drop in the term-score count.
    """
    global _engine
    if _engine is None:
        _engine = Eg2()
    return _engine
