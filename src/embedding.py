"""
Multimodal embeddings with EmbeddingGemma 2.

Text, code, images, audio and video are projected into one shared vector
space, so any modality can be used to search any other. Prompts follow the
EmbeddingGemma 2 developer guide: task prompts for text *queries*, the
"Document" format (``title: … | text: …``) for indexed text, and no prompt for
media.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch

from .config import (
    DEFAULT_QUERY_INTENT,
    DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    ENABLED_MODALITIES,
    IMAGE_SOFT_TOKENS,
    QUERY_CACHE_SIZE,
    QUERY_PROMPTS,
    VIDEO_FRAMES_PER_WINDOW,
)
from .segments import prepare_file
from .utils import normalize_vectors

logger = logging.getLogger(__name__)

NATIVE_DIMENSION = 768

# Pre-decoded frame lists carry no fps metadata, so sample by count only
FRAME_LIST_PROCESSING = {"video": {"fps": None, "max_frames": VIDEO_FRAMES_PER_WINDOW}}


class GemmaEmbedder:
    """Loads EmbeddingGemma 2 and embeds queries and segments of every modality."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = DEVICE,
        embedding_dim: int = EMBEDDING_DIMENSION,
    ):
        # Imported lazily so that importing the package stays cheap
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        # The model is not guaranteed to be thread-safe; serialise access to it
        self._lock = threading.Lock()
        self._query_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

        # float16 is not supported (NaNs / degraded vectors): bf16 on capable GPUs, else fp32
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Modular loading: skip encoders for modalities that are switched off.
        # Every configuration shares the same vector space.
        config_kwargs = {}
        if not {"image", "video"} & set(ENABLED_MODALITIES):
            config_kwargs["vision_config"] = None
        if "audio" not in ENABLED_MODALITIES:
            config_kwargs["audio_config"] = None

        logger.info(
            f"Loading {model_name} on {device} ({dtype}); modalities: {', '.join(ENABLED_MODALITIES)} ..."
        )
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": dtype},
            config_kwargs=config_kwargs or None,
            truncate_dim=None if embedding_dim == NATIVE_DIMENSION else embedding_dim,
        )
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------ core
    def _encode_many(self, inputs: List[Any], **kwargs) -> np.ndarray:
        with self._lock, torch.inference_mode():
            out = self.model.encode(
                inputs,
                batch_size=len(inputs),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                **kwargs,
            )
        # Re-normalise in float32 (removes bf16 drift; required after Matryoshka truncation)
        return normalize_vectors(np.asarray(out, dtype=np.float32).reshape(len(inputs), -1))

    # --------------------------------------------------------------- queries
    def embed_query(self, text: str, intent: str = DEFAULT_QUERY_INTENT) -> np.ndarray:
        """
        Embed a text query with the task prompt for ``intent``:
        search (default, also used for cross-modal search), question, code, fact, similarity.
        Results are cached, so retyping or paging a query costs nothing.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Query text is empty.")
        prompt = QUERY_PROMPTS.get(intent)
        if prompt is None:
            raise ValueError(f"Unknown query intent '{intent}'. Use one of: {', '.join(QUERY_PROMPTS)}")
        key = (intent, text)
        cached = self._query_cache.get(key)
        if cached is not None:
            self._query_cache.move_to_end(key)
            return cached
        vector = self._encode_many([text], prompt_name=prompt)[0]
        self._query_cache[key] = vector
        while len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return vector

    def embed_file(self, path: Union[str, Path]) -> np.ndarray:
        """
        Embed a whole file as a query ("search with this file"): its segments are
        embedded exactly like indexed ones and averaged into one vector.
        """
        prepared = prepare_file(Path(path))
        vectors = []
        for kind, segs in _group_by_embed_kind(prepared.segments).items():
            for result in self.embed_batch(kind, [s.payload for s in segs]):
                if isinstance(result, Exception):
                    raise result
                vectors.append(result)
        return normalize_vectors(np.mean(np.stack(vectors), axis=0))

    # ------------------------------------------------------------- batching
    def embed_batch(self, kind: str, payloads: List[Any]) -> List[Union[np.ndarray, Exception]]:
        """
        Embed a homogeneous batch of segment payloads. ``kind`` is the model
        input type: text | image | audio | video.

        Robust by design: a CUDA OOM halves the batch and retries, and any other
        failure bisects the batch so that only the offending item(s) fail.

        Returns:
            One entry per payload: its embedding, or the exception it raised.
        """
        if not payloads:
            return []
        try:
            return list(self._encode_kind(kind, payloads))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if len(payloads) == 1:
                return [e]
            logger.warning(f"CUDA OOM on a {kind} batch of {len(payloads)}; splitting it")
        except Exception as e:
            if len(payloads) == 1:
                return [e]
        mid = len(payloads) // 2
        return self.embed_batch(kind, payloads[:mid]) + self.embed_batch(kind, payloads[mid:])

    def _encode_kind(self, kind: str, payloads: List[Any]) -> np.ndarray:
        if kind == "text":
            return self._encode_many(payloads)  # already in the Document format
        inputs = [{kind: p} for p in payloads]
        if kind == "video":
            return self._encode_many(inputs, processing_kwargs=FRAME_LIST_PROCESSING)
        if kind == "image" and IMAGE_SOFT_TOKENS != 280:
            return self._encode_many(inputs, processing_kwargs={"image": {"max_soft_tokens": IMAGE_SOFT_TOKENS}})
        return self._encode_many(inputs)


def _group_by_embed_kind(segments) -> "OrderedDict[str, list]":
    groups: "OrderedDict[str, list]" = OrderedDict()
    for seg in segments:
        groups.setdefault(seg.embed_as, []).append(seg)
    return groups
