"""Qwen3-Embedding-0.6B text embeddings, 1024-d, on CPU via llama.cpp.

CPU-resident by design: it must never contend with pool A or pool B for
the 6 GB budget (spec 3).
"""
from __future__ import annotations

import logging
from pathlib import Path

from server.config import load
from server.models.embeddings import pack
from server.models.llama_server import LlamaServer, find_binary

log = logging.getLogger(__name__)

# Qwen3-Embedding is instruction-aware; queries and documents get different
# prefixes and mixing them measurably hurts retrieval.
QUERY_PREFIX = ("Instruct: Given a search query, retrieve relevant notes and "
                "images that answer it\nQuery: ")


class QwenEmbedder:
    def __init__(self) -> None:
        cfg = load()
        d = cfg.path("paths.models") / "qwen_embed"
        self.path = next(iter(sorted(d.rglob("*.gguf"))), None) if d.exists() else None
        self.dim = cfg.get("models.qwen_embed.dim", 1024)
        self.available = bool(self.path and find_binary())
        self._srv: LlamaServer | None = None

    def describe(self) -> str:
        return f"{self.path.name if self.path else 'missing'} (cpu, {self.dim}d)"

    def _server(self) -> LlamaServer:
        if self._srv is None:
            self._srv = LlamaServer(self.path, ctx=2048, ngl=0, embedding=True, cpu_only=True)
        return self._srv

    def embed_many(self, texts: list[str], *, is_query: bool = False) -> list[bytes]:
        if not self.available or not texts:
            return []
        payload = [(QUERY_PREFIX + t if is_query else t)[:8000] for t in texts]
        return [pack(v) for v in self._server().embed(payload)]

    def embed(self, text: str) -> bytes:
        """Query-side embedding, used by search."""
        out = self.embed_many([text], is_query=True)
        if not out:
            raise RuntimeError("qwen embedder unavailable")
        return out[0]

    def stop(self) -> None:
        if self._srv:
            self._srv.stop()
            self._srv = None


_engine: QwenEmbedder | None = None


def engine() -> QwenEmbedder:
    """Process-wide singleton.

    Every instance owns a llama-server subprocess. resolve_or_propose()
    constructed one per proposed tag name, so a single captioning batch
    tried to spawn ~40 servers and took the process down with it.
    """
    global _engine
    if _engine is None:
        _engine = QwenEmbedder()
    return _engine
