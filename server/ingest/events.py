"""SSE hub. Lets the canvas show a card the instant intake finishes and
fill it in as later stages land (spec 4.2, 12)."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any

_MAX_SUBS = 32


class EventHub:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recent: deque = deque(maxlen=100)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        if len(self._subs) < _MAX_SUBS:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        """Thread-safe: the ingest worker calls this from its own thread."""
        payload = json.dumps({"event": event, "data": data})
        self._recent.append(payload)
        if self._loop is None:
            return
        for q in list(self._subs):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, payload)
            except (RuntimeError, asyncio.QueueFull):
                pass


hub = EventHub()
