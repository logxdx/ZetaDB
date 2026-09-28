"""Source-agnostic text box.

The layout engine must never know whether boxes came from PP-OCR or from a
PDF text layer: that is what lets the v1.5 PDF path reuse stages 2-7
unchanged (spec 6.4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 1.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class LayoutResult:
    layout_text: str   # grid-projected; for the VLM, Needle and display
    flow_text: str     # plain reading order; for FTS5 and embeddings
    n_lines: int
    n_anchors: int
    flowing: bool      # True when the whole page took the prose bypass
