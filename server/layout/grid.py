"""Grid projection - reading order from coordinates.

Adapted from LiteParse's algorithm for PDFs, which solves the same problem
OCR has: boxes carry coordinates but no reading order. Changes for OCR are
deskewing (PDFs are axis-aligned, photos are not) and deriving character
width from box_width/len(text) since there are no font metrics (spec 6).

Emits two renditions. layout_text preserves columns for the VLM and for
display; flow_text is plain reading order for FTS5 and embeddings, because
indexing padded text destroys BM25 length normalisation.
"""
from __future__ import annotations

import math
from statistics import median

from server.config import load
from server.layout.types import LayoutResult, TextBox


def deskew(boxes: list[TextBox], min_deg: float) -> list[TextBox]:
    """Estimate global skew and rotate it out.

    Skew is measured from baseline drift WITHIN provisional line groups, not
    from consecutive boxes in y order. Consecutive-pair angles mix same-line
    pairs (where dx can be negative, so atan2 returns values near +/-pi) with
    cross-line pairs, and the median of that is meaningless - it rotated
    pages by ~180 degrees, flipping the text both ways.
    """
    if len(boxes) < 4:
        return boxes
    heights = [b.height for b in boxes if b.height > 0]
    if not heights:
        return boxes
    h = median(heights)

    # Provisional grouping with a generous tolerance, purely to find baselines.
    groups: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if groups and abs(b.cy - groups[-1][0].cy) <= h * 1.2:
            groups[-1].append(b)
        else:
            groups.append([b])

    slopes = []
    for g in groups:
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda b: b.cx)
        dx = g[-1].cx - g[0].cx
        if dx > 4 * h:                       # need a real horizontal span
            slopes.append((g[-1].cy - g[0].cy) / dx)
    if not slopes:
        return boxes

    theta = math.atan(median(slopes))
    if abs(math.degrees(theta)) < min_deg or abs(math.degrees(theta)) > 20:
        return boxes

    cx = median([b.cx for b in boxes])
    cy = median([b.cy for b in boxes])
    cos, sin = math.cos(-theta), math.sin(-theta)
    out = []
    for b in boxes:
        dx, dy = b.cx - cx, b.cy - cy
        nx, ny = cx + dx * cos - dy * sin, cy + dx * sin + dy * cos
        hw, hh = b.width / 2, b.height / 2
        out.append(TextBox(nx - hw, ny - hh, nx + hw, ny + hh, b.text, b.score))
    return out


def group_lines(boxes: list[TextBox], tol_factor: float, tol_min: float
                ) -> tuple[list[list[TextBox]], float]:
    """Group by y-centre; tolerance absorbs subscripts and OCR jitter."""
    if not boxes:
        return [], 0.0
    h = median([b.height for b in boxes]) or tol_min
    tol = max(h * tol_factor, tol_min)
    lines: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: (b.cy, b.x0)):
        if lines and abs(b.cy - median([x.cy for x in lines[-1]])) <= tol:
            lines[-1].append(b)
        else:
            lines.append([b])
    for ln in lines:
        ln.sort(key=lambda b: b.x0)
    return lines, h


def char_width(boxes: list[TextBox]) -> float:
    widths = [b.width / max(1, len(b.text)) for b in boxes if b.text]
    return max(1.0, median(widths)) if widths else 1.0


def extract_anchors(lines: list[list[TextBox]], w: float, rnd: float,
                    min_frac: float, min_abs: int) -> dict[float, int]:
    """Recurring x positions in grid units, rounded to `rnd` units.

    Votes are counted per distinct LINE, not per box: an anchor means
    "text aligns here down the page". Two words happening to share an x
    within a single line is not alignment, and counting per box lets prose
    accumulate phantom anchors and never take the flowing bypass.
    """
    votes: dict[float, set[int]] = {}
    for idx, ln in enumerate(lines):
        seen: set[float] = set()
        for b in ln:
            for x in (b.x0, b.x1, b.cx):
                key = round(round((x / w) / rnd) * rnd, 4)
                seen.add(key)
        for key in seen:
            votes.setdefault(key, set()).add(idx)

    n_lines = max(1, len(lines))
    threshold = max(min_abs, int(math.ceil(min_frac * n_lines)))
    kept = {k: len(v) for k, v in votes.items() if len(v) >= threshold}
    # Suppress weaker anchors within 1.0 grid unit of a stronger one.
    out: dict[float, int] = {}
    for k in sorted(kept, key=lambda k: (-kept[k], k)):
        if all(abs(k - o) >= 1.0 for o in out):
            out[k] = kept[k]
    return out


def _snap(b: TextBox, w: float, anchors: dict[float, int]) -> tuple[str, float | None]:
    """Classify a box against the anchor set. Priority left > right > centre;
    ties resolve to the anchor shared by the most boxes."""
    best: tuple[str, float, int] | None = None
    for kind, x in (("left", b.x0), ("right", b.x1), ("centre", b.cx)):
        g = x / w
        for a, votes in anchors.items():
            if abs(g - a) <= 0.5:
                rank = {"left": 0, "right": 1, "centre": 2}[kind]
                cand = (kind, a, votes)
                if best is None:
                    best = cand
                else:
                    br = {"left": 0, "right": 1, "centre": 2}[best[0]]
                    if rank < br or (rank == br and votes > best[2]):
                        best = cand
    return (best[0], best[1]) if best else ("float", None)


def project(boxes: list[TextBox], *, page_width: float | None = None,
            cfg=None) -> LayoutResult:
    cfg = cfg or load()
    g = lambda k, d: cfg.get(f"layout.{k}", d)  # noqa: E731

    boxes = [b for b in boxes if b.text and b.text.strip()]
    if not boxes:
        return LayoutResult("", "", 0, 0, False)

    boxes = deskew(boxes, g("deskew_min_deg", 0.5))
    lines, line_h = group_lines(boxes, g("line_tol_factor", 0.5), g("line_tol_min_px", 5.0))
    w = char_width(boxes)
    anchors = extract_anchors(lines, w, g("anchor_round", 0.25),
                              g("anchor_min_count_frac", 0.08),
                              g("anchor_min_count_abs", 3))
    page_w = page_width or max(b.x1 for b in boxes)

    # Blank-line boundaries split the page into blocks. The threshold is a
    # multiple of the typical line PITCH (centre-to-centre), not of glyph
    # height: normal single spacing is already ~1.4x glyph height, so using
    # height here splits every single line into its own block.
    centres = [median([b.cy for b in ln]) for ln in lines]
    pitches = [b - a for a, b in zip(centres, centres[1:], strict=False)]
    pitch = median(pitches) if pitches else line_h
    blank_gap = max(pitch, line_h) * g("blank_line_factor", 1.4)
    blocks: list[list[list[TextBox]]] = [[]]
    prev_y = None
    for ln in lines:
        y = median([b.cy for b in ln])
        if prev_y is not None and (y - prev_y) > blank_gap:
            blocks.append([])
        blocks[-1].append(ln)
        prev_y = y
    blocks = [b for b in blocks if b]

    rendered: list[str] = []
    flow: list[str] = []
    all_flowing = True

    for block in blocks:
        block_boxes = [b for ln in block for b in ln]
        block_anchors = {a: v for a, v in anchors.items()
                         if any(abs(bb.x0 / w - a) <= 0.5 or abs(bb.x1 / w - a) <= 0.5
                                for bb in block_boxes)}
        widths = [max(bb.x1 for bb in ln) - min(bb.x0 for bb in ln) for ln in block]
        # Prose runs edge to edge with word-sized gaps. A right-aligned or
        # two-column table also spans the width and can have very few
        # anchors, so width alone misclassifies it; the giveaway is a large
        # internal gap. Measure the widest gap within each line, in columns.
        gaps = []
        for ln in block:
            ordered = sorted(ln, key=lambda b: b.x0)
            widest = 0.0
            for a, bb in zip(ordered, ordered[1:], strict=False):
                widest = max(widest, (bb.x0 - a.x1) / w)
            gaps.append(widest)
        flowing = (median(gaps) <= g("flowing_max_gap_cols", 3.0)
                   and median(widths) > g("flowing_min_width_frac", 0.5) * page_w)

        for ln in block:
            flow.append(" ".join(bb.text.strip() for bb in ln))

        if flowing:
            rendered.extend(" ".join(bb.text.strip() for bb in ln) for ln in block)
        else:
            all_flowing = False
            rendered.extend(_project_block(block, w, block_anchors))
        rendered.append("")
        flow.append("")

    layout_text = _post(rendered, g("sparse_whitespace_frac", 0.8))
    flow_text = "\n".join(flow).strip()
    while "\n\n\n" in flow_text:
        flow_text = flow_text.replace("\n\n\n", "\n\n")
    return LayoutResult(layout_text, flow_text, len(lines), len(anchors), all_flowing)


def _project_block(block: list[list[TextBox]], w: float,
                   anchors: dict[float, int]) -> list[str]:
    """Render one block onto a character grid.

    Boxes are *assigned* columns in priority order (float > left > right >
    centre) so a contested cell goes to the higher-priority item, but the
    assignment must not renumber them left-to-right: a monotonic cursor
    would push an early-priority box on the right side of the line into
    displacing everything after it, which reorders the text. Collisions are
    resolved against actually-occupied intervals instead.

    Forward anchors hold table columns aligned: once an anchored box lands
    at column N, that is a floor for the same anchor on every later line.
    """
    # Forward anchors are tracked separately per alignment: a left column
    # propagates its START, a right column propagates its END. Propagating a
    # left column for a right-aligned number walks it rightward each row.
    forward_left: dict[float, int] = {}
    forward_right: dict[float, int] = {}
    out: list[str] = []
    order = {"float": 0, "left": 1, "right": 2, "centre": 3}

    for ln in block:
        classified = [(b, *_snap(b, w, anchors)) for b in ln]
        # Priority decides whose desired column wins a contested cell...
        classified.sort(key=lambda t: (order[t[1]], t[0].x0))
        desired: dict[int, int] = {}
        for b, kind, anchor in classified:
            text = b.text.strip()
            if not text:
                continue
            if kind == "right":
                end = int(round(b.x1 / w))
                if anchor is not None:
                    end = max(end, forward_right.get(anchor, 0))
                desired[id(b)] = max(0, end - len(text))
            else:
                if kind == "centre":
                    target = int(round(b.cx / w)) - len(text) // 2
                else:
                    target = int(round(b.x0 / w))
                floor = forward_left.get(anchor, 0) if anchor is not None else 0
                desired[id(b)] = max(0, target, floor)

        # ...but rendering follows x0 order with monotonically increasing
        # columns. Sliding a box right past its neighbours to dodge a
        # collision would reorder the text, which is never acceptable.
        placed: list[tuple[int, int, str]] = []
        cursor = -1
        for b, kind, anchor in sorted(classified, key=lambda t: t[0].x0):
            text = b.text.strip()
            if not text:
                continue
            col = max(desired.get(id(b), 0), cursor + 1)
            placed.append((col, col + len(text), text))
            cursor = col + len(text)
            if anchor is not None:
                if kind == "right":
                    forward_right[anchor] = col + len(text)
                else:
                    forward_left[anchor] = col

        row = ""
        for start, _, text in sorted(placed):
            if len(row) > start:
                row += " "
            else:
                row += " " * (start - len(row))
            row += text
        out.append(row.rstrip())
    return out


def _post(lines: list[str], sparse_frac: float) -> str:
    """Compress mostly-empty blocks and strip the common indent."""
    text = "\n".join(lines)
    body = [ln for ln in lines if ln.strip()]
    if body:
        total = sum(len(ln) for ln in body)
        spaces = sum(ln.count(" ") for ln in body)
        if total and spaces / total > sparse_frac:
            out = []
            for ln in lines:
                while "    " in ln:
                    ln = ln.replace("    ", "  ")
                out.append(ln)
            lines = out
        indent = min((len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()), default=0)
        if indent:
            lines = [ln[indent:] if ln.strip() else ln for ln in lines]
    text = "\n".join(ln.rstrip() for ln in lines).strip("\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text
