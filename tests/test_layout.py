"""Regression tests for the grid projection layout engine (spec 6).

Each case encodes a bug found during M2:
  - columns must stay aligned across rows (forward anchors)
  - prose must take the flowing bypass and never be reordered
  - right-aligned columns propagate their END, not their start
  - flow_text must be unpadded; layout_text must be padded
  - blank-line block separation uses line PITCH, not glyph height
"""
from __future__ import annotations

import math

from server.layout.grid import project
from server.layout.types import TextBox

CW, LH = 8.0, 20.0


def row(y, cells):
    return [TextBox(c * CW, y * LH, (c + len(t)) * CW, y * LH + 14, t) for c, t in cells]


def table_boxes():
    b = row(0, [(0, "Model"), (20, "Params"), (34, "Det"), (44, "Rec")])
    b += row(1, [(0, "PP-OCRv6_tiny"), (20, "1.5M"), (34, "80.6"), (44, "73.5")])
    b += row(2, [(0, "PP-OCRv6_small"), (20, "7.7M"), (34, "84.1"), (44, "81.3")])
    b += row(3, [(0, "PP-OCRv6_medium"), (20, "34.5M"), (34, "86.2"), (44, "83.2")])
    return b


PROSE = ("The model returns a sigmoid score for every one of its five thousand "
         "Danbooru tags independently so we read off only our own vocabulary "
         "terms and keep a per term threshold that adapts from feedback").split()


def prose_boxes():
    b, x, y = [], 0, 0
    for wd in PROSE:
        if x + len(wd) > 58:
            x, y = 0, y + 1
        b.append(TextBox(x * CW, y * LH, (x + len(wd)) * CW, y * LH + 14, wd))
        x += len(wd) + 1
    return b


def test_table_columns_stay_aligned():
    r = project(table_boxes(), page_width=60 * CW)
    lines = r.layout_text.splitlines()
    assert len(lines) == 4, r.layout_text
    for header, values in (("Det", ["80.6", "84.1", "86.2"]),
                           ("Rec", ["73.5", "81.3", "83.2"])):
        cols = [lines[0].index(header)] + [
            lines[i + 1].index(v) for i, v in enumerate(values)]
        assert len(set(cols)) == 1, f"{header} misaligned: {cols}\n{r.layout_text}"


def test_prose_takes_bypass_and_keeps_word_order():
    r = project(prose_boxes(), page_width=60 * CW)
    assert r.flowing, f"prose was gridded, anchors={r.n_anchors}"
    assert " ".join(r.layout_text.split()) == " ".join(PROSE)


def test_right_aligned_column_propagates_its_end():
    b = row(0, [(0, "Subtotal")]) + [TextBox(40 * CW, 0, 48 * CW, 14, "1,240.00")]
    b += row(1, [(0, "Tax")]) + [TextBox(42 * CW, LH, 48 * CW, LH + 14, "99.20")]
    b += row(2, [(0, "Total")]) + [TextBox(40 * CW, 2 * LH, 48 * CW, 2 * LH + 14, "1,339.20")]
    r = project(b, page_width=60 * CW)
    ends = [len(ln.rstrip()) for ln in r.layout_text.splitlines() if ln.strip()]
    assert len(set(ends)) == 1, f"right edges {ends}\n{r.layout_text}"


def test_flow_text_is_unpadded_layout_text_is_padded():
    r = project(table_boxes(), page_width=60 * CW)
    assert "  " not in r.flow_text, "flow_text must not carry grid padding"
    assert "  " in r.layout_text, "layout_text must preserve columns"
    assert "86.2" in r.flow_text


def test_blank_line_separates_blocks_but_normal_spacing_does_not():
    tight = row(0, [(0, "Header one")]) + row(1, [(0, "body line")])
    assert "\n\n" not in project(tight, page_width=60 * CW).layout_text
    gapped = tight + row(5, [(0, "Second block")])
    assert "\n\n" in project(gapped, page_width=60 * CW).layout_text


def test_deskew_recovers_alignment():
    b = []
    for t, xx in (("alpha", 0), ("beta", 20), ("gamma", 40)):
        for j in range(3):
            x = xx * CW
            y = j * LH + x * math.tan(math.radians(2.0))
            b.append(TextBox(x, y, (xx + len(t)) * CW, y + 14, t))
    r = project(b, page_width=60 * CW)
    cols = [tuple(ln.index(t) for t in ("alpha", "beta", "gamma") if t in ln)
            for ln in r.layout_text.splitlines() if ln.strip()]
    assert len(set(cols)) == 1, f"{cols}\n{r.layout_text}"


def test_empty_and_whitespace_input():
    assert project([]).layout_text == ""
    assert project([TextBox(0, 0, 10, 10, "   ")]).flow_text == ""


def test_deskew_does_not_flip_an_unskewed_page():
    """Regression: deskew measured angles between consecutive boxes in y
    order. Same-line pairs can have negative dx, so atan2 returned values
    near +/-pi and the median rotated the page ~180 degrees, reversing both
    reading order and line order on real OCR output."""
    b = table_boxes()
    r = project(b, page_width=60 * CW)
    lines = r.layout_text.splitlines()
    assert lines[0].startswith("Model"), f"line order flipped:\n{r.layout_text}"
    assert lines[-1].startswith("PP-OCRv6_medium"), r.layout_text
    # And the first cell of a row must still be its leftmost box.
    assert lines[1].index("PP-OCRv6_tiny") < lines[1].index("1.5M")


def test_deskew_ignores_implausible_angles():
    """A near-vertical estimate must be rejected, not applied."""
    b = [TextBox(0, 0, 40, 14, "a"), TextBox(0, 400, 40, 414, "b"),
         TextBox(0, 800, 40, 814, "c"), TextBox(0, 1200, 40, 1214, "d")]
    r = project(b, page_width=480)
    assert [ln.strip() for ln in r.layout_text.splitlines() if ln.strip()] == ["a", "b", "c", "d"]
