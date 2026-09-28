"""Stage 5 - structured extraction with Needle 3. CPU only.

Scope is deliberately narrow, and narrower than the spec first assumed.

M0 measured Needle 3 as deterministic and accurate on short extractive
spans. M5 measured it on REAL pipeline text and found it much weaker: on a
779-character caption+OCR blob it either returned None or hallucinated
freely - dates that are not in the source, a dozen invented "$1.00"
amounts, URLs like example.com/price/1.00. On the shorter caption alone it
got dates, amounts and the URL right but invented one total and filled
`emails`, `orgs` and `people` with the same two junk values. Re-run against
a differently-worded caption of the same invoice it returned None 3 times
out of 3.

So the PRIMARY extractor is a regex pass (`scan`) over the full text. These
four field types are defined by their shape: a regex has perfect precision,
runs on everything rather than a 320-character window, needs no model and
cannot fabricate. Needle runs after it as a supplement that may only add
spans the regex missed, under the guards below.

So this stage does three things the naive version did not (spec 8.1):

  1. Caps input length. Quality collapses on long text; the caption is a
     better source than caption+layout_text.
  2. Verifies every extracted span appears VERBATIM in the source. For an
     extractive task that is exactly the right validator, and it converts
     hallucination into silence rather than into database rows.
  3. Does not attempt todos at all. Asked for action items on images it
     returned "Todos" and "PP-OCRv6"; asked on a clear 114-character note
     with two explicit intentions ("recalibrate the CLIP thresholds before
     Friday, and email the team") it returned None on 3 of 3 attempts. The
     `todos` table stays in the schema, but nothing writes to it until a
     model that can actually do this is wired up - the VLM under a GBNF
     grammar is the obvious route, and notes would need to pass through a
     text-only captioning call to reach it.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import unicodedata

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

from server.db.conn import reader
from server.db.writer import writer

log = logging.getLogger(__name__)

MIN_CHARS = 40       # below this Needle reliably returns None
MAX_CHARS = 320      # above this it starts fabricating

# Deterministic extractors over the FULL text. These four field types are
# defined by their shape, so a regex has perfect precision, needs no model,
# and cannot hallucinate. Needle runs afterwards as a supplement and may
# only add spans these missed - it never overrides them.
_SCAN = {
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "url": re.compile(r"https?://[^\s<>\"\')]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "amount": re.compile(r"(?<![\w.])(?:[$£€]\s?)?\d{1,3}(?:,\d{3})+(?:\.\d{2})?|(?<![\w.])[$£€]\s?\d+(?:\.\d{2})?"),
}


# A bare decimal is only money in context. "86.2" on a benchmark row is a
# score, not an amount, so bare decimals are accepted only on a line that
# names one.
_MONEY_LINE = re.compile(
    r"\b(total|subtotal|tax|vat|amount|price|cost|due|balance|paid|fee|"
    r"discount|invoice|charge)\b", re.I)
_BARE_DECIMAL = re.compile(r"(?<![\w.,])\d+\.\d{2}(?![\w.,])")


def scan(text: str) -> list[tuple[str, str]]:
    """Regex pass. Returns (kind, value) pairs, deduplicated, in order."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, raw: str) -> None:
        v = raw.strip().rstrip(".,;:")
        if v and (kind, v) not in seen:
            seen.add((kind, v))
            out.append((kind, v))

    for kind, rx in _SCAN.items():
        for m in rx.finditer(text):
            add(kind, m.group(0))
    for line in text.splitlines():
        if _MONEY_LINE.search(line):
            for m in _BARE_DECIMAL.finditer(line):
                add("amount", m.group(0))
    return out
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FIELD_SHAPE = {
    "date": _ISO,
    "url": re.compile(r"^https?://\S+$"),
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    # Anchored and explicit. The first attempt began with [^A-Za-z]* which
    # silently swallowed "2026-03-1" and then matched the "4", so every
    # ISO date was also stored as an amount.
    "amount": re.compile(r"^[$£€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$"),
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold().replace(",", "").replace(" ", "")


def verbatim(value: str, source: str) -> bool:
    """Is this span actually present in the source text?

    Needle invents plausible-looking values. Anything it reports that is
    not literally in the input is a fabrication, so drop it. Commas and
    whitespace are normalised away because OCR spacing varies.
    """
    return _norm(value) in _norm(source)


def _schemas():
    from pydantic import BaseModel

    class OcrEntities(BaseModel):
        # Only field types with a machine-checkable SHAPE. `orgs` and
        # `people` were dropped after M5: Needle fills them with whatever
        # it found in another field (both came back as
        # ["finance@acme.example", "portal link"]), and with no shape to
        # test against, the verbatim check alone cannot catch it. It also
        # missed the one real org in the M0 probe. A field we cannot
        # validate is a field we should not store.
        dates: list[str]
        amounts: list[str]
        urls: list[str]
        emails: list[str]

    return OcrEntities


def _available() -> bool:
    try:
        import needle  # noqa: F401
        return True
    except Exception:
        return False


def full_text(row: sqlite3.Row) -> str:
    """Everything we know about the item, for the regex pass."""
    if row["kind"] == "note":
        return (row["body"] or "").strip()
    parts = [row["caption"] or "", row["layout_text"] or ""]
    return "\n".join(p for p in parts if p.strip()).strip()


def source_text(row: sqlite3.Row) -> str:
    """Shortest text that still carries the facts, for the Needle pass.

    An image's caption already restates its dates, amounts and links and is
    far shorter than caption + layout_text, which is where Needle degraded
    into invention.
    """
    if row["kind"] == "note":
        return (row["body"] or "").strip()[:MAX_CHARS]
    return (row["caption"] or row["layout_text"] or "").strip()[:MAX_CHARS]


def run(item_ids: list[int]) -> None:
    if not _available():
        log.info("needle unavailable; skipping extraction for %d item(s)", len(item_ids))
        return
    import needle

    OcrEntities = _schemas()
    conn = reader()
    ents: list[tuple] = []
    dropped = 0

    for iid in item_ids:
        row = conn.execute(
            "SELECT id, kind, body, caption, layout_text FROM items WHERE id = ?", (iid,)
        ).fetchone()
        if row is None:
            continue
        # --- deterministic pass over everything we have
        whole = full_text(row)
        for kind, value in scan(whole):
            ents.append((iid, kind, value, 1.0))

        # --- model pass, capped and guarded, purely additive
        text = source_text(row)
        if len(text) < MIN_CHARS:
            continue

        try:
            got = needle.extract(text, OcrEntities, strict=False)
        except Exception:
            log.debug("entity extraction failed for %d", iid, exc_info=True)
            got = None
        if got is not None:
            for field, values in got.model_dump().items():
                kind = field.rstrip("s")
                shape = _FIELD_SHAPE.get(kind)
                for v in (values or [])[:12]:
                    v = str(v).strip()
                    if not v or len(v) > 200:
                        continue
                    if not shape or not shape.match(v):
                        dropped += 1
                        continue
                    if not verbatim(v, whole):
                        dropped += 1
                        continue
                    if any(e[0] == iid and e[1] == kind and e[2] == v for e in ents):
                        continue
                    ents.append((iid, kind, v, 0.8))

    if dropped:
        log.info("dropped %d fabricated or malformed span(s)", dropped)
    if not ents:
        return

    def work(c: sqlite3.Connection) -> None:
        for row in ents:
            c.execute("INSERT INTO entities(item_id, kind, value, confidence) VALUES (?,?,?,?) "
                      "ON CONFLICT(item_id, kind, value) DO NOTHING", row)
    writer.submit(work)
    log.info("extracted %d verified entit(ies)", len(ents))
