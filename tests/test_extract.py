"""Extraction guards (spec 8.1).

Needle 3 hallucinates on real pipeline text. These tests pin the
validators that turn that into silence rather than into database rows.
"""
from __future__ import annotations

from server.ingest.stages.extract import _FIELD_SHAPE, MAX_CHARS, source_text, verbatim

SRC = ("A minimalist invoice from ACME SUPPLY CO dated 2026-03-14, subtotal "
       "1,240.00, tax 99.20, total 1,339.20, finance@acme.example, "
       "https://billing.acme.example/inv/8813")


def test_verbatim_accepts_real_spans():
    for v in ("2026-03-14", "1,240.00", "99.20", "finance@acme.example",
              "https://billing.acme.example/inv/8813"):
        assert verbatim(v, SRC), v


def test_verbatim_rejects_measured_hallucinations():
    """Every one of these was actually produced by Needle 3 against this
    document and appears nowhere in it."""
    for v in ("2026-09-28", "2026-09-29", "$12.50", "$1.00", "113.20",
              "https://www.example.com/price/1.00"):
        assert not verbatim(v, SRC), v


def test_verbatim_normalises_separators():
    """OCR spacing and thousands separators vary; the check must survive it."""
    assert verbatim("1339.20", SRC)
    assert verbatim("1,339.20", SRC)
    assert verbatim("1 339.20", SRC)


def test_field_shapes_reject_cross_contamination():
    """Needle filled emails/orgs/people with the same junk pair. Fields
    without a checkable shape were removed; the ones that remain must
    actually check."""
    assert _FIELD_SHAPE["date"].match("2026-03-14")
    assert not _FIELD_SHAPE["date"].match("14/03/2026")
    assert _FIELD_SHAPE["email"].match("finance@acme.example")
    assert not _FIELD_SHAPE["email"].match("portal link")
    assert _FIELD_SHAPE["url"].match("https://billing.acme.example/inv/8813")
    assert not _FIELD_SHAPE["url"].match("finance@acme.example")
    assert _FIELD_SHAPE["amount"].match("1,240.00")
    assert not _FIELD_SHAPE["amount"].match("portal link")
    assert set(_FIELD_SHAPE) == {"date", "url", "email", "amount"}, \
        "every extracted field must have a machine-checkable shape"


def test_source_text_is_capped_and_prefers_caption():
    """Quality collapsed on caption+layout_text; the caption alone is both
    shorter and denser."""
    row = {"kind": "image", "body": None, "caption": "C" * 50,
           "layout_text": "L" * 5000}
    assert source_text(row).startswith("C")
    row = {"kind": "image", "body": None, "caption": None,
           "layout_text": "L" * 5000}
    assert len(source_text(row)) == MAX_CHARS
    row = {"kind": "note", "body": "N" * 5000, "caption": None, "layout_text": None}
    assert len(source_text(row)) == MAX_CHARS


INVOICE = """ACME SUPPLY CO - INVOICE
Issued                      2026-03-14
Due                         2026-04-13
Portal                      https://billing.acme.example/inv/8813
Contact                     finance@acme.example
Subtotal                    1,240.00
Tax                         99.20
Total                       1,339.20"""

DASHBOARD = """Tag Fusion Dashboard
PP-OCRv6_tiny       1.5M    80.6   73.5
PP-OCRv6_small      7.7M    84.1   81.3
PP-OCRv6_medium     34.5M   86.2   83.2"""


def test_scan_finds_every_invoice_field():
    from server.ingest.stages.extract import scan
    got = set(scan(INVOICE))
    for pair in (("date", "2026-03-14"), ("date", "2026-04-13"),
                 ("url", "https://billing.acme.example/inv/8813"),
                 ("email", "finance@acme.example"),
                 ("amount", "1,240.00"), ("amount", "99.20"), ("amount", "1,339.20")):
        assert pair in got, f"{pair} missing from {sorted(got)}"


def test_scan_does_not_mistake_benchmark_scores_for_money():
    """Precision matters more than recall here: 86.2 on a benchmark row is
    a score. Bare decimals count as amounts only on a line naming money."""
    from server.ingest.stages.extract import scan
    assert [kv for kv in scan(DASHBOARD) if kv[0] == "amount"] == []


def test_scan_is_deterministic():
    from server.ingest.stages.extract import scan
    assert scan(INVOICE) == scan(INVOICE)


def test_amount_shape_does_not_swallow_dates():
    """Regression: the amount pattern started with [^A-Za-z]*, which
    consumed "2026-03-1" and matched the trailing "4", so every ISO date
    was also stored as an amount."""
    from server.ingest.stages.extract import _FIELD_SHAPE
    amount = _FIELD_SHAPE["amount"]
    for good in ("1,240.00", "99.20", "1,339.20", "5", "12.5", "$40.00"):
        assert amount.match(good), good
    for bad in ("2026-03-14", "2026-04-13", "1.2.3", "80.6.1", "v1.2"):
        assert not amount.match(bad), bad


def test_clip_score_requires_an_outlier_not_an_average():
    """Regression: the calibrated z-score was squashed with logistic(z),
    and logistic(0)=0.5 — so a term scoring merely AVERAGE for the corpus
    already cleared a 0.40 threshold. Measured on 29 real photos, that
    accepted 53 of 54 vocabulary terms for a single image."""
    from server.search.vocab import Term, clip_score

    t = Term(id=1, term="bird", joytag_alias=None, prompt_template="a photo of {term}",
             threshold=0.40, clip_mu=0.20, clip_sigma=0.03)
    at_mean = clip_score(t, 0.20)             # z = 0
    one_sigma = clip_score(t, 0.23)           # z = 1
    three_sigma = clip_score(t, 0.29)         # z = 3

    assert at_mean < 0.20, f"an average term must not pass: {at_mean}"
    assert one_sigma < t.threshold, f"1 sigma must not pass: {one_sigma}"
    assert three_sigma > t.threshold, f"3 sigma must pass: {three_sigma}"
    assert at_mean < one_sigma < three_sigma


def test_clip_score_falls_back_when_uncalibrated():
    from server.search.vocab import Term, clip_score
    t = Term(id=1, term="x", joytag_alias=None, prompt_template="{term}",
             threshold=0.4, clip_mu=None, clip_sigma=None)
    assert 0.0 <= clip_score(t, 0.30) <= 1.0


def test_clip_score_has_an_absolute_floor():
    """A relative z-score says "unusually X for THIS corpus", not "is X".
    On a homogeneous library every term finds its own outliers whether or
    not the concept is present, so a raw-cosine floor is needed too."""
    from server.search.vocab import Term, clip_score

    t = Term(id=1, term="whiteboard", joytag_alias=None, prompt_template="{term}",
             threshold=0.40, clip_mu=0.14, clip_sigma=0.02)
    # 3 sigma above a LOW mean is still a low absolute similarity
    assert clip_score(t, 0.20) == 0.0, "below the absolute floor must score zero"
    assert clip_score(t, 0.26) > 0.0


def test_text_bearing_terms_are_declared():
    """Terms that only make sense on an image containing text carry
    requires_text, and the tagging stage skips them when OCR found none.
    Measured: this removed 4 of 7 UI false positives on a photo corpus."""
    import yaml
    from server.config import ROOT

    entries = yaml.safe_load((ROOT / "vocab" / "seed.yaml").read_text())
    flagged = {e["term"] for e in entries if e.get("requires_text")}
    for term in ("screenshot", "dashboard", "code", "terminal", "invoice",
                 "spreadsheet", "webpage", "form", "handwriting", "whiteboard"):
        assert term in flagged, f"{term} should require textual evidence"
    for term in ("tree", "mountain", "bird", "sunset", "ocean"):
        assert term not in flagged, f"{term} must not require text"


def test_tag_stage_honours_requires_text():
    import inspect

    from server.ingest.stages import tag_visual
    src = inspect.getsource(tag_visual.run)
    assert "requires_text" in src and "has_text" in src


def test_item_sorts_are_declared_and_safe():
    """Sort keys map to fixed ORDER BY fragments — never interpolated from
    user input — and an unknown key is rejected rather than defaulted."""
    from server.api.items import SORTS

    assert set(SORTS) == {"date", "title", "tags"}
    for frag in SORTS.values():
        assert ";" not in frag and "--" not in frag
    assert "created_at DESC" in SORTS["date"]
    assert "top_tag" in SORTS["tags"], "tag sort must group by the strongest tag"


def test_related_modes_are_declared():
    """Two different questions: `similar` is a live embedding KNN, `tag` is
    set overlap on accepted tags. Anything else is rejected, not defaulted."""
    import inspect

    from server.api import search as search_api
    src = inspect.getsource(search_api.related)
    assert 'mode not in ("similar", "tag")' in src
    assert "cosine_from_l2" in src, "similar must convert L2 to cosine"
    assert "state = 'accepted'" in src, "tag overlap must ignore rejected tags"
