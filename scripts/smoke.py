#!/usr/bin/env python
"""End-to-end smoke test against a running server.

Exercises every endpoint the SPA calls, checks the response SHAPE matches
what the frontend destructures, and verifies SSE delivers. This is the
contract test between web/src/lib/api.ts and server/api/* - a mismatch
there breaks the UI silently, with no server-side error.

    uv run scripts/smoke.py [--base http://127.0.0.1:8777]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PASS, FAIL = "\033[32m PASS \033[0m", "\033[31m FAIL \033[0m"
results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"{PASS if ok else FAIL} {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def keys(obj: dict, *required: str) -> str:
    missing = [k for k in required if k not in obj]
    return f"missing {missing}" if missing else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8777")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=60)

    print(f"\n\033[1msmoke test → {a.base}\033[0m\n")

    # ---- health -----------------------------------------------------
    try:
        h = c.get("/api/health").json()
    except Exception as exc:
        print(f"{FAIL} server unreachable — {exc}")
        return 1
    check("GET /api/health", True, f"{h['db']['counts']['items']} items")
    check("  health shape", not keys(h, "host", "db", "gpu", "arbiter", "models"))
    if h.get("gpu"):
        g = h["gpu"]
        check("  gpu within budget", g["used_mb"] < g["budget_mb"],
              f"{g['used_mb']}/{g['budget_mb']} MiB")

    # ---- SPA served -------------------------------------------------
    r = c.get("/")
    check("GET / (SPA shell)", r.status_code == 200 and "<div id=\"app\">" in r.text)
    for asset in ("css", "js"):
        import re
        m = re.search(rf'/assets/[^"]+\.{asset}', r.text)
        if m:
            check(f"  /assets *.{asset}", c.get(m.group(0)).status_code == 200, m.group(0))
        else:
            check(f"  /assets *.{asset}", False, "not referenced in index.html")

    # ---- items list (Masonry) ---------------------------------------
    r = c.get("/api/items", params={"limit": 60}).json()
    check("GET /api/items", not keys(r, "items", "next_cursor"))
    items = r["items"]
    if items:
        it = items[0]
        # exactly the fields Card.svelte reads
        check("  item shape for Card.svelte",
              not keys(it, "id", "kind", "status", "title", "preview", "tags",
                       "width", "height", "near_dup_of"))
        check("  tags are objects, not strings",
              all(isinstance(t, dict) and "term" in t and "confidence" in t
                  and "source" in t and "state" in t for t in it["tags"]),
              f"{len(it['tags'])} tag(s)")
        imgs = [i for i in items if i["kind"] == "image"]
        if imgs:
            check("  image has thumb_url", "thumb_url" in imgs[0])
            t = c.get(imgs[0]["thumb_url"])
            check("  thumbnail loads", t.status_code == 200 and t.headers
                  .get("content-type", "").startswith("image/"),
                  f"{len(t.content)}B {t.headers.get('content-type')}")
            f = c.get(imgs[0]["file_url"])
            check("  full image loads", f.status_code == 200, f"{len(f.content)}B")

    # ---- detail (Focus.svelte) --------------------------------------
    if items:
        d = c.get(f"/api/items/{items[0]['id']}").json()
        check("GET /api/items/{id}",
              not keys(d, "body", "layout_text", "flow_text", "boxes",
                       "entities", "todos", "errors", "near_dups"))

        rel = c.get(f"/api/items/{items[0]['id']}/related").json()
        check("GET /api/items/{id}/related", "related" in rel,
              f"{len(rel.get('related', []))} related")

        g = c.get(f"/api/items/{items[0]['id']}/graph", params={"hops": 2}).json()
        check("GET /api/items/{id}/graph", not keys(g, "focus", "nodes", "edges"),
              f"{len(g.get('nodes', []))} nodes / {len(g.get('edges', []))} edges")

    # ---- search -----------------------------------------------------
    s = c.get("/api/search", params={"q": "invoice"}).json()
    check("GET /api/search", not keys(s, "query", "seeds", "retrievers", "results"))
    if s["results"]:
        check("  result carries why/score",
              not keys(s["results"][0], "why", "score", "ranks"),
              f"why={s['results'][0]['why']}")
    check("  multiple retrievers fired", len(s["retrievers"]) >= 2, str(s["retrievers"]))
    check("GET /api/search empty q rejected", c.get("/api/search", params={"q": " "})
          .status_code == 400)

    # ---- queue ------------------------------------------------------
    q = c.get("/api/queue").json()
    check("GET /api/queue", not keys(q, "depths", "total", "handlers"),
          f"{len(q['handlers'])} stage handlers")

    # ---- write path + SSE -------------------------------------------
    got_event: list[str] = []
    try:
        with c.stream("GET", "/api/events", timeout=25) as es:
            body = {"body": f"smoke test note {time.time():.0f} — ingest and index me"}
            created = httpx.post(f"{a.base}/api/items", data=body, timeout=30).json()
            new_id = created["created"][0]["id"]
            check("POST /api/items (note)", bool(new_id), f"id={new_id}")
            deadline = time.time() + 20
            for line in es.iter_lines():
                if time.time() > deadline:
                    break
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    got_event.append(ev["event"])
                    if ev.get("data", {}).get("id") == new_id:
                        break
    except Exception as exc:
        check("SSE /api/events", False, str(exc)[:80])
    else:
        check("SSE /api/events delivers", bool(got_event), ",".join(got_event[:4]) or "no events")

    # ---- cleanup ----------------------------------------------------
    if "new_id" in dir() or True:
        try:
            c.delete(f"/api/items/{new_id}")
            check("DELETE /api/items/{id}", c.get(f"/api/items/{new_id}").status_code == 404)
        except Exception:
            pass

    # ---- vocab / review ---------------------------------------------
    v = c.get("/api/vocab").json()
    check("GET /api/vocab", "terms" in v, f"{len(v.get('terms', []))} terms")
    rv = c.get("/api/review").json()
    check("GET /api/review", not keys(rv, "tags", "proposals"),
          f"{len(rv.get('tags', []))} pending tags")

    failed = [n for ok, n, _ in results if not ok]
    print(f"\n\033[1m{len(results) - len(failed)}/{len(results)} passed\033[0m")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
