"""Load the demo library into a running ZetaDB through its public API.

    uv run scripts/demo/load.py ~/zetadb-demo [http://127.0.0.1:8777]

Uploads the notes, photos, rendered documents and audio/video, then gives
each file a readable title (with the photographer for photos). Ingestion
runs in the background; watch the server log or the canvas fill in.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DOC_TITLES = {
    "boarding_pass_sfo_tokyo": "Boarding pass SFO → Tokyo",
    "chart_monthly_active_users": "Monthly active users 2026",
    "code_snippet_debounce": "debounce() snippet",
    "concert_ticket_jazz": "Midnight Quartet ticket",
    "handwritten_grocery_list": "Saturday market list",
    "receipt_harbor_and_bean_cafe": "Harbor & Bean receipt",
    "recipe_card_shakshuka": "Weekend shakshuka",
    "slide_roadmap_2027": "Roadmap 2027 slide",
    "terminal_connection_refused": "Worker crash: connection refused",
    "whiteboard_pricing_meeting": "Pricing v2 whiteboard",
}


def titles() -> dict[str, str]:
    out = dict(DOC_TITLES)
    for line in (HERE / "sources.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            f, _url, _lic, _author, title = line.split("\t")
            out[Path(f).stem] = title
    return out


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    api = httpx.Client(base_url=(sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8777")
                       + "/api", timeout=120)
    for n in json.loads((HERE / "notes.json").read_text()):
        api.post("/items", data={"title": n["title"], "body": n["body"]}).raise_for_status()
    print("notes", "ok")
    for folder in ("photos", "docs", "media"):
        for p in sorted((root / folder).glob("*")):
            with p.open("rb") as f:
                api.post("/items", files=[("files", (p.name, f))]).raise_for_status()
            print(folder, p.name)
            time.sleep(0.1)

    want = titles()
    cursor, done = None, 0
    while True:
        page = api.get("/items", params={"limit": 200, **({"cursor": cursor} if cursor else {})}).json()
        for it in page["items"]:
            stem = Path(it.get("title") or "").stem
            if stem in want and it["title"] != want[stem]:
                api.patch(f"/items/{it['id']}", json={"title": want[stem]}).raise_for_status()
                done += 1
        cursor = page.get("next_cursor")
        if not cursor:
            break
    print("titled", done)


if __name__ == "__main__":
    main()
