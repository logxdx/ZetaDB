"""Download the demo library listed in sources.tsv.

    uv run scripts/demo/fetch.py ~/zetadb-demo

Photos are scaled to a 2400 px long edge; audio and video are fetched as
listed (480p transcodes for video). Already-downloaded files are skipped,
and Wikimedia's rate limit is respected with a back-off. Then render the
documents (make_docs.py) and load everything (load.py).
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "zetadb-demo/1.0 (+https://github.com/logxdx/ZetaDB)"}


def sources() -> list[tuple[str, str]]:
    out = []
    for line in (HERE / "sources.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            f, url, *_ = line.split("\t")
            out.append((f, url))
    return out


def fetch(url: str, dest: Path) -> None:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                dest.write_bytes(r.read())
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 15 * (attempt + 1)))


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    for rel, url in sources():
        dest = root / rel
        if dest.exists() and dest.stat().st_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("↓", rel)
        fetch(url, dest)
        if rel.startswith("photos/"):
            from PIL import Image, ImageOps
            with Image.open(dest) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
                im.save(dest, quality=90)
        time.sleep(1.5)
    print("done:", root)


if __name__ == "__main__":
    main()
