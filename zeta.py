"""
Zeta DB — multimodal vector search for text, images, audio and video.

Usage:
    python zeta.py                                  # start the web app (default)
    python zeta.py serve --port 7860
    python zeta.py index ~/Pictures -c photos       # index a folder from the CLI
    python zeta.py search "a dog on a beach" -c photos
    python zeta.py search --file clip.mp3 -c music --only audio
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import (
    DEFAULT_QUERY_INTENT,
    DEFAULT_SEARCH_K,
    DEFAULT_TABLE,
    HOST,
    MODALITIES,
    PORT,
    QUERY_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cmd_serve(args):
    import uvicorn

    print(f"\n  Zeta DB is running at http://{args.host}:{args.port}\n")
    uvicorn.run("src.api:app", host=args.host, port=args.port, log_level="info")


def cmd_index(args):
    from src.engine import Engine

    engine = Engine()
    engine.load_model()
    job = engine.wait(engine.index_folder(args.collection, args.folder))
    print(
        f"{job.status}: {job.indexed} indexed, {job.skipped} already present, "
        f"{job.failed} failed (of {job.total})"
    )
    for err in job.errors:
        print(f"  ! {err}")
    return 0 if job.status == "completed" else 1


def cmd_search(args):
    from src.engine import Engine

    engine = Engine()
    engine.load_model()
    results = engine.search(
        args.collection,
        args.top_k,
        modalities=args.only,
        text=args.query,
        file_path=Path(args.file) if args.file else None,
        intent=args.intent,
    )
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. {r['score']:.3f}  [{r['modality']:<5}]  {r['path']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zeta DB — multimodal vector search")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="start the web app")
    p_serve.add_argument("--host", default=HOST)
    p_serve.add_argument("--port", type=int, default=PORT)
    p_serve.set_defaults(func=cmd_serve)

    p_index = sub.add_parser("index", help="index a folder of text, images, audio and video")
    p_index.add_argument("folder")
    p_index.add_argument("-c", "--collection", default=DEFAULT_TABLE)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="search a collection")
    p_search.add_argument("query", nargs="?", help="text query")
    p_search.add_argument("-f", "--file", help="query by example (image, audio, video or text file)")
    p_search.add_argument("-c", "--collection", default=DEFAULT_TABLE)
    p_search.add_argument("-k", "--top-k", type=int, default=DEFAULT_SEARCH_K)
    p_search.add_argument("--only", nargs="+", choices=MODALITIES, help="restrict result modalities")
    p_search.add_argument(
        "-i",
        "--intent",
        default=DEFAULT_QUERY_INTENT,
        choices=list(QUERY_PROMPTS),
        help="task prompt for the text query: search (default), question, code, fact, similarity",
    )
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["serve", *(argv or [])])
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
