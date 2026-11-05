"""Retrieval CLI.

uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?"
    [--mode dense|bm25|graph|dense+bm25|dense+bm25+graph|dense+bm25+rerank|...+graph+rerank]
    [--k 5] [--doc DOC_ID ...] [--file NAME ...] [--type table ...] [--no-generated]
    [--all-versions] [--identifiers terms|boost|require] [--full]
"""

from __future__ import annotations

import argparse
import json
import sys

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.retrieval import build_retrieval
from omnirag.retrieval.models import RetrievalFilters


def cmd_search(args: argparse.Namespace) -> int:
    service = build_retrieval(get_settings())
    filters = RetrievalFilters(
        latest_only=not args.all_versions, document_ids=args.doc or None,
        source_files=args.file or None, element_types=args.type or None,
        exclude_generated=args.no_generated,
    )  # fmt: skip
    try:
        if args.mode == "bm25" and args.identifiers:
            bm25 = service.retrievers["bm25"]
            result = bm25.retrieve(args.query, args.k, filters, identifier_mode=args.identifiers)
        else:
            result = service.search(args.query, args.mode, args.k, filters)
    finally:
        service.close()
    print(json.dumps(result.details, ensure_ascii=False))
    for hit in result.hits:
        section = " > ".join(hit.section_path)
        print(f"\n#{hit.rank}  {hit.score:.4f}  {hit.source_file}  {section}  "
              f"[{hit.element_type}; {'/'.join(hit.origins)}]  {hit.chunk_id}")  # fmt: skip
        text = hit.content if args.full else hit.content[:240].replace("\n", " ")
        print(f"    {text}")
        if hit.sources:
            print("    via " + ", ".join(f"{n} #{s['rank']:.0f}" for n, s in hit.sources.items()))
        for path in hit.graph_paths[:4]:
            print(f"    graph: {path}")
    print(f"\n{len(result.hits)} hits in {result.latency_ms} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument(
        "--mode",
        default=get_settings().retrieval_default_mode,
        choices=[
            "dense",
            "bm25",
            "graph",
            "dense+bm25",
            "dense+bm25+graph",
            "dense+bm25+rerank",
            "dense+bm25+graph+rerank",
        ],
    )
    p.add_argument("--k", type=int, default=get_settings().retrieval_k)
    p.add_argument("--doc", action="append", help="restrict to a document ID (repeatable)")
    p.add_argument("--file", action="append", help="restrict to a source file name")
    p.add_argument("--type", action="append", help="element type: paragraph, table, figure, ...")
    p.add_argument("--no-generated", action="store_true", help="skip chunks with model text")
    p.add_argument("--all-versions", action="store_true")
    p.add_argument("--identifiers", choices=["terms", "boost", "require"], default=None)
    p.add_argument("--full", action="store_true", help="print whole chunks")
    p.set_defaults(fn=cmd_search)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
