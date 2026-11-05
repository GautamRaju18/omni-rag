"""Knowledge-graph CLI.

uv run python -m omnirag.knowledge_base init             # constraints + indexes (idempotent)
uv run python -m omnirag.knowledge_base load --all       # (re)write documents from disk
uv run python -m omnirag.knowledge_base stats            # node / relationship counts
uv run python -m omnirag.knowledge_base outline DOC_ID   # sections, pages, tables, figures
uv run python -m omnirag.knowledge_base show-chunk CHUNK_ID
uv run python -m omnirag.knowledge_base versions FILE_NAME
uv run python -m omnirag.knowledge_base search-vector "question" [--k 5]
uv run python -m omnirag.knowledge_base search-text "lucene query" [--k 5]
uv run python -m omnirag.knowledge_base unload DOC_ID    # graph only; files on disk stay
"""

from __future__ import annotations

import argparse
import json
import sys

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.ingestion import build_graph_indexer
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base import traversal


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_init(args: argparse.Namespace) -> int:
    indexer = build_graph_indexer(get_settings())
    try:
        _print(indexer.store.verify())
        indexer.schema_options["recreate_vector_index"] = args.recreate_vector_index
        report = indexer.ensure_schema()
        print(f"created   {report.created}\nexisting  {report.existing}\n"
              f"recreated {report.recreated}")  # fmt: skip
    finally:
        indexer.close()
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    indexer = build_graph_indexer(settings)
    records = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED
               and (args.all or r.document_id in args.doc)]  # fmt: skip
    try:
        for record in records:
            sets = {cid: store.read_chunks(record.document_id, cid) for cid in indexer.config_ids}
            sets = {k: v for k, v in sets.items() if v}
            if not sets:
                print(f"{record.document_id}  {record.source_file:<36} no chunks: skipped")
                continue
            stats = indexer.index(record, store.read_elements(record.document_id), sets)
            record.metadata["graph"] = stats
            store.write_record(record)
            n = stats["nodes"]
            print(f"{record.document_id}  {record.source_file:<36} chunks={n['chunks']:<4}"
                  f" sections={n['sections']:<3} tables={n['tables']} figures={n['figures']}"
                  f" vectors={stats['vectors']} missing={stats['vectors_missing']}"
                  f" {stats['ms']} ms")  # fmt: skip
    finally:
        indexer.close()
    return 0


def _with_store(fn):
    def run(args: argparse.Namespace) -> int:
        indexer = build_graph_indexer(get_settings())
        try:
            return fn(indexer, args)
        finally:
            indexer.close()

    return run


@_with_store
def cmd_stats(indexer, _: argparse.Namespace) -> int:
    _print(traversal.stats(indexer.store))
    return 0


@_with_store
def cmd_outline(indexer, args: argparse.Namespace) -> int:
    outline = traversal.document_outline(indexer.store, args.document_id)
    if outline is None:
        print(f"{args.document_id} is not in the graph", file=sys.stderr)
        return 1
    _print(outline)
    return 0


@_with_store
def cmd_show_chunk(indexer, args: argparse.Namespace) -> int:
    context = traversal.chunk_context(indexer.store, args.chunk_id)
    if context is None:
        print(f"{args.chunk_id} is not in the graph", file=sys.stderr)
        return 1
    _print(context)
    return 0


@_with_store
def cmd_versions(indexer, args: argparse.Namespace) -> int:
    _print(traversal.versions(indexer.store, args.source_file))
    return 0


@_with_store
def cmd_search_vector(indexer, args: argparse.Namespace) -> int:
    from omnirag.chunking import build_embedder

    settings = get_settings()
    vector = build_embedder(settings).embed_query(args.query).tolist()
    _print(traversal.vector_search(indexer.store, vector, k=args.k,
                                   index=settings.graph_vector_index, config_id=args.config,
                                   latest_only=not args.all_versions))  # fmt: skip
    return 0


@_with_store
def cmd_search_text(indexer, args: argparse.Namespace) -> int:
    settings = get_settings()
    _print(traversal.fulltext_search(indexer.store, args.query, k=args.k,
                                     index=settings.graph_fulltext_index, config_id=args.config,
                                     latest_only=not args.all_versions))  # fmt: skip
    return 0


@_with_store
def cmd_unload(indexer, args: argparse.Namespace) -> int:
    found = indexer.delete(args.document_id)
    print("removed from graph" if found else f"{args.document_id} is not in the graph")
    return 0 if found else 1


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.knowledge_base")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="create constraints and indexes")
    p.add_argument(
        "--recreate-vector-index",
        action="store_true",
        help="drop and recreate a vector index whose dimension/quantization changed",
    )
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("load", help="write documents from disk into the graph")
    p.add_argument("doc", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_load)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("outline")
    p.add_argument("document_id")
    p.set_defaults(fn=cmd_outline)
    p = sub.add_parser("show-chunk")
    p.add_argument("chunk_id")
    p.set_defaults(fn=cmd_show_chunk)
    p = sub.add_parser("versions")
    p.add_argument("source_file")
    p.set_defaults(fn=cmd_versions)
    for name, fn in (("search-vector", cmd_search_vector), ("search-text", cmd_search_text)):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--config", default=None, help="restrict to one chunk set")
        p.add_argument("--all-versions", action="store_true")
        p.set_defaults(fn=fn)
    p = sub.add_parser("unload", help="remove a document from the graph (files stay)")
    p.add_argument("document_id")
    p.set_defaults(fn=cmd_unload)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
