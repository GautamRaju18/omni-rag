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
uv run python -m omnirag.knowledge_base extract --all    # LLM entities/relations (Phase 5b)
uv run python -m omnirag.knowledge_base revalidate --all # re-apply validation, no LLM calls
uv run python -m omnirag.knowledge_base entities [--type PERSON]
uv run python -m omnirag.knowledge_base entity "Marco Lindqvist"
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


def _records(store: DocumentStore, args: argparse.Namespace):
    return [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED
            and (args.all or r.document_id in args.doc)]  # fmt: skip


def _load_one(store, indexer, record, extraction=None) -> dict | None:
    """Write one document (structure + any stored extraction) from disk into the graph."""
    sets = {cid: store.read_chunks(record.document_id, cid) for cid in indexer.config_ids}
    sets = {k: v for k, v in sets.items() if v}
    if not sets:
        return None
    if extraction is None:
        extraction = store.read_extraction(record.document_id, get_settings().extraction_config_id)
    stats = indexer.index(record, store.read_elements(record.document_id), sets, extraction)
    record.metadata["graph"] = stats
    store.write_record(record)
    return stats


def cmd_load(args: argparse.Namespace) -> int:
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    indexer = build_graph_indexer(settings)
    try:
        for record in _records(store, args):
            stats = _load_one(store, indexer, record)
            if stats is None:
                print(f"{record.document_id}  {record.source_file:<36} no chunks: skipped")
                continue
            n = stats["nodes"]
            print(f"{record.document_id}  {record.source_file:<36} chunks={n['chunks']:<4}"
                  f" sections={n['sections']:<3} tables={n['tables']} figures={n['figures']}"
                  f" vectors={stats['vectors']} missing={stats['vectors_missing']}"
                  f" mentions={n['mentions']} relations={n['relations']}"
                  f" {stats['ms']} ms")  # fmt: skip
    finally:
        indexer.close()
    return 0


def _print_counts(record, counts: dict) -> None:
    keys = ("entities", "relations", "stated", "inferred", "rejected", "errors")
    print(f"{record.document_id}  {record.source_file:<36} "
          + " ".join(f"{k}={counts[k]}" for k in keys))  # fmt: skip


def cmd_extract(args: argparse.Namespace) -> int:
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    from omnirag.ingestion import build_extraction_step

    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    step = build_extraction_step(settings)
    indexer = build_graph_indexer(settings)
    try:
        for record in _records(store, args):
            doc = record.document_id
            elements = store.read_elements(doc)
            sets = {step.config_id: store.read_chunks(doc, step.config_id)}
            previous = None if args.force else store.read_extraction(doc, step.config_id)
            extraction = step.run(record, elements, sets, previous)
            if extraction is None:
                print(f"{doc}  {record.source_file:<36} no {step.config_id} chunks: skipped")
                continue
            store.write_extraction(doc, step.config_id, extraction)
            record.metadata["extraction"] = extraction.counts()
            _load_one(store, indexer, record, extraction)
            _print_counts(record, extraction.counts())
    finally:
        indexer.close()
    stats = step.extractor.stats
    print(
        f"\nmodel calls={stats['calls']:.0f} cached={stats['cached']:.0f} "
        f"seconds={stats['seconds']:.0f} output_tokens={stats['tokens']:.0f}"
    )
    return 0


def cmd_revalidate(args: argparse.Namespace) -> int:
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    from omnirag.knowledge_base.extractor import revalidate

    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    indexer = build_graph_indexer(settings)
    cid = settings.extraction_config_id
    try:
        for record in _records(store, args):
            extraction = store.read_extraction(record.document_id, cid)
            if extraction is None:
                continue
            extraction = revalidate(extraction, store.read_elements(record.document_id),
                                    store.read_chunks(record.document_id, cid))  # fmt: skip
            store.write_extraction(record.document_id, cid, extraction)
            record.metadata["extraction"] = extraction.counts()
            _load_one(store, indexer, record, extraction)
            _print_counts(record, extraction.counts())
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
def cmd_entities(indexer, args: argparse.Namespace) -> int:
    for e in traversal.entities(indexer.store, entity_type=args.type, limit=args.limit):
        aliases = [a for a in e["aliases"] if a != e["name"]]
        print(f"{e['type']:<13} {e['name']:<34} mentions={e['mentions']:<3} "
              f"docs={e['documents']:<2} relations={e['relations']:<3}"
              + (f" aliases={aliases}" if aliases else ""))  # fmt: skip
    return 0


@_with_store
def cmd_entity(indexer, args: argparse.Namespace) -> int:
    detail = traversal.entity_detail(indexer.store, args.name,
                                     latest_only=not args.all_versions)  # fmt: skip
    if detail is None:
        print(f"No entity named {args.name!r}", file=sys.stderr)
        return 1
    _print(detail)
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
    for name, fn, help_text in (
        ("extract", cmd_extract, "LLM entity/relation extraction, then reload into the graph"),
        ("revalidate", cmd_revalidate, "re-apply validation to stored replies (no LLM)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("doc", nargs="*")
        p.add_argument("--all", action="store_true")
        if name == "extract":
            p.add_argument("--force", action="store_true", help="ignore stored replies")
        p.set_defaults(fn=fn)
    p = sub.add_parser("entities")
    p.add_argument("--type", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_entities)
    p = sub.add_parser("entity")
    p.add_argument("name")
    p.add_argument("--all-versions", action="store_true")
    p.set_defaults(fn=cmd_entity)
    p = sub.add_parser("unload", help="remove a document from the graph (files stay)")
    p.add_argument("document_id")
    p.set_defaults(fn=cmd_unload)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
