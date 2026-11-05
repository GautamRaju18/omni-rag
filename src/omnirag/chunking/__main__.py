"""Chunking CLI.

uv run python -m omnirag.chunking compare                    # all strategies, all documents
uv run python -m omnirag.chunking compare --size 256 --size 512
uv run python -m omnirag.chunking rechunk --all              # rewrite chunk sets from settings
uv run python -m omnirag.chunking show doc_2b1c659c54266d54 multimodal-512 --limit 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict

from omnirag.chunking import build_chunker, load_tokenizer
from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import describe, timed_chunk
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore


def _documents(store: DocumentStore, ids: list[str]):
    records = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED]
    return [r for r in records if not ids or r.document_id in ids]


def cmd_compare(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    counter = load_tokenizer(settings)
    strategies = [ChunkingStrategy(s) for s in (args.strategy or settings.chunking_strategies)]
    docs = [(r, store.read_elements(r.document_id)) for r in _documents(store, args.doc)]
    print(f"{len(docs)} documents, tokenizer {counter.name}\n")
    header = (f"{'CONFIG':<28} {'CHUNKS':>6} {'INDEXED':>7} {'MED':>5} {'P95':>5} {'MAX':>5} "
              f"{'OVER':>4} {'INTACT':>7} {'KB':>7} {'MS':>7}")  # fmt: skip
    print(header)
    for size in args.size or [settings.chunk_size]:
        for strategy in strategies:
            config = ChunkingConfig(
                strategy=strategy, chunk_size=size,
                chunk_overlap=min(settings.chunk_overlap, size // 4),
                child_size=min(settings.chunk_child_size, size // 2),
                parent_size=max(settings.chunk_parent_size, size * 2),
                contextual_headers=args.ctx,
            )  # fmt: skip
            totals: dict[str, list[float]] = defaultdict(list)
            all_tokens: list[int] = []
            for record, elements in docs:
                chunks, ms = timed_chunk(record, elements, config, counter)
                stats = describe(chunks, elements, config)
                for key in ("chunks", "retrievable", "over_limit", "bytes"):
                    totals[key].append(stats[key])
                totals["ms"].append(ms)
                totals["integrity"].append(stats["integrity"])
                all_tokens += [c.token_count for c in chunks if c.retrievable]
            all_tokens.sort()
            p95 = all_tokens[int(0.95 * (len(all_tokens) - 1))] if all_tokens else 0
            print(
                f"{config.config_id:<28} {sum(totals['chunks']):>6.0f} "
                f"{sum(totals['retrievable']):>7.0f} {statistics.median(all_tokens or [0]):>5.0f} "
                f"{p95:>5} {max(all_tokens or [0]):>5} {sum(totals['over_limit']):>4.0f} "
                f"{statistics.mean(totals['integrity']):>7.1%} "
                f"{sum(totals['bytes']) / 1024:>7.1f} {sum(totals['ms']):>7.1f}"
            )
    print("\nMED/P95/MAX: tokens of indexed chunks · OVER: chunks above their level's size limit · "
          "INTACT: mean share of elements kept whole in some chunk")  # fmt: skip
    return 0


def cmd_rechunk(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    chunker = build_chunker(settings)
    docs = _documents(store, args.doc)
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    for record in docs:
        sets = chunker.chunk(record, store.read_elements(record.document_id))
        for config_id, chunks in sets.items():
            store.write_chunks(record.document_id, config_id, chunks)
        record.metadata["chunk_sets"] = {cid: len(c) for cid, c in sets.items()}
        store.write_record(record)
        counts = " ".join(f"{cid}={n}" for cid, n in record.metadata["chunk_sets"].items())
        print(f"{record.document_id}  {record.source_file:<36} {counts}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = DocumentStore(get_settings().data_dir)
    chunks = store.read_chunks(args.doc_id, args.config_id)
    if not chunks:
        available = ", ".join(store.chunk_config_ids(args.doc_id)) or "none"
        print(f"No chunks for {args.doc_id} / {args.config_id} (available: {available})")
        return 1
    for c in chunks[: args.limit]:
        flags = "" if c.retrievable else " (parent, not indexed)"
        print(f"[{c.chunk_id}] L{c.level} {c.element_type} {c.token_count} tok "
              f"p={c.page_numbers or '-'} section={c.section or '-'}{flags}")  # fmt: skip
        print(f"    elements={len(c.element_ids)} partial={len(c.partial_element_ids)} "
              f"origins={','.join(c.origins)} parent={c.parent_chunk_id or '-'}")  # fmt: skip
        if c.context:
            print("    context: " + c.context.replace("\n", " | "))
        print("    " + c.content.replace("\n", "\n    ")[:600])
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.chunking")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare", help="Measure strategies on ingested documents (no writes)")
    p.add_argument("--strategy", action="append", choices=[s.value for s in ChunkingStrategy])
    p.add_argument("--size", action="append", type=int)
    p.add_argument("--doc", action="append", default=[])
    p.add_argument("--ctx", action="store_true", help="with contextual headers")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("rechunk", help="Rewrite chunk sets from current settings")
    p.add_argument("doc", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_rechunk)
    p = sub.add_parser("show", help="Print chunks of one document and config")
    p.add_argument("doc_id")
    p.add_argument("config_id")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_show)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
