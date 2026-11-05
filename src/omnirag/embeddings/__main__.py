"""Embeddings CLI.

uv run python -m omnirag.embeddings info            # model space (digest, dim, limit), cache
uv run python -m omnirag.embeddings embed --all     # (re)embed the configured chunk sets
uv run python -m omnirag.embeddings verify          # every indexed chunk has a current vector
"""

from __future__ import annotations

import argparse
import sys

from omnirag.chunking import build_embedder, load_tokenizer
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.embeddings.ollama import EmbeddingCache
from omnirag.embeddings.service import ChunkEmbedder
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore


def _docs(store: DocumentStore, ids: list[str]):
    done = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED]
    return [r for r in done if not ids or r.document_id in ids]


def cmd_info(_: argparse.Namespace) -> int:
    settings = get_settings()
    space = build_embedder(settings).space
    print(f"space_id   {space.space_id}")
    for field in ("provider", "model", "revision", "kind", "dim", "max_tokens", "normalized",
                  "query_prefix", "document_prefix"):  # fmt: skip
        print(f"{field:<10} {getattr(space, field)!r}")
    print(f"indexed    {settings.embedding_index_configs}")
    print("\ncache (space, dim, vectors):")
    for row in EmbeddingCache(settings.data_dir / "cache" / "embeddings.sqlite").spaces():
        marker = "  <- current" if row[0] == space.space_id else ""
        print(f"  {row[0]:<44} {row[1]:>5} {row[2]:>7}{marker}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    counter = load_tokenizer(settings)
    configs = args.config or settings.embedding_index_configs
    embedder = ChunkEmbedder(build_embedder(settings, counter.count), configs)
    if not args.all and not args.doc:
        print("Give document IDs or --all", file=sys.stderr)
        return 2
    for record in _docs(store, args.doc):
        sets = {cid: store.read_chunks(record.document_id, cid) for cid in configs}
        stats = embedder.embed_sets({k: v for k, v in sets.items() if v})
        for cid, chunks in sets.items():
            if chunks:
                store.write_chunks(record.document_id, cid, chunks)
        record.metadata["embeddings"] = stats
        store.write_record(record)
        print(f"{record.document_id}  {record.source_file:<36} {stats['chunks']} {stats['ms']} ms")
    print(f"\n{embedder.embedder.stats}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    space = build_embedder(settings).space
    cache = EmbeddingCache(settings.data_dir / "cache" / "embeddings.sqlite")
    problems = 0
    for record in _docs(store, []):
        for cid in args.config or settings.embedding_index_configs:
            chunks = [c for c in store.read_chunks(record.document_id, cid) if c.retrievable]
            refs = [c.embedding_ref for c in chunks if c.embedding_ref]
            prefix = space.space_id + "/"
            current = [r.removeprefix(prefix) for r in refs if r.startswith(prefix)]
            no_ref, stale = len(chunks) - len(refs), len(refs) - len(current)
            missing = len(current) - len(cache.get_many(space.space_id, current))
            if no_ref or stale or missing or not chunks:
                problems += 1
                print(f"✖ {record.source_file:<36} {cid}: {len(chunks)} chunks, {no_ref} without "
                      f"ref, {stale} stale, {missing} missing vectors")  # fmt: skip
            else:
                print(f"✔ {record.source_file:<36} {cid}: {len(chunks)} chunks embedded")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.embeddings")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show the embedding space and cache").set_defaults(func=cmd_info)
    p = sub.add_parser("embed", help="Embed configured chunk sets (backfill / re-embed)")
    p.add_argument("doc", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--config", action="append")
    p.set_defaults(func=cmd_embed)
    p = sub.add_parser("verify", help="Check every indexed chunk has a current vector")
    p.add_argument("--config", action="append")
    p.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
