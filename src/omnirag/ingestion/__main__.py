"""Command-line ingestion (until the API endpoints arrive in Phase 2e).

uv run python -m omnirag.ingestion ingest data/samples [--force]
uv run python -m omnirag.ingestion list
uv run python -m omnirag.ingestion show <document_id> [--limit 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omnirag.core.config import get_settings
from omnirag.core.errors import OmniRAGError
from omnirag.core.logging import configure_logging
from omnirag.ingestion import build_pipeline
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.validation import EXTENSION_KINDS


def _expand(paths: list[Path]) -> list[Path]:
    """Files as given; directories expanded to supported files (README* files are skipped)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in EXTENSION_KINDS
                and not p.name.lower().startswith("readme")
            )
        else:
            files.append(path)
    return files


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(get_settings())
    failures = 0
    print(f"{'STATUS':<10} {'DOCUMENT ID':<22} {'KIND':<9} {'ELEMS':>5} {'MS':>8}  FILE")
    for path in _expand(args.paths):
        try:
            result = pipeline.ingest(path, force=args.force)
        except OmniRAGError as exc:
            failures += 1
            print(f"{'REJECTED':<10} {'-':<22} {'-':<9} {'-':>5} {'-':>8}  {path.name}: {exc}")
            continue
        r = result.record
        status = "DUPLICATE" if result.duplicate else r.status.value.upper()
        failures += r.status is IngestionStatus.FAILED
        print(
            f"{status:<10} {r.document_id:<22} {r.file_kind.value:<9} {r.element_count:>5} "
            f"{r.duration_ms or 0:>8.1f}  {r.source_file}"
        )
        for warning in r.warnings:
            print(f"{'':<10} ⚠ {warning}")
        if r.error:
            print(f"{'':<10} ✖ {r.error}")
    return 1 if failures else 0


def cmd_list(_: argparse.Namespace) -> int:
    store = build_pipeline(get_settings()).store
    for r in store.list_records():
        counts = ", ".join(f"{k}={v}" for k, v in r.element_counts_by_type.items())
        print(f"{r.document_id}  {r.status.value:<10} {r.source_file:<32} {counts}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = build_pipeline(get_settings()).store
    record = store.read_record(args.document_id)
    if record is None:
        print(f"No document {args.document_id}", file=sys.stderr)
        return 1
    print(record.model_dump_json(indent=2, exclude={"metadata"}))
    for e in store.read_elements(args.document_id)[: args.limit]:
        where = " > ".join(e.section_path) or "-"
        preview = e.content.replace("\n", " ⏎ ")[:110]
        parent = f" parent={e.parent_element_id[-6:]}" if e.parent_element_id else ""
        print(f"[{e.element_id[-6:]}] {e.element_type.value:<9} conf={e.confidence:.2f}{parent}")
        print(f"         section: {where}\n         {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging("WARNING", json_logs=False)  # keep CLI output readable
    parser = argparse.ArgumentParser(prog="omnirag.ingestion", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest files or directories")
    p_ingest.add_argument("paths", nargs="+", type=Path)
    p_ingest.add_argument("--force", action="store_true", help="re-process duplicates")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("list", help="list ingested documents").set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a document's record and elements")
    p_show.add_argument("document_id")
    p_show.add_argument("--limit", type=int, default=20)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    print(f"data dir: {settings.data_dir}\n")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
