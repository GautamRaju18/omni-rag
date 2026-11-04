"""Ingestion pipeline: validate → hash/dedup → preserve original → load → persist.

The pipeline is deliberately synchronous and side-effect ordered:

1. **Validate** extension, size and magic bytes (nothing is stored for invalid files).
2. **Hash** the bytes; the document ID is derived from the hash, so duplicates are detected
   before any parsing work is done.
3. **Preserve the original** before parsing, so a parser crash never loses the source.
4. **Load** with the loader registered for the file kind.
5. **Persist** elements and a ``DocumentRecord`` whose ``status`` says what happened.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import structlog

from omnirag.core.errors import LoaderNotAvailableError
from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import Loader
from omnirag.ingestion.models import DocumentRecord, FileKind, IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.ingestion.validation import document_id_for, sha256_file, validate_file

log = structlog.get_logger(module=__name__)


@dataclass(frozen=True)
class IngestionResult:
    record: DocumentRecord
    duplicate: bool = False


class IngestionPipeline:
    def __init__(
        self,
        store: DocumentStore,
        loaders: Mapping[FileKind, Loader],
        *,
        max_bytes: int,
    ) -> None:
        self.store = store
        self.loaders = loaders
        self.max_bytes = max_bytes

    def ingest(
        self, path: Path, *, source_name: str | None = None, force: bool = False
    ) -> IngestionResult:
        """Ingest one file. Raises for invalid input; returns FAILED records for parse errors."""
        source_name = source_name or path.name
        kind = validate_file(path, max_bytes=self.max_bytes)
        loader = self.loaders.get(kind)
        if loader is None:
            raise LoaderNotAvailableError(
                f"No loader available for {kind.value} files yet", details={"file": source_name}
            )

        sha256 = sha256_file(path)
        document_id = document_id_for(sha256)
        existing = self.store.read_record(document_id)
        if existing and existing.status is IngestionStatus.COMPLETED and not force:
            log.info("ingest_duplicate", document_id=document_id, source_file=source_name)
            return IngestionResult(existing, duplicate=True)

        record = DocumentRecord(
            document_id=document_id,
            source_file=source_name,
            file_kind=kind,
            sha256=sha256,
            size_bytes=path.stat().st_size,
            status=IngestionStatus.PROCESSING,
            loader=loader.name,
        )
        if existing:
            record.created_at = existing.created_at
        original = self.store.save_original(document_id, path)
        self.store.clear_derived(document_id)
        self.store.write_record(record)

        ctx = ExtractionContext(
            document_id,
            source_name,
            asset_saver=partial(self.store.save_asset, document_id),
            original_ref=original.relative_to(self.store.data_dir).as_posix(),
        )
        started = time.perf_counter()
        try:
            metadata = loader.load(path, ctx)
        except Exception as exc:  # any parser bug: record it, keep the original, don't crash
            log.exception("ingest_failed", document_id=document_id, loader=loader.name)
            record.status = IngestionStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.warnings = ctx.warnings
            return self._finish(record, started)

        if not ctx.elements:
            ctx.warn("No content was extracted from this file")
        self.store.write_elements(document_id, ctx.elements)
        record.status = IngestionStatus.COMPLETED
        record.metadata = metadata
        record.warnings = ctx.warnings
        record.element_count = len(ctx.elements)
        record.element_counts_by_type = dict(
            Counter(e.element_type.value for e in ctx.elements).most_common()
        )
        result = self._finish(record, started)
        log.info(
            "ingest_completed",
            document_id=document_id,
            source_file=source_name,
            elements=record.element_count,
            duration_ms=record.duration_ms,
        )
        return result

    def _finish(self, record: DocumentRecord, started: float) -> IngestionResult:
        record.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        record.updated_at = datetime.now(UTC)
        self.store.write_record(record)
        return IngestionResult(record)
