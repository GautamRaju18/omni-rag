"""Ingestion pipeline: validate → hash/dedup → preserve original → load → persist.

The pipeline is deliberately synchronous and side-effect ordered:

1. **Validate** extension, size and magic bytes (nothing is stored for invalid files).
2. **Hash** the bytes; the document ID is derived from the hash, so duplicates are detected
   before any parsing work is done.
3. **Preserve the original** before parsing, so a parser crash never loses the source.
4. **Load** with the loader registered for the file kind.
5. **Enrich** (optional): e.g. vision-model image descriptions. Enrichers only *append*
   elements, so loader element IDs never depend on them, and an enricher failure is a warning,
   never a failed document.
6. **Persist** elements and a ``DocumentRecord`` whose ``status`` says what happened.
7. **Chunk, embed, graph** (optional, Phases 3-5): each step's failure is a warning; the files
   on disk stay the source of truth and every later step can be re-run from the CLI.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

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


class Enricher(Protocol):
    """Post-load step that appends derived elements (never edits or removes existing ones)."""

    name: str

    def enrich(
        self, ctx: ExtractionContext, read_asset: Callable[[str], Path]
    ) -> dict[str, Any]: ...


class IngestionPipeline:
    def __init__(
        self,
        store: DocumentStore,
        loaders: Mapping[FileKind, Loader],
        *,
        max_bytes: int,
        enrichers: Sequence[Enricher] = (),
        chunker: Any = None,  # omnirag.chunking.Chunker (untyped here to avoid an import cycle)
        embedder: Any = None,  # omnirag.embeddings.service.ChunkEmbedder
        graph: Any = None,  # omnirag.knowledge_base.graph_builder.GraphIndexer
        extraction: Any = None,  # omnirag.knowledge_base.extractor.ExtractionStep
    ) -> None:
        self.store = store
        self.loaders = loaders
        self.max_bytes = max_bytes
        self.enrichers = enrichers
        self.chunker = chunker
        self.embedder = embedder
        self.graph = graph
        self.extraction = extraction

    def close(self) -> None:
        """End of a batch: let enrichers release resources (e.g. unload the vision model)."""
        for enricher in self.enrichers:
            close = getattr(enricher, "close", None)
            if close is not None:
                close()

    def delete(self, document_id: str) -> bool:
        """Remove a document from the graph, then from disk. Returns False if unknown.

        Graph first: if Neo4j is down this raises and the files stay, so nothing is orphaned
        in the graph.
        """
        if self.graph is not None:
            self.graph.delete(document_id)
        return self.store.delete(document_id)

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

        for enricher in self.enrichers:
            try:
                metadata[enricher.name] = enricher.enrich(ctx, self.store.resolve_asset)
            except Exception as exc:  # enrichment is optional: keep the loaded content
                log.exception("enrich_failed", document_id=document_id, enricher=enricher.name)
                ctx.warn(f"Enricher {enricher.name!r} failed: {type(exc).__name__}: {exc}")

        if not ctx.elements:
            ctx.warn("No content was extracted from this file")
        self.store.write_elements(document_id, ctx.elements)
        record.status = IngestionStatus.COMPLETED
        record.metadata = metadata
        record.element_count = len(ctx.elements)
        record.element_counts_by_type = dict(
            Counter(e.element_type.value for e in ctx.elements).most_common()
        )
        if self.chunker is not None and ctx.elements:
            # Chunking failure (e.g. tokenizer download offline) is a warning: the elements
            # are stored and the document can be re-chunked later from the CLI.
            try:
                sets = self.chunker.chunk(record, ctx.elements)
            except Exception as exc:
                log.exception("chunking_failed", document_id=document_id)
                ctx.warn(f"Chunking failed: {type(exc).__name__}: {exc}")
                sets = {}
            if sets and self.embedder is not None:
                # Embedding failure (Ollama down, model missing) is a warning too: chunks are
                # stored without refs and `python -m omnirag.embeddings embed` fills them in.
                try:
                    record.metadata["embeddings"] = self.embedder.embed_sets(sets)
                except Exception as exc:
                    log.exception("embedding_failed", document_id=document_id)
                    ctx.warn(f"Embedding failed: {type(exc).__name__}: {exc}")
            for config_id, chunks in sets.items():
                self.store.write_chunks(document_id, config_id, chunks)
            if sets:
                record.metadata["chunk_sets"] = {cid: len(c) for cid, c in sets.items()}
            extracted = None
            if sets and self.extraction is not None:
                # LLM extraction failure (model missing, server down) is a warning; run
                # `python -m omnirag.knowledge_base extract` later.
                try:
                    extracted = self.extraction.run(record, ctx.elements, sets)
                    if extracted is not None:
                        self.store.write_extraction(document_id, extracted.config_id, extracted)
                        record.metadata["extraction"] = extracted.counts()
                except Exception as exc:
                    log.exception("extraction_failed", document_id=document_id)
                    ctx.warn(f"Entity extraction failed: {type(exc).__name__}: {exc}")
            if sets and self.graph is not None:
                # Graph failure (Neo4j down) is a warning: everything is on disk and
                # `python -m omnirag.knowledge_base load` rebuilds the subgraph later.
                try:
                    record.metadata["graph"] = self.graph.index(
                        record, ctx.elements, sets, extracted
                    )
                except Exception as exc:
                    log.exception("graph_failed", document_id=document_id)
                    ctx.warn(f"Graph write failed: {type(exc).__name__}: {exc}")
        record.warnings = ctx.warnings
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
