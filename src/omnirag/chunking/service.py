"""Run the configured chunking strategies for a document and describe the results."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

from omnirag.chunking.models import Chunk, ChunkingConfig, ChunkingStrategy
from omnirag.chunking.semantic import Embedder
from omnirag.chunking.strategies import chunk_document
from omnirag.chunking.tokenizer import TokenCounter
from omnirag.ingestion.models import DocumentElement, DocumentRecord, ElementType


class Chunker:
    """Holds the token counter (loaded lazily: the tokenizer file may need downloading)."""

    def __init__(
        self,
        counter_factory: Callable[[], TokenCounter],
        configs: Sequence[ChunkingConfig],
        embedder: Embedder | None = None,  # only needed for the semantic strategy
    ) -> None:
        self._factory = counter_factory
        self._counter: TokenCounter | None = None
        self.configs = list(configs)
        self.embedder = embedder

    @property
    def counter(self) -> TokenCounter:
        if self._counter is None:
            self._counter = self._factory()
        return self._counter

    def chunk(
        self, record: DocumentRecord, elements: list[DocumentElement]
    ) -> dict[str, list[Chunk]]:
        return {
            config.config_id: chunk_document(record, elements, config, self.counter, self.embedder)
            for config in self.configs
        }


def default_configs(
    strategies: Sequence[ChunkingStrategy],
    *,
    chunk_size: int,
    chunk_overlap: int,
    child_size: int,
    parent_size: int,
    contextual_headers: bool,
) -> list[ChunkingConfig]:
    return [
        ChunkingConfig(
            strategy=s,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            child_size=child_size,
            parent_size=parent_size,
            contextual_headers=contextual_headers,
        )  # fmt: skip
        for s in strategies
    ]


def integrity(elements: list[DocumentElement], chunks: list[Chunk]) -> float:
    """Share of content elements contained *whole* in at least one chunk.

    A proxy for "can a single retrieved chunk answer a question about this element?".
    It needs no retrieval and no labels. A table row counts as whole if its table is.
    """
    whole: set[str] = set()
    for c in chunks:
        partial = set(c.partial_element_ids)
        whole.update(e for e in c.element_ids if e not in partial)
    content = [e for e in elements if e.content and e.element_type is not ElementType.HEADING]
    if not content:
        return 1.0
    ok = sum(
        1
        for e in content
        if e.element_id in whole
        or (e.element_type is ElementType.TABLE_ROW and e.parent_element_id in whole)
    )
    return ok / len(content)


def level_limit(config: ChunkingConfig, level: int) -> int:
    """The size a chunk at ``level`` is allowed to have under ``config``."""
    if config.strategy is ChunkingStrategy.HIERARCHICAL:
        return [config.parent_size, config.chunk_size, config.child_size][min(level, 2)]
    if config.strategy is ChunkingStrategy.PARENT_CHILD:
        return [config.chunk_size, config.child_size][min(level, 1)]
    return config.chunk_size


def describe(
    chunks: list[Chunk], elements: list[DocumentElement], config: ChunkingConfig
) -> dict[str, Any]:
    tokens = [c.token_count for c in chunks if c.retrievable] or [0]
    return {
        "chunks": len(chunks),
        "retrievable": sum(c.retrievable for c in chunks),
        "tokens_median": statistics.median(tokens),
        "tokens_p95": sorted(tokens)[int(0.95 * (len(tokens) - 1))],
        "tokens_max": max(tokens),
        "over_limit": sum(c.token_count > level_limit(config, c.level) for c in chunks),
        "integrity": round(integrity(elements, chunks), 3),
        "bytes": sum(len(c.model_dump_json()) + 1 for c in chunks),
    }


def timed_chunk(
    record: DocumentRecord,
    elements: list[DocumentElement],
    config: ChunkingConfig,
    counter: TokenCounter,
    embedder: Embedder | None = None,
) -> tuple[list[Chunk], float]:
    started = time.perf_counter()
    chunks = chunk_document(record, elements, config, counter, embedder)
    return chunks, round((time.perf_counter() - started) * 1000, 2)
