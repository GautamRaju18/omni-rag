"""Retrieval request/response models shared by every retriever (dense, lexical; graph in Phase 7).

A retriever returns ranked ``RetrievedChunk``s with everything a citation or a later fusion
step needs: provenance (document, pages, section, element IDs), what the text is made of
(origins, OCR confidence) and how it was found (retriever, rank, raw score). Raw scores are
only comparable *within* one retriever (cosine vs BM25), which is why Phase 7 fuses ranks
with RRF instead of adding scores.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrieverName = Literal["dense", "bm25", "graph"]
# Retrieval modes (brief section 9): one retriever, several fused with RRF, optionally
# re-scored by a cross-encoder (Phase 8). Query rewriting joins in Phase 9.
RetrievalMode = Literal[
    "dense", "bm25", "graph", "dense+bm25", "dense+bm25+graph",
    "dense+bm25+rerank", "dense+bm25+graph+rerank",
    "advanced",  # the Phase 9 LangGraph workflow (transforms + context check + retries)
]  # fmt: skip


class RetrievalFilters(BaseModel):
    """Metadata filters. ``None`` means "no constraint"."""

    config_id: str | None = None  # chunk set; None = the configured search set
    latest_only: bool = True  # skip superseded document versions
    document_ids: list[str] | None = Field(default=None, max_length=200)
    source_files: list[str] | None = Field(default=None, max_length=200)
    element_types: list[str] | None = Field(default=None, max_length=20)  # table, figure, ...
    exclude_generated: bool = False  # drop chunks containing model-written text
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)  # e.g. OCR quality


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    rank: int  # 1-based, within this retriever's result
    score: float  # raw retriever score (cosine, BM25, graph weight, RRF); per retriever only
    retriever: RetrievalMode
    content: str
    context: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    element_type: str
    element_ids: list[str] = Field(default_factory=list)
    origins: list[str] = Field(default_factory=list)
    min_confidence: float = 1.0
    config_id: str
    graph_paths: list[str] = Field(default_factory=list)  # how the graph retriever found it
    # Fused results: each member retriever's rank and raw score for this chunk.
    sources: dict[str, dict[str, float]] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    retriever: RetrievalMode
    k: int
    hits: list[RetrievedChunk]
    latency_ms: float
    details: dict[str, Any] = Field(default_factory=dict)  # query actually run, fetch size, ...


class RetrievalError(RuntimeError):
    """Retrieval can't run correctly (e.g. the query and the index use different embeddings)."""


# Properties every retriever returns (one Cypher projection, so both stay in sync).
CHUNK_FIELDS = (
    "c.chunk_id AS chunk_id, c.document_id AS document_id, c.source_file AS source_file, "
    "c.content AS content, c.context AS context, c.section_path AS section_path, "
    "c.page_numbers AS page_numbers, c.element_type AS element_type, "
    "c.element_ids AS element_ids, c.origins AS origins, c.min_confidence AS min_confidence, "
    "c.config_id AS config_id"
)
