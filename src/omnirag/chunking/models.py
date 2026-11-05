"""Chunk schema and chunking configuration.

A chunk is the unit that gets embedded, indexed and retrieved. It records **which elements it
was built from** (``element_ids``). Element IDs are deterministic, while chunk IDs change with
strategy and size, so evaluation ground truth is written once at element level and mapped onto
any strategy's chunks.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"  # token windows over the plain text, ignoring structure (baseline)
    RECURSIVE = "recursive"  # split at paragraph > line > sentence > word boundaries
    STRUCTURE = "structure"  # never cross a section; elements kept whole when they fit
    TABLE = "table"  # structure + tables as their own chunks, header repeated per part
    MULTIMODAL = "multimodal"  # table + image/figure groups (caption, OCR, description)
    HIERARCHICAL = "hierarchical"  # 3 nested levels, every level retrievable
    PARENT_CHILD = "parent_child"  # small children retrieved, larger parents returned
    # SEMANTIC (embedding-similarity boundaries) arrives in Phase 3b.


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy
    chunk_size: int = Field(default=512, ge=16, le=8192)  # tokens (bge-m3 tokenizer)
    chunk_overlap: int = Field(default=64, ge=0)  # fixed / recursive / child splits
    child_size: int = Field(default=128, ge=16)  # parent_child children, hierarchical leaves
    parent_size: int = Field(default=2048, ge=16)  # hierarchical top level
    contextual_headers: bool = False  # prepend "title > section" to the embedding text

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            if "chunk_overlap" in self.model_fields_set:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
            # Only the default is too big (e.g. chunk_size=64): scale it instead of failing.
            self.chunk_overlap = self.chunk_size // 4
        if self.strategy is ChunkingStrategy.HIERARCHICAL and not (
            self.child_size < self.chunk_size < self.parent_size
        ):
            raise ValueError("hierarchical needs child_size < chunk_size < parent_size")
        if self.strategy is ChunkingStrategy.PARENT_CHILD and self.child_size >= self.chunk_size:
            raise ValueError("parent_child needs child_size < chunk_size")
        return self

    @property
    def config_id(self) -> str:
        """Stable, readable ID used in chunk IDs and file names, e.g. ``recursive-512-64``."""
        s = self.strategy
        if s in (ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE):
            base = f"{s.value}-{self.chunk_size}-{self.chunk_overlap}"
        elif s is ChunkingStrategy.HIERARCHICAL:
            base = f"hierarchical-{self.parent_size}-{self.chunk_size}-{self.child_size}"
        elif s is ChunkingStrategy.PARENT_CHILD:
            base = f"parentchild-{self.chunk_size}-{self.child_size}-{self.chunk_overlap}"
        else:
            base = f"{s.value}-{self.chunk_size}"
        return base + ("-ctx" if self.contextual_headers else "")


class Chunk(BaseModel):
    chunk_id: str  # f"{document_id}-{config_id}-{chunk_index:05d}"
    document_id: str
    source_file: str
    chunking_strategy: ChunkingStrategy
    config_id: str
    chunk_index: int
    content: str  # the document's own text (or labelled figure text), what a citation shows
    context: str | None = None  # optional enrichment, prepended only for embedding
    token_count: int  # of ``content``, bge-m3 tokens without special tokens
    page_number: int | None = None  # first page; None for page-less formats
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    element_type: str  # dominant source element type ("paragraph", "table", "figure", ...)
    element_ids: list[str]  # every element this chunk contains (fully or partly)
    partial_element_ids: list[str] = Field(default_factory=list)  # split across chunks
    parent_chunk_id: str | None = None
    level: int = 0  # depth in hierarchical / parent_child trees (0 = top)
    retrievable: bool = True  # False for parent_child parents (returned, not searched)
    origins: list[str] = Field(default_factory=list)  # original / ocr / generated
    min_confidence: float = 1.0
    embedding_ref: str | None = None  # set by the embedding engine (Phase 4)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def section(self) -> str:
        return " > ".join(self.section_path)

    def embedding_text(self) -> str:
        return f"{self.context}\n\n{self.content}" if self.context else self.content
