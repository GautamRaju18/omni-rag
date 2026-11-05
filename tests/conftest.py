"""Shared fixtures."""

from __future__ import annotations

import pytest

from omnirag.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore the developer's local .env file."""
    return Settings(_env_file=None, env="test", log_json=False, neo4j_password="test-password")


@pytest.fixture
def graph_doc():
    """Factory for a small synthetic document for graph tests (unit and integration).

    ``make(doc_id, source_file, created_at)`` returns (record, elements, chunk_sets). Layout:
    title > heading > paragraph, a table with one row, an image with caption + OCR +
    generated description, a 3-chunk multimodal set and a parent/child pair.
    """
    from datetime import UTC, datetime

    from omnirag.chunking.models import Chunk, ChunkingStrategy
    from omnirag.ingestion.models import (
        ContentOrigin,
        DocumentElement,
        DocumentRecord,
        ElementType,
        ExtractionMethod,
        FileKind,
    )

    def make(
        doc_id: str = "doc_00000000000000aa",
        source_file: str = "notes.md",
        created_at: datetime | None = None,
        text: str = "Cell AUR-C103 failed at cycle 412.",
    ):
        record = DocumentRecord(
            document_id=doc_id, source_file=source_file, file_kind=FileKind.MARKDOWN,
            sha256=doc_id[4:] * 4, size_bytes=100, element_count=9,
            metadata={"title": "Aurora Notes"},
            created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        )  # fmt: skip
        top, sub = ["Aurora Notes"], ["Aurora Notes", "Results"]

        def el(i, kind, content, path, *, page=None, parent=None, origin="original", **meta):
            return DocumentElement(
                document_id=doc_id, element_id=f"{doc_id}-e{i:05d}", element_type=kind,
                content=content, source_file=source_file, page_number=page, section_path=path,
                order=i, metadata=meta, extraction_method=ExtractionMethod.MARKDOWN_IT,
                origin=ContentOrigin(origin), confidence=0.8 if origin == "ocr" else 1.0,
                parent_element_id=f"{doc_id}-e{parent:05d}" if parent is not None else None,
                asset_ref="assets/img.png" if kind is ElementType.IMAGE else None,
            )  # fmt: skip

        elements = [
            el(0, ElementType.TITLE, "Aurora Notes", []),
            el(1, ElementType.HEADING, "Results", top),
            el(2, ElementType.PARAGRAPH, text, sub, page=1),
            el(3, ElementType.TABLE, "| Cell | Swelling |\n| AUR-C103 | 14.0 |", sub, page=1,
               columns=["Cell", "Swelling"], n_rows=1, n_cols=2),
            el(4, ElementType.TABLE_ROW, "Cell: AUR-C103 | Swelling: 14.0", sub, page=1,
               parent=3),
            el(5, ElementType.IMAGE, "", sub, page=2),
            el(6, ElementType.CAPTION, "Figure 1: swelling by cell", sub, page=2, parent=5),
            el(7, ElementType.PARAGRAPH, "14%", sub, page=2, parent=5, origin="ocr"),
            el(8, ElementType.DESCRIPTION, "A bar chart of swelling.", sub, page=2, parent=5,
               origin="generated", model="qwen3-vl:2b-instruct"),
        ]  # fmt: skip

        def chunk(config, i, ids, content, *, pages=(), level=0, parent=None, retrievable=True):
            return Chunk(
                chunk_id=f"{doc_id}-{config}-{i:05d}", document_id=doc_id,
                source_file=source_file, chunking_strategy=ChunkingStrategy.MULTIMODAL
                if config.startswith("multimodal") else ChunkingStrategy.PARENT_CHILD,
                config_id=config, chunk_index=i, content=content, token_count=8,
                page_number=pages[0] if pages else None, page_numbers=list(pages),
                section_path=sub, element_type="paragraph",
                element_ids=[f"{doc_id}-e{n:05d}" for n in ids], level=level,
                parent_chunk_id=f"{doc_id}-{config}-{parent:05d}" if parent is not None else None,
                retrievable=retrievable, origins=["original"],
                embedding_ref=f"space-test/{doc_id}-{config}-{i}" if retrievable else None,
            )  # fmt: skip

        mm, pc = "multimodal-256-ctx", "parentchild-256-128-32"
        sets = {
            mm: [
                chunk(mm, 0, [0, 1, 2], f"Aurora Notes\n\nResults\n\n{text}", pages=[1]),
                chunk(mm, 1, [4], "Cell: AUR-C103 | Swelling: 14.0", pages=[1]),
                chunk(mm, 2, [5, 6, 7, 8], "Figure 1: swelling by cell\n14%", pages=[2]),
            ],
            pc: [
                chunk(pc, 0, [0, 1, 2, 3, 4], text, pages=[1], retrievable=False),
                chunk(pc, 1, [2], text, pages=[1], level=1, parent=0),
            ],
        }
        return record, elements, sets

    return make
