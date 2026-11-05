"""The seven strategies on one hand-built document (whitespace tokens = words)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import integrity, level_limit
from omnirag.chunking.strategies import chunk_document
from omnirag.chunking.tokenizer import WhitespaceTokenizer
from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import (
    ContentOrigin,
    DocumentRecord,
    ElementType,
    ExtractionMethod,
    FileKind,
)
from omnirag.ingestion.tables import add_table

WS = WhitespaceTokenizer()
M = ExtractionMethod.PYMUPDF
S = ChunkingStrategy


def _doc():
    ctx = ExtractionContext("doc_00000000000000aa", "r.pdf")
    ctx.add(ElementType.HEADING, "Report", extraction_method=M, page_number=1,
            metadata={"level": 1})  # fmt: skip
    ctx.add(ElementType.PARAGRAPH, " ".join(f"intro{i}" for i in range(30)), extraction_method=M,
            page_number=1, section_path=["Report"])  # fmt: skip
    ctx.add(ElementType.HEADING, "Results", extraction_method=M, page_number=2,
            section_path=["Report"], metadata={"level": 2})  # fmt: skip
    ctx.add(ElementType.PARAGRAPH, " ".join(f"res{i}" for i in range(20)), extraction_method=M,
            page_number=2, section_path=["Report", "Results"])  # fmt: skip
    rows = [[f"C{i}", str(80 + i)] for i in range(12)]
    add_table(ctx, ["Cell", "Retention"], rows, extraction_method=M, page_number=2,
              section_path=["Report", "Results"])  # fmt: skip
    image = ctx.add(ElementType.IMAGE, "", extraction_method=M, page_number=3,
                    section_path=["Report", "Results"], asset_ref="assets/x.png")  # fmt: skip
    ctx.add(ElementType.CAPTION, "Figure 1: retention bars", extraction_method=M, page_number=3,
            section_path=["Report", "Results"], parent_id=image)  # fmt: skip
    ctx.add(ElementType.PARAGRAPH, "C103 72.5", extraction_method=ExtractionMethod.TESSERACT,
            origin=ContentOrigin.OCR, confidence=0.91, page_number=3, parent_id=image,
            metadata={"words": 2})  # fmt: skip
    ctx.add(ElementType.DESCRIPTION, "A bar chart of retention per cell", parent_id=image,
            extraction_method=ExtractionMethod.VLM, origin=ContentOrigin.GENERATED,
            confidence=0.5, page_number=3, metadata={"model": "vlm:test"})  # fmt: skip
    # A second, tiny top-level section: its hierarchical parent fits the child size, which is
    # where a naive splitter would create a child identical to its parent.
    ctx.add(ElementType.HEADING, "Appendix", extraction_method=M, page_number=4,
            metadata={"level": 1})  # fmt: skip
    ctx.add(ElementType.PARAGRAPH, "short appendix note", extraction_method=M, page_number=4,
            section_path=["Appendix"])  # fmt: skip
    record = DocumentRecord(
        document_id=ctx.document_id,
        source_file="r.pdf",
        file_kind=FileKind.PDF,
        sha256="0" * 64,
        size_bytes=1,
        metadata={"title": "Aurora Report"},
    )
    return record, ctx.elements


def _chunks(strategy: S, **kw):
    record, elements = _doc()
    sizes = {"chunk_size": 40, "chunk_overlap": 5, "child_size": 16, "parent_size": 200}
    config = ChunkingConfig(strategy=strategy, **{**sizes, **kw})
    return chunk_document(record, elements, config, WS), elements, config


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_every_strategy_respects_size_limits_and_provenance(strategy: S) -> None:
    chunks, elements, config = _chunks(strategy)
    ids = {e.element_id for e in elements}
    assert chunks
    for i, c in enumerate(chunks):
        assert (
            c.chunk_index == i and c.chunk_id == f"doc_00000000000000aa-{config.config_id}-{i:05d}"
        )
        assert c.token_count == WS.count(c.content) <= level_limit(config, c.level)
        assert c.element_ids and set(c.element_ids) <= ids
        assert set(c.partial_element_ids) <= set(c.element_ids)
        assert c.content.strip() == c.content and c.content


def test_chunking_is_deterministic() -> None:
    first, _, _ = _chunks(S.MULTIMODAL)
    second, _, _ = _chunks(S.MULTIMODAL)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_fixed_cuts_through_elements() -> None:
    chunks, elements, _ = _chunks(S.FIXED)
    assert all(c.token_count == 40 for c in chunks[:-1])  # exact windows, last one shorter
    assert any(c.partial_element_ids for c in chunks)
    assert integrity(elements, chunks) < 1.0


def test_structure_keeps_sections_apart_and_headings_first() -> None:
    chunks, _, _ = _chunks(S.STRUCTURE)
    results = [c for c in chunks if c.section_path == ["Report", "Results"]]
    assert results[0].content.startswith("Results\n\nres0")
    assert all("intro" not in c.content for c in results)  # no chunk spans two sections


def test_table_strategy_splits_tables_by_rows_with_header_repeated() -> None:
    chunks, elements, _ = _chunks(S.TABLE)
    parts = [c for c in chunks if c.element_type == "table"]
    assert len(parts) >= 2
    assert all("| Cell | Retention |" in c.content for c in parts)  # header in every part
    row_ids = {e.element_id for e in elements if e.element_type is ElementType.TABLE_ROW}
    covered = {eid for c in parts for eid in c.element_ids} & row_ids
    assert covered == row_ids  # every row lands in exactly the parts that show it
    assert all(not ({eid for eid in c.element_ids if eid in row_ids} & set(c.partial_element_ids))
               for c in parts)  # fmt: skip


def test_multimodal_groups_image_caption_ocr_and_description() -> None:
    chunks, elements, _ = _chunks(S.MULTIMODAL)
    (figure,) = [c for c in chunks if c.element_type == "figure"]
    image = next(e for e in elements if e.element_type is ElementType.IMAGE)
    assert image.element_id in figure.element_ids and figure.page_number == 3
    assert "Figure: Figure 1: retention bars" in figure.content
    assert "Text in image (OCR, confidence 0.91): C103 72.5" in figure.content
    assert "Image description (generated by vlm:test): A bar chart" in figure.content
    assert figure.origins == ["generated", "ocr", "original"] and figure.min_confidence == 0.5
    others = [c for c in chunks if c is not figure]
    assert all("bar chart" not in c.content for c in others)  # descendants not repeated


def test_hierarchical_levels_nest_without_duplicates() -> None:
    chunks, _, config = _chunks(S.HIERARCHICAL)
    by_id = {c.chunk_id: c for c in chunks}
    assert {c.level for c in chunks} >= {0, 1}
    for c in chunks:
        if c.parent_chunk_id:
            parent = by_id[c.parent_chunk_id]
            assert parent.level == c.level - 1
            assert set(c.element_ids) <= set(parent.element_ids)
            assert c.content != parent.content  # no child identical to its parent
    assert all(c.retrievable for c in chunks)


def test_parent_child_indexes_children_only() -> None:
    chunks, _, config = _chunks(S.PARENT_CHILD)
    parents = [c for c in chunks if c.level == 0]
    children = [c for c in chunks if c.level == 1]
    assert parents and all(not p.retrievable for p in parents)
    assert all(c.retrievable and c.token_count <= config.child_size for c in children)
    assert {c.parent_chunk_id for c in children} == {p.chunk_id for p in parents}


def test_contextual_headers_only_change_the_embedding_text() -> None:
    plain, _, _ = _chunks(S.STRUCTURE)
    enriched, _, config = _chunks(S.STRUCTURE, contextual_headers=True)
    assert config.config_id == "structure-40-ctx"
    assert [c.content for c in plain] == [c.content for c in enriched]
    c = next(c for c in enriched if c.section_path == ["Report", "Results"])
    assert c.context == "Document: Aurora Report\nSection: Report > Results"
    assert c.embedding_text() == f"{c.context}\n\n{c.content}"


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"strategy": "fixed", "chunk_size": 64, "chunk_overlap": 64}, "overlap"),
        ({"strategy": "hierarchical", "chunk_size": 64, "child_size": 128}, "child_size"),
        ({"strategy": "parent_child", "chunk_size": 64, "child_size": 64}, "child_size"),
    ],
)
def test_invalid_configs_are_rejected(kw, message) -> None:
    with pytest.raises(ValidationError, match=message):
        ChunkingConfig(**kw)


def test_config_ids() -> None:
    ids = [ChunkingConfig(strategy=s).config_id for s in ChunkingStrategy]
    assert ids == [
        "fixed-512-64", "recursive-512-64", "structure-512", "table-512", "multimodal-512",
        "hierarchical-2048-512-128", "parentchild-512-128-64",
    ]  # fmt: skip
