"""Golden checks on the real sample corpus (whitespace tokens, so no tokenizer download)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import integrity
from omnirag.chunking.strategies import chunk_document
from omnirag.chunking.tokenizer import WhitespaceTokenizer
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore

SAMPLES = Path(__file__).parents[3] / "data" / "samples"
WS = WhitespaceTokenizer()


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    store = DocumentStore(tmp_path_factory.mktemp("data"))
    pipeline = IngestionPipeline(store, default_loaders(), max_bytes=10 << 20)
    out = {}
    for name in ("aurora_safety_review.pdf", "aurora_q1_review.pptx", "aurora_budget_2026.xlsx"):
        record = pipeline.ingest(SAMPLES / name).record
        out[name] = (record, store.read_elements(record.document_id))
    return out


def _run(docs, name, strategy, size=120):
    record, elements = docs[name]
    config = ChunkingConfig(strategy=strategy, chunk_size=size, chunk_overlap=16, child_size=40,
                            parent_size=600)  # fmt: skip
    return chunk_document(record, elements, config, WS), elements


def test_table_chunk_keeps_the_swelling_table_whole_with_its_heading(docs) -> None:
    chunks, _ = _run(docs, "aurora_safety_review.pdf", ChunkingStrategy.TABLE)
    table = next(c for c in chunks if c.element_type == "table")
    assert table.content.startswith("3. Swelling Measurements")
    assert "| AUR-C105 | E-7 | 45 | 600 | 2.1 |" in table.content
    assert table.page_number == 2 and table.section_path[-1] == "3. Swelling Measurements"


def test_pptx_chart_data_and_its_figure_are_one_multimodal_chunk(docs) -> None:
    chunks, _ = _run(docs, "aurora_q1_review.pptx", ChunkingStrategy.MULTIMODAL)
    chart = next(c for c in chunks if "Chart: Capacity retention" in c.content)
    assert chart.element_type == "figure" and chart.page_number == 3
    assert "| 600 | 89.6 | 90.2 |" in chart.content


def test_structure_strategies_keep_elements_whole_where_fixed_does_not(docs) -> None:
    fixed, elements = _run(docs, "aurora_budget_2026.xlsx", ChunkingStrategy.FIXED, size=60)
    table, _ = _run(docs, "aurora_budget_2026.xlsx", ChunkingStrategy.TABLE, size=60)
    assert integrity(elements, table) == 1.0
    assert integrity(elements, fixed) < 1.0
    assert all(c.section_path for c in table)  # every chunk knows its sheet
