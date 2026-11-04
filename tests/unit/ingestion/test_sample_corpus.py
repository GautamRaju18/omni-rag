"""Golden tests: ingest the real sample corpus and check facts land where expected.

These protect against regressions that unit tests on tiny inputs miss: e.g. a loader change
that silently drops the table the evaluation dataset will later ask about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import DocumentElement, ElementType, IngestionStatus
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore

SAMPLES = Path(__file__).parents[3] / "data" / "samples"
FILES = [
    "aurora_research_notes.md",
    "cell_test_results.csv",
    "aurora_config.json",
    "meeting_notes_2026-03-12.txt",
    "helios_aurora_s1.html",
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[DocumentElement]]:
    store = DocumentStore(tmp_path_factory.mktemp("data"))
    pipeline = IngestionPipeline(store, default_loaders(), max_bytes=10 * 1024 * 1024)
    out = {}
    for name in FILES:
        record = pipeline.ingest(SAMPLES / name).record
        assert record.status is IngestionStatus.COMPLETED, record.error
        assert record.warnings == [], record.warnings
        out[name] = store.read_elements(record.document_id)
    return out


def _find(
    elements: list[DocumentElement], needle: str, etype: ElementType | None = None
) -> DocumentElement:
    return next(
        e for e in elements if needle in e.content and (etype is None or e.element_type is etype)
    )


def test_markdown_results_table_rows_are_self_describing(corpus) -> None:
    row = _find(corpus["aurora_research_notes.md"], "AUR-C103 | Electrolyte")
    assert row.element_type is ElementType.TABLE_ROW
    assert row.content == (
        "Cell: AUR-C103 | Electrolyte: E-5 | Temp (°C): 45 | Cycles: 412 | Retention (%): 72.5"
    )
    assert row.section_path == ["Project Aurora: Research Notes", "Results (as of 2026-03-10)"]


def test_markdown_keeps_chemical_subscripts(corpus) -> None:
    assert "Na₂Fe[Fe(CN)₆]" in _find(corpus["aurora_research_notes.md"], "Prussian white (").content


def test_csv_failed_cell(corpus) -> None:
    row = _find(corpus["cell_test_results.csv"], "AUR-C103", ElementType.TABLE_ROW)
    assert "status: failed" in row.content and "temperature_c: 45" in row.content


def test_json_owner_of_e7(corpus) -> None:
    record = _find(corpus["aurora_config.json"], "E-7 qualification")
    assert record.content.startswith("name: Marco Lindqvist")
    assert record.metadata["json_path"] == "$.team[1]"


def test_meeting_notes_decision(corpus) -> None:
    para = _find(corpus["meeting_notes_2026-03-12.txt"], "Decision: move")
    assert "Marco Lindqvist owns the E-7 qualification" in para.content


def test_html_spec_table_without_boilerplate(corpus) -> None:
    elements = corpus["helios_aurora_s1.html"]
    text = "\n".join(e.content for e in elements)
    for noise in ("Careers", "cookies", "Privacy", "analytics"):
        assert noise not in text
    row = _find(elements, "Cycle life", ElementType.TABLE_ROW)
    assert row.content == "Parameter: Cycle life | Value: 3000 cycles to 80% | Condition: 25 °C, 1C"
    caption = _find(elements, "Figure 1")
    image = next(e for e in elements if e.element_id == caption.parent_element_id)
    assert image.element_type is ElementType.IMAGE
