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
    "aurora_status_report_2026-03.docx",
    "aurora_budget_2026.xlsx",
    "aurora_q1_review.pptx",
]
# Warnings that are correct for a file; anything else is a regression.
EXPECTED_WARNINGS = {"aurora_status_report_2026-03.docx": ["tracked changes"]}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> DocumentStore:
    return DocumentStore(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def corpus(store: DocumentStore) -> dict[str, list[DocumentElement]]:
    pipeline = IngestionPipeline(store, default_loaders(), max_bytes=10 * 1024 * 1024)
    out = {}
    for name in FILES:
        record = pipeline.ingest(SAMPLES / name).record
        assert record.status is IngestionStatus.COMPLETED, record.error
        expected = EXPECTED_WARNINGS.get(name, [])
        assert len(record.warnings) == len(expected), record.warnings
        assert all(e in w for e, w in zip(expected, record.warnings, strict=True))
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


# ---------------------------------------------------------------- Office (Phase 2b)
def test_docx_shows_tracked_changes_as_accepted(corpus) -> None:
    elements = corpus["aurora_status_report_2026-03.docx"]
    summary = _find(elements, "Room-temperature cycling")
    assert "high-temperature testing moved to electrolyte E-7" in summary.content  # insertion
    assert summary.section_path == ["Summary"]
    text = "\n".join(e.content for e in elements)
    assert "preferred electrolyte for all temperatures" not in text  # deletion
    assert "CONFIDENTIAL" not in text  # page header


def test_docx_risk_register_and_budget(corpus) -> None:
    elements = corpus["aurora_status_report_2026-03.docx"]
    row = _find(elements, "ID: R1", ElementType.TABLE_ROW)
    assert "Owner: Marco Lindqvist" in row.content and row.confidence == 1.0
    assert "EUR 240,000" in _find(elements, "steering committee approved").content


def test_docx_figure_asset_and_caption(corpus, store) -> None:
    elements = corpus["aurora_status_report_2026-03.docx"]
    caption = _find(elements, "Figure 1", ElementType.CAPTION)
    image = next(e for e in elements if e.element_id == caption.parent_element_id)
    assert image.element_type is ElementType.IMAGE
    assert store.resolve_asset(image.asset_ref).read_bytes().startswith(b"\x89PNG")


def test_xlsx_budget_row_matches_docx(corpus) -> None:
    row = _find(corpus["aurora_budget_2026.xlsx"], "Workstream: Electrolyte", ElementType.TABLE_ROW)
    assert row.content == (
        "Workstream: Electrolyte | Owner: Marco Lindqvist | Planned (EUR): 240000 | "
        "Spent to date (EUR): 97200 | Spent (%): 40.5%"
    )
    assert row.metadata["excel_row"] == 3 and row.section_path == ["Budget"]


def test_xlsx_offset_supplier_table(corpus) -> None:
    elements = corpus["aurora_budget_2026.xlsx"]
    table = _find(elements, "Carbonix Materials", ElementType.TABLE)
    assert table.metadata["cell_range"] == "B4:E8"
    assert "Scratch" not in {p for e in elements for p in e.section_path}  # empty sheet


def test_pptx_chart_values_match_csv(corpus) -> None:
    # The CSV says AUR-C105 / C106 are at 89.6% / 90.2% after 600 cycles: the chart agrees.
    row = _find(corpus["aurora_q1_review.pptx"], "Category: 600", ElementType.TABLE_ROW)
    assert row.content == "Category: 600 | AUR-C105: 89.6 | AUR-C106: 90.2"
    assert row.page_number == 3


def test_pptx_speaker_notes_and_hidden_slide(corpus) -> None:
    elements = corpus["aurora_q1_review.pptx"]
    notes = _find(elements, "EUR 85,000")
    assert notes.metadata["speaker_notes"] is True and notes.page_number == 2
    backup = _find(elements, "EC decomposition")
    assert backup.metadata["hidden"] is True and backup.page_number == 6
