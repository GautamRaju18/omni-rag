"""PDF engine routing (fast, fake engine) and the real Docling engine (``-m docling``)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from omnirag.ingestion.loaders.docling_engine import DoclingEngine, docling_available
from omnirag.ingestion.loaders.pdf_loader import PdfLoader
from omnirag.ingestion.models import ElementType, ExtractionMethod

SAMPLES = Path(__file__).parents[3] / "data" / "samples"
PROSE = "This paragraph is long enough to count as prose for the column detector."


def _two_column_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(50, 60, 280, 400), f"<p>Left. {PROSE}</p>")
    page.insert_htmlbox(pymupdf.Rect(315, 60, 545, 400), f"<p>Right. {PROSE}</p>")
    return doc.tobytes()


def _one_column_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_htmlbox(pymupdf.Rect(50, 60, 545, 400), f"<p>{PROSE}</p><p>{PROSE}</p>")
    return doc.tobytes()


class FakeEngine:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[Path] = []

    def convert(self, path: Path, ctx) -> dict:
        self.calls.append(path)
        ctx.add(
            ElementType.PARAGRAPH, "from layout engine", extraction_method=ExtractionMethod.DOCLING
        )
        if self.fail:
            raise RuntimeError("model crashed")
        return {"pictures": 0, "tables": 0}


def test_auto_routes_multi_column_pdf_to_layout_engine(run_loader) -> None:
    engine = FakeEngine()
    ctx, meta = run_loader(PdfLoader(layout_engine=engine, engine_mode="auto"), "c.pdf",
                           _two_column_pdf())  # fmt: skip
    assert len(engine.calls) == 1
    assert meta["engine"] == "docling" and meta["multi_column_pages"] == [1]
    assert [e.content for e in ctx.elements] == ["from layout engine"]


def test_auto_keeps_single_column_pdf_on_pymupdf(run_loader) -> None:
    engine = FakeEngine()
    _, meta = run_loader(PdfLoader(layout_engine=engine, engine_mode="auto"), "s.pdf",
                         _one_column_pdf())  # fmt: skip
    assert engine.calls == [] and meta["engine"] == "pymupdf"
    assert meta["engine_reason"] == "single-column layout"


def test_layout_engine_failure_falls_back_without_duplicates(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(layout_engine=FakeEngine(fail=True), engine_mode="docling"),
                           "c.pdf", _two_column_pdf())  # fmt: skip
    assert meta["engine"] == "pymupdf" and "Docling failed" in ctx.warnings[0]
    assert "from layout engine" not in [e.content for e in ctx.elements]  # partial output dropped
    assert ctx.elements[0].element_id == "doc_test-e00000"  # IDs restart cleanly


def test_docling_mode_without_docling_installed_warns(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(engine_mode="docling"), "s.pdf", _one_column_pdf())
    assert meta["engine"] == "pymupdf" and "not installed" in ctx.warnings[0]


# ------------------------------------------------------------- real Docling
@pytest.mark.docling
@pytest.mark.skipif(not docling_available(), reason="docling not installed")
def test_docling_reads_columns_in_order_and_finds_borderless_table(run_loader) -> None:
    loader = PdfLoader(layout_engine=DoclingEngine(), engine_mode="auto")
    ctx, meta = run_loader(loader, "e7.pdf", (SAMPLES / "e7_technical_note.pdf").read_bytes())
    assert meta["engine"] == "docling" and meta["tables"] == 1
    text = [e.content for e in ctx.elements]
    abstract = next(i for i, t in enumerate(text) if t.startswith("Abstract"))
    results = next(i for i, t in enumerate(text) if "3. Results" in t)
    assert abstract < results  # left column before right column
    rows = [e.content for e in ctx.elements if e.element_type is ElementType.TABLE_ROW]
    assert "Component: NaDFOB | Role: SEI additive | Amount: 1 wt%" in rows
    assert all(e.extraction_method is ExtractionMethod.DOCLING for e in ctx.elements)
