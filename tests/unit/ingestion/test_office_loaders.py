"""DOCX / XLSX / PPTX loaders on small documents built in the test itself.

Building inputs with the same libraries keeps every case readable and explicit: the test shows
exactly which XML structure is being exercised.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import docx
import openpyxl
import pytest
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from omnirag.ingestion.loaders.docx_loader import DocxLoader
from omnirag.ingestion.loaders.pptx_loader import PptxLoader
from omnirag.ingestion.loaders.xlsx_loader import XlsxLoader, render_value
from omnirag.ingestion.models import ElementType


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(out, format="PNG")
    return out.getvalue()


def _save(obj) -> bytes:
    out = io.BytesIO()
    obj.save(out)
    return out.getvalue()


def _types(ctx) -> list[str]:
    return [e.element_type.value for e in ctx.elements]


# ------------------------------------------------------------------ docx
def _tracked_doc() -> bytes:
    d = docx.Document()
    p = d.add_paragraph("Kept.")
    p._p.append(
        parse_xml(
            f'<w:del {nsdecls("w")} w:id="1" w:author="t" w:date="2026-01-01T00:00:00Z">'
            "<w:r><w:delText> Deleted.</w:delText></w:r></w:del>"
        )
    )
    p._p.append(
        parse_xml(
            f'<w:ins {nsdecls("w")} w:id="2" w:author="t" w:date="2026-01-01T00:00:00Z">'
            '<w:r><w:t xml:space="preserve"> Inserted.</w:t></w:r></w:ins>'
        )
    )
    return _save(d)


def test_docx_tracked_insertions_kept_deletions_dropped(run_loader) -> None:
    ctx, meta = run_loader(DocxLoader(), "t.docx", _tracked_doc())
    assert ctx.elements[0].content == "Kept. Inserted."
    assert meta["tracked_changes"] == {"insertions": 1, "deletions": 1}
    assert any("tracked changes" in w for w in ctx.warnings)


def test_python_docx_paragraph_text_misses_insertions(tmp_path: Path) -> None:
    """Why ``_element_text`` exists. If this fails, python-docx fixed it: revisit the loader."""
    path = tmp_path / "t.docx"
    path.write_bytes(_tracked_doc())
    assert docx.Document(str(path)).paragraphs[0].text == "Kept."


def test_docx_heading_by_outline_level_not_style_name(run_loader) -> None:
    d = docx.Document()
    style = d.styles.add_style("Kapitel", WD_STYLE_TYPE.PARAGRAPH)  # a localized/custom name
    style.element.get_or_add_pPr().append(parse_xml(f'<w:outlineLvl {nsdecls("w")} w:val="0"/>'))
    d.add_paragraph("Einleitung", style="Kapitel")
    d.add_heading("Details", level=2)
    d.add_paragraph("Body text.")
    ctx, _ = run_loader(DocxLoader(), "h.docx", _save(d))
    assert _types(ctx) == ["heading", "heading", "paragraph"]
    assert ctx.elements[0].metadata["level"] == 1
    assert ctx.elements[2].section_path == ["Einleitung", "Details"]


def test_docx_lists_nesting_and_numbering(run_loader) -> None:
    d = docx.Document()
    d.add_paragraph("a", style="List Bullet")
    d.add_paragraph("a.1", style="List Bullet 2")
    d.add_paragraph("b", style="List Bullet")
    d.add_paragraph("first", style="List Number")
    d.add_paragraph("second", style="List Number")
    ctx, _ = run_loader(DocxLoader(), "l.docx", _save(d))
    bullets, numbered = ctx.elements
    assert bullets.content == "- a\n  - a.1\n- b"
    assert numbered.content == "1. first\n2. second"
    assert numbered.metadata["ordered"] is True


def test_docx_tables_header_confidence_and_layout_tables(run_loader) -> None:
    d = docx.Document()
    data = d.add_table(rows=2, cols=2)
    for r, row in enumerate([("Key", "Value"), ("a", "1")]):
        for c, value in enumerate(row):
            data.cell(r, c).text = value
    layout = d.add_table(rows=1, cols=2)  # one row: layout, not data
    layout.cell(0, 0).text = "Left column"
    layout.cell(0, 1).text = "Right column"
    ctx, _ = run_loader(DocxLoader(), "t.docx", _save(d))
    assert _types(ctx) == ["table", "table_row", "paragraph", "paragraph"]
    assert ctx.elements[0].confidence == pytest.approx(0.9)  # header row not declared
    assert ctx.elements[1].content == "Key: a | Value: 1"
    assert ctx.elements[2].metadata["layout_table"] is True


def test_docx_image_saved_with_alt_text_and_caption(run_loader, tmp_path: Path) -> None:
    d = docx.Document()
    shape = d.add_picture(io.BytesIO(_png()))
    shape._inline.docPr.set("descr", "A red square")
    d.add_paragraph("Figure 1: Red.", style="Caption")
    d.sections[0].header.paragraphs[0].text = "CONFIDENTIAL"
    ctx, meta = run_loader(DocxLoader(), "i.docx", _save(d))
    image, caption = ctx.elements  # the page header is not extracted
    assert image.element_type is ElementType.IMAGE and image.content == "A red square"
    assert (tmp_path / image.asset_ref).read_bytes() == _png()  # original bytes preserved
    assert caption.parent_element_id == image.element_id
    assert meta["images"] == 1


# ------------------------------------------------------------------ xlsx
@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (0.725, "0.0%", "72.5%"),
        (0.5, "0%", "50%"),
        (1000.0, "General", "1000"),
        (3.25, "General", "3.25"),
        (True, "General", "TRUE"),
        (datetime(2026, 5, 30), "yyyy-mm-dd", "2026-05-30"),
        (datetime(2026, 5, 30, 14, 5), "yyyy-mm-dd h:mm", "2026-05-30 14:05:00"),
        (None, "General", ""),
    ],
)
def test_xlsx_render_value(value, fmt, expected) -> None:
    assert render_value(value, fmt) == expected


def test_xlsx_blocks_offsets_and_row_numbers(run_loader) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["C3"], ws["D3"] = "Item", "Qty"
    ws["C4"], ws["D4"] = "bolt", 40
    ws["C5"], ws["D5"] = "nut", 25
    ws["C8"] = "Checked by Aiko"  # second block after blank rows
    wb.create_sheet("Empty")
    ctx, meta = run_loader(XlsxLoader(), "b.xlsx", _save(wb))
    assert _types(ctx) == ["table", "table_row", "table_row", "paragraph"]
    table, row1, _, note = ctx.elements
    assert table.metadata["cell_range"] == "C3:D5"
    assert row1.content == "Item: bolt | Qty: 40"
    assert row1.metadata["excel_row"] == 4
    assert note.content == "Checked by Aiko" and note.section_path == ["Data"]
    assert [s["name"] for s in meta["sheets"]] == ["Data", "Empty"]


def test_xlsx_numeric_block_gets_generated_headers(run_loader) -> None:
    wb = openpyxl.Workbook()
    for row in [(1, 2.5), (3, 4.5)]:
        wb.active.append(row)
    ctx, _ = run_loader(XlsxLoader(), "n.xlsx", _save(wb))
    assert ctx.elements[0].metadata["header_detected"] is False
    assert ctx.elements[1].content == "column_1: 1 | column_2: 2.5"


def test_xlsx_uncached_formula_shows_formula_and_warns(run_loader) -> None:
    wb = openpyxl.Workbook()  # openpyxl never calculates, so no cached values are stored
    ws = wb.active
    ws.append(["Item", "Cost"])
    ws.append(["a", 2])
    ws.append(["Total", "=SUM(B2:B2)"])
    ctx, meta = run_loader(XlsxLoader(), "f.xlsx", _save(wb))
    assert ctx.elements[-1].content == "Item: Total | Cost: =SUM(B2:B2)"
    assert ctx.elements[0].confidence == pytest.approx(0.95 * 0.8)
    assert meta["uncached_formulas"] == 1 and "no cached value" in ctx.warnings[0]


def test_xlsx_dates_from_real_cells(run_loader) -> None:
    wb = openpyxl.Workbook()
    wb.active.append(["Milestone", "Due"])
    wb.active.append(["M2", date(2026, 5, 30)])
    ctx, _ = run_loader(XlsxLoader(), "d.xlsx", _save(wb))
    assert ctx.elements[1].content == "Milestone: M2 | Due: 2026-05-30"


def test_openpyxl_uses_defusedxml() -> None:
    assert openpyxl.DEFUSEDXML is True


# ------------------------------------------------------------------ pptx
def _deck() -> bytes:
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content
    s.shapes.title.text = "Status"
    tf = s.placeholders[1].text_frame
    tf.text = "Top point"
    sub = tf.add_paragraph()
    sub.text, sub.level = "Detail\vcontinued", 1
    s.notes_slide.notes_text_frame.text = "Say the budget\vnumber."  # soft break, not a bullet

    s = prs.slides.add_slide(prs.slide_layouts[6])  # Blank: no title
    low = s.shapes.add_textbox(Inches(1), Inches(4), Inches(4), Inches(1))
    low.text_frame.text = "Lower box"
    high = s.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    high.text_frame.text = "Upper box"  # added second, but read first
    table = s.shapes.add_table(2, 2, Inches(1), Inches(5), Inches(4), Inches(1)).table
    table.first_row = False
    for r, row in enumerate([("k", "v"), ("x", "1")]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    s.shapes.title.text = "Chart"
    data = CategoryChartData()
    data.categories = ["Q1", "Q2"]
    data.add_series("Revenue", (1.5, 2.0))
    s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(2), Inches(6), Inches(4), data
    )
    pic = s.shapes.add_picture(io.BytesIO(_png()), Inches(7), Inches(2))
    pic._element.xpath("./p:nvPicPr/p:cNvPr")[0].set("descr", "Logo")
    s._element.set("show", "0")
    return _save(prs)


def test_pptx_slides_lists_notes_and_reading_order(run_loader) -> None:
    ctx, meta = run_loader(PptxLoader(), "d.pptx", _deck())
    slide1 = [e for e in ctx.elements if e.page_number == 1]
    assert [e.element_type.value for e in slide1] == ["heading", "list", "paragraph"]
    assert slide1[1].content == "- Top point\n  - Detail continued"  # "\v" inside one bullet
    assert slide1[1].section_path == ["Status"]
    assert slide1[2].content == "Say the budget\nnumber."  # relies on normalize_text's "\v" fix
    assert slide1[2].metadata["speaker_notes"] is True
    assert slide1[2].section_path == ["Status", "Speaker notes"]
    slide2 = [e for e in ctx.elements if e.page_number == 2]
    assert [e.content for e in slide2[:2]] == ["Upper box", "Lower box"]
    assert slide2[0].section_path == ["Slide 2"]  # untitled slide
    assert slide2[0].bbox is not None and slide2[0].bbox.y0 == pytest.approx(72.0)  # 1 inch
    assert meta["slide_count"] == 3 and meta["hidden_slides"] == [3]


def test_pptx_table_without_header_flag_has_lower_confidence(run_loader) -> None:
    ctx, _ = run_loader(PptxLoader(), "d.pptx", _deck())
    table = next(e for e in ctx.elements if e.element_type is ElementType.TABLE)
    assert table.confidence == pytest.approx(0.9) and table.metadata["header_marked"] is False


def test_pptx_chart_becomes_figure_with_data_table(run_loader) -> None:
    ctx, _ = run_loader(PptxLoader(), "d.pptx", _deck())
    figure = next(e for e in ctx.elements if e.element_type is ElementType.FIGURE)
    assert figure.metadata["chart_type"] == "COLUMN_CLUSTERED"
    assert figure.metadata["hidden"] is True
    rows = [e for e in ctx.elements if e.page_number == 3 and e.element_type.value == "table_row"]
    assert [r.content for r in rows] == ["Category: Q1 | Revenue: 1.5", "Category: Q2 | Revenue: 2"]
    table = next(e for e in ctx.elements if e.element_id == rows[0].parent_element_id)
    assert table.parent_element_id == figure.element_id


def test_pptx_picture_saved_as_asset(run_loader, tmp_path: Path) -> None:
    ctx, _ = run_loader(PptxLoader(), "d.pptx", _deck())
    image = next(e for e in ctx.elements if e.element_type is ElementType.IMAGE)
    assert image.content == "Logo" and image.page_number == 3
    assert (tmp_path / image.asset_ref).read_bytes() == _png()
