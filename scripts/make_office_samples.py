"""Generate the Office samples in ``data/samples`` (Project Aurora, entirely fictional).

The committed .docx/.xlsx/.pptx files are the source of truth. Re-running this script produces
new bytes (ZIP entries carry timestamps), so the files get new document IDs. Only regenerate
deliberately, then update the golden tests.

Each file deliberately contains the structures the loaders must handle: headings, nested and
numbered lists, a table with a declared header row, an image with alt text and a caption,
tracked changes and a page header (DOCX); percentages, dates, an offset table, several blocks on
one sheet and an empty sheet (XLSX); bullets with levels, a chart, a table, a picture, a group,
speaker notes and a hidden slide (PPTX).

    uv run python scripts/make_office_samples.py
"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import docx
import openpyxl
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches as PptxInches
from pptx.util import Pt

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
RETENTION = {  # cell -> latest retention %, consistent with cell_test_results.csv
    "C101": 91.4, "C102": 88.0, "C103": 72.5, "C104": 83.1,
    "C105": 89.6, "C106": 90.2, "C107": 94.8, "C108": 95.1,
}  # fmt: skip


def bar_chart_png() -> bytes:
    img = Image.new("RGB", (640, 320), "white")
    draw = ImageDraw.Draw(img)
    draw.line([(40, 280), (620, 280)], fill="black")
    for i, (cell, value) in enumerate(RETENTION.items()):
        x = 60 + i * 70
        top = 280 - int((value - 60) * 6)
        draw.rectangle([x, top, x + 40, 280], fill="#3b6ea5" if value >= 80 else "#c0392b")
        draw.text((x, 288), cell, fill="black")
        draw.text((x + 4, top - 14), f"{value}", fill="black")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def pouch_png() -> bytes:
    img = Image.new("RGB", (400, 260), "#f4f4f4")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([60, 40, 340, 220], radius=12, fill="#b8b8b8", outline="black")
    draw.rectangle([150, 10, 180, 40], fill="#d4a017")
    draw.rectangle([220, 10, 250, 40], fill="#8c8c8c")
    draw.text((140, 120), "Aurora S1-P3", fill="black")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def make_docx(path: Path) -> None:
    d = docx.Document()
    d.core_properties.title = "Project Aurora Monthly Status Report, March 2026"
    d.core_properties.author = "Dr. Priya Raman"
    d.core_properties.created = datetime(2026, 3, 31, 9, 0)
    d.core_properties.modified = datetime(2026, 3, 31, 9, 0)  # else the template's 2013 date
    d.sections[0].header.paragraphs[0].text = "CONFIDENTIAL - Helios Energy Lab internal"

    d.add_heading("Project Aurora: Monthly Status Report", level=0)  # level 0 = Title style
    d.add_paragraph(
        "Reporting period: March 2026. Prepared by Dr. Priya Raman for the Helios Energy Lab "
        "steering committee."
    )

    d.add_heading("Summary", level=1)
    p = d.add_paragraph("Room-temperature cycling remains on track.")
    # A reviewer's tracked changes: one deletion, one insertion.
    p._p.append(
        parse_xml(
            f'<w:del {nsdecls("w")} w:id="1" w:author="Marco Lindqvist" '
            'w:date="2026-03-20T10:00:00Z"><w:r><w:delText xml:space="preserve"> E-5 remains '
            "the preferred electrolyte for all temperatures.</w:delText></w:r></w:del>"
        )
    )
    p._p.append(
        parse_xml(
            f'<w:ins {nsdecls("w")} w:id="2" w:author="Marco Lindqvist" '
            'w:date="2026-03-20T10:00:00Z"><w:r><w:t xml:space="preserve"> The E-5 baseline '
            "failed at 45 °C (AUR-C103), so high-temperature testing moved to electrolyte E-7."
            "</w:t></w:r></w:ins>"
        )
    )
    d.add_paragraph("Milestone M2 (45 °C stability demonstrated, due 2026-05-30) is at risk.")

    d.add_heading("Cycling Progress", level=2)
    d.add_paragraph(
        "All eight cells completed their latest checkpoint. Cells on E-7 at 45 °C are above "
        "89% retention after 600 cycles."
    )
    shape = d.add_picture(io.BytesIO(bar_chart_png()), width=Inches(5))
    shape._inline.docPr.set(
        "descr", "Bar chart of capacity retention per cell, AUR-C101 to AUR-C108"
    )
    d.add_paragraph(
        "Figure 1: Capacity retention at the latest checkpoint, AUR-C101 to AUR-C108.",
        style="Caption",
    )

    d.add_heading("Budget", level=1)
    d.add_paragraph(
        "On 2026-03-19 the steering committee approved an additional EUR 85,000 for the E-7 "
        "qualification, raising the electrolyte workstream budget to EUR 240,000."
    )

    d.add_heading("Risk Register", level=1)
    risks = [
        ("ID", "Risk", "Owner", "Likelihood", "Mitigation"),
        ("R1", "E-7 swelling exceeds the 5% limit", "Marco Lindqvist", "Medium",
         "Fall back to E-5 + FEC"),
        ("R2", "Pilot-line pouch tooling arrives late", "Samuel Okafor", "Low",
         "Order tooling by 2026-04-15"),
        ("R3", "Single hard-carbon supplier", "Dr. Priya Raman", "Medium",
         "Qualify Carbonix Materials as second supplier"),
    ]  # fmt: skip
    table = d.add_table(rows=len(risks), cols=len(risks[0]))
    table.style = "Table Grid"
    for r, row in enumerate(risks):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))

    d.add_heading("Next Steps", level=1)
    d.add_paragraph("Run AUR-C105 to AUR-C108 to 1500 cycles", style="List Bullet")
    d.add_paragraph("Report swelling at every 100-cycle checkpoint", style="List Bullet 2")
    d.add_paragraph("Prepare pilot-line tooling for the E-7 fill volume", style="List Bullet")
    d.add_paragraph("Submit the E-7 qualification draft (2026-03-26)", style="List Number")
    d.add_paragraph("Freeze the pilot-line bill of materials (2026-04-30)", style="List Number")
    d.save(path)


def make_xlsx(path: Path) -> None:
    wb = openpyxl.Workbook()
    wb.properties.title = "Project Aurora Budget 2026"
    wb.properties.creator = "Dr. Priya Raman"

    ws = wb.active
    ws.title = "Budget"
    ws.append(["Workstream", "Owner", "Planned (EUR)", "Spent to date (EUR)", "Spent (%)"])
    budget = [
        ("Cathode", "Dr. Priya Raman", 180000, 121500),
        ("Electrolyte", "Marco Lindqvist", 240000, 97200),
        ("Cell testing", "Aiko Tanaka", 150000, 88500),
        ("Pilot line", "Samuel Okafor", 320000, 64000),
    ]
    budget.append(("Total", None, sum(b[2] for b in budget), sum(b[3] for b in budget)))
    for name, owner, planned, spent in budget:
        ws.append([name, owner, planned, spent, spent / planned])
        ws.cell(ws.max_row, 5).number_format = "0.0%"

    ms = wb.create_sheet("Milestones")
    ms.append(["ID", "Milestone", "Due", "Done"])
    for row in [
        ("M1", "Baseline chemistry frozen", date(2026, 2, 15), True),
        ("M2", "45 °C stability demonstrated", date(2026, 5, 30), False),
        ("M3", "Pilot line 100-cell batch", date(2026, 9, 30), False),
    ]:
        ms.append(list(row))
    ms.append([])
    ms.append(["Note: M2 depends on E-7 swelling staying below the 5% limit."])

    sup = wb.create_sheet("Suppliers")
    sup["B2"] = "Qualified suppliers (updated 2026-03-15)"
    rows = [
        ("Material", "Supplier", "Lead time (weeks)", "Qualified"),
        ("Hard carbon", "Kokos Carbon Ltd", 6, True),
        ("Hard carbon", "Carbonix Materials", 10, False),
        ("NaPF6 salt", "Nordic Salts AB", 4, True),
        ("Pouch film", "Laminex", 8, True),
    ]
    for r, row in enumerate(rows, start=4):  # offset table: starts at B4, not A1
        for c, value in enumerate(row, start=2):
            sup.cell(row=r, column=c, value=value)

    wb.create_sheet("Scratch")  # empty sheet: must produce nothing
    wb.save(path)


def make_pptx(path: Path) -> None:
    prs = Presentation()
    prs.core_properties.title = "Project Aurora Q1 2026 Review"
    prs.core_properties.author = "Dr. Priya Raman"

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Project Aurora: Q1 2026 Review"
    s.placeholders[1].text = "Helios Energy Lab steering committee, 2026-04-02"

    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Highlights"
    tf = s.placeholders[1].text_frame
    tf.text = "Room-temperature retention above 88% after 1000 cycles"
    for text, level in [
        ("Best cell: AUR-C101 (E-5 + FEC) at 91.4%", 1),
        ("E-5 baseline failed at 45 °C;\vhigh-temperature testing moved to E-7", 0),
        ("Pilot-line pouch tooling ordered", 0),
    ]:
        para = tf.add_paragraph()
        para.text, para.level = text, level
    s.notes_slide.notes_text_frame.text = (
        "Mention that the steering committee approved an extra EUR 85,000 for the E-7 "
        "qualification on 2026-03-19."
    )

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "E-7 Retention at 45 °C"
    data = CategoryChartData()
    data.categories = ["100", "200", "300", "400", "500", "600"]
    data.add_series("AUR-C105", (99.1, 97.8, 95.9, 93.6, 91.5, 89.6))
    data.add_series("AUR-C106", (99.3, 98.0, 96.2, 94.1, 92.0, 90.2))
    chart = s.shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS,
        PptxInches(0.7), PptxInches(1.6), PptxInches(8.6), PptxInches(5),
        data,
    ).chart  # fmt: skip
    chart.has_title = True
    chart.chart_title.text_frame.text = "Capacity retention (%) vs. cycles, E-7 at 45 °C"

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Milestones"
    rows = [
        ("ID", "Milestone", "Due", "Status"),
        ("M1", "Baseline chemistry frozen", "2026-02-15", "Done"),
        ("M2", "45 °C stability demonstrated", "2026-05-30", "At risk"),
        ("M3", "Pilot line 100-cell batch", "2026-09-30", "On track"),
    ]
    table = s.shapes.add_table(
        len(rows), 4, PptxInches(0.7), PptxInches(1.8), PptxInches(8.6), PptxInches(2.5)
    ).table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Pouch Cell Prototype"
    pic = s.shapes.add_picture(
        io.BytesIO(pouch_png()), PptxInches(0.7), PptxInches(1.8), width=PptxInches(5)
    )
    pic._element.xpath("./p:nvPicPr/p:cNvPr")[0].set(
        "descr", "Aurora S1-P3 pouch cell prototype on the test bench"
    )
    group = s.shapes.add_group_shape()
    for i, text in enumerate(["Capacity: 50 Ah", "Assembled on the pilot line, 2026-03-28"]):
        box = group.shapes.add_textbox(
            PptxInches(6), PptxInches(2 + i * 0.6), PptxInches(3.5), PptxInches(0.5)
        )
        box.text_frame.text = text
        box.text_frame.paragraphs[0].runs[0].font.size = Pt(14)

    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Backup: E-5 Gas Analysis"
    s.placeholders[
        1
    ].text_frame.text = "CO₂ and C₂H₄ detected after failure, consistent with EC decomposition"
    s._element.set("show", "0")  # hidden slide
    prs.save(path)


if __name__ == "__main__":
    make_docx(OUT / "aurora_status_report_2026-03.docx")
    make_xlsx(OUT / "aurora_budget_2026.xlsx")
    make_pptx(OUT / "aurora_q1_review.pptx")
    for f in sorted(OUT.glob("aurora_*.[dpx]*x")):
        print(f"{f.name}: {f.stat().st_size} bytes")
