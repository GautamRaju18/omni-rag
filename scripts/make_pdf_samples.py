"""Generate the PDF and image samples in ``data/samples`` (Project Aurora, entirely fictional).

* ``aurora_safety_review.pdf``: born-digital, 3 pages. Headings in larger fonts, a ruled table,
  a bullet list, an embedded figure with caption, and a running header/footer on every page.
* ``aurora_lab_notebook_scan.pdf``: one page that is only a slightly rotated, noisy JPEG with
  no text layer (what a scanner produces).
* ``whiteboard_2026-03-20.jpg``: a phone-style photo stored sideways with EXIF orientation 6,
  so OCR fails unless the loader applies the EXIF rotation.

As with the Office samples, the committed files are the source of truth (regenerating changes
their document IDs).

    uv run python scripts/make_pdf_samples.py
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import pymupdf
from make_office_samples import bar_chart_png  # same figure as in the Word report
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"

CSS = """
body { font-family: sans-serif; font-size: 11px; line-height: 1.4; }
h1 { font-size: 20px; } h2 { font-size: 15px; }
table { border-collapse: collapse; } th, td { border: 1px solid black; padding: 3px; }
"""

PAGES = [
    """
<h1>Aurora E-5 High-Temperature Failure: Safety Review</h1>
<p>Prepared by Lena Brandt (Safety Officer) on 2026-03-24. Reviewed by Dr. Priya Raman.</p>
<h2>1. Incident Summary</h2>
<p>Cell AUR-C103 (electrolyte E-5) failed at cycle 412 during 45 °C cycling. The pouch swelled
by 14% and vented without flame. The cell was moved to the safety cabinet within 10 minutes and
no one was exposed.</p>
<h2>2. Gas Analysis</h2>
<p>GC-MS of the pouch gas found CO2 and ethylene (C2H4), consistent with decomposition of
ethylene carbonate (EC) in the E-5 electrolyte at elevated temperature. No manufacturing defect
was found: the seal integrity tests on the same batch passed.</p>
""",
    """
<h2>3. Swelling Measurements</h2>
<table>
<tr><th>Cell</th><th>Electrolyte</th><th>Temp (°C)</th><th>Cycles</th><th>Swelling (%)</th></tr>
<tr><td>AUR-C103</td><td>E-5</td><td>45</td><td>412</td><td>14.0</td></tr>
<tr><td>AUR-C104</td><td>E-5 + FEC</td><td>45</td><td>800</td><td>3.2</td></tr>
<tr><td>AUR-C105</td><td>E-7</td><td>45</td><td>600</td><td>2.1</td></tr>
<tr><td>AUR-C106</td><td>E-7</td><td>45</td><td>600</td><td>2.4</td></tr>
</table>
<p>All cells except AUR-C103 meet the 5% swelling limit defined in the test thresholds.</p>
""",
    """
<h2>4. Recommendations</h2>
<ul>
<li>Stop 45 °C testing with E-5 (already decided on 2026-03-12).</li>
<li>Continue the E-7 qualification with swelling checks every 100 cycles.</li>
<li>Add a gas-sampling port to pilot-line pouches before milestone M3.</li>
</ul>
<h2>5. Approval</h2>
<p>Continuation of E-7 testing approved by Dr. Priya Raman on 2026-03-25.</p>
""",
]


def make_safety_review(path: Path) -> None:
    doc = pymupdf.open()
    for number, html in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)  # A4 in points
        page.insert_text((56, 30), "Helios Energy Lab - Internal safety review", fontsize=8)
        page.insert_text((56, 820), f"Page {number} of {len(PAGES)}", fontsize=8)
        spare, _ = page.insert_htmlbox(pymupdf.Rect(56, 60, 539, 520), html, css=CSS)
        if number == 2:  # figure + caption below the table
            top = 520 - spare + 16
            page.insert_image(pymupdf.Rect(56, top, 396, top + 170), stream=bar_chart_png())
            page.insert_htmlbox(
                pymupdf.Rect(56, top + 176, 539, top + 210),
                "<p>Figure 1: Capacity retention at the latest checkpoint, AUR-C101 to "
                "AUR-C108.</p>",
                css=CSS,
            )
    doc.set_metadata(
        {"title": "Aurora E-5 Failure Safety Review", "author": "Lena Brandt", "producer": ""}
    )
    doc.save(path, garbage=4, deflate=True)


NOTEBOOK = [
    "Lab Notebook - Project Aurora",
    "Date: 2026-03-14        Operator: Aiko Tanaka",
    "",
    "AUR-C105 (E-7, 45 C): 600 cycles, retention 89.6 %, swelling 2.1 %",
    "AUR-C106 (E-7, 45 C): 600 cycles, retention 90.2 %, swelling 2.4 %",
    "",
    "Observation: no gas pocket visible on either cell.",
    "Next checkpoint: 700 cycles on 2026-03-21.",
    "",
    "Signed: A. Tanaka",
]


def notebook_scan_jpeg() -> bytes:
    rng = random.Random(42)  # noqa: S311 (fixed seed for reproducible noise, not crypto)
    img = Image.new("L", (1700, 2200), 250)  # letter size at 200 dpi
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(NOTEBOOK):
        draw.text((140, 200 + i * 70), line, fill=25, font=font)
    for _ in range(6000):  # scanner dust
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        img.putpixel((x, y), rng.randrange(120, 200))
    img = img.rotate(0.7, fillcolor=250, resample=Image.Resampling.BICUBIC)  # page fed crooked
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return out.getvalue()


def make_notebook_scan(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=notebook_scan_jpeg())
    doc.set_metadata({"title": "", "producer": "Office scanner", "creator": ""})
    doc.save(path, garbage=4, deflate=True)


WHITEBOARD = [
    "E-7 QUALIFICATION PLAN",
    "1. Swelling limit: 5 %",
    "2. Checkpoint every 100 cycles",
    "3. Pilot batch: 100 cells (M3)",
    "Owner: Marco",
]


def make_whiteboard(path: Path) -> None:
    upright = Image.new("RGB", (1400, 900), (232, 234, 228))
    draw = ImageDraw.Draw(upright)
    font = ImageFont.load_default(size=58)
    for i, line in enumerate(WHITEBOARD):
        draw.text((90, 90 + i * 140), line, fill=(20, 40, 120), font=font)
    stored = upright.transpose(Image.Transpose.ROTATE_90)  # pixels sideways, like a phone
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    stored.save(path, format="JPEG", quality=88, exif=exif.tobytes())


LEFT_COLUMN = """
<h1>E-7 Electrolyte: Technical Note</h1>
<p><b>Abstract.</b> E-7 is a carbonate electrolyte with a borate additive developed to replace
the E-5 baseline at elevated temperature. This note summarises its formulation and the first
600 cycles at 45 °C.</p>
<p><b>1. Motivation.</b> The E-5 baseline decomposed at 45 °C and cell AUR-C103 failed at cycle
412. A solvent blend with a higher boiling point and a film-forming additive was needed.</p>
<p><b>2. Formulation.</b> The solvent blend replaces DEC with PC and DMC. The additive
NaDFOB forms a stable interphase on hard carbon.</p>
"""
RIGHT_COLUMN = """
<p><b>3. Results.</b> Cells AUR-C105 and AUR-C106 retained 89.6% and 90.2% of their capacity
after 600 cycles at 45 °C, with 2.1% and 2.4% swelling.</p>
<p><b>4. Conclusion.</b> E-7 meets the 5% swelling limit and is the candidate for milestone M2.
Qualification continues to 1500 cycles under Marco Lindqvist.</p>
"""
# A borderless table: aligned columns, no ruling lines (common in papers and reports).
BORDERLESS_TABLE = """
<table>
<tr><td><b>Component</b></td><td><b>Role</b></td><td><b>Amount</b></td></tr>
<tr><td>NaPF6</td><td>Salt</td><td>1.0 M</td></tr>
<tr><td>EC</td><td>Solvent</td><td>30 vol%</td></tr>
<tr><td>PC</td><td>Solvent</td><td>30 vol%</td></tr>
<tr><td>DMC</td><td>Co-solvent</td><td>40 vol%</td></tr>
<tr><td>NaDFOB</td><td>SEI additive</td><td>1 wt%</td></tr>
</table>
"""
LAYOUT_CSS = CSS.replace("border: 1px solid black;", "") + " td { padding: 2px 18px 2px 0; }"


def make_layout_paper(path: Path) -> None:
    """Two text columns, then a full-width borderless table: hard cases for PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_htmlbox(pymupdf.Rect(50, 60, 287, 520), LEFT_COLUMN, css=LAYOUT_CSS)
    page.insert_htmlbox(pymupdf.Rect(308, 60, 545, 520), RIGHT_COLUMN, css=LAYOUT_CSS)
    page.insert_htmlbox(pymupdf.Rect(50, 540, 545, 760), BORDERLESS_TABLE, css=LAYOUT_CSS)
    doc.set_metadata({"title": "E-7 Electrolyte: Technical Note", "author": "Marco Lindqvist"})
    doc.save(path, garbage=4, deflate=True)


BUILDERS = {
    "aurora_safety_review.pdf": make_safety_review,
    "aurora_lab_notebook_scan.pdf": make_notebook_scan,
    "whiteboard_2026-03-20.jpg": make_whiteboard,
    "e7_technical_note.pdf": make_layout_paper,
}

if __name__ == "__main__":
    import sys

    # Regenerating a sample changes its document ID, so name the ones to build (default: all).
    for name in sys.argv[1:] or BUILDERS:
        BUILDERS[name](OUT / name)
        print(f"{name}: {(OUT / name).stat().st_size} bytes")
