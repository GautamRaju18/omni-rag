"""PDF loader, image loader and the Tesseract wrapper.

Tests that run the real ``tesseract`` binary are skipped when it isn't installed. Everything
else (TSV parsing, scan detection without OCR, PDF structure) runs anywhere.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from omnirag.ingestion.loaders.image_loader import ImageLoader
from omnirag.ingestion.loaders.pdf_loader import PdfLoader
from omnirag.ingestion.models import ContentOrigin, ElementType
from omnirag.ingestion.ocr import OcrError, TesseractOcr, parse_tsv

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed"
)


def _text_image(lines: list[str], size: tuple[int, int] = (1200, 400)) -> Image.Image:
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 80), line, fill=0, font=font)
    return img


def _encode(img: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    out = io.BytesIO()
    img.save(out, format=fmt, **kw)
    return out.getvalue()


def _pdf(doc: pymupdf.Document) -> bytes:
    return doc.tobytes()


# ------------------------------------------------------------------- ocr
TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t800\t600\t-1\t\n"
    "4\t1\t1\t1\t1\t0\t10\t10\t200\t20\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t10\t10\t80\t20\t96.5\tHello\n"
    "5\t1\t1\t1\t1\t2\t100\t10\t110\t20\t90.0\tworld\n"
    "5\t1\t1\t1\t2\t1\t10\t40\t60\t20\t40.0\tagain\n"
    "5\t1\t2\t1\t1\t1\t10\t300\t50\t20\t88.0\tSecond\n"
)


def test_parse_tsv_groups_words_into_paragraphs() -> None:
    first, second = parse_tsv(TSV)
    assert first.text == "Hello world\nagain"  # OCR line breaks are kept
    assert first.confidence == pytest.approx((96.5 + 90 + 40) / 300)
    assert first.bbox == (10, 10, 210, 60)
    assert (first.words, first.low_confidence_words) == (3, 1)
    assert second.text == "Second"


def test_missing_tesseract_is_reported_not_crashing() -> None:
    ocr = TesseractOcr(cmd="definitely-not-tesseract")
    assert ocr.available() is False
    with pytest.raises(OcrError, match="not found"):
        ocr.recognize(b"")


@requires_tesseract
def test_real_ocr_reads_rendered_text() -> None:
    result = TesseractOcr().recognize(_encode(_text_image(["Hello OmniRAG 2026"])))
    assert "Hello OmniRAG 2026" in result.paragraphs[0].text
    assert result.mean_confidence > 0.8


# ------------------------------------------------------------------- pdf
def _structured_pdf() -> bytes:
    doc = pymupdf.open()
    for n in (1, 2, 3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 30), "ACME internal", fontsize=8)  # running header
        page.insert_text((50, 825), f"Page {n}", fontsize=8)  # running footer
        if n == 1:
            page.insert_text((50, 100), "Main Title", fontsize=22)
        page.insert_text((50, 160), f"Section {n}", fontsize=15)
        page.insert_text((50, 200), f"Body text of section {n} with enough words.", fontsize=11)
        page.insert_text((50, 230), "- first point", fontsize=11)
        page.insert_text((50, 245), "- second point", fontsize=11)
    return _pdf(doc)


def test_pdf_headings_sections_lists_and_boilerplate(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(), "s.pdf", _structured_pdf())
    text = "\n".join(e.content for e in ctx.elements)
    assert "ACME internal" not in text and "Page 2" not in text  # running header/footer
    assert sorted(meta["boilerplate_removed"]) == ["acme internal", "page #"]
    assert meta["body_font_size"] == 11.0
    title, sec1, body1 = ctx.elements[:3]
    assert (title.element_type, title.metadata["level"]) == (ElementType.HEADING, 1)
    assert (sec1.content, sec1.metadata["level"]) == ("Section 1", 2)
    assert body1.section_path == ["Main Title", "Section 1"] and body1.page_number == 1
    assert body1.bbox is not None and body1.bbox.unit == "pt"
    lists = [e for e in ctx.elements if e.element_type is ElementType.LIST]
    assert lists[0].content == "- first point\n- second point"
    # Section 2 replaces Section 1 under the same title (stack carried across pages).
    body2 = next(e for e in ctx.elements if "section 2" in e.content)
    assert body2.section_path == ["Main Title", "Section 2"] and body2.page_number == 2


def test_pdf_table_rows_not_duplicated_as_text(run_loader) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    html = (
        "<table style='border-collapse: collapse'>"
        "<tr><th style='border: 1px solid black'>Cell</th>"
        "<th style='border: 1px solid black'>Swelling (%)</th></tr>"
        "<tr><td style='border: 1px solid black'>AUR-C105</td>"
        "<td style='border: 1px solid black'>2.1</td></tr></table>"
    )
    page.insert_htmlbox(pymupdf.Rect(50, 50, 400, 300), html)
    ctx, _ = run_loader(PdfLoader(), "t.pdf", _pdf(doc))
    assert [e.element_type.value for e in ctx.elements] == ["table", "table_row"]
    assert ctx.elements[1].content == "Cell: AUR-C105 | Swelling (%): 2.1"
    assert ctx.elements[0].bbox is not None


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_pdf_image_and_caption(run_loader, tmp_path: Path, fmt: str) -> None:
    source = Image.new("RGB", (200, 100), "blue")
    data = _encode(source, fmt)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Some introduction text for the page body.", fontsize=11)
    page.insert_image(pymupdf.Rect(50, 100, 250, 200), stream=data)
    page.insert_text((50, 220), "Figure 3: A blue box.", fontsize=11)
    ctx, _ = run_loader(PdfLoader(), "i.pdf", _pdf(doc))
    image = next(e for e in ctx.elements if e.element_type is ElementType.IMAGE)
    caption = next(e for e in ctx.elements if e.element_type is ElementType.CAPTION)
    assert caption.parent_element_id == image.element_id
    stored = (tmp_path / image.asset_ref).read_bytes()
    if fmt == "JPEG":
        # PDF stores JPEG natively (DCTDecode): we get the exact original bytes back.
        assert stored == data
    else:
        # PDF has no PNG format: pixels are stored Flate-compressed and re-wrapped as a new PNG
        # on extraction. Lossless, so the pixels match even though the file bytes differ.
        assert Image.open(io.BytesIO(stored)).convert("RGB").tobytes() == source.tobytes()


def test_pdf_password_protected_is_refused(run_loader) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(ValueError, match="password-protected"):
        run_loader(PdfLoader(), "p.pdf", data)


def _scanned_pdf() -> bytes:
    scan = _encode(_text_image(["Retention 89.6 percent", "Swelling 2.1 percent"], (1700, 600)))
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=216)  # the image fills the whole page
    page.insert_image(page.rect, stream=scan)
    return _pdf(doc)


def test_pdf_scanned_page_without_ocr_warns(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(ocr=None), "scan.pdf", _scanned_pdf())
    assert [e.element_type for e in ctx.elements] == [ElementType.IMAGE]  # scan kept as asset
    assert meta["scanned_pages"] == [1] and meta["ocr_pages"] == []
    assert "looks scanned" in ctx.warnings[0]


@requires_tesseract
def test_pdf_scanned_page_is_ocred(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(TesseractOcr(), ocr_dpi=200), "scan.pdf", _scanned_pdf())
    image, *paragraphs = ctx.elements
    assert meta["ocr_pages"] == [1]
    ocr = paragraphs[0]
    assert ocr.origin is ContentOrigin.OCR and ocr.parent_element_id == image.element_id
    assert "89.6" in ocr.content and "2.1" in ocr.content
    assert 0.8 < ocr.confidence <= 1.0
    assert ocr.bbox is not None and ocr.bbox.x1 <= 612  # converted from pixels to points


@requires_tesseract
def test_pdf_ocr_page_limit(run_loader) -> None:
    ctx, meta = run_loader(PdfLoader(TesseractOcr(), ocr_max_pages=0), "s.pdf", _scanned_pdf())
    assert meta["ocr_pages"] == [] and "not OCR'd" in ctx.warnings[0]


# ----------------------------------------------------------------- image
def test_image_without_ocr_keeps_original_reference(run_loader) -> None:
    ctx, meta = run_loader(ImageLoader(ocr=None), "photo.png", _encode(_text_image(["x"])))
    (image,) = ctx.elements
    assert image.asset_ref == "documents/doc_test/photo.png"  # the original, not a copy
    assert meta["width_px"] == 1200 and meta["ocr"] == {"performed": False}
    assert "OCR unavailable" in ctx.warnings[0]


@requires_tesseract
def test_image_exif_rotation_applied_before_ocr(run_loader) -> None:
    upright = _text_image(["QUALIFICATION PLAN", "Swelling limit 5 percent"])
    sideways = upright.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    data = _encode(sideways, "JPEG", quality=92, exif=exif.tobytes())
    ctx, meta = run_loader(ImageLoader(TesseractOcr()), "board.jpg", data)
    image, *paragraphs = ctx.elements
    text = "\n".join(p.content for p in paragraphs)
    assert "QUALIFICATION PLAN" in text and "Swelling limit 5 percent" in text
    assert all(p.origin is ContentOrigin.OCR for p in paragraphs)
    assert all(p.parent_element_id == image.element_id for p in paragraphs)
    assert paragraphs[0].bbox is not None and paragraphs[0].bbox.unit == "px"
    assert meta["exif_orientation"] == 6 and meta["ocr"]["mean_confidence"] > 0.8
