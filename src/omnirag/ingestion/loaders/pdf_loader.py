"""PDF via PyMuPDF, with Tesseract OCR for pages that have no usable text layer.

A PDF is a set of drawing instructions ("put glyph 'A' at x,y in font F, size 11"), not a
document with paragraphs. Structure has to be *inferred*. Two passes:

**Pass 1 (whole document)** learns what's normal for this file:
* the body font size (the size covering the most characters) and the larger sizes used by
  short blocks, which become heading levels 1-3;
* text repeated in the top/bottom 8% of at least half the pages ("Page 3 of 12",
  "Internal"). That's running headers and footers, removed as boilerplate. Digits are masked so
  page numbers match.

**Pass 2 (per page)** emits elements in reading order (top-to-bottom, then left-to-right):
* tables from ``page.find_tables()`` (ruling lines), with the text inside them not repeated;
* headings, list blocks (bullet or "1." markers) and paragraphs, with ``bbox`` in points;
* embedded images, saved once per document as assets. A following "Figure N" line becomes their
  CAPTION;
* **scanned pages**: almost no text plus an image covering most of the page. The page is
  rendered at ``ocr_dpi`` and OCR'd, with ``origin=OCR`` and Tesseract's confidence, linked to
  the scan image;
* **garbled text layers** (many U+FFFD characters from fonts without a Unicode map) are OCR'd
  too.

Known limits of these heuristics: multi-column reading order and tables without ruling lines.
Pass 1 therefore also detects multi-column pages; with ``pdf_engine=auto`` such documents go to
the Docling layout model instead (``docling_engine.py``), and scanned ones stay here for OCR.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pymupdf

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.docling_engine import DoclingEngine
from omnirag.ingestion.models import (
    BoundingBox,
    ContentOrigin,
    ElementType,
    ExtractionMethod,
    FileKind,
)
from omnirag.ingestion.ocr import OcrError, TesseractOcr
from omnirag.ingestion.tables import add_table

# PyMuPDF prints an advert for its optional layout package to stdout on first table/layout
# use; it would corrupt CLI output and logs. Layout models are evaluated with Docling in 2d.
pymupdf.no_recommend_layout()

_METHOD = ExtractionMethod.PYMUPDF
# Not TEXTFLAGS_DICT: that embeds image bytes in the result and replaces unknown glyphs with
# raw codes, which would hide a garbled text layer. We want U+FFFD for unknown glyphs.
_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_DEHYPHENATE
_MARGIN = 0.08  # top/bottom fraction of the page where running headers/footers live
_HEADING_RATIO = 1.15  # a heading is at least 15% larger than body text
_MAX_HEADING_CHARS = 200
_MIN_TEXT_CHARS = 25  # less text than this + a page-sized image = scanned page
_MIN_IMAGE_COVERAGE = 0.5
_MAX_GARBLED_RATIO = 0.1
_MIN_IMAGE_SIDE = 32  # px; smaller images are rules, bullets and spacers
_MIN_OCR_CONFIDENCE = 0.3  # below this, Tesseract is reading noise, not text
_MAX_PAGES = 2000
_BULLET = re.compile(r"^(?:[•▪◦‣●○■\-–*]|(\d{1,3}[.)]))\s+")
_FIGURE_CAPTION = re.compile(r"^(?:Figure|Fig\.)\s*\d+", re.IGNORECASE)
_BOLD = 16  # span flag bit 4


def _block_info(block: dict[str, Any]) -> tuple[list[str], float]:
    """Non-empty line texts and the dominant font size of a text block."""
    lines, sizes = [], Counter()
    for line in block["lines"]:
        text = "".join(span["text"] for span in line["spans"]).strip()
        if text:
            lines.append(text)
        for span in line["spans"]:
            sizes[round(span["size"], 1)] += len(span["text"].strip())
    return lines, (sizes.most_common(1)[0][0] if sizes else 0.0)


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def _in_margin(bbox: tuple[float, ...], height: float) -> bool:
    return bbox[3] < height * _MARGIN or bbox[1] > height * (1 - _MARGIN)


def _overlap(inner: pymupdf.Rect, outer: pymupdf.Rect) -> float:
    area = inner.get_area()
    return (inner & outer).get_area() / area if area else 0.0


def _bbox(rect: Any) -> BoundingBox:
    x0, y0, x1, y1 = tuple(rect)
    return BoundingBox(x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1))


def _is_multi_column(prose: list[pymupdf.Rect], width: float) -> bool:
    """Two prose blocks side by side: one in the left part, one in the right, overlapping
    vertically. Prose only (>= 40 chars), so table cells next to each other don't count."""
    left = [r for r in prose if r.x1 <= width * 0.55]
    right = [r for r in prose if r.x0 >= width * 0.45]
    return any(a.y0 < b.y1 and b.y0 < a.y1 for a in left for b in right)


class _Plan:
    """Pass-1 results: boilerplate keys, font size -> heading level, and layout signals."""

    def __init__(self, doc: pymupdf.Document, pages: int) -> None:
        margin_pages: Counter[str] = Counter()
        body_sizes: Counter[float] = Counter()
        candidates: set[float] = set()
        self.multi_column_pages: list[int] = []
        self.scan_like_pages: list[int] = []
        for number, page in enumerate(doc.pages(0, pages), start=1):
            keys = set()
            prose: list[pymupdf.Rect] = []
            chars = 0
            for block in page.get_text("dict", flags=_FLAGS)["blocks"]:
                if block["type"] != 0:
                    continue
                lines, size = _block_info(block)
                text = " ".join(lines)
                if not text:
                    continue
                chars += len(text)
                if _in_margin(block["bbox"], page.rect.height):
                    keys.add(_margin_key(text))
                    continue
                body_sizes[size] += len(text)
                if len(text) <= _MAX_HEADING_CHARS:
                    candidates.add(size)
                if len(text) >= 40:
                    prose.append(pymupdf.Rect(block["bbox"]))
            margin_pages.update(keys)
            if _is_multi_column(prose, page.rect.width):
                self.multi_column_pages.append(number)
            if chars < _MIN_TEXT_CHARS and page.get_images():
                self.scan_like_pages.append(number)
        self.boilerplate = {
            key for key, n in margin_pages.items() if pages >= 2 and n >= max(2, pages / 2)
        }
        self.body_size = body_sizes.most_common(1)[0][0] if body_sizes else 0.0
        larger = sorted(
            (s for s in candidates if s >= self.body_size * _HEADING_RATIO), reverse=True
        )
        self.levels = {size: min(i + 1, 3) for i, size in enumerate(larger)}


class PdfLoader:
    name = "pdf"
    kinds = (FileKind.PDF,)

    def __init__(
        self,
        ocr: TesseractOcr | None = None,
        *,
        ocr_dpi: int = 300,
        ocr_max_pages: int = 50,
        layout_engine: DoclingEngine | None = None,
        engine_mode: Literal["pymupdf", "docling", "auto"] = "pymupdf",
    ) -> None:
        self.ocr = ocr
        self.ocr_dpi = ocr_dpi
        self.ocr_max_pages = ocr_max_pages
        self.layout_engine = layout_engine
        self.engine_mode = engine_mode

    def _choose_engine(self, plan: _Plan, ctx: ExtractionContext) -> tuple[str, str]:
        """``(engine, reason)``. Scanned pages always stay on PyMuPDF + Tesseract."""
        if self.engine_mode == "pymupdf":
            return "pymupdf", "configured"
        if self.layout_engine is None:
            if self.engine_mode == "docling":
                ctx.warn("pdf_engine=docling but Docling is not installed; used PyMuPDF")
            return "pymupdf", "docling unavailable"
        if plan.scan_like_pages:
            return "pymupdf", f"scanned pages {plan.scan_like_pages} need OCR"
        if self.engine_mode == "docling":
            return "docling", "configured"
        if plan.multi_column_pages:
            return "docling", f"multi-column pages {plan.multi_column_pages}"
        return "pymupdf", "single-column layout"

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("PDF is password-protected; decrypt it before ingesting")
            pages = min(doc.page_count, _MAX_PAGES)
            if doc.page_count > _MAX_PAGES:
                ctx.warn(f"Only the first {_MAX_PAGES} of {doc.page_count} pages were processed")
            plan = _Plan(doc, pages)
            meta = doc.metadata or {}
            base = {
                "page_count": doc.page_count,
                "author": meta.get("author") or None,
                "producer": meta.get("producer") or None,
                "creation_date": meta.get("creationDate") or None,
                "toc": [{"level": lvl, "title": t, "page": p} for lvl, t, p in doc.get_toc()][:200],
                "multi_column_pages": plan.multi_column_pages,
            }
            engine, reason = self._choose_engine(plan, ctx)
            if engine == "docling" and self.layout_engine is not None:
                emitted = len(ctx.elements)
                try:
                    counts = self.layout_engine.convert(path, ctx)
                except Exception as exc:  # model/runtime failure: fall back, don't lose the doc
                    del ctx.elements[emitted:]  # drop partial output; IDs restart from here
                    ctx.warn(f"Docling failed ({type(exc).__name__}: {exc}); used PyMuPDF")
                    engine, reason = "pymupdf", "docling failed"
                else:
                    return {**base, "title": meta.get("title") or None, "engine": "docling",
                            "engine_reason": reason, **counts}  # fmt: skip
            state: dict[str, Any] = {
                "headings": [],  # (level, text) stack, carried across pages
                "seen_xrefs": set(),  # an image reused on many pages (logo) is stored once
                "ocr_pages": [],
                "scanned_pages": [],
                "small_images": 0,
                "first_heading": None,
            }
            for number, page in enumerate(doc.pages(0, pages), start=1):
                self._page(doc, page, number, plan, ctx, state)
            return {
                **base,
                "title": meta.get("title") or state["first_heading"],
                "engine": "pymupdf",
                "engine_reason": reason,
                "body_font_size": plan.body_size,
                "boilerplate_removed": sorted(plan.boilerplate),
                "scanned_pages": state["scanned_pages"],
                "ocr_pages": state["ocr_pages"],
                "skipped_small_images": state["small_images"],
            }

    # ------------------------------------------------------------------ page
    def _page(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        number: int,
        plan: _Plan,
        ctx: ExtractionContext,
        state: dict[str, Any],
    ) -> None:
        blocks = [
            b for b in page.get_text("dict", flags=_FLAGS, sort=True)["blocks"] if b["type"] == 0
        ]
        text = "".join(" ".join(_block_info(b)[0]) for b in blocks)
        images = self._images_on(doc, page, state)
        page_area = page.rect.get_area() or 1.0
        coverage = min(1.0, sum((r & page.rect).get_area() for _, r in images) / page_area)
        scanned = len(text.strip()) < _MIN_TEXT_CHARS and coverage >= _MIN_IMAGE_COVERAGE
        garbled = len(text) >= _MIN_TEXT_CHARS and text.count("�") / len(text) > (
            _MAX_GARBLED_RATIO
        )
        page_meta = {"page_width_pt": round(page.rect.width, 1)}

        if scanned or garbled:
            if scanned:
                state["scanned_pages"].append(number)
            image_ids = [
                self._emit_image(doc, xref, rect, number, ctx, state) for xref, rect in images
            ]
            if self._ocr_page(page, number, ctx, state, parent_id=next(iter(image_ids), None)):
                return
            reason = "looks scanned (no text layer)" if scanned else "has a garbled text layer"
            ctx.warn(f"Page {number} {reason} and was not OCR'd (OCR unavailable or page limit)")
            if scanned:
                return

        items: list[tuple[float, float, str, Any]] = []
        table_rects = []
        for table in page.find_tables().tables:
            rect = pymupdf.Rect(table.bbox)
            table_rects.append(rect)
            items.append((rect.y0, rect.x0, "table", table))
        for block in blocks:
            rect = pymupdf.Rect(block["bbox"])
            lines, _ = _block_info(block)
            if not lines or any(_overlap(rect, t) > 0.5 for t in table_rects):
                continue
            if _in_margin(block["bbox"], page.rect.height) and (
                _margin_key(" ".join(lines)) in plan.boilerplate
            ):
                continue
            items.append((rect.y0, rect.x0, "text", block))
        for xref, rect in images:
            items.append((rect.y0, rect.x0, "image", (xref, rect)))
        items.sort(key=lambda item: (round(item[0]), item[1]))

        confidence = 0.5 if garbled else 1.0
        pending_list: list[tuple[list[str], pymupdf.Rect]] = []
        last_image: str | None = None

        def section() -> list[str]:
            return [t for _, t in state["headings"]]

        def flush_list() -> None:
            if not pending_list:
                return
            rendered, ordered = [], False
            for lines, _ in pending_list:
                for line in lines:
                    match = _BULLET.match(line)
                    if match:
                        ordered = ordered or bool(match.group(1))
                        marker = match.group(1) or "-"
                        rendered.append(f"{marker} {line[match.end() :]}")
                    elif rendered:  # wrapped continuation of the previous item
                        rendered[-1] += f" {line}"
                    else:
                        rendered.append(line)
            union = pymupdf.Rect(pending_list[0][1])
            for _, rect in pending_list[1:]:
                union |= rect
            ctx.add(
                ElementType.LIST,
                "\n".join(rendered),
                extraction_method=_METHOD,
                page_number=number,
                section_path=section(),
                confidence=confidence,
                bbox=_bbox(union),
                metadata={**page_meta, "ordered": ordered},
            )
            pending_list.clear()

        for _, _, kind, payload in items:
            if kind == "text":
                lines, size = _block_info(payload)
                rect = pymupdf.Rect(payload["bbox"])
                joined = " ".join(lines)
                # Font size first: "1. Incident Summary" in a heading font is a numbered
                # heading, not a list item.
                level = plan.levels.get(size) if len(joined) <= _MAX_HEADING_CHARS else None
                if not level and _BULLET.match(lines[0]):
                    pending_list.append((lines, rect))
                    continue
                flush_list()
                common = {
                    "extraction_method": _METHOD,
                    "page_number": number,
                    "confidence": confidence,
                    "bbox": _bbox(rect),
                }
                if level:
                    headings = [h for h in state["headings"] if h[0] < level]
                    state["headings"] = headings
                    ctx.add(
                        ElementType.HEADING,
                        joined,
                        section_path=section(),
                        metadata={**page_meta, "level": level, "font_size": size},
                        **common,
                    )
                    state["headings"].append((level, joined))
                    state["first_heading"] = state["first_heading"] or joined
                elif last_image and _FIGURE_CAPTION.match(joined):
                    ctx.add(
                        ElementType.CAPTION,
                        joined,
                        section_path=section(),
                        parent_id=last_image,
                        metadata=page_meta,
                        **common,
                    )
                else:
                    ctx.add(
                        ElementType.PARAGRAPH,
                        joined,
                        section_path=section(),
                        metadata={**page_meta, "font_size": size},
                        **common,
                    )
                last_image = None
            elif kind == "table":
                flush_list()
                self._emit_table(payload, number, section(), ctx)
                last_image = None
            else:
                flush_list()
                xref, rect = payload
                last_image = self._emit_image(doc, xref, rect, number, ctx, state, section())
        flush_list()

    # ------------------------------------------------------------- pieces
    def _images_on(
        self, doc: pymupdf.Document, page: pymupdf.Page, state: dict[str, Any]
    ) -> list[tuple[int, pymupdf.Rect]]:
        """``(xref, rect)`` of meaningful images placed on this page."""
        found = []
        for info in page.get_images(full=True):
            xref, width, height = info[0], info[2], info[3]
            if width < _MIN_IMAGE_SIDE or height < _MIN_IMAGE_SIDE:
                state["small_images"] += 1
                continue
            rects = page.get_image_rects(xref)
            if rects:
                found.append((xref, rects[0]))
        return found

    @staticmethod
    def _emit_image(
        doc: pymupdf.Document,
        xref: int,
        rect: pymupdf.Rect,
        number: int,
        ctx: ExtractionContext,
        state: dict[str, Any],
        section_path: list[str] | None = None,
    ) -> str | None:
        if xref in state["seen_xrefs"]:
            return None
        state["seen_xrefs"].add(xref)
        extracted = doc.extract_image(xref)  # the embedded stream as stored: original bytes
        if not extracted:
            return None
        ref = ctx.save_asset(f"page{number:03d}_xref{xref}.{extracted['ext']}", extracted["image"])
        return ctx.add(
            ElementType.IMAGE,
            "",
            extraction_method=_METHOD,
            page_number=number,
            section_path=section_path or [t for _, t in state["headings"]],
            asset_ref=ref,
            bbox=_bbox(rect),
            metadata={
                "xref": xref,
                "width_px": extracted.get("width"),
                "height_px": extracted.get("height"),
                "format": extracted["ext"],
            },
        )

    @staticmethod
    def _emit_table(
        table: Any, number: int, section_path: list[str], ctx: ExtractionContext
    ) -> None:
        rows = [
            [(cell or "").replace("\n", " ").strip() for cell in row] for row in table.extract()
        ]
        if table.header.external:  # header detected above the table's ruled area
            header, data = list(table.header.names), rows
        else:
            header, data = rows[0], rows[1:]
        if not data:
            return
        add_table(
            ctx,
            header,
            data,
            extraction_method=_METHOD,
            # Detected from ruling lines, so structure is inferred: not 1.0.
            confidence=0.85,
            page_number=number,
            section_path=section_path,
            metadata={"detector": "pymupdf.find_tables"},
            bbox=_bbox(table.bbox),
        )

    def _ocr_page(
        self,
        page: pymupdf.Page,
        number: int,
        ctx: ExtractionContext,
        state: dict[str, Any],
        parent_id: str | None,
    ) -> bool:
        if self.ocr is None or not self.ocr.available():
            return False
        if len(state["ocr_pages"]) >= self.ocr_max_pages:
            return False
        pixmap = page.get_pixmap(dpi=self.ocr_dpi, colorspace=pymupdf.csGRAY)
        try:
            result = self.ocr.recognize(pixmap.tobytes("png"), dpi=self.ocr_dpi)
        except OcrError as exc:
            ctx.warn(f"Page {number}: OCR failed ({exc})")
            return False
        state["ocr_pages"].append(number)
        scale = 72 / self.ocr_dpi  # pixels at ocr_dpi -> PDF points
        dropped = 0
        for para in result.paragraphs:
            if para.confidence < _MIN_OCR_CONFIDENCE:
                dropped += 1
                continue
            x0, y0, x1, y1 = (round(v * scale, 1) for v in para.bbox)
            ctx.add(
                ElementType.PARAGRAPH,
                para.text,
                extraction_method=ExtractionMethod.TESSERACT,
                origin=ContentOrigin.OCR,
                confidence=para.confidence,
                page_number=number,
                section_path=[t for _, t in state["headings"]],
                parent_id=parent_id,
                bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                metadata={
                    "ocr_engine": self.ocr.version,
                    "ocr_dpi": self.ocr_dpi,
                    "words": para.words,
                    "low_confidence_words": para.low_confidence_words,
                    "ocr_ms": result.duration_ms,
                },
            )
        if dropped:
            ctx.warn(f"Page {number}: dropped {dropped} OCR fragments below 30% confidence")
        return True
