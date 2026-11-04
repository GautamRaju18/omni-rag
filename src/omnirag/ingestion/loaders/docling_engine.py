"""Docling layout engine for PDFs that PyMuPDF's heuristics get wrong.

Docling runs layout and table-structure **models** (on CPU here) instead of font-size rules:
* reading order across columns (left column fully before the right one);
* tables without ruling lines (TableFormer predicts rows, columns and header cells);
* page headers/footers recognised by the layout model (``furniture``), not by repetition.

Costs (measured 2026-11-04, Docling 2.141.0, CPU): ~3.4 s for a 1-page note vs 0.09 s with
PyMuPDF, ~1.4 GB RAM, a 506 MB one-time model download from Hugging Face, and ~60 s for the
first pipeline initialisation. That's why ``PdfLoader`` only routes documents here when needed
(``pdf_engine=auto``) or when asked (``pdf_engine=docling``).

The converter is created lazily, once per process, and guarded by a lock. Loaders are shared
across threads, and two concurrent conversions would only compete for the same CPU cores.
Docling's own OCR is off: scanned pages stay on the PyMuPDF + Tesseract path.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import BoundingBox, ElementType, ExtractionMethod
from omnirag.ingestion.tables import add_table

_METHOD = ExtractionMethod.DOCLING
_TABLE_CONFIDENCE = 0.9  # structure predicted by a model, text from the PDF text layer
_IMAGES_SCALE = 2.0  # picture crops rendered at 2x (144 dpi)


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


class DoclingEngine:
    def __init__(self, *, num_threads: int = 4, timeout_s: float = 300.0) -> None:
        self.num_threads = num_threads
        self.timeout_s = timeout_s
        self._converter: Any = None
        self._lock = threading.Lock()

    def _get_converter(self) -> Any:
        # Imported here: importing docling pulls in torch (~1 s, hundreds of MB), and most
        # documents never need it.
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if self._converter is None:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()  # "Loading weights" bars would corrupt CLI output
            options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                generate_picture_images=True,
                images_scale=_IMAGES_SCALE,
                document_timeout=self.timeout_s,
                accelerator_options=AcceleratorOptions(
                    device=AcceleratorDevice.CPU, num_threads=self.num_threads
                ),
            )
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def convert(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        """Emit elements for ``path`` into ``ctx``. Raises if Docling fails before emitting."""
        with self._lock:
            result = self._get_converter().convert(str(path), raises_on_error=True)
        return _emit(result.document, ctx)


def _emit(doc: Any, ctx: ExtractionContext) -> dict[str, Any]:
    from docling_core.types.doc import (
        CodeItem,
        DocItemLabel,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    ids: dict[str, str] = {}  # Docling self_ref -> our element ID (for caption parents)
    headings: list[tuple[int, str]] = []
    pending: list[tuple[Any, int]] = []
    counts = {"pictures": 0, "tables": 0}

    def section() -> list[str]:
        return [text for _, text in headings]

    def where(item: Any) -> dict[str, Any]:
        if not item.prov:
            return {"page_number": None, "bbox": None}
        prov = item.prov[0]
        box = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)
        return {
            "page_number": prov.page_no,
            "bbox": BoundingBox(
                x0=round(box.l, 1), y0=round(box.t, 1), x1=round(box.r, 1), y1=round(box.b, 1)
            ),
        }

    def flush_list() -> None:
        if not pending:
            return
        base = min(level for _, level in pending)
        lines, ordered = [], False
        for item, level in pending:
            ordered = ordered or item.enumerated
            marker = item.marker if item.enumerated and item.marker else "-"
            lines.append(f"{'  ' * (level - base)}{marker} {item.text}")
        first = where(pending[0][0])
        ctx.add(
            ElementType.LIST,
            "\n".join(lines),
            extraction_method=_METHOD,
            section_path=section(),
            metadata={"ordered": ordered, "items": len(pending)},
            **first,
        )
        pending.clear()

    for item, level in doc.iterate_items():  # body only: headers/footers are "furniture"
        if isinstance(item, ListItem):
            pending.append((item, level))
            continue
        flush_list()
        loc = where(item)
        if isinstance(item, TitleItem):
            ids[item.self_ref] = ctx.add(ElementType.TITLE, item.text, extraction_method=_METHOD,
                                         **loc)  # fmt: skip
        elif isinstance(item, SectionHeaderItem):
            headings[:] = [h for h in headings if h[0] < item.level]
            ids[item.self_ref] = ctx.add(
                ElementType.HEADING,
                item.text,
                extraction_method=_METHOD,
                section_path=section(),
                metadata={"level": item.level},
                **loc,
            )
            headings.append((item.level, item.text))
        elif isinstance(item, TableItem):
            grid = [[cell.text for cell in row] for row in item.data.grid]
            if not grid:
                continue
            has_header = any(cell.column_header for cell in item.data.grid[0])
            header, rows = (grid[0], grid[1:]) if has_header else ([""] * len(grid[0]), grid)
            table_id = add_table(
                ctx,
                header,
                rows,
                extraction_method=_METHOD,
                confidence=_TABLE_CONFIDENCE,
                page_number=loc["page_number"],
                section_path=section(),
                metadata={"detector": "docling tableformer", "header_detected": has_header},
                bbox=loc["bbox"],
            )
            if table_id:
                ids[item.self_ref] = table_id
            counts["tables"] += 1
        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None:
                continue
            counts["pictures"] += 1
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            ref = ctx.save_asset(
                f"page{loc['page_number'] or 0:03d}_picture{counts['pictures']}.png",
                buffer.getvalue(),
            )
            ids[item.self_ref] = ctx.add(
                ElementType.IMAGE,
                "",
                extraction_method=_METHOD,
                section_path=section(),
                asset_ref=ref,
                # A rendered crop of the page region, not the embedded stream; the original
                # PDF is preserved byte-for-byte next to it.
                metadata={"source": "rendered crop", "scale": _IMAGES_SCALE},
                **loc,
            )
        elif isinstance(item, CodeItem):
            ctx.add(ElementType.CODE, item.text, extraction_method=_METHOD,
                    section_path=section(), **loc)  # fmt: skip
        elif isinstance(item, TextItem):
            if item.label is DocItemLabel.CAPTION:
                parent = ids.get(item.parent.cref) if item.parent else None
                ctx.add(ElementType.CAPTION, item.text, extraction_method=_METHOD,
                        section_path=section(), parent_id=parent, **loc)  # fmt: skip
            else:
                ctx.add(
                    ElementType.PARAGRAPH,
                    item.text,
                    extraction_method=_METHOD,
                    section_path=section(),
                    metadata={"docling_label": item.label.value},
                    **loc,
                )
    flush_list()
    return counts
