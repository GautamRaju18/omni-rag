"""PowerPoint (.pptx) via python-pptx.

One slide is one page: ``page_number`` is the slide number, and the slide title becomes the
section. Within a slide, shapes are sorted top-to-bottom, then left-to-right. XML order is
z-order (what was drawn last), not reading order. Per slide we emit:

* the title as a HEADING;
* body placeholders as a LIST (indented by bullet level), other text boxes as PARAGRAPHs;
* tables through ``add_table``;
* **charts as data**: a FIGURE plus a TABLE of the chart's cached series values. A chart's
  numbers live in the file, so no vision model is needed to read them;
* pictures saved as assets, with their alt text, for the vision model in Phase 2d;
* speaker notes, which often hold the facts the presenter actually says.

Footer, date and slide-number placeholders are skipped as boilerplate. Hidden slides are kept
and flagged in their metadata. Not extracted yet: SmartArt, embedded OLE objects and audio/video.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.shapes.graphfrm import GraphicFrame
from pptx.shapes.group import GroupShape
from pptx.shapes.picture import Picture

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.xlsx_loader import render_value
from omnirag.ingestion.models import BoundingBox, ElementType, ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_METHOD = ExtractionMethod.PYTHON_PPTX
_TITLES = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE}
_BOILERPLATE = {
    PP_PLACEHOLDER.DATE,
    PP_PLACEHOLDER.FOOTER,
    PP_PLACEHOLDER.SLIDE_NUMBER,
    PP_PLACEHOLDER.HEADER,
}
_BULLETED = {
    PP_PLACEHOLDER.BODY,
    PP_PLACEHOLDER.OBJECT,
    PP_PLACEHOLDER.VERTICAL_BODY,
    PP_PLACEHOLDER.VERTICAL_OBJECT,
}


def _placeholder_type(shape: Any) -> PP_PLACEHOLDER | None:
    return shape.placeholder_format.type if shape.is_placeholder else None


def _walk(shapes: Any, top_level: bool = True) -> Iterator[tuple[Any, bool]]:
    """Shapes in reading order, descending into groups."""
    for shape in sorted(shapes, key=lambda s: (s.top or 0, s.left or 0)):
        if isinstance(shape, GroupShape):
            yield from _walk(shape.shapes, top_level=False)
        else:
            yield shape, top_level


def _bbox(shape: Any) -> BoundingBox | None:
    # Group children use the group's internal coordinate space, so only top-level shapes
    # get a box; ``.pt`` converts EMU (914,400 per inch) to points.
    if None in (shape.left, shape.top, shape.width, shape.height):
        return None
    x0, y0 = shape.left.pt, shape.top.pt
    return BoundingBox(x0=x0, y0=y0, x1=x0 + shape.width.pt, y1=y0 + shape.height.pt)


class PptxLoader:
    name = "pptx"
    kinds = (FileKind.PPTX,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        prs = Presentation(str(path))
        hidden_slides, skipped_frames = [], 0
        for number, slide in enumerate(prs.slides, start=1):
            hidden = slide._element.get("show") == "0"
            if hidden:
                hidden_slides.append(number)
            skipped_frames += self._slide(slide, number, hidden, ctx)
        if skipped_frames:
            ctx.warn(f"Skipped {skipped_frames} SmartArt/OLE objects (not extracted yet)")
        props = prs.core_properties
        return {
            "title": props.title or None,
            "author": props.author or None,
            "slide_count": len(prs.slides),
            "hidden_slides": hidden_slides,
        }

    def _slide(self, slide: Any, number: int, hidden: bool, ctx: ExtractionContext) -> int:
        title_shape = slide.shapes.title
        title = title_shape.text_frame.text.strip() if title_shape is not None else ""
        section = [title or f"Slide {number}"]
        base = {"slide_number": number, "hidden": hidden}
        common = {"extraction_method": _METHOD, "page_number": number}
        if title:
            ctx.add(
                ElementType.HEADING,
                title,
                metadata={**base, "level": 1},
                bbox=_bbox(title_shape),
                **common,
            )

        skipped = 0
        for shape, top_level in _walk(slide.shapes):
            kind = _placeholder_type(shape)
            if kind in _BOILERPLATE or (title_shape and shape.shape_id == title_shape.shape_id):
                continue
            kw = {**common, "section_path": section, "bbox": _bbox(shape) if top_level else None}
            if isinstance(shape, Picture):
                self._picture(shape, number, ctx, base, kw)
            elif isinstance(shape, GraphicFrame) and shape.has_chart:
                self._chart(shape.chart, ctx, base, kw)
            elif isinstance(shape, GraphicFrame) and shape.has_table:
                self._table(shape.table, ctx, base, kw)
            elif isinstance(shape, GraphicFrame):
                skipped += 1
            elif shape.has_text_frame:
                self._text(shape.text_frame, kind in _BULLETED, ctx, base, kw)

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                ctx.add(
                    ElementType.PARAGRAPH,
                    notes.text,
                    metadata={**base, "speaker_notes": True},
                    section_path=[*section, "Speaker notes"],
                    **common,
                )
        return skipped

    @staticmethod
    def _text(frame: Any, bulleted: bool, ctx: ExtractionContext, base: dict, kw: dict) -> None:
        paragraphs = [(p.level, p.text.strip()) for p in frame.paragraphs if p.text.strip()]
        if not paragraphs:
            return
        if bulleted:
            # A soft line break ("\v") inside one bullet is still one item: join with a space.
            lines = [f"{'  ' * level}- {text.replace(chr(11), ' ')}" for level, text in paragraphs]
            ctx.add(ElementType.LIST, "\n".join(lines), metadata=base, **kw)
        else:
            ctx.add(ElementType.PARAGRAPH, "\n".join(t for _, t in paragraphs), metadata=base, **kw)

    @staticmethod
    def _table(table: Any, ctx: ExtractionContext, base: dict, kw: dict) -> None:
        # A horizontally merged cell is "spanned" with empty text: it is omitted from row text.
        rows = [[cell.text.replace("\v", " ").strip() for cell in row.cells] for row in table.rows]
        if len(rows) < 2:
            ctx.add(ElementType.PARAGRAPH, " | ".join(c for c in rows[0] if c), metadata=base, **kw)
            return
        add_table(
            ctx,
            rows[0],
            rows[1:],
            extraction_method=_METHOD,
            # "Header row" formatting switched on in PowerPoint makes row 1 a declared header.
            confidence=1.0 if table.first_row else 0.9,
            page_number=kw["page_number"],
            section_path=kw["section_path"],
            metadata={**base, "header_marked": table.first_row},
        )

    @staticmethod
    def _chart(chart: Any, ctx: ExtractionContext, base: dict, kw: dict) -> None:
        try:
            title = chart.chart_title.text_frame.text.strip() if chart.has_title else ""
            categories = [str(c) for c in chart.plots[0].categories]
            series = [(s.name, list(s.values)) for plot in chart.plots for s in plot.series]
            chart_type = getattr(chart.chart_type, "name", str(chart.chart_type))
        except Exception as exc:  # unusual chart XML: keep the slide, report the chart
            ctx.warn(f"Slide {base['slide_number']}: chart not extracted ({exc})")
            return
        n_points = max((len(values) for _, values in series), default=0)
        categories = categories or [str(i + 1) for i in range(n_points)]
        figure_id = ctx.add(
            ElementType.FIGURE,
            f"Chart: {title}" if title else "Chart",
            metadata={**base, "chart_type": chart_type, "series": [name for name, _ in series]},
            **kw,
        )
        rows = [
            [cat, *(render_value(v[i]) if i < len(v) else "" for _, v in series)]
            for i, cat in enumerate(categories)
        ]
        add_table(
            ctx,
            ["Category", *(name for name, _ in series)],
            rows,
            extraction_method=_METHOD,
            page_number=kw["page_number"],
            section_path=kw["section_path"],
            metadata={**base, "source": "chart data cache"},
            parent_id=figure_id,
        )

    @staticmethod
    def _picture(shape: Any, number: int, ctx: ExtractionContext, base: dict, kw: dict) -> None:
        try:
            image = shape.image
        except ValueError:  # python-pptx raises this for linked (not embedded) pictures
            ctx.warn(f"Slide {number}: skipped a linked image; remote URLs are never fetched")
            return
        alt = (shape._element.xpath("./p:nvPicPr/p:cNvPr/@descr") or [""])[0]
        asset_ref = ctx.save_asset(
            f"slide{number:03d}_shape{shape.shape_id}.{image.ext}", image.blob
        )
        ctx.add(
            ElementType.IMAGE,
            alt,
            asset_ref=asset_ref,
            metadata={**base, "alt_text": alt or None, "content_type": image.content_type},
            **kw,
        )
