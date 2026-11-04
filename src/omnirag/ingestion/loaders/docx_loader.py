"""Word (.docx) via python-docx, with our own text extraction where python-docx falls short.

Reading order comes from ``Document.iter_inner_content()``, which yields body paragraphs and
tables in document order.

Why not ``paragraph.text``? In python-docx 1.2.0 it is
``"".join(e.text for e in p.xpath("w:r | w:hyperlink"))``, which reads only the paragraph's
*direct* children. Text inside a tracked insertion (``<w:ins><w:r>``), a simple field
(``<w:fldSimple>``), a smart tag or an inline content control (``<w:sdt>``) is silently dropped.
``_element_text`` reads every ``w:t`` inside a run instead, and skips deleted, moved-away and
text-box content.

Headings: the paragraph's outline level (``w:outlineLvl``, set on the paragraph or inherited
through its style chain) does not depend on the UI language. Style names like "Heading 1" are
localized ("Überschrift 1" in German Word), so the name is only a fallback.

Not extracted yet: headers and footers (mostly boilerplate such as "Confidential" and page
numbers), footnotes, comments, text boxes, charts, and body-level content controls (cover
pages, tables of contents).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import docx
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_METHOD = ExtractionMethod.PYTHON_DOCX
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
# Text under these ancestors is not part of the visible body text.
_SKIP_ANCESTORS = {qn("w:del"), qn("w:moveFrom"), qn("w:txbxContent"), _MC_FALLBACK}
_RUN_TEXT = {qn("w:t"): "", qn("w:tab"): "\t", qn("w:br"): "\n", qn("w:cr"): "\n"}
_HEADING_NAME = re.compile(r"^heading (\d)$", re.IGNORECASE)
_LEVEL_SUFFIX = re.compile(r" (\d)$")  # "List Bullet 2" -> nesting level 1


def _element_text(element: Any) -> str:
    """Visible text of a ``w:p`` element (see the module docstring for why this exists)."""
    parts: list[str] = []
    for node in element.iter(*_RUN_TEXT):
        if node.getparent().tag != qn("w:r"):
            continue  # e.g. <w:tab> inside <w:tabs> is a tab-stop definition, not text
        if any(a.tag in _SKIP_ANCESTORS for a in node.iterancestors()):
            continue
        parts.append((node.text or "") if node.tag == qn("w:t") else _RUN_TEXT[node.tag])
    return "".join(parts)


def _style_chain(paragraph: Paragraph) -> Iterator[Any]:
    style = paragraph.style
    while style is not None:
        yield style
        style = style.base_style


def _from_pPr_or_styles(paragraph: Paragraph, xpath: str) -> list[str]:
    """Look up a paragraph property directly, then through the style inheritance chain."""
    values = paragraph._p.xpath(f"./w:pPr/{xpath}")
    for style in _style_chain(paragraph):
        if values:
            break
        values = style.element.xpath(f"./w:pPr/{xpath}")
    return values


def _heading_level(paragraph: Paragraph) -> int | None:
    outline = _from_pPr_or_styles(paragraph, "w:outlineLvl/@w:val")
    if outline and outline[0].isdigit() and int(outline[0]) < 9:  # 9 = body text
        return int(outline[0]) + 1
    for style in _style_chain(paragraph):
        if match := _HEADING_NAME.match(style.name or ""):
            return int(match.group(1))
    return None


def _has_style(paragraph: Paragraph, name: str) -> bool:
    return any((style.name or "").lower() == name for style in _style_chain(paragraph))


def _list_item(paragraph: Paragraph) -> tuple[str, int] | None:
    """``(numId, level)`` when the paragraph is a list item, else ``None``."""
    num_ids = _from_pPr_or_styles(paragraph, "w:numPr/w:numId/@w:val")
    if not num_ids or num_ids[0] == "0":  # numId 0 explicitly switches numbering off
        return None
    ilvl = _from_pPr_or_styles(paragraph, "w:numPr/w:ilvl/@w:val")
    if ilvl and ilvl[0].isdigit():
        return num_ids[0], int(ilvl[0])
    # Built-in styles nest by style ("List Bullet 2") rather than by ilvl.
    match = _LEVEL_SUFFIX.search(paragraph.style.name or "") if paragraph.style else None
    return num_ids[0], int(match.group(1)) - 1 if match else 0


def _numbering_formats(document: Any) -> dict[tuple[str, int], str]:
    """``(numId, level) -> numFmt`` (``bullet``, ``decimal``, ``lowerLetter`` ...)."""
    try:
        # Not ``document.part.numbering_part``: that property tries to create a missing part.
        numbering = document.part.part_related_by(RT.NUMBERING)
    except KeyError:
        return {}
    root = numbering.element
    abstract: dict[str, dict[int, str]] = {}
    for definition in root.findall(qn("w:abstractNum")):
        levels = {}
        for lvl in definition.findall(qn("w:lvl")):
            fmt = lvl.find(qn("w:numFmt"))
            ilvl = lvl.get(qn("w:ilvl"), "0")
            if ilvl.isdigit():
                levels[int(ilvl)] = fmt.get(qn("w:val")) if fmt is not None else "decimal"
        abstract[definition.get(qn("w:abstractNumId"))] = levels
    formats = {}
    for num in root.findall(qn("w:num")):
        ref = num.find(qn("w:abstractNumId"))
        if ref is not None:
            for level, fmt in abstract.get(ref.get(qn("w:val")), {}).items():
                formats[(num.get(qn("w:numId")), level)] = fmt
    return formats


class DocxLoader:
    name = "docx"
    kinds = (FileKind.DOCX,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        document = docx.Document(str(path))
        formats = _numbering_formats(document)
        headings: list[tuple[int, str]] = []
        list_items: list[tuple[str, int, str]] = []
        state: dict[str, Any] = {"title": None, "last_image": None, "images": 0, "tables": 0}

        def section() -> list[str]:
            return [text for _, text in headings]

        def flush_list() -> None:
            if not list_items:
                return
            lines, counters, ordered = [], {}, False
            for num_id, level, text in list_items:
                fmt = formats.get((num_id, level), "bullet")
                if fmt in ("bullet", "none"):
                    marker = "-"
                else:
                    ordered = True
                    counters = {k: v for k, v in counters.items() if k[1] <= level}
                    counters[(num_id, level)] = counters.get((num_id, level), 0) + 1
                    marker = f"{counters[(num_id, level)]}."
                lines.append(f"{'  ' * level}{marker} {text}")
            ctx.add(
                ElementType.LIST,
                "\n".join(lines),
                extraction_method=_METHOD,
                section_path=section(),
                metadata={"ordered": ordered, "items": len(list_items)},
            )
            list_items.clear()
            state["last_image"] = None

        for block in document.iter_inner_content():
            if isinstance(block, Table):
                flush_list()
                self._table(block, ctx, section())
                state["tables"] += 1
                state["last_image"] = None
                continue

            text = _element_text(block._p).strip()
            item = _list_item(block)
            if item and text:
                # Nested levels often use their own numId ("List Bullet 2"), so only a level-0
                # item with a different numId than the list's first item starts a new list.
                if list_items and item[1] == 0 and item[0] != list_items[0][0]:
                    flush_list()
                list_items.append((item[0], item[1], text))
                continue
            flush_list()

            level = _heading_level(block)
            if _has_style(block, "title") and text:
                ctx.add(ElementType.TITLE, text, extraction_method=_METHOD)
                state["title"] = state["title"] or text
            elif level is not None and text:
                headings[:] = [h for h in headings if h[0] < level]
                ctx.add(
                    ElementType.HEADING,
                    text,
                    extraction_method=_METHOD,
                    section_path=section(),
                    metadata={"level": level},
                )
                headings.append((level, text))
            elif _has_style(block, "caption") and text:
                ctx.add(
                    ElementType.CAPTION,
                    text,
                    extraction_method=_METHOD,
                    section_path=section(),
                    parent_id=state["last_image"],
                )
            elif text:
                ctx.add(
                    ElementType.PARAGRAPH, text, extraction_method=_METHOD, section_path=section()
                )
            # Images go after the paragraph's own text so a following caption can link to them.
            image_id = self._images(block, document, ctx, section(), state)
            state["last_image"] = image_id if image_id else None
        flush_list()

        props = document.core_properties
        body = document.element.body
        insertions = len(body.findall(f".//{qn('w:ins')}"))
        deletions = len(body.findall(f".//{qn('w:del')}"))
        if insertions or deletions:
            ctx.warn(
                f"Document contains tracked changes ({insertions} insertions, {deletions} "
                "deletions); extracted text shows them as if accepted"
            )
        return {
            "title": props.title or state["title"],
            "author": props.author or None,
            "created": props.created.isoformat() if props.created else None,
            "modified": props.modified.isoformat() if props.modified else None,
            "tables": state["tables"],
            "images": state["images"],
            "tracked_changes": {"insertions": insertions, "deletions": deletions},
        }

    @staticmethod
    def _table(table: Table, ctx: ExtractionContext, section_path: list[str]) -> None:
        rows = []
        for row in table.rows:
            # row.cells repeats a merged cell once per grid column it spans, so merged headers
            # still label every column. Nested tables are flattened into their cell's text.
            rows.append(
                [
                    "\n".join(t for p in cell._tc.iter(qn("w:p")) if (t := _element_text(p)))
                    for cell in row.cells
                ]
            )
        rows = [r for r in rows if any(c.strip() for c in r)]
        width = max((len(r) for r in rows), default=0)
        if len(rows) < 2 or width < 2:
            # One row or one column: almost always a layout table, not data.
            for cell in (c for r in rows for c in r if c.strip()):
                ctx.add(
                    ElementType.PARAGRAPH,
                    cell,
                    extraction_method=_METHOD,
                    section_path=section_path,
                    metadata={"layout_table": True},
                )
            return
        header_marked = bool(table.rows[0]._tr.xpath("./w:trPr/w:tblHeader"))
        add_table(
            ctx,
            rows[0],
            rows[1:],
            extraction_method=_METHOD,
            # Word marks real header rows ("Repeat as header row"); otherwise row 1 is a guess.
            confidence=1.0 if header_marked else 0.9,
            section_path=section_path,
            metadata={"header_marked": header_marked},
        )

    @staticmethod
    def _images(
        paragraph: Paragraph,
        document: Any,
        ctx: ExtractionContext,
        section_path: list[str],
        state: dict[str, Any],
    ) -> str | None:
        """Save embedded pictures as assets; return the last IMAGE element ID."""
        last_id = None
        for drawing in paragraph._p.iter(qn("w:drawing")):
            if any(a.tag == _MC_FALLBACK for a in drawing.iterancestors()):
                continue
            doc_pr = next(drawing.iter(qn("wp:docPr")), None)
            alt = (doc_pr.get("descr") or doc_pr.get("title") or "") if doc_pr is not None else ""
            blip = next(drawing.iter(qn("a:blip")), None)
            rid = blip.get(qn("r:embed")) if blip is not None else None
            if rid is None:
                linked = blip is not None and blip.get(qn("r:link"))
                ctx.warn(
                    "Skipped a linked (external) image: remote URLs are never fetched"
                    if linked
                    else "Skipped a drawing without an embedded picture (chart, SmartArt or shape)"
                )
                continue
            part = document.part.related_parts[rid]
            state["images"] += 1
            suffix = PurePosixPath(str(part.partname)).suffix or ".bin"
            asset_ref = ctx.save_asset(f"image_{state['images']:03d}{suffix}", part.blob)
            last_id = ctx.add(
                ElementType.IMAGE,
                alt,
                extraction_method=_METHOD,
                section_path=section_path,
                asset_ref=asset_ref,
                metadata={
                    "alt_text": alt or None,
                    "name": doc_pr.get("name") if doc_pr is not None else None,
                    "content_type": part.content_type,
                },
            )
        return last_id
