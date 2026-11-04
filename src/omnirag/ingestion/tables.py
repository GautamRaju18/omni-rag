"""Shared table handling for every format that contains tables (CSV, HTML, DOCX, XLSX, PPTX, PDF).

A table becomes two levels of elements:

* one ``TABLE`` element holding the whole table as Markdown (keeps structure for the LLM),
* one ``TABLE_ROW`` child per data row rendered as ``Header: value | Header: value``.
  A bare row like ``| C-104 | 87.2 |`` means nothing on its own, but with headers attached it can
  be retrieved and cited by itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import ElementType, ExtractionMethod


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def to_markdown(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a table as GitHub-flavoured Markdown."""
    width = len(header)
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [_cell(c) for c in list(row)[:width]]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def row_to_text(header: Sequence[str], row: Sequence[object]) -> str:
    """``Header: value | Header: value``; empty cells are omitted."""
    pairs = []
    for name, value in zip(header, row, strict=False):
        cell = _cell(value)
        if cell:
            pairs.append(f"{_cell(name)}: {cell}")
    return " | ".join(pairs)


def add_table(
    ctx: ExtractionContext,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    extraction_method: ExtractionMethod,
    confidence: float = 1.0,
    page_number: int | None = None,
    section_path: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
    parent_id: str | None = None,
    row_metadata: Sequence[dict[str, Any]] | None = None,
) -> str | None:
    """Add a TABLE element plus one TABLE_ROW child per row. Returns the table element ID.

    ``row_metadata[i]`` is merged into row ``i``'s metadata, e.g. ``{"excel_row": 7}`` so a
    citation can point at the exact spreadsheet row.
    """
    header = [h if _cell(h) else f"column_{i + 1}" for i, h in enumerate(header)]
    table_meta = {"columns": list(header), "n_rows": len(rows), "n_cols": len(header)}
    table_id = ctx.add(
        ElementType.TABLE,
        to_markdown(header, rows),
        extraction_method=extraction_method,
        confidence=confidence,
        page_number=page_number,
        section_path=section_path,
        metadata={**table_meta, **(metadata or {})},
        parent_id=parent_id,
    )
    if table_id is None:
        return None
    for index, row in enumerate(rows):
        ctx.add(
            ElementType.TABLE_ROW,
            row_to_text(header, row),
            extraction_method=extraction_method,
            confidence=confidence,
            page_number=page_number,
            section_path=section_path,
            metadata={"row_index": index, **(row_metadata[index] if row_metadata else {})},
            parent_id=table_id,
        )
    return table_id
