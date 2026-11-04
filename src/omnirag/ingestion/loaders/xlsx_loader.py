"""Excel (.xlsx) via openpyxl in read-only (streaming) mode.

A spreadsheet is a grid, not a document. One sheet often holds several tables separated by blank
rows, offset from A1, plus free-text notes. For each sheet we:

1. Stream the rows twice in lockstep: once for **cached values** (``data_only=True``, what
   Excel last displayed) and once for **formulas**. Files written by scripts (openpyxl, pandas)
   contain formulas but no cached values. In that case we keep the formula text and warn,
   rather than silently emitting an empty cell.
2. Render values the way Excel displays them where it changes meaning: percentages (``0.725``
   with format ``0.0%`` becomes ``72.5%``) and dates (``2026-05-30``, not serial ``46172``).
3. Split the grid into blocks at fully empty rows and trim empty edge columns.
4. Emit multi-row, multi-column blocks as tables (row 1 is the header when it looks like one)
   and 1-row or 1-column blocks as plain paragraphs.

Read-only mode keeps memory flat on large workbooks. openpyxl uses ``defusedxml`` (a project
dependency) automatically to block XML-bomb attacks.

Not extracted yet: charts, images, comments and pivot caches (read-only mode skips them).
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.utils import get_column_letter

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_METHOD = ExtractionMethod.OPENPYXL
_MAX_ROWS_PER_SHEET = 100_000
_PERCENT = re.compile(r"0(?:\.(0+))?%")


def _number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_value(value: Any, number_format: str | None = None) -> str:
    """Cell value as display text."""
    if value is None:
        return ""
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):  # before date: datetime is a subclass of date
        return value.date().isoformat() if value.time() == time() else value.isoformat(sep=" ")
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, int | float):
        if match := _PERCENT.search(number_format or ""):
            return f"{value * 100:.{len(match.group(1) or '')}f}%"
        return _number(value)
    return str(value).strip()


def _looks_like_header(row: list[str]) -> bool:
    """Unique, non-empty, non-numeric names. Typed cells make this simpler than in CSV."""
    if not all(row) or len(set(row)) != len(row):
        return False
    return not any(re.fullmatch(r"[-+]?\d[\d,.]*%?", cell) for cell in row)


def _blocks(grid: list[list[str]]) -> list[tuple[int, int, list[list[str]]]]:
    """Split at fully empty rows: ``(first_row_idx, first_col_idx, rows)`` per block."""
    runs: list[tuple[int, list[list[str]]]] = []
    for r, row in enumerate(grid):
        if any(row):
            if runs and runs[-1][0] + len(runs[-1][1]) == r:
                runs[-1][1].append(row)
            else:
                runs.append((r, [row]))
    blocks = []
    for start, rows in runs:
        used = [c for row in rows for c, cell in enumerate(row) if cell]
        c0, c1 = min(used), max(used)
        trimmed = [(row + [""] * (c1 + 1 - len(row)))[c0 : c1 + 1] for row in rows]
        blocks.append((start, c0, trimmed))
    return blocks


class XlsxLoader:
    name = "xlsx"
    kinds = (FileKind.XLSX,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        values_wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
        formulas_wb = openpyxl.load_workbook(
            path, read_only=True, data_only=False, keep_links=False
        )
        try:
            sheets = []
            uncached_total = 0
            for index, (ws_v, ws_f) in enumerate(
                zip(values_wb.worksheets, formulas_wb.worksheets, strict=True)
            ):
                uncached = self._sheet(ws_v, ws_f, index, ctx)
                uncached_total += uncached
                sheets.append({"name": ws_v.title, "state": ws_v.sheet_state})
            props = values_wb.properties
            return {
                "title": props.title or None,
                "author": props.creator or None,
                "sheets": sheets,
                "uncached_formulas": uncached_total,
            }
        finally:
            values_wb.close()  # read-only workbooks keep the ZIP file open until closed
            formulas_wb.close()

    @staticmethod
    def _sheet(ws_v: Any, ws_f: Any, index: int, ctx: ExtractionContext) -> int:
        # Dimensions are written by the producing app and are sometimes wrong; ignore them and
        # read from A1 so grid indices map directly to Excel row/column numbers.
        ws_v.reset_dimensions()
        ws_f.reset_dimensions()
        grid: list[list[str]] = []
        uncached = 0
        rows_v = ws_v.iter_rows(min_row=1, min_col=1)
        rows_f = ws_f.iter_rows(min_row=1, min_col=1, values_only=True)
        for row_v, row_f in zip(rows_v, rows_f, strict=False):
            if len(grid) >= _MAX_ROWS_PER_SHEET:
                ctx.warn(f"Sheet {ws_v.title!r} truncated at {_MAX_ROWS_PER_SHEET} rows")
                break
            cells = []
            for c, cell in enumerate(row_v):
                text = render_value(cell.value, getattr(cell, "number_format", None))
                formula = row_f[c] if c < len(row_f) else None
                if not text and isinstance(formula, str) and formula.startswith("="):
                    text, uncached = formula, uncached + 1
                cells.append(text)
            grid.append(cells)
        if uncached:
            ctx.warn(
                f"Sheet {ws_v.title!r}: {uncached} formula cells have no cached value (file "
                "was not saved by a calculating app); formula text is shown instead"
            )

        section = [ws_v.title]
        sheet_meta = {"sheet": ws_v.title, "sheet_index": index, "sheet_state": ws_v.sheet_state}
        confidence_factor = 0.8 if uncached else 1.0
        for start, col, rows in _blocks(grid):
            width = len(rows[0])
            cell_range = (
                f"{get_column_letter(col + 1)}{start + 1}:"
                f"{get_column_letter(col + width)}{start + len(rows)}"
            )
            meta = {**sheet_meta, "cell_range": cell_range}
            if len(rows) < 2 or width < 2:
                text = "\n".join(" | ".join(c for c in row if c) for row in rows)
                ctx.add(
                    ElementType.PARAGRAPH,
                    text,
                    extraction_method=_METHOD,
                    confidence=confidence_factor,
                    section_path=section,
                    metadata=meta,
                )
                continue
            has_header = _looks_like_header(rows[0])
            header = rows[0] if has_header else [""] * width
            data = rows[1:] if has_header else rows
            first_data_row = start + 1 + (1 if has_header else 0)
            add_table(
                ctx,
                header,
                data,
                extraction_method=_METHOD,
                confidence=(0.95 if has_header else 0.85) * confidence_factor,
                section_path=section,
                metadata={**meta, "header_detected": has_header},
                row_metadata=[{"excel_row": first_data_row + i} for i in range(len(data))],
            )
        return uncached
