"""CSV (.csv): dialect + header detection.

Real-world CSVs use ``,`` ``;`` ``\\t`` or ``|`` and may or may not have a header row.

* Dialect: ``csv.Sniffer().sniff`` is reliable for delimiters.
* Header: ``csv.Sniffer().has_header`` is *not* reliable on its own. Its string-length vote
  rejects ``id;value`` over ``A1;10`` because ``id`` and ``A1`` have the same length. We first
  apply a type rule: a header has unique, non-empty, non-numeric names, and at least one
  column below it is numeric. Only when that rule is inconclusive (all-text tables) do we fall
  back to the Sniffer. The method used is recorded in metadata for debugging.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import read_text
from omnirag.ingestion.models import ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_SAMPLE_CHARS = 64 * 1024
_DELIMITERS = ",;\t|"


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", "").strip())
    except ValueError:
        return False
    return True


def _detect_header(rows: list[list[str]], sample: str) -> tuple[bool, str]:
    """Return ``(has_header, method)``."""
    first, data = rows[0], rows[1:]
    names = [c.strip() for c in first]
    if not all(names) or len(set(names)) != len(names) or any(_is_number(n) for n in names):
        return False, "type_rule"
    numeric_columns = [
        i
        for i in range(len(first))
        if (values := [r[i] for r in data if i < len(r) and r[i].strip()])
        and all(_is_number(v) for v in values)
    ]
    if numeric_columns or not data:
        return True, "type_rule"
    try:
        return csv.Sniffer().has_header(sample), "sniffer"
    except csv.Error:
        return True, "assumed"


class CsvLoader:
    name = "csv"
    kinds = (FileKind.CSV,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        decoded = read_text(path)
        sample = decoded.text[:_SAMPLE_CHARS]
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
                sample, delimiters=_DELIMITERS
            )
        except csv.Error:
            dialect = csv.excel
            ctx.warn("Could not detect CSV dialect; assumed comma-separated")

        rows = [
            r for r in csv.reader(io.StringIO(decoded.text), dialect) if any(c.strip() for c in r)
        ]
        if not rows:
            return {"encoding": decoded.encoding, "n_rows": 0}
        has_header, header_method = _detect_header(rows, sample)
        if header_method == "assumed":
            ctx.warn("Could not detect a header row; assumed the first row is the header")

        if has_header:
            header, data = rows[0], rows[1:]
        else:
            header, data = [f"column_{i + 1}" for i in range(len(rows[0]))], rows
        ragged = sum(1 for r in data if len(r) != len(header))
        if ragged:
            ctx.warn(f"{ragged} rows have a different number of fields than the header")

        add_table(
            ctx,
            header,
            data,
            extraction_method=ExtractionMethod.CSV_SNIFFER,
            confidence=decoded.confidence * 0.95,  # header/dialect are heuristic guesses
            metadata={"delimiter": dialect.delimiter, "header_detected": has_header},
        )
        return {
            "encoding": decoded.encoding,
            "delimiter": dialect.delimiter,
            "header_detected": has_header,
            "header_detection_method": header_method,
            "columns": header,
            "n_rows": len(data),
            "n_cols": len(header),
            "ragged_rows": ragged,
        }
