"""JSON (.json): one ``RECORD`` element per object, nested structure kept as a path.

Embedding a whole JSON file as one blob makes field-level questions ("who leads the
electrolyte workstream?") hard to retrieve. We emit one element per object containing its
*scalar* fields as ``key: value`` lines and recurse into nested objects/arrays, linking children
to their parent record. ``metadata.json_path`` (e.g. ``$.team[2].skills``) is exact provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import read_text
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind

_METHOD = ExtractionMethod.JSON_PARSE
_MAX_INLINE_LIST = 50  # short scalar lists stay inline: "tags: a, b, c"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _is_inline(value: Any) -> bool:
    return _is_scalar(value) or (
        isinstance(value, list)
        and len(value) <= _MAX_INLINE_LIST
        and all(_is_scalar(v) for v in value)
    )


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class JsonLoader:
    name = "json"
    kinds = (FileKind.JSON,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        decoded = read_text(path)
        data = json.loads(decoded.text)  # JSONDecodeError → pipeline marks document FAILED
        # Loaders are shared across worker threads, so per-file state is passed, not stored.
        self._emit(ctx, data, ["$"], None, decoded.confidence)
        return {
            "encoding": decoded.encoding,
            "top_level_type": type(data).__name__,
            "top_level_keys": list(data)[:50] if isinstance(data, dict) else None,
        }

    def _emit(
        self, ctx: ExtractionContext, value: Any, path: list[str], parent: str | None, conf: float
    ) -> None:
        json_path = ".".join(path)
        if isinstance(value, dict):
            inline = {k: v for k, v in value.items() if _is_inline(v)}
            record_id = parent
            if inline:
                added = ctx.add(
                    ElementType.RECORD,
                    "\n".join(f"{k}: {_fmt(v)}" for k, v in inline.items()),
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=path[1:],
                    metadata={"json_path": json_path, "keys": list(inline)},
                    parent_id=parent,
                )
                record_id = added or parent
            for key, child in value.items():
                if not _is_inline(child):
                    self._emit(ctx, child, [*path, str(key)], record_id, conf)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._emit(ctx, item, [*path[:-1], f"{path[-1]}[{index}]"], parent, conf)
        else:  # scalar reached directly (top-level scalar or long scalar list items)
            ctx.add(
                ElementType.RECORD,
                f"{path[-1]}: {_fmt(value)}",
                extraction_method=_METHOD,
                confidence=conf,
                section_path=path[1:],
                metadata={"json_path": json_path},
                parent_id=parent,
            )
