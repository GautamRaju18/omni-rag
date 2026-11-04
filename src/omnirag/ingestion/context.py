"""``ExtractionContext``: the only object a loader talks to while extracting.

Loaders call ``ctx.add(...)`` for each element. The context:

* normalizes the text (one normalization layer for every format),
* skips empty elements,
* assigns **deterministic** IDs (``<document_id>-e00042``) in reading order, so re-ingesting the
  same file yields the same IDs. Evaluation ground truth that references them stays valid,
* stores binary assets (images) through the document store, so originals are never lost.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from omnirag.ingestion.models import (
    BoundingBox,
    ContentOrigin,
    DocumentElement,
    ElementType,
    ExtractionMethod,
)
from omnirag.ingestion.normalize import normalize_text

# Indentation carries meaning here (code blocks, Markdown tables, nested lists).
_LAYOUT_SENSITIVE = {ElementType.CODE, ElementType.TABLE, ElementType.LIST}


class ExtractionContext:
    def __init__(
        self,
        document_id: str,
        source_file: str,
        asset_saver: Callable[[str, bytes], str] | None = None,
        original_ref: str | None = None,
    ) -> None:
        self.document_id = document_id
        self.source_file = source_file
        # Reference to the stored original file. An image file *is* its own asset, so the image
        # loader points at it instead of storing a second copy.
        self.original_ref = original_ref
        self.elements: list[DocumentElement] = []
        self.warnings: list[str] = []
        self._asset_saver = asset_saver

    def add(
        self,
        element_type: ElementType,
        content: str,
        *,
        extraction_method: ExtractionMethod,
        page_number: int | None = None,
        section_path: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        origin: ContentOrigin = ContentOrigin.ORIGINAL,
        confidence: float = 1.0,
        parent_id: str | None = None,
        asset_ref: str | None = None,
        bbox: BoundingBox | None = None,
    ) -> str | None:
        """Append an element; return its ID, or ``None`` if it was empty and skipped."""
        text = normalize_text(content, preserve_layout=element_type in _LAYOUT_SENSITIVE)
        if not text and asset_ref is None:
            return None
        order = len(self.elements)
        element_id = f"{self.document_id}-e{order:05d}"
        self.elements.append(
            DocumentElement(
                document_id=self.document_id,
                element_id=element_id,
                element_type=element_type,
                content=text,
                source_file=self.source_file,
                page_number=page_number,
                section_path=list(section_path),
                order=order,
                metadata=metadata or {},
                extraction_method=extraction_method,
                origin=origin,
                confidence=confidence,
                parent_element_id=parent_id,
                asset_ref=asset_ref,
                bbox=bbox,
            )
        )
        return element_id

    def save_asset(self, name: str, data: bytes) -> str:
        """Persist an original binary asset; returns a reference relative to the data dir."""
        if self._asset_saver is None:
            raise RuntimeError("This context was created without an asset store")
        return self._asset_saver(name, data)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
