"""The unified, normalized element schema every loader produces.

Whatever the source format (Markdown heading, DOCX paragraph, PDF text block, OCR'd image
region, spreadsheet row), it becomes a ``DocumentElement``. Everything downstream (chunking,
embedding, graph construction, citations) depends only on this schema, never on file formats.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FileKind(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    CSV = "csv"
    TXT = "txt"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    IMAGE = "image"


class ElementType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    TABLE_ROW = "table_row"
    RECORD = "record"  # one structured object (JSON object, ...)
    IMAGE = "image"
    FIGURE = "figure"
    CAPTION = "caption"
    # Model-written description of an IMAGE (always origin=generated, child of the image).
    DESCRIPTION = "description"


class ExtractionMethod(StrEnum):
    TEXT_DECODE = "text_decode"
    MARKDOWN_IT = "markdown_it"
    BEAUTIFULSOUP = "beautifulsoup"
    JSON_PARSE = "json_parse"
    CSV_SNIFFER = "csv_sniffer"
    PYTHON_DOCX = "python_docx"
    OPENPYXL = "openpyxl"
    PYTHON_PPTX = "python_pptx"
    PYMUPDF = "pymupdf"
    TESSERACT = "tesseract"
    PILLOW = "pillow"
    DOCLING = "docling"
    VLM = "vlm"  # vision-language model (generated content)


class ContentOrigin(StrEnum):
    """Where the ``content`` text came from: crucial for trust and citations."""

    ORIGINAL = "original"  # text that exists verbatim in the source
    OCR = "ocr"  # recognized from pixels; may contain recognition errors
    GENERATED = "generated"  # written by a model (e.g. image description); never a quote


class IngestionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundingBox(BaseModel):
    """Region on a page or image. ``unit`` is ``pt`` for PDFs and ``px`` for images."""

    x0: float
    y0: float
    x1: float
    y1: float
    unit: str = "pt"


class DocumentElement(BaseModel):
    """One extracted unit of content with full provenance."""

    document_id: str
    element_id: str
    element_type: ElementType
    content: str
    source_file: str
    page_number: int | None = None  # 1-based; None for formats without pages
    # Heading trail, e.g. ["Results", "Cycling"]; JSON uses the key trail.
    section_path: list[str] = Field(default_factory=list)
    order: int  # position in reading order within the document
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_method: ExtractionMethod
    origin: ContentOrigin = ContentOrigin.ORIGINAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    parent_element_id: str | None = None
    asset_ref: str | None = None  # relative path of a preserved original asset (image, ...)
    bbox: BoundingBox | None = None


class DocumentRecord(BaseModel):
    """Document-level metadata and ingestion state, persisted next to the original file."""

    document_id: str
    source_file: str
    file_kind: FileKind
    sha256: str
    size_bytes: int
    status: IngestionStatus = IngestionStatus.PENDING
    loader: str | None = None
    element_count: int = 0
    element_counts_by_type: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
