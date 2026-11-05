"""API response models for documents and uploads."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from omnirag.ingestion.jobs import IngestionJob
from omnirag.ingestion.models import DocumentElement, DocumentRecord, FileKind, IngestionStatus


class DocumentSummary(BaseModel):
    """List view of a document (the full record is at ``GET /documents/{id}``)."""

    document_id: str
    source_file: str
    file_kind: FileKind
    status: IngestionStatus
    size_bytes: int
    element_count: int
    element_counts_by_type: dict[str, int]
    warning_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, record: DocumentRecord) -> DocumentSummary:
        return cls(
            **record.model_dump(
                include={
                    "document_id", "source_file", "file_kind", "status", "size_bytes",
                    "element_count", "element_counts_by_type", "created_at", "updated_at",
                }
            ),
            warning_count=len(record.warnings),
        )  # fmt: skip


class DocumentPage(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class ElementPage(BaseModel):
    document_id: str
    items: list[DocumentElement]
    total: int  # after filters
    limit: int
    offset: int


class JobPage(BaseModel):
    items: list[IngestionJob]
    total: int
    limit: int
    offset: int


class UploadResponse(BaseModel):
    """``202`` with a job when work was queued; ``200`` with ``duplicate=true`` when the same
    bytes were already ingested (nothing to do)."""

    document_id: str
    duplicate: bool
    job: IngestionJob | None = None
    document: DocumentSummary | None = None
