"""Document endpoints: upload, browse, download, re-ingest, delete.

* ``POST   /documents``                     upload a file → 202 + job (or 200 if duplicate)
* ``GET    /documents``                     list (filter by status / kind, paginated)
* ``GET    /documents/{id}``                full record (metadata, warnings, counts)
* ``GET    /documents/{id}/elements``       extracted elements (filter by type, page, origin)
* ``GET    /documents/{id}/original``       the original file, byte-for-byte
* ``GET    /documents/{id}/assets/{name}``  an extracted image
* ``POST   /documents/{id}/reingest``       process the stored original again → 202 + job
* ``DELETE /documents/{id}``                remove it from the graph, then original/elements/assets

Upload safety, in order: ``Content-Length`` pre-check (413 before reading), filename reduced to
a safe basename, extension allow-list (415), size cap enforced while copying (413), magic-byte
and ZIP validation (415/422), then hashing for duplicate detection. Nothing is parsed in the
request; parsing happens in the background job.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from omnirag.api.dependencies import JobsDep, SettingsDep, StoreDep
from omnirag.core.errors import (
    ConflictError,
    DependencyUnavailableError,
    FileValidationError,
    NotFoundError,
    PayloadTooLargeError,
)
from omnirag.ingestion.models import (
    ContentOrigin,
    DocumentRecord,
    ElementType,
    FileKind,
    IngestionStatus,
)
from omnirag.ingestion.validation import document_id_for, kind_for_path, sha256_file, validate_file
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.schemas.documents import DocumentPage, DocumentSummary, ElementPage, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

DocumentId = Annotated[str, PathParam(pattern=r"^doc_[0-9a-f]{16}$")]
AssetName = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_MULTIPART_OVERHEAD = 64 * 1024  # boundaries and part headers on top of the file itself
_CHUNK = 1024 * 1024
_NOSNIFF = {"X-Content-Type-Options": "nosniff"}
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def safe_filename(raw: str | None) -> str:
    """Basename only, no control characters, at most 150 chars (extension kept)."""
    name = Path((raw or "").replace("\\", "/")).name
    name = "".join(
        c for c in unicodedata.normalize("NFC", name) if unicodedata.category(c)[0] != "C"
    )
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise FileValidationError("Upload has no usable filename")
    stem, suffix = Path(name).stem, Path(name).suffix
    return f"{stem[: 150 - len(suffix)]}{suffix}"


def _copy_limited(src: BinaryIO, dest: Path, max_bytes: int) -> int:
    size = 0
    with dest.open("wb") as out:
        while chunk := src.read(_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise PayloadTooLargeError(
                    f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
                    details={"max_bytes": max_bytes},
                )
            out.write(chunk)
    return size


def _record_or_404(store: StoreDep, document_id: str) -> DocumentRecord:
    record = store.read_record(document_id)
    if record is None:
        raise NotFoundError(f"Document {document_id} not found")
    return record


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Upload a file for background ingestion",
    responses={200: {"description": "Same bytes already ingested (duplicate)"}},
)
async def upload_document(
    request: Request,
    response: Response,
    settings: SettingsDep,
    store: StoreDep,
    jobs: JobsDep,
    file: Annotated[UploadFile, File(description="Document to ingest")],
    force: Annotated[bool, Query(description="Re-process even if already ingested")] = False,
) -> UploadResponse:
    max_bytes = settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + _MULTIPART_OVERHEAD:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_mb} MB upload limit",
            details={"max_bytes": max_bytes, "content_length": int(declared)},
        )
    name = safe_filename(file.filename)
    kind_for_path(Path(name))  # 415 for unsupported extensions, before copying anything

    job_id, upload_dir = jobs.new_upload_dir()
    path = upload_dir / name
    try:
        await asyncio.to_thread(_copy_limited, file.file, path, max_bytes)
        await asyncio.to_thread(validate_file, path, max_bytes=max_bytes)
        document_id = document_id_for(await asyncio.to_thread(sha256_file, path))
        existing = store.read_record(document_id)
        if existing and existing.status is IngestionStatus.COMPLETED and not force:
            shutil.rmtree(upload_dir, ignore_errors=True)
            response.status_code = status.HTTP_200_OK
            return UploadResponse(
                document_id=document_id, duplicate=True, document=DocumentSummary.of(existing)
            )
        if active := jobs.active_job_for(document_id):
            # Same bytes already queued or running: return that job (idempotent upload).
            shutil.rmtree(upload_dir, ignore_errors=True)
            response.headers["Location"] = f"{settings.api_prefix}/ingestion/jobs/{active.job_id}"
            return UploadResponse(document_id=document_id, duplicate=False, job=active)
        job = jobs.submit(job_id, name, document_id, force=force)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        await file.close()
    response.headers["Location"] = f"{settings.api_prefix}/ingestion/jobs/{job.job_id}"
    return UploadResponse(document_id=document_id, duplicate=False, job=job)


@router.get("", response_model=DocumentPage, summary="List ingested documents")
async def list_documents(
    store: StoreDep,
    status_filter: Annotated[IngestionStatus | None, Query(alias="status")] = None,
    kind: FileKind | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    records = await asyncio.to_thread(store.list_records)
    records = [
        r for r in records
        if (status_filter is None or r.status is status_filter)
        and (kind is None or r.file_kind is kind)
    ]  # fmt: skip
    records.sort(key=lambda r: r.created_at, reverse=True)
    page = [DocumentSummary.of(r) for r in records[offset : offset + limit]]
    return DocumentPage(items=page, total=len(records), limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentRecord, summary="Full document record")
async def get_document(document_id: DocumentId, store: StoreDep) -> DocumentRecord:
    return _record_or_404(store, document_id)


@router.get("/{document_id}/elements", response_model=ElementPage, summary="Extracted elements")
async def get_elements(
    document_id: DocumentId,
    store: StoreDep,
    element_type: Annotated[ElementType | None, Query(alias="type")] = None,
    page: Annotated[int | None, Query(ge=1)] = None,
    origin: ContentOrigin | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ElementPage:
    _record_or_404(store, document_id)
    elements = await asyncio.to_thread(store.read_elements, document_id)
    elements = [
        e for e in elements
        if (element_type is None or e.element_type is element_type)
        and (page is None or e.page_number == page)
        and (origin is None or e.origin is origin)
    ]  # fmt: skip
    return ElementPage(
        document_id=document_id,
        items=elements[offset : offset + limit],
        total=len(elements),
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}/original", summary="Download the original file")
async def get_original(document_id: DocumentId, store: StoreDep) -> FileResponse:
    record = _record_or_404(store, document_id)
    path = store.original_path(document_id)
    if path is None:
        raise NotFoundError(f"Original of {document_id} not found")
    # Always a download with a neutral type: an uploaded .html must never render (and run
    # scripts) on the API's origin.
    return FileResponse(
        path, media_type="application/octet-stream", filename=record.source_file, headers=_NOSNIFF
    )


@router.get("/{document_id}/assets/{name}", summary="An extracted image")
async def get_asset(document_id: DocumentId, name: AssetName, store: StoreDep) -> FileResponse:
    _record_or_404(store, document_id)
    path = store.doc_dir(document_id) / "assets" / name
    if not path.is_file():
        raise NotFoundError(f"Asset {name} not found for {document_id}")
    media_type = _IMAGE_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path, media_type=media_type, filename=name, content_disposition_type="inline",
        headers=_NOSNIFF,
    )  # fmt: skip


@router.post(
    "/{document_id}/reingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Process the stored original again (e.g. after a loader upgrade)",
)
async def reingest_document(
    document_id: DocumentId,
    response: Response,
    settings: SettingsDep,
    store: StoreDep,
    jobs: JobsDep,
) -> UploadResponse:
    record = _record_or_404(store, document_id)
    if active := jobs.active_job_for(document_id):
        return UploadResponse(document_id=document_id, duplicate=False, job=active)
    original = store.original_path(document_id)
    if original is None:
        raise NotFoundError(f"Original of {document_id} not found")
    job_id, upload_dir = jobs.new_upload_dir()
    name = safe_filename(record.source_file)
    try:
        await asyncio.to_thread(shutil.copyfile, original, upload_dir / name)
        job = jobs.submit(job_id, name, document_id, force=True)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    response.headers["Location"] = f"{settings.api_prefix}/ingestion/jobs/{job.job_id}"
    return UploadResponse(document_id=document_id, duplicate=False, job=job)


@router.delete(
    "/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document"
)
async def delete_document(document_id: DocumentId, jobs: JobsDep) -> Response:
    if active := jobs.active_job_for(document_id):
        raise ConflictError(
            f"{document_id} is being ingested; wait for job {active.job_id}",
            details={"job_id": active.job_id, "status": active.status.value},
        )
    try:
        deleted = await asyncio.to_thread(jobs.pipeline.delete, document_id)
    except GraphUnavailableError as exc:  # files are kept, so nothing is orphaned in the graph
        raise DependencyUnavailableError(f"Neo4j unavailable; {document_id} not deleted") from exc
    if not deleted:
        raise NotFoundError(f"Document {document_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
