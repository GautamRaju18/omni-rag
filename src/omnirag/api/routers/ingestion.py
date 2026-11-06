"""Ingestion job endpoints.

* ``GET /ingestion/jobs``           recent jobs (filter by status, paginated)
* ``GET /ingestion/jobs/{job_id}``  one job: poll this after an upload
* ``GET /ingestion/status``         worker and queue state
* ``POST /ingestion/start``         (re)ingest a stored document: ``{"document_id": ...}``
* ``GET /ingestion/{job_id}/status``  the brief's path for one job (same as ``/jobs/{id}``)

New files start ingesting when uploaded (``POST /documents/upload``); ``start`` is for files
already in the store (after a loader upgrade, or a failed job).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, Response
from pydantic import BaseModel, Field

from omnirag.api.dependencies import JobsDep, SettingsDep, StoreDep
from omnirag.api.routers.documents import reingest_document
from omnirag.core.errors import NotFoundError
from omnirag.ingestion.jobs import IngestionJob, JobStatus
from omnirag.schemas.documents import JobPage, UploadResponse

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

JobId = Annotated[str, Path(pattern=r"^job_[0-9a-f]{12}$")]


@router.get("/jobs", response_model=JobPage, summary="List ingestion jobs (newest first)")
async def list_jobs(
    jobs: JobsDep,
    status: JobStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobPage:
    items, total = jobs.list(status=status, limit=limit, offset=offset)
    return JobPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=IngestionJob, summary="One ingestion job")
async def get_job(job_id: JobId, jobs: JobsDep) -> IngestionJob:
    job = jobs.jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    return job


@router.get("/status", summary="Worker and queue state")
async def ingestion_status(jobs: JobsDep) -> dict[str, Any]:
    return jobs.stats()


class StartRequest(BaseModel):
    document_id: str = Field(pattern=r"^doc_[0-9a-f]{16}$")


@router.post("/start", status_code=202, response_model=UploadResponse,
             summary="(Re)ingest a stored document in the background")  # fmt: skip
async def start_ingestion(request: StartRequest, response: Response, settings: SettingsDep,
                          store: StoreDep, jobs: JobsDep) -> UploadResponse:  # fmt: skip
    return await reingest_document(request.document_id, response, settings, store, jobs)


@router.get("/{job_id}/status", response_model=IngestionJob, summary="One ingestion job")
async def job_status(job_id: JobId, jobs: JobsDep) -> IngestionJob:
    return await get_job(job_id, jobs)
