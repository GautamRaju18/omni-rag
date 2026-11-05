"""Background ingestion jobs: an in-process queue with persistent job records.

Uploading returns at once with a job ID. A worker ingests in a thread (parsing, OCR, Docling
and the vision model are blocking, CPU/GPU-bound work that must not run on the event loop),
and clients poll ``GET /ingestion/jobs/{id}``.

Why in-process and not Celery/RQ/arq: this is a single-user, single-machine system. A broker
(Redis/RabbitMQ) would add a service to run and monitor and buy nothing at this scale. What we
do need from a "real" queue, we implement:

* **Persistence:** every state change is written to ``<data_dir>/jobs/<job_id>.json``
  (atomic replace).
* **Restart recovery:** on start-up, jobs left ``queued`` or ``running`` are queued again. This is
  safe because ingestion is idempotent (same bytes, same document and element IDs). A job that
  keeps dying (``max_attempts``) is marked failed instead of crash-looping the server.
* **Back-pressure:** ``queue_max`` queued jobs, then ``IngestionQueueFullError`` (HTTP 503).
* **Batch resources:** when the queue drains, ``pipeline.close()`` unloads the vision model.

All job state is changed on the event-loop thread only (the worker awaits the thread and then
updates the record), so no locks are needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import shutil
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import structlog
from pydantic import BaseModel, Field

from omnirag.core.errors import IngestionQueueFullError, OmniRAGError
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import atomic_write_bytes

log = structlog.get_logger(module=__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


ACTIVE = {JobStatus.QUEUED, JobStatus.RUNNING}


class JobError(BaseModel):
    code: str
    message: str


class IngestionJob(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    source_file: str
    document_id: str  # known at upload time: it is derived from the file's hash
    force: bool = False
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    duplicate: bool = False
    element_count: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: JobError | None = None


class _Pipeline(Protocol):
    def ingest(self, path: Path, *, source_name: str | None = None, force: bool = False) -> Any: ...

    def close(self) -> None: ...

    def delete(self, document_id: str) -> bool: ...


def new_job_id() -> str:
    return f"job_{secrets.token_hex(6)}"


class JobManager:
    def __init__(
        self,
        pipeline: _Pipeline,
        data_dir: Path,
        *,
        workers: int = 1,
        queue_max: int = 100,
        max_attempts: int = 3,
    ) -> None:
        self.pipeline = pipeline
        self.jobs_dir = data_dir / "jobs"
        self.uploads_dir = data_dir / "uploads"
        self.workers = workers
        self.queue_max = queue_max
        self.max_attempts = max_attempts
        self.jobs: dict[str, IngestionJob] = {}
        self._queue: asyncio.Queue[str] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._running: set[str] = set()
        self._idle = asyncio.Event()
        self._stopping = False

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._queue = asyncio.Queue()
        self._idle.set()
        recovered = 0
        for path in sorted(self.jobs_dir.glob("job_*.json")):
            job = IngestionJob.model_validate_json(path.read_bytes())
            self.jobs[job.job_id] = job
            if job.status in ACTIVE:
                if job.attempts >= self.max_attempts or not self.upload_path(job).exists():
                    self._finish(job, JobStatus.FAILED, JobError(
                        code="interrupted",
                        message=f"Server stopped during processing ({job.attempts} attempts)",
                    ))  # fmt: skip
                else:
                    job.status = JobStatus.QUEUED
                    self._persist(job)
                    self._queue.put_nowait(job.job_id)
                    recovered += 1
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.workers)]
        log.info("jobs_started", workers=self.workers, known=len(self.jobs), requeued=recovered)

    async def stop(self, grace_s: float = 30.0) -> None:
        """Stop taking jobs; give running ones ``grace_s`` to finish. Unfinished jobs stay
        ``running`` on disk and are re-queued on the next start."""
        self._stopping = True
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._idle.wait(), timeout=grace_s)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await asyncio.to_thread(self.pipeline.close)
        log.info("jobs_stopped", still_running=sorted(self._running))

    # ------------------------------------------------------------------- API
    def new_upload_dir(self) -> tuple[str, Path]:
        job_id = new_job_id()
        path = self.uploads_dir / job_id
        path.mkdir(parents=True)
        return job_id, path

    def upload_path(self, job: IngestionJob) -> Path:
        return self.uploads_dir / job.job_id / job.source_file

    def submit(
        self, job_id: str, source_file: str, document_id: str, *, force: bool
    ) -> IngestionJob:
        if self._queue is None or self._stopping:
            raise IngestionQueueFullError("Ingestion is not accepting jobs (server stopping)")
        queued = sum(1 for j in self.jobs.values() if j.status is JobStatus.QUEUED)
        if queued >= self.queue_max:
            raise IngestionQueueFullError(
                f"{queued} jobs are already queued; retry later", details={"queued": queued}
            )
        job = IngestionJob(
            job_id=job_id, source_file=source_file, document_id=document_id, force=force
        )
        self.jobs[job_id] = job
        self._persist(job)
        self._queue.put_nowait(job_id)
        log.info("job_queued", job_id=job_id, document_id=document_id, source_file=source_file)
        return job

    def active_job_for(self, document_id: str) -> IngestionJob | None:
        return next(
            (j for j in self.jobs.values() if j.document_id == document_id and j.status in ACTIVE),
            None,
        )

    def list(
        self, *, status: JobStatus | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[IngestionJob], int]:
        jobs = [j for j in self.jobs.values() if status is None or j.status is status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[offset : offset + limit], len(jobs)

    def stats(self) -> dict[str, Any]:
        counts = {s.value: 0 for s in JobStatus}
        for job in self.jobs.values():
            counts[job.status.value] += 1
        return {
            "workers": self.workers,
            "workers_alive": sum(not t.done() for t in self._tasks),
            "accepting": self._queue is not None and not self._stopping,
            "queue_max": self.queue_max,
            "running": sorted(self._running),
            "counts": counts,
        }

    # ---------------------------------------------------------------- worker
    async def _worker(self, index: int) -> None:
        assert self._queue is not None  # noqa: S101 - set in start()
        while not self._stopping:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                continue
            if self._stopping:  # leave it queued on disk for the next start
                break
            await self._run(job)
            if self._queue.empty() and not self._running:
                # End of a batch: unload the vision model so the chat model gets the GPU back.
                await asyncio.to_thread(self.pipeline.close)

    async def _run(self, job: IngestionJob) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        job.attempts += 1
        self._persist(job)
        self._running.add(job.job_id)
        self._idle.clear()
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                self.pipeline.ingest, self.upload_path(job), source_name=job.source_file,
                force=job.force,
            )  # fmt: skip
        except asyncio.CancelledError:
            # Shutdown while the thread runs: keep "running" on disk and keep the upload,
            # so the next start re-queues it.
            raise
        except OmniRAGError as exc:
            self._finish(job, JobStatus.FAILED, JobError(code=exc.code, message=exc.message),
                         started)  # fmt: skip
        except Exception as exc:  # loader bugs are caught by the pipeline; this is a backstop
            log.exception("job_crashed", job_id=job.job_id)
            self._finish(job, JobStatus.FAILED,
                         JobError(code="internal_error", message=f"{type(exc).__name__}: {exc}"),
                         started)  # fmt: skip
        else:
            record = result.record
            job.duplicate = result.duplicate
            job.element_count = record.element_count
            job.warnings = record.warnings
            if record.status is IngestionStatus.FAILED:
                error = JobError(code="extraction_failed", message=record.error or "unknown")
                self._finish(job, JobStatus.FAILED, error, started)
            else:
                self._finish(job, JobStatus.COMPLETED, None, started)
        finally:
            self._running.discard(job.job_id)
            if not self._running:
                self._idle.set()

    def _finish(
        self,
        job: IngestionJob,
        status: JobStatus,
        error: JobError | None,
        started: float | None = None,
    ) -> None:
        job.status = status
        job.error = error
        job.finished_at = datetime.now(UTC)
        if started is not None:
            job.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        self._persist(job)
        # The original is now in the document store (or the file was rejected): drop the copy.
        shutil.rmtree(self.uploads_dir / job.job_id, ignore_errors=True)
        log.info("job_finished", job_id=job.job_id, status=status.value,
                 error=error.code if error else None, duration_ms=job.duration_ms)  # fmt: skip

    def _persist(self, job: IngestionJob) -> None:
        atomic_write_bytes(
            self.jobs_dir / f"{job.job_id}.json", job.model_dump_json(indent=2).encode()
        )
