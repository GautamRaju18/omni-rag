"""JobManager with a fake pipeline: state transitions, back-pressure, restart recovery."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnirag.core.errors import FileValidationError, IngestionQueueFullError
from omnirag.ingestion.jobs import IngestionJob, JobManager, JobStatus
from omnirag.ingestion.models import IngestionStatus

FINAL = {JobStatus.COMPLETED, JobStatus.FAILED}


class FakePipeline:
    def __init__(self, behavior: str = "ok", gate: threading.Event | None = None) -> None:
        self.behavior = behavior
        self.gate = gate  # when set, ingest blocks until the test releases it
        self.calls: list[tuple[str, bool]] = []
        self.closed = 0

    def ingest(self, path: Path, *, source_name: str | None = None, force: bool = False):
        self.calls.append((path.name, force))
        if self.gate is not None:
            self.gate.wait(5)
        if self.behavior == "invalid":
            raise FileValidationError("bad bytes")
        if self.behavior == "crash":
            raise RuntimeError("boom")
        failed = self.behavior == "record_failed"
        record = SimpleNamespace(
            status=IngestionStatus.FAILED if failed else IngestionStatus.COMPLETED,
            element_count=3,
            warnings=["w1"],
            error="ValueError: broken" if failed else None,
        )
        return SimpleNamespace(record=record, duplicate=False)

    def close(self) -> None:
        self.closed += 1


async def _wait(manager: JobManager, job_id: str, statuses=FINAL) -> IngestionJob:
    for _ in range(500):
        job = manager.jobs[job_id]
        if job.status in statuses:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} stuck in {manager.jobs[job_id].status}")


def _submit(manager: JobManager, name: str = "a.txt", doc: str = "doc_0000000000000001") -> str:
    job_id, upload_dir = manager.new_upload_dir()
    (upload_dir / name).write_text("hello")
    manager.submit(job_id, name, doc, force=False)
    return job_id


async def test_job_completes_persists_and_cleans_up(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    manager = JobManager(pipeline, tmp_path)
    await manager.start()
    job_id = _submit(manager)
    job = await _wait(manager, job_id)
    await asyncio.sleep(0.05)  # let the worker reach the end-of-batch close()
    assert job.status is JobStatus.COMPLETED and job.attempts == 1
    assert (job.element_count, job.warnings) == (3, ["w1"])
    assert job.started_at and job.finished_at and job.duration_ms is not None
    assert not (tmp_path / "uploads" / job_id).exists()  # upload copy removed
    on_disk = IngestionJob.model_validate_json((tmp_path / "jobs" / f"{job_id}.json").read_bytes())
    assert on_disk.status is JobStatus.COMPLETED
    assert pipeline.closed >= 1  # VLM unloaded when the queue drained
    await manager.stop()


@pytest.mark.parametrize(
    ("behavior", "code"),
    [
        ("record_failed", "extraction_failed"),
        ("invalid", "invalid_file"),
        ("crash", "internal_error"),
    ],
)
async def test_job_failures_carry_an_error_code(tmp_path: Path, behavior: str, code: str) -> None:
    manager = JobManager(FakePipeline(behavior), tmp_path)
    await manager.start()
    job = await _wait(manager, _submit(manager))
    assert job.status is JobStatus.FAILED and job.error is not None and job.error.code == code
    await manager.stop()


async def test_queue_full_is_back_pressure(tmp_path: Path) -> None:
    gate = threading.Event()
    manager = JobManager(FakePipeline(gate=gate), tmp_path, queue_max=1)
    await manager.start()
    first = _submit(manager, doc="doc_0000000000000001")
    await _wait(manager, first, {JobStatus.RUNNING})
    _submit(manager, doc="doc_0000000000000002")  # fills the queue
    with pytest.raises(IngestionQueueFullError):
        _submit(manager, doc="doc_0000000000000003")
    assert manager.stats()["counts"] == {"queued": 1, "running": 1, "completed": 0, "failed": 0}
    gate.set()
    await manager.stop()


async def test_active_job_lookup_by_document(tmp_path: Path) -> None:
    gate = threading.Event()
    manager = JobManager(FakePipeline(gate=gate), tmp_path)
    await manager.start()
    job_id = _submit(manager, doc="doc_00000000000000aa")
    assert manager.active_job_for("doc_00000000000000aa").job_id == job_id
    gate.set()
    await _wait(manager, job_id)
    assert manager.active_job_for("doc_00000000000000aa") is None
    await manager.stop()


def _write_job(tmp_path: Path, job: IngestionJob, with_upload: bool = True) -> None:
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "jobs" / f"{job.job_id}.json").write_text(job.model_dump_json())
    if with_upload:
        (tmp_path / "uploads" / job.job_id).mkdir(parents=True)
        (tmp_path / "uploads" / job.job_id / job.source_file).write_text("x")


async def test_restart_requeues_interrupted_jobs(tmp_path: Path) -> None:
    doc = "doc_0000000000000001"
    was_running = IngestionJob(job_id="job_00000000000a", source_file="a.txt", document_id=doc,
                               status=JobStatus.RUNNING, attempts=1)  # fmt: skip
    was_queued = IngestionJob(job_id="job_00000000000b", source_file="b.txt", document_id=doc)
    gave_up = IngestionJob(job_id="job_00000000000c", source_file="c.txt", document_id=doc,
                           status=JobStatus.RUNNING, attempts=3)  # fmt: skip
    lost_upload = IngestionJob(job_id="job_00000000000d", source_file="d.txt", document_id=doc)
    for job in (was_running, was_queued, gave_up):
        _write_job(tmp_path, job)
    _write_job(tmp_path, lost_upload, with_upload=False)

    pipeline = FakePipeline()
    manager = JobManager(pipeline, tmp_path, max_attempts=3)
    await manager.start()
    assert (await _wait(manager, "job_00000000000a")).attempts == 2
    assert (await _wait(manager, "job_00000000000b")).status is JobStatus.COMPLETED
    for job_id in ("job_00000000000c", "job_00000000000d"):
        job = manager.jobs[job_id]
        assert job.status is JobStatus.FAILED and job.error.code == "interrupted"
    assert sorted(name for name, _ in pipeline.calls) == ["a.txt", "b.txt"]
    await manager.stop()


async def test_stop_leaves_unfinished_job_for_next_start(tmp_path: Path) -> None:
    gate = threading.Event()
    manager = JobManager(FakePipeline(gate=gate), tmp_path)
    await manager.start()
    job_id = _submit(manager)
    await _wait(manager, job_id, {JobStatus.RUNNING})
    await manager.stop(grace_s=0.05)
    on_disk = IngestionJob.model_validate_json((tmp_path / "jobs" / f"{job_id}.json").read_bytes())
    assert on_disk.status is JobStatus.RUNNING  # re-queued by the next start()
    assert (tmp_path / "uploads" / job_id / "a.txt").exists()
    assert manager.stats()["accepting"] is False
    with pytest.raises(IngestionQueueFullError, match="not accepting"):
        _submit(manager)
    gate.set()
