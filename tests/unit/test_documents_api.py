"""/documents and /ingestion through the real app lifespan (no external services).

Neo4j/Ollama clients are created but never contacted; vision and OCR are off, Docling is not
used (pdf_engine=pymupdf). Ingestion runs in the real background worker.
"""

from __future__ import annotations

import asyncio
import io
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import docx
import httpx
import pytest
from PIL import Image

from omnirag.api.main import create_app
from omnirag.api.routers.documents import safe_filename
from omnirag.core.config import Settings
from omnirag.core.errors import FileValidationError

API = "/api/v1"
TEXT = b"Aurora status.\n\nE-7 is the candidate electrolyte for milestone M2.\n"


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, object]]:
    settings = Settings(
        _env_file=None, env="test", log_json=False, neo4j_password="x", data_dir=tmp_path,
        vision_enabled=False, ocr_enabled=False, pdf_engine="pymupdf", max_upload_mb=1,
    )  # fmt: skip
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app


async def _upload(client, name: str, data: bytes, **params) -> httpx.Response:
    return await client.post(f"{API}/documents", files={"file": (name, data)}, params=params)


async def _finish(client, job_id: str) -> dict:
    for _ in range(500):
        job = (await client.get(f"{API}/ingestion/jobs/{job_id}")).json()
        if job["status"] in ("completed", "failed"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {job}")


async def test_upload_ingest_browse_download(api) -> None:
    client, _ = api
    response = await _upload(client, "status.txt", TEXT)
    assert response.status_code == 202
    body = response.json()
    assert body["duplicate"] is False and body["job"]["status"] == "queued"
    assert response.headers["location"] == f"{API}/ingestion/jobs/{body['job']['job_id']}"
    doc_id = body["document_id"]

    job = await _finish(client, body["job"]["job_id"])
    assert job["status"] == "completed" and job["element_count"] == 2

    listing = (await client.get(f"{API}/documents")).json()
    assert listing["total"] == 1 and listing["items"][0]["document_id"] == doc_id
    record = (await client.get(f"{API}/documents/{doc_id}")).json()
    assert record["status"] == "completed" and record["file_kind"] == "txt"

    elements = (await client.get(f"{API}/documents/{doc_id}/elements", params={"limit": 1})).json()
    assert elements["total"] == 2 and len(elements["items"]) == 1
    assert elements["items"][0]["element_id"] == f"{doc_id}-e00000"

    original = await client.get(f"{API}/documents/{doc_id}/original")
    assert original.content == TEXT
    assert original.headers["content-type"] == "application/octet-stream"
    assert original.headers["content-disposition"].startswith("attachment")
    assert original.headers["x-content-type-options"] == "nosniff"


async def test_duplicate_and_force(api) -> None:
    client, _ = api
    first = (await _upload(client, "a.txt", TEXT)).json()
    await _finish(client, first["job"]["job_id"])

    again = await _upload(client, "renamed.txt", TEXT)  # same bytes, other name
    assert again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["job"] is None
    assert again.json()["document"]["source_file"] == "a.txt"

    forced = await _upload(client, "a.txt", TEXT, force="true")
    assert forced.status_code == 202 and forced.json()["job"]["force"] is True
    assert (await _finish(client, forced.json()["job"]["job_id"]))["status"] == "completed"


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("tool.exe", b"MZ\x90\x00", 415, "unsupported_file_type"),
        ("report.pdf", b"MZ not a pdf", 422, "invalid_file"),
        ("empty.txt", b"", 422, "invalid_file"),
        ("big.txt", b"x" * (1024 * 1024 + 200 * 1024), 413, "payload_too_large"),  # header check
        ("edge.txt", b"x" * (1024 * 1024 + 10 * 1024), 413, "payload_too_large"),  # copy check
    ],
)
async def test_rejected_uploads_leave_nothing_behind(api, tmp_path: Path, name, data, status, code):
    client, _ = api
    response = await _upload(client, name, data)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert list((tmp_path / "uploads").iterdir()) == []
    assert (await client.get(f"{API}/ingestion/jobs")).json()["total"] == 0


async def test_filename_is_reduced_to_a_safe_basename(api) -> None:
    client, _ = api
    body = (await _upload(client, "../../etc/evil notes.txt", TEXT)).json()
    job = await _finish(client, body["job"]["job_id"])
    assert job["source_file"] == "evil notes.txt"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("..\\..\\windows\\report.pdf", "report.pdf"),  # Windows-style traversal
        ("a\x00b\x1f c.txt", "ab c.txt"),  # control characters (httpx would %-encode these)
        ("  spaced   name .txt ", "spaced name .txt"),
        ("x" * 300 + ".docx", "x" * 145 + ".docx"),  # capped at 150, extension kept
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "../", "\x00\x01", " . "])
def test_safe_filename_rejects_unusable_names(raw) -> None:
    with pytest.raises(FileValidationError):
        safe_filename(raw)


async def test_invalid_ids_and_missing_resources(api) -> None:
    client, _ = api
    assert (await client.get(f"{API}/documents/not-an-id")).status_code == 422
    assert (await client.get(f"{API}/documents/doc_0000000000000000")).status_code == 404
    assert (await client.get(f"{API}/ingestion/jobs/job_000000000000")).status_code == 404
    assert (await client.get(f"{API}/ingestion/jobs/../../etc")).status_code == 404
    assert (await client.delete(f"{API}/documents/doc_0000000000000000")).status_code == 404


async def test_assets_served_inline_with_safe_names(api) -> None:
    client, _ = api
    png = io.BytesIO()
    Image.new("RGB", (40, 20), "blue").save(png, format="PNG")
    d = docx.Document()
    d.add_picture(io.BytesIO(png.getvalue()))
    out = io.BytesIO()
    d.save(out)
    body = (await _upload(client, "fig.docx", out.getvalue())).json()
    await _finish(client, body["job"]["job_id"])
    doc_id = body["document_id"]

    images = (
        await client.get(f"{API}/documents/{doc_id}/elements", params={"type": "image"})
    ).json()
    name = images["items"][0]["asset_ref"].rsplit("/", 1)[-1]
    asset = await client.get(f"{API}/documents/{doc_id}/assets/{name}")
    assert asset.status_code == 200 and asset.content == png.getvalue()
    assert asset.headers["content-type"] == "image/png"
    assert asset.headers["content-disposition"].startswith("inline")
    for bad in ("..%2Fdocument.json", ".hidden", "a%2Fb.png"):
        assert (await client.get(f"{API}/documents/{doc_id}/assets/{bad}")).status_code in (
            404,
            422,
        )


async def test_delete_conflict_reingest_and_status(api) -> None:
    client, app = api
    body = (await _upload(client, "a.txt", TEXT)).json()
    await _finish(client, body["job"]["job_id"])
    doc_id = body["document_id"]

    # Hold the next job inside the worker so the document is "being ingested".
    gate = threading.Event()
    pipeline = app.state.jobs.pipeline
    real_ingest = pipeline.ingest

    def slow_ingest(*args, **kwargs):
        gate.wait(5)
        return real_ingest(*args, **kwargs)

    pipeline.ingest = slow_ingest
    reingest = await client.post(f"{API}/documents/{doc_id}/reingest")
    assert reingest.status_code == 202 and reingest.json()["job"]["force"] is True
    job_id = reingest.json()["job"]["job_id"]
    for _ in range(200):  # wait until the worker picked it up
        if (await client.get(f"{API}/ingestion/jobs/{job_id}")).json()["status"] == "running":
            break
        await asyncio.sleep(0.01)
    status = (await client.get(f"{API}/ingestion/status")).json()
    assert status["running"] == [job_id] and status["accepting"] is True

    conflict = await client.delete(f"{API}/documents/{doc_id}")
    assert conflict.status_code == 409 and conflict.json()["error"]["details"]["job_id"] == job_id
    again = await client.post(f"{API}/documents/{doc_id}/reingest")  # idempotent
    assert again.json()["job"]["job_id"] == job_id

    gate.set()
    assert (await _finish(client, job_id))["status"] == "completed"
    assert (await client.delete(f"{API}/documents/{doc_id}")).status_code == 204
    assert (await client.get(f"{API}/documents/{doc_id}")).status_code == 404
    jobs = (await client.get(f"{API}/ingestion/jobs", params={"status": "completed"})).json()
    assert jobs["total"] == 2
