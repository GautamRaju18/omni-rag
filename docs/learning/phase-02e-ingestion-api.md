# Phase 2e: Ingestion API with background jobs

> Completes Phase 2. Builds on [2a](phase-02a-ingestion-core.md)–[2d](phase-02d-layout-and-vision.md). Decision record: [ADR-0007](../adr/0007-in-process-ingestion-jobs.md).

## 1. Problem being solved
Everything from Phase 2 so far ran from the command line. A frontend (Phase 14) and other services need HTTP. Ingestion can take more than 10 s (Docling, vision), so an upload must **return immediately** while the work runs in the background, with a way to ask "is it done?". All of this has to survive restarts and stay safe against hostile uploads.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **202 Accepted + polling** | The server accepted the work but hasn't done it. The body contains a job; the `Location` header points at `GET /ingestion/jobs/{id}`, which the client polls until `completed`/`failed`. |
| **Job queue / worker** | `asyncio.Queue` of job IDs. The worker coroutine takes one and runs the blocking `pipeline.ingest` in a **thread** (`asyncio.to_thread`), so the event loop keeps serving requests. |
| **Back-pressure** | A bounded queue. When it's full the API answers 503, and the client retries later, instead of the server accepting unlimited work. |
| **Idempotency** | Repeating a request has the same effect as doing it once. Here: re-uploading the same bytes returns the existing document (200) or the job already in progress, never a second ingestion. |
| **Restart recovery** | Job state is on disk. After a crash, unfinished jobs run again, which is safe because ingestion is idempotent (content-addressed IDs). |
| **Defence in depth** | Several independent checks, so one bug doesn't let a bad upload through. Shown in §9: with the size check removed, file validation still rejected the file. |
| **`Content-Disposition: attachment` + `nosniff`** | Downloaded originals are never rendered by the browser. Otherwise an uploaded `.html` file could run scripts on the API's origin (stored XSS). |

## 3. API
| Method | Path | Result |
|---|---|---|
| POST | `/api/v1/documents` (`multipart/form-data`, field `file`, `?force=true`) | **202** job · **200** duplicate · 413 / 415 / 422 / 503 |
| GET | `/api/v1/documents?status=&kind=&limit=&offset=` | Summaries, newest first |
| GET | `/api/v1/documents/{doc_id}` | Full `DocumentRecord` (metadata, warnings, engine…) |
| GET | `/api/v1/documents/{doc_id}/elements?type=&page=&origin=&limit=&offset=` | Elements, e.g. `?origin=generated` |
| GET | `/api/v1/documents/{doc_id}/original` | The original bytes, as an attachment |
| GET | `/api/v1/documents/{doc_id}/assets/{name}` | An extracted image (inline, `image/png` / `image/jpeg`) |
| POST | `/api/v1/documents/{doc_id}/reingest` | **202** job (`force=true`), e.g. after a loader upgrade |
| DELETE | `/api/v1/documents/{doc_id}` | **204** · 404 · **409** while a job runs |
| GET | `/api/v1/ingestion/jobs?status=&limit=&offset=` | Jobs, newest first |
| GET | `/api/v1/ingestion/jobs/{job_id}` | One job: status, attempts, timings, element count, warnings, error code |
| GET | `/api/v1/ingestion/status` | Workers, accepting, running job IDs, counts per status |

IDs are validated by pattern (`doc_` + 16 hex, `job_` + 12 hex), so `../` can never reach the filesystem. Asset names allow `[A-Za-z0-9._-]` only and can't start with a dot.

## 4. Upload pipeline (in the request) and job (in the background)
```
POST /documents
  Content-Length > limit + 64 KiB?  → 413 (before reading the body)
  safe_filename(): basename, no control chars, ≤150 chars  → 422 if nothing usable is left
  extension allow-list                                     → 415
  copy to <data_dir>/uploads/<job_id>/ with a running size cap → 413
  validate_file (magic bytes, OOXML layout, NUL bytes)     → 415 / 422
  sha256 → doc_id ── completed already & !force → 200 duplicate (upload removed)
                  └─ job queued/running for doc_id → 202 with *that* job
  JobManager.submit → jobs/<job_id>.json (queued)          → 202 + Location (or 503 queue full)

worker: queued → running (attempts+1, persisted) → pipeline.ingest in a thread
        → completed | failed{code: extraction_failed | invalid_file | internal_error}
        → upload copy deleted (the original now lives in documents/<doc_id>/)
        queue empty → pipeline.close() (unload the vision model)
start-up: queued/running on disk → queued again (attempts < 3 and upload present) else failed{interrupted}
```

## 5. File map
| File | Purpose |
|---|---|
| `src/omnirag/ingestion/jobs.py` | `IngestionJob`, `JobManager` (queue, worker, persistence, recovery) |
| `src/omnirag/api/routers/documents.py` | Upload, browse, download, re-ingest, delete; `safe_filename` |
| `src/omnirag/api/routers/ingestion.py` | Job list/detail, worker status |
| `src/omnirag/schemas/documents.py` | `DocumentSummary`, pages, `UploadResponse` |
| `src/omnirag/api/main.py` | Lifespan creates the store, pipeline and job manager, and stops them |
| `src/omnirag/core/errors.py` | + 413 `payload_too_large`, 409 `conflict`, 503 `ingestion_queue_full` |
| `tests/unit/test_jobs.py` | 8 tests with a fake pipeline (failures, back-pressure, recovery, shutdown) |
| `tests/unit/test_documents_api.py` | 20 tests through the real lifespan and real background worker |

New dependency: `python-multipart==0.0.32` (FastAPI needs it for `UploadFile`). New settings: `OMNIRAG_INGESTION_WORKERS`, `OMNIRAG_INGESTION_QUEUE_MAX`, `OMNIRAG_INGESTION_SHUTDOWN_GRACE_S`.

## 6. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run uvicorn omnirag.api.main:app --no-access-log      # don't use --workers N (one queue per process)
# second terminal:
curl -s -F "file=@data/samples/aurora_safety_review.pdf" localhost:8000/api/v1/documents | jq
curl -s localhost:8000/api/v1/ingestion/jobs/<job_id> | jq '{status, element_count, duration_ms}'
curl -s "localhost:8000/api/v1/documents/<doc_id>/elements?type=table_row&limit=3" | jq '.items[].content'
```
Or use the interactive docs at http://localhost:8000/docs ("Try it out" on `POST /documents`).

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                          # 176 unit tests (~10 s), no services needed
uv run pytest -m "integration or docling"
```

## 8. Expected outputs (captured 2026-11-05, fresh data dir, real models, one worker)
All 12 samples uploaded: every response was `202`, and the queue drained in **27.3 s**:
```text
completed  11612.8 ms   9 elems  e7_technical_note.pdf          (Docling)
completed   7686.5 ms  26 elems  aurora_q1_review.pptx          (first vision call: cold load)
completed   2844.8 ms  20 elems  aurora_safety_review.pdf       (+ description)
completed   2234.8 ms   6 elems  whiteboard_2026-03-20.jpg      (OCR + description)
completed    780.4 ms   5 elems  aurora_lab_notebook_scan.pdf   (OCR, description skipped)
completed      1.7 ms   6 elems  meeting_notes_2026-03-12.txt
```
- Re-uploading `cell_test_results.csv` as `renamed.csv` returned `200 {"duplicate": true, "document": "cell_test_results.csv"}`.
- A 101 MB upload got `413 payload_too_large` from the `Content-Length` pre-check.
- The HTML original came back as `content-type: application/octet-stream`, `content-disposition: attachment; filename="helios_aurora_s1.html"`, with `x-content-type-options: nosniff`.
- `?origin=generated` found exactly one description in each of the 4 documents with described images.

**Crash test:** `POST /documents/{e7}/reingest`, wait for `running`, then `kill -9` the server. The job file said `{"status":"running","attempts":1}` and the upload copy was still there. After restart: `{"status":"completed","attempts":2,"element_count":9}`. A graceful `SIGTERM` logged `jobs_stopped` and `shutdown`; afterwards the vision model was unloaded (GPU back to 1028 MiB) and `uploads/` was empty.

## 9. Bugs and surprises found during verification
1. **My filename test was wrong, not the code.** httpx percent-encodes control characters in multipart filenames, so the server got `%00evil notes.txt` (harmless text), not a NUL byte. Control-character removal is now tested directly on `safe_filename`.
2. **A Phase 1 test pinned the exact set of OpenAPI routes.** It now checks what it meant: every public route is under `/api/v1/` and the root `/health` alias stays hidden.
3. **Mutation checks:** with the copy-time size check removed, the oversized-file test failed. The file was *still* rejected, as 422 by `validate_file`'s own size limit, which is defence in depth working, but the status code was wrong. With the download headers removed, the browse/download test failed.
4. During the live run, two `jq: parse error` lines appeared at the start of my script's output, and its first lines were garbled. Re-running those uploads separately gave valid JSON (HTTP 202) and both jobs had completed, so the API wasn't the cause. I couldn't reproduce it and don't claim a root cause.

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `Form data requires "python-multipart"` | Dependency missing | `uv sync` |
| `413 payload_too_large` | File above `OMNIRAG_MAX_UPLOAD_MB` (100) | Raise the setting, or split the file |
| `503 ingestion_queue_full` | More than `OMNIRAG_INGESTION_QUEUE_MAX` jobs waiting | Retry later; check `/ingestion/status` |
| `409 conflict` on delete | A job for that document is queued or running | Wait for the job in `details.job_id` |
| Job `failed` with `interrupted` | The server died 3 times while processing that file | Check the logs for OOM; try `pdf_engine=pymupdf` or vision off for that file |
| Jobs processed twice or out of order | API started with `--workers N` | Run a single API process (ADR-0007) |
| A job stays `queued` forever | Worker crashed (`workers_alive: 0` in `/ingestion/status`) | Restart the API; jobs are re-queued from disk |

## 11. Exercise (≈45 min)
Add **job cancellation**: `DELETE /api/v1/ingestion/jobs/{job_id}`.
1. A `queued` job becomes a new status `cancelled` (persisted), and its upload directory is removed. The worker already skips jobs that aren't `queued`.
2. A `running` job can't be interrupted safely (it's inside a thread): return 409 with a clear message.
3. A finished job: 409 (`already completed` / `already failed`).

Acceptance: unit tests in `test_jobs.py` (use the `gate` pattern so a second job is still queued), the job never reaches `pipeline.ingest`, the restart-recovery test still passes (cancelled jobs are not re-queued), and ruff is clean.

## 12. Acceptance criteria — Phase 2e (and Phase 2 overall)
- [x] Upload endpoint with validation before queueing (413/415/422), duplicate detection (200) and idempotent submits
- [x] Background worker with persistent jobs, back-pressure (503), restart recovery (verified with `kill -9`) and graceful shutdown
- [x] Document browse/filter/download endpoints; originals served as safe downloads; assets inline with strict names
- [x] Re-ingest and delete (409 while busy)
- [x] 176 unit tests + 6 live tests pass; ruff clean
- [x] **Phase 2:** 10 file types, OCR, layout engine, image descriptions, originals preserved, and every element carrying provenance, confidence and origin
