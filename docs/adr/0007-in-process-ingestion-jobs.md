# ADR-0007: In-process background ingestion jobs with persistent job records

**Status:** accepted (2026-11-05, Phase 2e)

## Context
Ingestion takes from milliseconds (a CSV) to more than 10 s (Docling, vision descriptions), so an upload request can't wait for it. This is a single-user system on one laptop. The heavy steps (OCR, Docling, the vision model) all share one CPU and one 6 GB GPU.

## Decision
- `POST /documents` validates the upload (extension, size, magic bytes, ZIP layout), hashes it, and returns **202 + a job**, or **200** if the same bytes were already ingested. The job runs in an **in-process worker** (`asyncio` queue, ingestion in `asyncio.to_thread`), and clients poll `GET /ingestion/jobs/{id}`.
- **One worker by default** (`OMNIRAG_INGESTION_WORKERS`, max 4). Parallel jobs would compete for the same CPU/GPU.
- **Job records are files** (`<data_dir>/jobs/<job_id>.json`, atomic replace). On start-up, `queued`/`running` jobs are queued again: ingestion is idempotent (content-addressed IDs). After `max_attempts=3` a job is marked `failed: interrupted`, so a file that crashes the process can't crash-loop it.
- **Back-pressure:** at most `OMNIRAG_INGESTION_QUEUE_MAX` queued jobs, then 503 `ingestion_queue_full`.
- **Idempotent submits:** uploading bytes whose document is already queued or running returns that job, instead of a second one. Deleting such a document is 409.
- When the queue drains, `pipeline.close()` unloads the vision model.

## Verified (2026-11-05)
All 12 samples uploaded over HTTP, done in 27 s with one worker. A `SIGKILL` during a 10 s Docling job left it `running` on disk with its upload kept; the restarted server re-queued it, and it completed on attempt 2. A graceful `SIGTERM` waits up to `OMNIRAG_INGESTION_SHUTDOWN_GRACE_S`.

## Consequences
- No extra service to run. But the job queue lives in the API process: running **two API processes** (e.g. `uvicorn --workers 2`) would give two independent queues writing the same data dir. Run one process; scale-out means moving to a broker (below).
- The CLI and the API can still write the same data dir at the same time. Avoid ingesting the same file through both at once.
- Starlette parses the whole multipart body (spooled to a temp file) before our handler runs. The `Content-Length` pre-check rejects honest oversized uploads early, but a client that lies about it is only stopped by the size check while copying. That's acceptable for a local service; a public deployment needs a size limit at the reverse proxy.

## Alternatives considered
| Option | Why not (now) |
|---|---|
| FastAPI `BackgroundTasks` | No queue limit, no status, no persistence, no restart recovery, no concurrency limit |
| Celery / RQ + Redis | A broker service for a single-user laptop app; worth it only with several machines or API processes |
| arq (asyncio + Redis) | Same broker cost; a good fit if we scale out later |
| Run ingestion synchronously in the request | Requests of more than 10 s, client timeouts, and a blocked worker thread |
