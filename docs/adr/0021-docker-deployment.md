# ADR-0021: One image for API and dashboard; Ollama stays on the host; data is a bind mount

**Status:** accepted (2026-11-06, Phase 15)

## Context
The brief asks for a Dockerized deployment. The constraints are measured, not assumed:
- one 6 GB GPU, which Ollama (native Windows, ADR-0002) already owns;
- a WSL VM capped at 8 GB that Docker shares;
- an API that must stay **one process** (in-process ingestion jobs, ADR-0007; in-process evaluation runs, ADR-0020);
- ingested data on the WSL ext4 disk (`~/omnirag-data`), which the CLI also writes.

## Decision
- **One image, two commands** (`Dockerfile`, `omnirag:0.1.0`): the API (`uvicorn`, the default CMD) and the dashboard (`streamlit run`). Both are the same Python package, so one build and one lock serve both.
  - **Build stage:** `python:3.11.15-slim-bookworm` + `uv` 0.11.32 copied from `ghcr.io/astral-sh/uv:0.11.32`. It runs `uv sync --frozen --no-dev` from the committed lock (exact versions, CPU-only torch via the pinned index), with a dependency layer cached separately from the source.
  - **Runtime stage:** the same Python base, plus Tesseract (OCR), libGL/GLib (Docling) and curl (healthchecks). It runs as non-root user `omnirag` (uid 1000, the host user's uid, so the bind mount stays writable).
- **Compose profile `app`** (`docker compose --profile app up -d --build`):
  - `api` (port 8000) waits for Neo4j to be healthy;
  - `dashboard` (port 8501) waits for the API;
  - Neo4j and the test Neo4j are unchanged.
- **Ollama stays on the host:** `OMNIRAG_OLLAMA_BASE_URL=http://host.docker.internal:11434`, with `extra_hosts: host-gateway`. Containerising it would need GPU passthrough and would compete with the host's Ollama for the same 6 GB.
- **Data is the host's directory**, bind-mounted at `/data` (`OMNIRAG_HOST_DATA_DIR`, default `~/omnirag-data`): originals, elements, chunks, the embedding cache, the pinned tokenizer, chat history and evaluation runs. The container and the CLI see the same state. `docs/benchmarks` is mounted read-only, so new reports appear without a rebuild.
- **Secrets only at runtime**: `env_file: .env` (`OMNIRAG_NEO4J_PASSWORD`). `.env` is in `.dockerignore`, and nothing secret is in the image.
- **Memory limits** inside the 8 GB VM: neo4j 2g, api 3g (torch reranker + Docling when used), dashboard 768m.
- **Healthchecks:**
  - API liveness is `GET /openapi.json`. `/health` returns 503 when a dependency is down, and restarting the API wouldn't fix that.
  - Dashboard: `/_stcore/health`.

## Measured (2026-11-06, Docker Desktop 29.8.1 on WSL2)
| | Value |
|---|---|
| Image size | 3.56 GB (most of it torch CPU + Docling) |
| Cold build | ~11 min (the layer export took 306 s) |
| Start to healthy | api ~25 s; dashboard ~13 s after the API |
| `/api/v1/health` from the container | Neo4j ok (95 ms, `bolt://neo4j:7687`); Ollama ok (76 ms via `host.docker.internal`, 0.40.14, no missing models) |
| Memory at idle → after one chat | api 468 → 763 MiB; dashboard 60 MiB; neo4j 834 MiB → 1.04 GiB |
| Chat through the container | "Who prepared the safety review…": Lena Brandt [1], 1/1 claim verified, citation page 1 |
| Duplicate upload | detected (`duplicate: true`), no re-ingestion |

Latency measured during this test is not representative: the 110-question evaluation was using the same GPU at the same time.

## Consequences
- The image is large because it carries torch for the CPU reranker and Docling. A slim variant (`uv sync --no-group layout`) would drop reranking and layout PDFs, so it isn't the default.
- The first Docling use downloads its models into `/data/hf` (on the mount, so only once).
- Scaling out (several API replicas) needs the in-process job and evaluation queues to move to a shared queue, and SQLite to PostgreSQL. That's out of scope for a single-user local system (ADR-0007, ADR-0019).
- **Open:** a cloud deployment profile (managed Neo4j, a GPU host for Ollama); image signing.
