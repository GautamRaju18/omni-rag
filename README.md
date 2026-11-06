# OmniRAG

**Multimodal Graph-Based Retrieval Intelligence**: a local-first RAG platform. It ingests heterogeneous documents (PDF, scans, Office files, images, HTML, JSON), builds a Neo4j knowledge graph with vector and full-text indexes, runs dense + BM25 + graph hybrid retrieval with reranking, and generates cited, grounded answers with local LLMs through Ollama. Every stage is measured by an evaluation harness.

> 🚧 Work in progress, built in 16 tested phases. Current status: **Phase 9b complete** (multimodal ingestion with upload API; 8 chunking strategies; versioned bge-m3 embeddings; Neo4j knowledge graph with vector + full-text indexes, document versioning, and LLM-extracted entities/relations validated against the source; dense, identifier-aware BM25 and graph retrieval with metadata filters, fused by RRF and optionally reranked by a BGE cross-encoder, benchmarked against exact baselines; a LangGraph retrieval workflow with a query router, a bounded search agent, query transforms and a context check; a Streamlit dashboard to run and inspect every step). See [docs/PROGRESS.md](docs/PROGRESS.md).

## Target hardware
Built for and measured on a 16 GB RAM laptop with a 6 GB RTX 3060 (Windows 11 + WSL2). Memory and VRAM budgets are documented in [docs/learning/phase-00-environment.md](docs/learning/phase-00-environment.md).

## Architecture (current)
```
Documents ─► validate ─► loader (TXT/MD/HTML/JSON/CSV/DOCX/XLSX/PPTX/PDF/image)
                            PDF: PyMuPDF │ Docling layout model (multi-column, auto) │ Tesseract OCR (scans)
                         ─► enrich: image descriptions (qwen3-vl:2b via Ollama, origin=generated)
                         ─► elements + originals + assets  (~/omnirag-data)
                         ─► chunk: fixed │ recursive │ structure │ table │ multimodal │
                                   hierarchical │ parent-child   (bge-m3 tokens, element provenance)
                         ─► embed: bge-m3 via Ollama (versioned space, cached, embedding_ref per chunk)

FastAPI (WSL2, uvicorn) ──Bolt──► Neo4j 2026.10 (Docker, ≤2 GB)
        └──────HTTP──────► Ollama (native Windows, GPU)
                              ├─ qwen3:4b-instruct     (generation)
                              ├─ bge-m3                (embeddings, 1024-d)
                              └─ qwen3-vl:2b-instruct  (image descriptions, loaded per batch)
```

## Tech stack
Python 3.11 · uv · FastAPI · Pydantic v2 · structlog · Neo4j (async driver) · Ollama · httpx · PyMuPDF · Docling (CPU) · Tesseract · python-docx/openpyxl/python-pptx · pytest · Ruff · Docker Compose. More components (LangGraph, LangSmith, Streamlit …) are added in their phases.

## Quick start
Prerequisites: WSL2 Ubuntu, Docker Desktop with WSL integration, Ollama for Windows, `uv`, and `sudo apt install tesseract-ocr`.

```bash
# PowerShell (Windows): models
ollama pull qwen3:4b-instruct
ollama pull bge-m3
ollama pull qwen3-vl:2b-instruct
```

```bash
# Ubuntu (WSL)
cd /mnt/p/omni-rag
cp .env.example .env               # set a real password in both NEO4J_PASSWORD and OMNIRAG_NEO4J_PASSWORD
docker compose up -d               # Neo4j
uv sync                            # install from uv.lock (~1.6 GB venv incl. CPU-only torch for Docling)
# uv sync --no-group layout        # slim install without Docling (PDFs then always use PyMuPDF)
uv run python -m omnirag.ingestion ingest data/samples   # first Docling use downloads ~506 MB of models
uv run uvicorn omnirag.api.main:app --reload --no-access-log   # one process: the job queue lives in it
curl -s localhost:8000/api/v1/health
```
API docs: http://localhost:8000/docs · Neo4j Browser: http://localhost:7474

## Environment variables
See [.env.example](.env.example). All application settings use the `OMNIRAG_` prefix and are validated at startup by [config.py](src/omnirag/core/config.py).

## API (so far)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Aggregate status (200 ok/degraded, 503 if a dependency is down) |
| GET | `/api/v1/health/ollama` | Ollama version, required/missing models, loaded models + VRAM |
| GET | `/api/v1/health/neo4j` | Neo4j connectivity, version, edition |
| POST | `/api/v1/documents` | Upload a file → 202 + background job (200 if duplicate) |
| GET | `/api/v1/documents` | List documents (filter by status/kind) |
| GET | `/api/v1/documents/{id}` · `/elements` · `/original` · `/assets/{name}` | Record, extracted elements, original file, extracted images |
| POST | `/api/v1/documents/{id}/reingest` | Process the stored original again |
| DELETE | `/api/v1/documents/{id}` | Delete (409 while being ingested) |
| GET | `/api/v1/ingestion/jobs` · `/jobs/{id}` · `/status` | Job list, one job (poll after upload), worker/queue state |
| POST | `/api/v1/retrieval/search` | One retrieval mode: dense, bm25, graph, their RRF fusions, `+rerank` |
| POST | `/api/v1/retrieval/advanced` | LangGraph workflow: router, transforms, agent, context check, retries; returns the per-node trace and every ranked list |
| GET | `/api/v1/graph/stats` · `/graph/entities` | Graph size; most-mentioned entities |

Upload example: `curl -F "file=@data/samples/aurora_safety_review.pdf" localhost:8000/api/v1/documents`

## Dashboard
A Streamlit app to try every retrieval feature and watch each step (it talks to the API only):
```bash
uv run uvicorn omnirag.api.main:app --no-access-log          # terminal 1
uv run streamlit run src/omnirag/frontend/app.py             # terminal 2 → http://localhost:8501
```
- **Overview**: health of Neo4j and Ollama, what is in the graph, the system's stages.
- **Retrieval lab**: one question through several modes side by side: where each chunk ranks in each mode, overlap, latency.
- **Workflow inspector**: every Phase 9 option (router, self-query, transforms, agent, rerank, context check, retries, per-document cap); shows the path through the LangGraph graph, a timeline, what each node did, every ranked list and how RRF built the final context.
- **Benchmarks**: the measured results in `docs/benchmarks/` (read, never recomputed).

Streamlit telemetry is off (`.streamlit/config.toml`).

## Testing
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                  # unit tests (no services needed, ~7 s)
uv run pytest -m integration   # against live Neo4j + Ollama (incl. the vision model)
uv run pytest -m docling       # real Docling layout models (~1 min first start-up)
```

## Documentation
- Progress and measured baselines: [docs/PROGRESS.md](docs/PROGRESS.md)
- Learning notes per phase: [docs/learning/](docs/learning/)
- Architecture decisions: [docs/adr/](docs/adr/)
