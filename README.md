# OmniRAG

**Multimodal Graph-Based Retrieval Intelligence**: a local-first RAG platform. It ingests heterogeneous documents (PDF, scans, Office files, images, HTML, JSON), builds a Neo4j knowledge graph with vector and full-text indexes, runs dense + BM25 + graph hybrid retrieval with reranking, and generates cited, grounded answers with local LLMs through Ollama. Every stage is measured by an evaluation harness.

> 🚧 Work in progress, built in 16 tested phases. Current status: **Phase 2d complete** (multimodal ingestion: 10 file types, OCR, layout model, image descriptions). See [docs/PROGRESS.md](docs/PROGRESS.md).

## Target hardware
Built for and measured on a 16 GB RAM laptop with a 6 GB RTX 3060 (Windows 11 + WSL2). Memory and VRAM budgets are documented in [docs/learning/phase-00-environment.md](docs/learning/phase-00-environment.md).

## Architecture (current)
```
Documents ─► validate ─► loader (TXT/MD/HTML/JSON/CSV/DOCX/XLSX/PPTX/PDF/image)
                            PDF: PyMuPDF │ Docling layout model (multi-column, auto) │ Tesseract OCR (scans)
                         ─► enrich: image descriptions (qwen3-vl:2b via Ollama, origin=generated)
                         ─► elements + originals + assets  (~/omnirag-data)

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
uv run uvicorn omnirag.api.main:app --reload --no-access-log
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
