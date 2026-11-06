# OmniRAG

**Multimodal Graph-Based Retrieval Intelligence**: a local-first RAG platform.
- It ingests heterogeneous documents (PDF, scans, Office files, images, HTML, JSON, CSV) and builds a Neo4j knowledge graph with vector and full-text indexes and validated entities.
- It retrieves with dense + BM25 + graph hybrid search, fused and optionally reranked, through a bounded LangGraph workflow.
- It answers with a local LLM (Ollama), and **every claim carries a quote that code checks against the cited source**.
- A 110-question evaluation harness measures every stage.

Built in 16 tested phases ([docs/PROGRESS.md](docs/PROGRESS.md)), with an architecture decision record per design choice ([docs/adr/](docs/adr/)) and a learning guide per phase ([docs/learning/](docs/learning/)).

## Highlights (measured; sources in docs/benchmarks)
- **Answers you can check.**
  - The model must return a verbatim quote per claim; code verifies quotes and numbers in the cited block, corrects wrong source numbers, and removes unsupported claims.
  - Questions about things no document mentions are refused without a model call (26 ms).
  - ([ADR-0019](docs/adr/0019-grounded-generation-and-citations.md))
- **110-question evaluation** (8 question types, 10 unanswerable), dense + BM25 configuration (E3): Recall@5 0.950, context recall 0.980, answer correctness 0.900, quote-verified faithfulness 0.853, 9/10 unanswerable refused. Reranking (E5) reaches 0.932 correct on the 76 questions it covered, at 4.6× the latency on CPU. Every failure is attributed to the pipeline stage that lost it. ([ADR-0020](docs/adr/0020-evaluation-framework.md))
- **Every component kept or dropped by measurement**, with per-question wins and losses:
  - kept: identifier-aware BM25, RRF fusion, a rule router over an LLM router, a grounded search agent for multi-hop questions, a token budget over a fixed top-k;
  - dropped: graph retrieval and query rewriting as defaults, because they didn't help on this corpus.
- **Runs on a laptop:** 6 GB GPU (qwen3:4b-instruct + bge-m3 on Ollama), 8 GB WSL VM, everything local. LangSmith is opt-in and redacting.

## Architecture
```
Documents ─► validate ─► loaders (TXT/MD/HTML/JSON/CSV/DOCX/XLSX/PPTX/PDF/image)
             PDF: PyMuPDF │ Docling layout model │ Tesseract OCR      images: qwen3-vl:2b descriptions (origin=generated)
          ─► elements + originals ─► chunks (bge-m3 tokens, element provenance) ─► bge-m3 embeddings (versioned, cached)
          ─► Neo4j 2026: Document/Section/Page/Chunk/Table/Figure, vector + full-text indexes,
             Entity/RELATED_TO extracted by the LLM and validated against the text, versions

Question ─► LangGraph workflow (bounded):  route ─► [transform] ─► retrieve ⇄ agent ─► [rerank] ─► check ─► [correct] ─► assemble
             retrieve = dense (vector) + BM25 (identifier-aware Lucene) [+ graph], fused by RRF
             assemble = 1,024-token context of numbered sources (optional expansion / extractive compression)
          ─► answer: qwen3:4b-instruct, structured claims + quotes ─► citation validation ─► SQLite trace
          ─► FastAPI (/chat, /chat/stream SSE) ─► Streamlit dashboard
```

## Tech stack
Python 3.11 · uv · FastAPI · Pydantic v2 · structlog · Neo4j 2026.10 (vector + full-text, Cypher 25) · Ollama (qwen3:4b-instruct, bge-m3, qwen3-vl:2b) · LangGraph · bge-reranker-v2-m3 (torch CPU) · PyMuPDF · Docling · Tesseract · python-docx / openpyxl / python-pptx · SQLite · Streamlit + Altair · LangSmith (opt-in) · pytest · Ruff · Docker Compose.

## Quick start
Prerequisites: WSL2 Ubuntu, Docker Desktop with WSL integration, Ollama for Windows, `uv`, and `sudo apt install tesseract-ocr`.
```bash
# PowerShell (Windows): models
ollama pull qwen3:4b-instruct
ollama pull bge-m3
ollama pull qwen3-vl:2b-instruct
```
**Option A: containers** (API + dashboard + Neo4j; Ollama stays on the host)
```bash
cd /mnt/p/omni-rag
cp .env.example .env                              # set NEO4J_PASSWORD and OMNIRAG_NEO4J_PASSWORD
docker compose --profile app up -d --build        # first build ~11 min → http://localhost:8501
```
**Option B: local development**
```bash
cp .env.example .env
docker compose up -d                              # Neo4j only
uv sync                                           # from uv.lock (CPU-only torch)
uv run python -m omnirag.ingestion ingest data/samples
uv run python -m omnirag.chunking rechunk --all && uv run python -m omnirag.embeddings embed --all
uv run python -m omnirag.knowledge_base init && uv run python -m omnirag.knowledge_base load --all
uv run python -m omnirag.knowledge_base extract --all   # entities + relations (optional, LLM)
uv run uvicorn omnirag.api.main:app --no-access-log                 # http://localhost:8000/docs
uv run streamlit run src/omnirag/frontend/app.py                    # http://localhost:8501
```
Ask from the command line:
```bash
curl -s localhost:8000/api/v1/chat -H 'content-type: application/json' \
     -d '{"question": "Why did cell AUR-C103 fail?"}' | jq '.answer | {text, grounded, citations}'
```

## Dashboard
| Group | Page | What it does |
|---|---|---|
| Use | **Ask** | Chat with streaming claims, numbered citations (file › section · page, the exact passage), refusals explained, "how this answer was checked" |
| Use | **Documents** | Library with preview (original + extracted elements), upload, re-ingest, delete; ingestion status |
| Use | **Graph explorer** | Documents ↔ entities overview; an entity's relations (stated solid, inferred dashed) and mentions; a document's chunks and entities |
| Inspect | **Overview** | Health of API, Neo4j, Ollama; system diagram; graph contents |
| Inspect | **Retrieval lab** | One question through every retrieval mode side by side |
| Inspect | **Workflow inspector** | Every workflow option; the path through the graph, timeline, ranked lists, RRF, the generator's context |
| Measure | **Evaluation** | E1–E7 reports compared, any metric charted, failure root causes, per-question results; start a run |
| Measure | **Performance** | Live latency per component, answer quality signals, failure flags; evaluation latency, memory, indexing |
| Measure | **Benchmarks** | Every measured benchmark file |

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` · `/api/v1/health` · `/health/ollama` · `/health/neo4j` | Liveness; aggregate and per-component health |
| POST | `/api/v1/documents/upload` (also `/documents`) | Upload a file → background ingestion job (duplicates detected) |
| GET / DELETE | `/api/v1/documents` · `/documents/{id}` (+ `/elements`, `/original`, `/assets/{name}`) | Library, record, elements, original file; delete |
| POST | `/api/v1/ingestion/start` | (Re)ingest a stored document |
| GET | `/api/v1/ingestion/{job_id}/status` · `/ingestion/jobs` · `/ingestion/status` | Job status, job list, worker and queue state |
| POST | `/api/v1/retrieval/search` · `/retrieval/advanced` | One retrieval mode; the LangGraph workflow with its full trace and context |
| POST | `/api/v1/chat` · `/chat/stream` | Grounded answer with verified citations; Server-Sent Events (claims as they're written) |
| GET / DELETE | `/api/v1/chat/history` · `/chat/history/{id}` · `/chat/turns/{id}` | Conversations and full query traces |
| POST / GET | `/api/v1/evaluate` · `/evaluate/{run_id}` · `/evaluate/experiments` | Start an evaluation run (background), results |
| GET | `/api/v1/analytics` | Answer quality, latency per component, failure flags, latest evaluation |
| GET | `/api/v1/graph/stats` · `/graph/entities` · `/graph/entity` · `/graph/neighborhood` | Graph size, entities, one entity's evidence, explorer data |

## Evaluation
```bash
uv run python -m omnirag.evaluation.rag_eval --experiments E3,E7 --limit 20 --out /tmp/quick.json
uv run python -m omnirag.evaluation.rag_eval --indexing --out docs/benchmarks/rag-eval-<date>.json   # all E1-E7
```
The 2026-11-06 report ([JSON](docs/benchmarks/rag-eval-2026-11-06.json)) covers:
- E1–E4 complete;
- E5 on the first 76 questions (the run was stopped early);
- E6–E7 not yet measured.

Results and per-question analysis are in [ADR-0020](docs/adr/0020-evaluation-framework.md).

## Testing
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                  # 419 unit tests, no services needed (~1 min)
uv run pytest -m integration   # 37 tests against live Neo4j + Ollama (needs: docker compose --profile test up -d neo4j-test)
uv run pytest -m docling       # real Docling layout models
```

## Configuration and privacy
- All settings use the `OMNIRAG_` prefix ([.env.example](.env.example)), validated by [config.py](src/omnirag/core/config.py). Secrets only come from `.env` (git-ignored, never in the image).
- Everything runs locally.
- LangSmith tracing needs both `OMNIRAG_LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`, and hides inputs/outputs by default. Evaluation uploads send questions and scores, not documents.
- Cypher is always parameterized; user text never reaches Lucene unescaped; uploads are validated; originals are served as attachments with `nosniff`.

## Documentation
- [docs/PROGRESS.md](docs/PROGRESS.md): phases, checklist, measured baselines
- [docs/adr/](docs/adr/): 21 architecture decision records
- [docs/learning/](docs/learning/): per-phase guides (concepts, walkthrough, how to run, real outputs, common errors, exercise, acceptance criteria)
- [docs/RESUME.md](docs/RESUME.md): portfolio summary and interview notes
- [CLAUDE.md](CLAUDE.md): working invariants for the codebase
