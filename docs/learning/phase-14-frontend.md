# Phase 14: The Streamlit dashboard: every screen the brief asks for

> Started in Phase 9b (retrieval lab, workflow inspector, benchmarks); finished here. Code: `src/omnirag/frontend/` (`app.py` entry, `pages.py`, `screens.py`, `views.py`, `api_client.py`). Tests: `tests/unit/test_frontend.py`, `tests/unit/test_screens.py` (every page rendered headless with Streamlit's AppTest).

## 1. Problem being solved
The brief lists eleven screens, from upload to evaluation. Every one has to show **real** system state: measured numbers, stored traces, the graph as it is. The dashboard can't hold credentials or compute its own metrics. And because a user judges a RAG system by its answers, the chat screen has to make the checking visible: which words of which document back each claim.

## 2. The screens (brief → page)
| Brief | Page (sidebar group) | What it shows / does | API |
|---|---|---|---|
| 1 Dashboard | **Overview** (Inspect) | Health of API, Neo4j, Ollama; the system diagram; what is in the graph; documents | `/health`, `/graph/stats`, `/documents` |
| 2 Upload | **Documents → Upload** (Use) | Multi-file upload; duplicate detection; job IDs | `POST /documents/upload` |
| 3 Library | **Documents → Library** | Every document with kind, status, elements, warnings; re-ingest, delete (with a confirmation) | `/documents`, `/ingestion/start`, `DELETE /documents/{id}` |
| 4 Ingestion status | **Documents → Ingestion status** | Workers, queue counts, every job with duration and error | `/ingestion/status`, `/ingestion/jobs` |
| 5 Chat | **Ask** (Use) | Conversations, streaming claims, the answer with [n] markers, refusals explained | `POST /chat/stream`, `/chat/history` |
| 6 Source citations | **Ask** → "Sources" | Each cited block: file › section · pages, the exact passage the model read, the claims it supports, chunk IDs; "How this answer was checked" lists every claim's status | (in the turn record) |
| 7 Document preview | **Documents → Library** (and "Open the document" from a citation) | The original (image shown, text formats as text, download for all), its extracted elements by type, origin, page and confidence | `/documents/{id}`, `/original`, `/elements` |
| 8 Graph explorer | **Graph explorer** (Use) | Overview (documents ↔ entities with mention counts), one entity (relations, solid = stated, dashed = inferred; chunks that mention it), one document (chunks, sections, entities) | `/graph/neighborhood`, `/graph/entity`, `/graph/entities` |
| 9 Retrieval lab | **Retrieval lab** (Inspect) | One question through several modes; ranks per mode, overlap, latency | `/retrieval/search` |
| (9b) | **Workflow inspector** (Inspect) | Every Phase 9 option; the path through the LangGraph graph, timeline, ranked lists, RRF, the context | `/retrieval/advanced` |
| 10 Evaluation | **Evaluation** (Measure) | E1–E7 compared (retrieval, generation, judge, latency), any metric charted, failure root causes per experiment, per-question results; start a new run | `/evaluate` |
| 11 Performance | **Performance** (Measure) | Live traffic: answered / grounded / refused, latency p50/p95 per component, failure flags, verdicts, recent turns; the latest evaluation's per-component latency, memory, indexing time | `/analytics`, `/evaluate` |

## 3. Design rules (kept by tests)
- **API only.** `api_client.py` is the only network code; no Neo4j or Ollama client, no secrets.
- **Pure helpers are unit-tested.** `views.py` turns API JSON into rows and DOT (`citation_cards`, `claim_rows`, `eval_rows`, `failure_rows`, `per_question_rows`, `graph_dot`, …).
- **Graphviz safety.** Node IDs in DOT are generated (`n0`, `n1`, …), never user text, and labels are escaped. A node literally called `graph` once blanked a diagram (Phase 9b).
- **Colour carries one job.** Charts are single-series (one hue, the title names it). Claim status uses the fixed status palette, always with an icon *and* a word (`✅ verified`, `↪️ citation corrected`, `⚠️ not verbatim`, `⛔ unsupported (removed)`). The graph explorer uses categorical slots 1–3 for documents, entities and chunks, with a legend.
- **Nothing computed in the UI.** Evaluation and performance numbers are displayed as the API returns them.

## 4. How to run
```bash
uv run uvicorn omnirag.api.main:app --no-access-log        # terminal 1
uv run streamlit run src/omnirag/frontend/app.py           # terminal 2 → http://localhost:8501
# or everything in containers (Phase 15):
docker compose --profile app up -d --build                 # → http://localhost:8501
```

## 5. Common errors
| Symptom | Cause / fix |
|---|---|
| "API not reachable at http://localhost:8000" | Start uvicorn, or set `OMNIRAG_FRONTEND_API_URL`. |
| A page still shows old behaviour | Streamlit caches imported modules: restart `streamlit run`. |
| The Evaluation page says "No completed evaluation yet" | No report in `docs/benchmarks/rag-eval-*.json` (or `OMNIRAG_BENCHMARKS_DIR`) and no finished `POST /evaluate` run. |
| Graph explorer shows no entities | Extraction is off at ingestion: `python -m omnirag.knowledge_base extract`. |
| The chat answers slowly | It streams claims as they're written; the first one needs retrieval and the context check (1–3 s). A hot laptop GPU throttles (Performance page). |

## 6. Exercise (≈30 min)
1. Ask "Which electrolyte swelled less at 45 °C, E-5 + FEC or E-7?". Open both citations. Does each passage contain the number its claim states?
2. In the Graph explorer, open the entity "E-7". Which relations are dashed (inferred), and which document states the solid ones?
3. On the Evaluation page, chart "hallucination rate", then open E1's per-question table filtered to failures. Which root cause dominates?

## 7. Acceptance criteria: Phase 14
- [ ] All eleven screens open without errors (`uv run pytest tests/unit/test_screens.py tests/unit/test_frontend.py`).
- [ ] An uploaded file appears in Ingestion status, then in the Library with its elements.
- [ ] A chat answer's citation card shows a passage that contains the claim's quote.
- [ ] The Evaluation page shows the measured E1–E7 report; the Performance page shows your own turns' latency.
