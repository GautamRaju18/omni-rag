# Phase 16: Testing, documentation, and what the project shows

## 1. The test pyramid as built
| Layer | Count | What it proves | Command |
|---|---|---|---|
| Unit | 419 | Every module with fakes (no services): loaders, chunking, embeddings, graph writes, retrieval, fusion, workflow, generation, citations, evaluation, API, every dashboard page (headless AppTest) | `uv run pytest` |
| Integration | 37 | Against live Neo4j (a throwaway instance on 7688) and Ollama: ingestion, graph, retrieval, the workflow, grounded answers, refusals, streaming, every evaluation quote in the live store | `uv run pytest -m integration` |
| Docling | marker | The real layout model on the two-column PDF | `uv run pytest -m docling` |
| Evaluation | 110 questions × E1–E7 | Measured quality and latency (not pass/fail) | `python -m omnirag.evaluation.rag_eval` |
| Deployment | manual, recorded | Containers healthy, chat through the container, no secrets in the image | ADR-0021 |

Rules the tests enforce, so later changes can't silently break them:
- every evidence quote exists in its file;
- reference answers pass their own answer terms;
- graph DOT diagrams never use DOT keywords as node IDs;
- the openapi route list matches the brief's endpoints;
- the worst-case workflow stays within its recursion bound;
- `defusedxml` stays installed.

## 2. Documentation map
- **README:** what it is, the measured highlights, quick start (containers or local), every API route, the dashboard.
- **PROGRESS:** each phase, its evidence and its measured result; the success-criteria checklist from the brief; the decisions log.
- **21 ADRs:** one per design decision, each with context, the decision, *measured* consequences and what's open.
- **Learning guides:** one per phase, each with the problem, concepts, architecture, walkthrough, how to run, real outputs, surprises, common errors, an exercise and acceptance criteria.
- **CLAUDE.md:** the invariants a future change must keep (IDs, grounding rules, citation checks, evaluation hygiene).
- **RESUME.md:** bullets, numbers to quote, interview talking points.

## 3. What is measured vs what is open (honest status)
- **Measured and complete:**
  - ingestion of 10 formats, chunking, embeddings, graph, retrieval;
  - fusion, reranking, workflow, context assembly;
  - generation and citations;
  - E1–E4 on all 110 questions; deployment.
- **Partial:**
  - E5 covers 76 of the 110 questions, and E6/E7 were not run: the evaluation was stopped early by request. `rag_eval --out docs/benchmarks/rag-eval-2026-11-06.json` resumes from the stored records.
  - LangSmith upload and tracing are implemented and tested with a fake client, but have not been run against the hosted service (no API key on this machine).
- **Known limits:**
  - one author wrote the dataset;
  - the judge is the generator's own 4B model;
  - quote checks can't catch a real quote attached to the wrong entity (q33);
  - the reranker runs on CPU only;
  - the API is a single process.

## 4. Final checks run (2026-11-06)
```
uv run ruff format --check . && uv run ruff check .   → clean
uv run pytest                                          → 419 passed
uv run pytest -m integration                           → 37 passed
docker compose --profile app up -d --build             → api, dashboard, neo4j healthy
```

## 5. Exercise: make it yours (≈2 h)
1. Ingest a folder of your own PDFs through the Documents page, then ask questions on the Ask page. Which answers come back `partial`, and why?
2. Write 10 questions about your documents in the dataset format, run `rag_eval --experiments E3,E7 --dataset your.json`, and read the root causes.
3. Finish the evaluation (`rag_eval --out docs/benchmarks/rag-eval-2026-11-06.json`), then compare E7 with E3: does the router + agent + check pay for its latency on your questions?

## 6. Acceptance criteria: Phase 16
- [ ] A fresh clone builds and runs from the README alone (containers or local).
- [ ] `uv run pytest` and `uv run pytest -m integration` pass.
- [ ] Every number in README, PROGRESS, ADRs and RESUME links to a file in `docs/benchmarks/` or a learning guide's recorded output.
