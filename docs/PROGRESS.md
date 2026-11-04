# OmniRAG — Implementation progress

A feature counts as done only when its tests pass and the result was observed, not merely when code exists.

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| 0 | Hardware & environment | ✅ Done 2026-11-04 | [learning/phase-00](learning/phase-00-environment.md) |
| 1 | Backend foundation (config, logging, DI, health) | ✅ Done 2026-11-04 | 24 unit + 4 integration tests; live checks in [learning/phase-01](learning/phase-01-backend-foundation.md) |
| 2a | Ingestion core + TXT/MD/HTML/JSON/CSV | ✅ Done 2026-11-04 | 83 unit tests; [learning/phase-02a](learning/phase-02a-ingestion-core.md) |
| 2b | DOCX / XLSX / PPTX loaders | ⏭ Next | |
| 2c | PDF (PyMuPDF) + Tesseract OCR + images | ⬜ | needs `sudo apt install tesseract-ocr` |
| 2d | Docling + vision-model descriptions | ⬜ | |
| 2e | `/documents` + `/ingestion` API, background jobs | ⬜ | |
| 3 | Chunking engine | ⬜ | |
| 4 | Embedding engine | ⬜ | |
| 5 | Neo4j graph construction | ⬜ | |
| 6 | Dense + sparse retrieval | ⬜ | |
| 7 | Graph-enhanced hybrid retrieval (RRF) | ⬜ | |
| 8 | Reranking | ⬜ | |
| 9 | Advanced RAG + LangGraph | ⬜ | |
| 10 | Ollama generation | ⬜ | |
| 11 | Citations + answer validation | ⬜ | |
| 12 | LangSmith evaluation | ⬜ | |
| 13 | Failure analysis + observability | ⬜ | |
| 14 | Frontend (Streamlit) | ⬜ | |
| 15 | Dockerization + deployment | ⬜ | |
| 16 | Tests, docs, resume prep | ⬜ | |

## Success-criteria tracker (from the brief)
- [ ] All document types ingest (5/10 kinds: txt, md, html, json, csv) · [ ] OCR on scans/images · [ ] Table/figure provenance (tables ✅ for text formats)
- [ ] Multiple chunking strategies · [ ] Embeddings in Neo4j · [ ] Graph queryable
- [ ] Dense · [ ] BM25 · [ ] Graph retrieval · [ ] RRF · [ ] Reranking
- [ ] Grounded answers · [ ] Verifiable citations · [ ] LangSmith tracing
- [ ] Retrieval eval · [ ] Faithfulness eval · [ ] Reproducible benchmarks
- [ ] FastAPI endpoints (health ✅; documents/ingestion/chat/evaluate/analytics pending)
- [ ] Frontend upload + QA · [x] Tests pass (Phase 1 scope) · [ ] Docker deploy · [ ] Docs complete

## Decisions log
- [ADR-0001](adr/0001-src-layout.md): `src/omnirag` package layout
- [ADR-0002](adr/0002-ollama-native-windows.md): Ollama native on Windows, mirrored WSL networking
- [ADR-0003](adr/0003-neo4j-primary-store.md): Neo4j for graph + vector + full-text

## Measured baselines (2026-11-04)
| Metric | Value |
|---|---|
| qwen3:4b-instruct VRAM (8k ctx, q8 KV) | 3.08 GB, 100% GPU |
| bge-m3 VRAM | 0.64 GB, 100% GPU |
| GPU total used with both loaded | 4989 / 6144 MiB |
| qwen3:4b-instruct warm generation | ~55 tok/s |
| qwen3:4b-instruct cold load | ~22 s |
| bge-m3 output | 1024-dim, L2 norm 1.0 |
| Neo4j idle RSS | 940 MiB / 2 GiB limit |
| `/api/v1/health` latency (both probes, concurrent) | ~50 ms |

## Phase 2 prerequisites (to be done at the start of Phase 2)
- `sudo apt install tesseract-ocr` (needs your sudo password) plus `jq` (optional, for nicer CLI JSON)
- Select and measure a vision model (3–4B VL) against the VRAM headroom
- Add Docling/PyMuPDF and measure their RAM inside the 8 GB WSL cap
