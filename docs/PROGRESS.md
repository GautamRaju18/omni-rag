# OmniRAG — Implementation progress

A feature counts as done only when its tests pass and the result was observed, not merely when code exists.

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| 0 | Hardware & environment | ✅ Done 2026-11-04 | [learning/phase-00](learning/phase-00-environment.md) |
| 1 | Backend foundation (config, logging, DI, health) | ✅ Done 2026-11-04 | 24 unit + 4 integration tests; live checks in [learning/phase-01](learning/phase-01-backend-foundation.md) |
| 2a | Ingestion core + TXT/MD/HTML/JSON/CSV | ✅ Done 2026-11-04 | 83 unit tests; [learning/phase-02a](learning/phase-02a-ingestion-core.md) |
| 2b | DOCX / XLSX / PPTX loaders | ✅ Done 2026-11-04 | 117 unit tests; [learning/phase-02b](learning/phase-02b-office-loaders.md) |
| 2c | PDF (PyMuPDF) + Tesseract OCR + images | ✅ Done 2026-11-04 | 136 unit tests; [learning/phase-02c](learning/phase-02c-pdf-ocr.md) |
| 2d | Docling + vision-model descriptions | ✅ Done 2026-11-04 | 147 unit + 5 integration + 1 docling tests; [learning/phase-02d](learning/phase-02d-layout-and-vision.md) |
| 2e | `/documents` + `/ingestion` API, background jobs | ✅ Done 2026-11-05 | 176 unit + 6 live tests; [learning/phase-02e](learning/phase-02e-ingestion-api.md) |
| 3 | Chunking engine | ⏭ Next | |
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
- [x] All document types ingest (10/10 kinds: txt, md, html, json, csv, docx, xlsx, pptx, pdf, image) · [x] OCR on scans/images (Tesseract, with confidence) · [x] Table/figure provenance (tables in all formats incl. borderless via Docling; PPTX chart data; image descriptions with origin=generated)
- [ ] Multiple chunking strategies · [ ] Embeddings in Neo4j · [ ] Graph queryable
- [ ] Dense · [ ] BM25 · [ ] Graph retrieval · [ ] RRF · [ ] Reranking
- [ ] Grounded answers · [ ] Verifiable citations · [ ] LangSmith tracing
- [ ] Retrieval eval · [ ] Faithfulness eval · [ ] Reproducible benchmarks
- [ ] FastAPI endpoints (health ✅, documents ✅, ingestion ✅; chat/evaluate/analytics pending)
- [ ] Frontend upload + QA · [x] Tests pass (Phase 1 scope) · [ ] Docker deploy · [ ] Docs complete

## Decisions log
- [ADR-0001](adr/0001-src-layout.md): `src/omnirag` package layout
- [ADR-0002](adr/0002-ollama-native-windows.md): Ollama native on Windows, mirrored WSL networking
- [ADR-0003](adr/0003-neo4j-primary-store.md): Neo4j for graph + vector + full-text
- [ADR-0004](adr/0004-pdf-pymupdf-and-tesseract-cli.md): PyMuPDF 1.28.2 (AGPL) for PDFs; Tesseract CLI for OCR
- [ADR-0005](adr/0005-pdf-engine-routing-docling.md): PDF engine routing (auto: multi-column → Docling on CPU)
- [ADR-0006](adr/0006-vision-model-image-descriptions.md): qwen3-vl:2b-instruct image descriptions as generated content
- [ADR-0007](adr/0007-in-process-ingestion-jobs.md): in-process background ingestion jobs with persistent job records

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
| Office loader time (docx / xlsx / pptx samples, 43 / 7 / 47 KB) | 69 / 40 / 76 ms (python-docx 1.2.0, openpyxl 3.1.5, python-pptx 1.0.2) |
| CLI ingest of one Office file (process total) | 1.8–2.0 s wall, 72–77 MB max RSS |

## Phase 2 prerequisites
- [x] `tesseract-ocr` 5.5.0 (eng + osd) and `jq` 1.8.1 installed (2026-11-04)
- [x] PyMuPDF added and measured (see PDF / OCR baselines below)
- [x] Vision model selected and measured: qwen3-vl:2b-instruct (ADR-0006)
- [x] Docling 2.141.0 on CPU measured: 1.2–1.45 GB RSS (ADR-0005)

## Measured baselines: PDF / OCR (2026-11-04, PyMuPDF 1.28.2, Tesseract 5.5.0, CPU only)
| Metric | Value |
|---|---|
| Born-digital PDF, 3 pages (loader) | 174 ms |
| Scanned page OCR @300 dpi (loader total / Tesseract only) | 756 ms / ~455–518 ms |
| OCR time vs DPI (one letter page) | 150: 254 ms · 200: 300 ms · 300: 455 ms · 400: 669 ms |
| Image OCR, 1400×900 photo | 216 ms, mean confidence 0.96 (0.34 without EXIF rotation) |
| CLI process max RSS (PDF / scan / image) | 113 / 135 / 116 MB |

## Measured baselines: layout + vision (2026-11-04, Docling 2.141.0 CPU, Ollama 0.40.14)
| Metric | Value |
|---|---|
| Docling vs PyMuPDF, 1-page two-column note | 3.4 s vs 0.09 s (first converter init ~60 s incl. 506 MB model download; ~10 s later) |
| Docling vs PyMuPDF, 3-page safety review | 7.1 s vs 0.17 s |
| Docling process max RSS | 1.2–1.45 GB |
| qwen3-vl:2b-instruct (chosen) | 24/24 facts, 8/8 chart label-value pairs, 0.9–1.7 s/image warm, 5.2 s cold load, 2.13 GB VRAM |
| gemma3:4b / qwen3-vl:4b-instruct | 24/24, 8/8, 2.4–4.7 s, 2.80 GB / 19/24, 0/8, 2.1–2.6 s, 3.76 GB |
| Full sample batch (12 files, 4 images described) | 24.9 s with per-batch VLM unload (40.9 s with per-document unload) |
| venv size | 1.6 GB with CPU torch (5.9 GB with the accidental CUDA build) |

## Measured baselines: ingestion API (2026-11-05, one worker, real models)
| Metric | Value |
|---|---|
| 12 sample uploads over HTTP → all jobs completed | 27.3 s (Docling note 11.6 s, deck with first vision call 7.7 s, text files < 6 ms) |
| Crash recovery (`kill -9` during a Docling job) | requeued at start-up, completed on attempt 2 (10.9 s) |
