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
| 3a | Chunking engine: 7 strategies, bge-m3 tokens | ✅ Done 2026-11-05 | 218 unit + 7 live tests; [learning/phase-03a](learning/phase-03a-chunking-strategies.md) |
| 3b | Semantic chunking, enrichment experiment, chunking benchmark | ✅ Done 2026-11-05 | 236 unit + 8 live tests; [learning/phase-03b](learning/phase-03b-semantic-chunking-and-benchmark.md), [results](benchmarks/chunking-2026-11-05.json) |
| 4 | Embedding engine | ✅ Done 2026-11-05 | 243 unit + 8 live tests; [learning/phase-04](learning/phase-04-embedding-engine.md), [results](benchmarks/embeddings-2026-11-05.json) |
| 5a | Neo4j graph: structure, vectors, versions | ✅ Done 2026-11-05 | 262 unit + 12 live tests; [learning/phase-05a](learning/phase-05a-graph-structure.md) |
| 5b | Entity/relationship extraction with validation | ✅ Done 2026-11-05 | 294 unit + 15 live tests; entity recall 36/37, typed facts 10/14; [learning/phase-05b](learning/phase-05b-entity-extraction.md), [results](benchmarks/extraction-2026-11-05.json) |
| 6 | Dense + sparse retrieval | ✅ Done 2026-11-05 | 315 unit + 25 live tests; neo4j dense = exact baseline, BM25 classic/require R@5 0.986 nDCG 0.861; [learning/phase-06](learning/phase-06-dense-and-lexical-retrieval.md), [results](benchmarks/retrieval-2026-11-05.json) |
| 7 | Graph-enhanced hybrid retrieval (RRF) | ✅ Done 2026-11-05 | 325 unit + 28 live tests; dense+bm25 RRF seed MRR 0.928 (default); graph measured, opt-in; [learning/phase-07](learning/phase-07-graph-retrieval-and-fusion.md), [results](benchmarks/fusion-2026-11-05.json) |
| 8 | Reranking | ✅ Done 2026-11-05 | 332 unit + 30 live tests; bge-reranker-v2-m3 depth 20: seed MRR 0.980 / nDCG 0.948, relational R@5 1.000; 17 s/query on CPU (opt-in mode; GPU open); [learning/phase-08](learning/phase-08-reranking.md), [results](benchmarks/rerank-2026-11-05.json) |
| 9a | Corrective retrieval workflow (LangGraph): transforms, context check, bounded retries | ✅ Done 2026-11-06 | 344 unit + 32 live tests; transforms don't help these sets; context check 37/40 seed, catches 3/3 unanswerable; [learning/phase-09a](learning/phase-09a-corrective-retrieval-workflow.md), [results](benchmarks/workflow-2026-11-05.json) |
| 9b | Query router, self-query, grounded search agent, per-document cap; Streamlit dashboard (Phase 14's retrieval lab, pulled forward) | ✅ Done 2026-11-06 | 373 unit tests; rules router 30/40 · 7/8 · 23/24 (LLM router worse); default (router + agent) relational R@5 0.875 → 0.938, MRR 0.838 → 0.917, +r02 no loss, seed unchanged; [learning/phase-09b](learning/phase-09b-routing-and-agentic-retrieval.md), [results](benchmarks/workflow-9b-2026-11-06.json) |
| 9c | Context assembly: parent-child / neighbour expansion, compression, token budget | ✅ Done 2026-11-06 | 383 unit + 33 live tests; budget 1,024 vs top 5: context recall seed 0.986 → 1.000, relational 0.875 → 0.938 (+q22 +r05, no loss); expansion and lexical/embedding compression lose evidence; LLM compression best at 256 tokens (relational 0.562 → 0.812); [learning/phase-09c](learning/phase-09c-context-assembly.md), [results](benchmarks/context-2026-11-06.json) |
| 10 | Ollama generation: claims with quotes, structured output, streaming (SSE), refusal without a model call, chat history + traces (SQLite) | ✅ Done 2026-11-06 | 418 unit + 37 live tests; live answers quote-verified (AUR-C103 2/2, 8-claim comparison 8/8), AUR-C109 refused in 26 ms with 0 model calls; [learning/phase-10](learning/phase-10-generation.md) |
| 11 | Citations + answer validation: quotes checked in every block (verified / relocated / elided on one line), numbers checked, unsupported claims removed, generated-only evidence → inferred | ✅ Done 2026-11-06 | 16 generation tests incl. every status; two rules found live (elided table quote, OCR vs description); [learning/phase-11](learning/phase-11-citations-and-validation.md) |
| 12 | Evaluation: 110-question set, E1–E7 runner, deterministic metrics + local LLM judge, LangSmith (opt-in, redacting), `POST /evaluate` | ✅ Done 2026-11-06 (run stopped early by request: E1–E4 complete, E5 76/110, E6–E7 not measured) | E3 dense+BM25: R@5 0.950, context recall 0.980, correct 0.900, faithfulness 0.853, 9/10 unanswerable refused; E5 rerank on 76 q: correct 0.932 vs E3 0.918, 4.6× latency; judge called 25/27 refusals "correct" → refusals scored by rule; [ADR-0020](adr/0020-evaluation-framework.md), [learning/phase-12](learning/phase-12-evaluation.md), [report](benchmarks/rag-eval-2026-11-06.json) |
| 13 | Failure analysis + observability: root cause per failed question (11 categories in pipeline order), query lifecycle traces, `/analytics` | ✅ Done 2026-11-06 | BM25 alone fails at retrieval (15), dense alone at context construction (10), dense+BM25 mostly at generation (6); no parsing/OCR/chunking/graph/embedding failures; [learning/phase-13](learning/phase-13-failure-analysis.md) |
| 14 | Frontend (Streamlit): all 11 screens | ✅ Done 2026-11-06 | Ask (streaming, citation cards, checks), Documents (library, upload, ingestion status, preview), Graph explorer, Retrieval lab, Workflow inspector, Evaluation, Performance, Benchmarks; every page rendered headless in tests; [learning/phase-14](learning/phase-14-frontend.md) |
| 15 | Dockerization + deployment: one image for API + dashboard, compose profile `app`, Ollama on the host | ✅ Done 2026-11-06 | 3.56 GB image, ~11 min cold build; api/dashboard/neo4j healthy; Ollama reached via host.docker.internal; grounded chat through the container; no secrets in the image; [learning/phase-15](learning/phase-15-deployment.md) |
| 16 | Tests, docs, resume prep | ✅ Done 2026-11-06 | 419 unit + 37 live tests; README, 21 ADRs, learning guide per phase, [RESUME.md](RESUME.md); [learning/phase-16](learning/phase-16-final.md) |

## Success-criteria tracker (from the brief)
- [x] All document types ingest (10/10 kinds: txt, md, html, json, csv, docx, xlsx, pptx, pdf, image) · [x] OCR on scans/images (Tesseract, with confidence) · [x] Table/figure provenance (tables in all formats incl. borderless via Docling; PPTX chart data; image descriptions with origin=generated)
- [x] Multiple chunking strategies (all 8, benchmarked) · [x] Embeddings in Neo4j (58 chunks, 1024-d, cosine, no quantization) · [x] Graph queryable (structure + validated entities/relations)
- [x] Dense (Neo4j vector index, in-index filters) · [x] BM25 (full-text, identifier-aware) · [x] Graph retrieval (linking, 2-hop traversal, provenance; opt-in, ADR-0014) · [x] RRF · [x] Reranking (cross-encoder, CPU; ADR-0015)
- [x] Grounded answers (claims with quotes, refusal without a model call) · [x] Verifiable citations (quotes + numbers checked by code) · [~] LangSmith tracing + experiment upload (implemented, redacting, tested with a fake client; not run against the hosted service: no API key)
- [x] Retrieval eval (110 questions) · [x] Faithfulness eval (quote-verified + judge) · [x] Reproducible benchmarks (resumable records, `--from-partial`) · [~] E6/E7 on the full set (run stopped early)
- [x] FastAPI endpoints (all the brief lists: health, documents/upload, ingestion/start + status, chat + history, evaluate, analytics; plus retrieval and graph)
- [x] Frontend (all 11 screens) · [x] Tests pass (419 unit + 37 live) · [x] Docker deploy (compose profile `app`) · [x] Docs complete (README, 21 ADRs, learning guides, resume notes)

## Decisions log
- [ADR-0001](adr/0001-src-layout.md): `src/omnirag` package layout
- [ADR-0002](adr/0002-ollama-native-windows.md): Ollama native on Windows, mirrored WSL networking
- [ADR-0003](adr/0003-neo4j-primary-store.md): Neo4j for graph + vector + full-text
- [ADR-0004](adr/0004-pdf-pymupdf-and-tesseract-cli.md): PyMuPDF 1.28.2 (AGPL) for PDFs; Tesseract CLI for OCR
- [ADR-0005](adr/0005-pdf-engine-routing-docling.md): PDF engine routing (auto: multi-column → Docling on CPU)
- [ADR-0006](adr/0006-vision-model-image-descriptions.md): qwen3-vl:2b-instruct image descriptions as generated content
- [ADR-0007](adr/0007-in-process-ingestion-jobs.md): in-process background ingestion jobs with persistent job records
- [ADR-0008](adr/0008-chunking-tokens-and-element-provenance.md): chunk sizes in bge-m3 tokens; element-level provenance and ground truth
- [ADR-0009](adr/0009-evidence-based-relevance-and-early-embeddings.md): relevance by verbatim evidence; minimal cached embedding client before Phase 4
- [ADR-0010](adr/0010-embedding-engine.md): versioned embedding spaces; Ollama dense in production; BGE-M3 sparse/ColBERT as local experiments
- [ADR-0011](adr/0011-graph-schema-and-versioning.md): graph schema, idempotent per-document writes, source-level versioning
- [ADR-0012](adr/0012-validated-entity-extraction.md): LLM entity/relation extraction validated against the source; stated vs inferred; SAME_AS instead of merging
- [ADR-0013](adr/0013-dense-and-lexical-retrieval.md): Neo4j dense + lexical retrieval; in-index filters; safe Lucene builder; classic analyzer + required identifiers
- [ADR-0014](adr/0014-graph-retrieval-and-rrf.md): graph retrieval + RRF; dense+bm25 default, graph opt-in (measured to lower ranking here)
- [ADR-0015](adr/0015-cross-encoder-reranking.md): bge-reranker-v2-m3 cross-encoder (pinned, CPU with CUDA auto-detect); quality mode, not default; its score doesn't detect unanswerable questions
- [ADR-0016](adr/0016-corrective-retrieval-workflow.md): LangGraph corrective loop (bounded); transforms off by default (no gain measured); identifier + LLM context check on
- [ADR-0017](adr/0017-routing-self-query-agentic-retrieval.md): rule router (beats the LLM router); grounded search agent for multi-hop routes (+r02); BM25 requires only the user's identifiers; no corrective retry after the agent; self-query opt-in (precise, no gain)
- [ADR-0018](adr/0018-context-assembly.md): token budget (1,024 bge-m3) instead of top-k; expansion and extractive compression opt-in (measured); LLM compression for small budgets
- [ADR-0019](adr/0019-grounded-generation-and-citations.md): claims with verbatim quotes checked by code; refusal without a model call; SQLite chat traces
- [ADR-0020](adr/0020-evaluation-framework.md): 110-question evaluation, deterministic metrics first, local LLM judge with reported agreement, failure root causes, LangSmith opt-in and redacting
- [ADR-0021](adr/0021-docker-deployment.md): one image for API + dashboard, Ollama on the host, data as a bind mount

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

## Measured baselines: chunking (2026-11-05, 12 sample documents, bge-m3 tokenizer @5617a9f6)
| Metric | Value |
|---|---|
| Tokenizer vs Ollama bge-m3 | 5/5 exact (ours + 2 special tokens = `prompt_eval_count`) |
| Tokenizer load / memory | 2.3 s / +302 MB RSS |
| Elements kept whole at 128 tokens | fixed 53.6% · recursive 77.7% · structure 97.3% · table 97.3% · multimodal 95.3% |
| Chunks at 512 (indexed) | fixed 16 · structure 48 · multimodal 55 · hierarchical 82 · parent–child 77 (+55 parents) |
| All 7 strategies, 12 documents | 18–90 ms per strategy; rechunk incl. tokenizer load 2.9 s |

## Measured baselines: chunking benchmark (2026-11-05, seed set 12 docs / 37 answerable questions, K=5)
Full tables: [docs/benchmarks/chunking-2026-11-05.json](benchmarks/chunking-2026-11-05.json) and [learning/phase-03b](learning/phase-03b-semantic-chunking-and-benchmark.md).
| Metric | Value |
|---|---|
| bge-m3 embedding throughput (Ollama) | 17 texts/s single, ~43 texts/s batched (8–64) |
| Dense Recall@5 / MRR, multimodal-256 → + contextual headers | 0.878 / 0.786 → 0.932 / 0.863 (2 questions better, 0 worse) |
| Dense Recall@5, structure-256 vs table-256 | 0.905 vs 0.851 (table-aware worse on 2 questions, 35 ties) |
| BM25 Recall@5, multimodal-256 | 0.959 (dense 0.878) |
| LLM contextual enrichment (qwen3:4b-instruct) | 58 chunks in 104 s (1.8 s/chunk); dense R@5 0.905, BM25 MRR 0.938 |
| Caveat | 1 question = 0.027 recall; 512-token baselines hit 1.000 only because 12 docs → 16 chunks |

## Measured baselines: embeddings (2026-11-05, Ollama 0.40.14, bge-m3 digest 79076464…)
Full table: [docs/benchmarks/embeddings-2026-11-05.json](benchmarks/embeddings-2026-11-05.json) and [learning/phase-04](learning/phase-04-embedding-engine.md).
| Metric | Value |
|---|---|
| Ollama bge-m3 usable input | 2,048 tokens incl. 2 special (2,046 OK, 2,047 → HTTP 413); default `truncate` silently cuts to 2,048 |
| Ollama vs original fp32 weights | cosine ≥ 0.99992 on 58 chunks; identical Recall@5/MRR |
| Encode 58 chunks | Ollama GPU 1.4 s (41.6/s) · local CPU 31 s (1.9/s, 2.7 GB RSS) · nomic 1.1 s |
| Recall@5 / MRR (multimodal-256-ctx) | BM25 0.986/0.900 · dense bge-m3 0.932/0.863 · nomic 0.919/0.821 · sparse M3 0.905/0.761 · ColBERT 0.959/0.901 |
| Storage per chunk | dense 4 KB · sparse 0.4 KB · ColBERT 424 KB |
| Full sample ingest incl. embeddings | 46 s (embedding ≈ 2 s of it) |
