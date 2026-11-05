# OmniRAG — working notes for Claude sessions

Local-first multimodal GraphRAG platform, built phase by phase (see `docs/PROGRESS.md`). The user is learning: each phase gets a `docs/learning/phase-XX-*.md` covering concepts, walkthrough, real outputs, common errors, an exercise and acceptance criteria. Stop at the end of each phase for the user to verify.

## Environment (do not change without discussion)
- Source on Windows `P:\omni-rag` = `/mnt/p/omni-rag` in WSL Ubuntu 26.04. P: has only ~5 GB free: never put data, models or venvs there.
- venv: `~/.venvs/omnirag` via `UV_PROJECT_ENVIRONMENT` (set in `~/.profile`, NOT `.bashrc`). Python 3.11 (uv-managed).
- Run Python/tests through WSL: `wsl.exe -d Ubuntu -- bash -l <script.sh>`. Inline `bash -lc '...'` via wsl.exe mangles `$()` and quotes, so write a script file in the scratchpad instead. From Git Bash, set `MSYS_NO_PATHCONV=1`.
- Ollama: native Windows 0.40.x at `localhost:11434` (mirrored networking). Models: `qwen3:4b-instruct` (generator), `qwen3:4b`, `bge-m3`, `qwen3-vl:2b-instruct` (vision), `nomic-embed-text:v1.5` (embedding comparison only). Max 2 resident models on a 6 GB GPU.
- A second, unrelated Ollama (0.32.3, systemd service inside WSL, another project's `llama3.1`/`nomic-embed-text`) was moved to `127.0.0.1:11435` on 2026-11-05 via `/etc/systemd/system/ollama.service.d/port.conf`. Mirrored networking shares ports, so if `/api/version` on 11434 ever says 0.32.x, that override is gone and it grabbed our port. Never repoint OmniRAG at 11435.
- Neo4j: `docker compose up -d` → `neo4j:2026.10.1-community`, mem_limit 2g. Password in `.env` (`NEO4J_PASSWORD` and `OMNIRAG_NEO4J_PASSWORD`). Docker Desktop must be running (`docker` is missing in WSL otherwise); it also starts another project's supabase containers.
- WSL VM capped at 8 GB (`C:\Users\gautam\.wslconfig`); Docker shares that cap.
- Ingested data lives in `OMNIRAG_DATA_DIR=/home/gautam/omnirag-data` (ext4). CLI: `uv run python -m omnirag.ingestion ingest|list|show`.

## Ingestion invariants (Phase 2)
- Every loader emits `DocumentElement`s only via `ExtractionContext.add()`: never construct IDs by hand.
- Element IDs are deterministic (`<doc_id>-eNNNNN`); document IDs are `doc_` + sha256[:16]. Don't change either scheme; eval ground truth depends on it.
- Originals are saved before parsing and never modified. Model-written text uses `origin=GENERATED`, OCR text uses `origin=OCR`.
- Tables go through `ingestion/tables.add_table()` (TABLE + self-describing TABLE_ROW children).
- Loaders are shared across threads: no per-file state on `self`.
- Don't fetch remote URLs found in documents (SSRF / offline-first).
- DOCX text: use `docx_loader._element_text`, never python-docx `Paragraph.text`/`cell.text` (they drop tracked insertions, fields, content controls).
- Office samples in `data/samples` are committed binaries; regenerating via `scripts/make_office_samples.py` changes their doc IDs.
- `defusedxml` must stay installed (openpyxl XML-bomb protection; asserted by a test).
- OCR text: `origin=OCR`, confidence = Tesseract mean word confidence, `parent_element_id` = the source IMAGE. OCR goes through `ingestion/ocr.TesseractOcr` (subprocess, list args, no shell); tests needing the binary use `requires_tesseract`.
- PyMuPDF is AGPL (ADR-0004); keep PDF parsing behind `PdfLoader` so it can be swapped.
- Enrichers (e.g. vision) only APPEND elements after the loader; never change loader element order/IDs. Generated text: `ElementType.DESCRIPTION`, `origin=GENERATED`, child of its IMAGE.
- torch/torchvision stay as direct deps in the `layout` group (CPU index); uv ignores sources for transitive deps (ADR-0005). Docling tests: `-m docling`.
- Call `pipeline.close()` after a batch (unloads the VLM).
- Evaluation: relevance = verbatim evidence quotes contained in returned chunk content (`evaluation/dataset.py`, ADR-0009). Never weaken a failing evidence test: fix the quote. Report per-question wins/losses, not only means (seed set: 1 question = 0.027 recall).
- Embeddings (ADR-0010): retrieval code uses only `DenseEmbedder`/`SparseEmbedder` + `EmbeddingSpace.space_id` (includes the model digest). Always `truncate: false`; Ollama bge-m3 accepts 2,048 tokens incl. specials (measured). `chunk.embedding_ref = <space_id>/<sha256>`; check with `python -m omnirag.embeddings verify`. `BGEM3Local` (CPU, 2.7 GB) is for experiments only, never at ingestion.
- Chunking: sizes are bge-m3 tokens (`chunking/tokenizer.HFTokenizer`, pinned revision + SHA-256). Chunks reference `element_ids` (from character spans, never string search); evaluation ground truth is element IDs, never chunk IDs (ADR-0008). Unit tests use `WhitespaceTokenizer`; API/unit tests must keep `chunking_enabled=False` or a fake counter (no 17 MB download).
- API ingestion runs through `ingestion/jobs.JobManager` (one process only: never `uvicorn --workers N`). Job files in `<data_dir>/jobs/`, upload copies in `<data_dir>/uploads/<job_id>/`. Originals are served as `application/octet-stream` attachments with `nosniff`.

## Knowledge graph invariants (Phase 5, ADR-0011)
- The graph is a rebuildable index of the files (`python -m omnirag.knowledge_base load --all`); files stay the source of truth.
- One managed transaction per document: merge Source/Document, delete owned nodes (Chunk/Page/Section/Table/Figure by `document_id`), recreate, recompute versions. Keep writes idempotent.
- Versions: Source keyed by lowercased file name; ordered by `created_at, document_id`; only the newest is `is_latest` (copied onto its chunks).
- Vector index: explicit `vector.quantization.type: NONE` (the 2026.x default is BINARY), filter properties `WITH [c.config_id, c.is_latest]`. Vectors via `db.create.setNodeVectorProperty` (native VECTOR type rejected, 51N77), read from the embedding cache by `embedding_ref`, never by calling the model.
- Searches use Cypher 25 `SEARCH` (`queryNodes` is deprecated). Index names can't be params: validate with `traversal._index_name`. Full-text SEARCH has no WHERE: post-filter.
- Delete order: graph first, then files (503 if Neo4j is down). A graph write failure during ingestion is a warning.
- Graph integration tests wipe the database: only against `neo4j-test` (`docker compose --profile test up -d neo4j-test`, port 7688); the fixture refuses other ports. Drop constraints before indexes.

## Entity extraction invariants (Phase 5b, ADR-0012)
- The LLM proposes, `knowledge_base/entities.py` decides: entities must be grounded in the chunk text (word boundaries, after `norm`), never only in GENERATED text; relations need accepted endpoints and a fitting type signature (`SIGNATURES`). Never relax grounding to raise recall.
- `support` is `stated` only when the evidence is found in original/OCR text and names both endpoints; everything else is `inferred`. Keep both, labelled; keep every rejection with its reason.
- Entity identity = `entity_key` (canonical name) → `entity_id`; type is the majority over mentions. Cross-document person aliases are `SAME_AS` links (recomputed), never merges.
- Raw replies live in `extractions/<config>.json` keyed by input hash + model digest + `PROMPT_VERSION`; bump `PROMPT_VERSION` whenever the prompt or schema changes. Rule changes: `revalidate` (no model calls).
- Extraction is off at ingestion (`extraction_enabled=false`); run `python -m omnirag.knowledge_base extract`. Quality is measured with `evaluation/extraction_eval.py` against `data/eval/entities_seed.json` (names and quotes verified against the samples by a test).

## Retrieval invariants (Phase 6, ADR-0013)
- Never pass user text to Lucene: always `retrieval.sparse.lucene_query` (escapes everything; identifiers by mode). Full-text SEARCH has no WHERE: all its filters are post-filters.
- Vector SEARCH WHERE = AND of `=`/`IN` on declared properties only (`schema.VECTOR_FILTER_PROPERTIES` == `retrieval.filters.VECTOR_INDEX_FILTERS`, asserted). A MATCH before SEARCH is not a pre-filter. Other filters over-fetch and report `underfilled`.
- Dense refuses to rank when the graph's `Document.embedding_space` differs from the query embedder's `space_id`.
- Index config changes (analyzer, vector filters/dimension/quantization) raise `SchemaMismatchError`; fix with `init --recreate-fulltext-index` / `--recreate-vector-index`, never by silently keeping the old index. Analyzer sweeps run only against `neo4j-test` (7688).
- Raw scores are per retriever (Neo4j cosine is (1+cos)/2; BM25 unbounded): never add them across retrievers; Phase 7 fuses ranks (RRF).

## Graph retrieval + fusion invariants (Phase 7, ADR-0014)
- Fuse ranks with `retrieval.fusion.rrf`, never raw scores. Fused hits keep `sources` (each member's rank/score) and graph hits keep `graph_paths`: don't drop provenance.
- Default mode is `dense+bm25` (measured best on both eval sets); graph is opt-in (`graph`, `dense+bm25+graph`) because it lowered MRR on both. Re-measure on both sets (`--dataset data/eval/relational_seed.json`) before changing the default or graph scoring; report per-question wins/losses.
- Graph scoring: chunk = Σ seed IDF weights of mentioned seeds + its *best* edge (max, not sum: hub entities). Tie-breaks are part of the ranking (seed weight, then fewer other entities), never chunk ID alone.
- A failing fusion member degrades (`details.failed`); only all-failed raises.

## Conventions
- Code lives in `src/omnirag/<module>/` (ADR-0001). Create a module's package only in its phase.
- Config only through `omnirag.core.config.Settings` (`OMNIRAG_*`). No hardcoded secrets.
- Clients are created once in the FastAPI lifespan and injected with `Depends` (see `api/dependencies.py`).
- Logging: `structlog.get_logger(module=__name__)`. Never `logger=` (reserved).
- Parameterized Cypher only. Neo4j health probes use `session.run` (no retry); catch `DriverError` + `Neo4jError`.
- Exact `==` pins in pyproject; add heavy deps only in the phase that needs them; `uv.lock` committed.
- Checks: `uv run ruff format --check . && uv run ruff check . && uv run pytest` (unit) and `uv run pytest -m integration` (live services).
- Never invent benchmark numbers; record measured values with date and versions.
