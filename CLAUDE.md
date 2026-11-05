# OmniRAG — working notes for Claude sessions

Local-first multimodal GraphRAG platform, built phase by phase (see `docs/PROGRESS.md`). The user is learning: each phase gets a `docs/learning/phase-XX-*.md` covering concepts, walkthrough, real outputs, common errors, an exercise and acceptance criteria. Stop at the end of each phase for the user to verify.

## Environment (do not change without discussion)
- Source on Windows `P:\omni-rag` = `/mnt/p/omni-rag` in WSL Ubuntu 26.04. P: has only ~5 GB free: never put data, models or venvs there.
- venv: `~/.venvs/omnirag` via `UV_PROJECT_ENVIRONMENT` (set in `~/.profile`, NOT `.bashrc`). Python 3.11 (uv-managed).
- Run Python/tests through WSL: `wsl.exe -d Ubuntu -- bash -l <script.sh>`. Inline `bash -lc '...'` via wsl.exe mangles `$()` and quotes, so write a script file in the scratchpad instead. From Git Bash, set `MSYS_NO_PATHCONV=1`.
- Ollama: native Windows 0.40.x at `localhost:11434` (mirrored networking). Models: `qwen3:4b-instruct` (generator), `qwen3:4b`, `bge-m3`, `qwen3-vl:2b-instruct` (vision). Max 2 resident models on a 6 GB GPU.
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
- API ingestion runs through `ingestion/jobs.JobManager` (one process only: never `uvicorn --workers N`). Job files in `<data_dir>/jobs/`, upload copies in `<data_dir>/uploads/<job_id>/`. Originals are served as `application/octet-stream` attachments with `nosniff`.

## Conventions
- Code lives in `src/omnirag/<module>/` (ADR-0001). Create a module's package only in its phase.
- Config only through `omnirag.core.config.Settings` (`OMNIRAG_*`). No hardcoded secrets.
- Clients are created once in the FastAPI lifespan and injected with `Depends` (see `api/dependencies.py`).
- Logging: `structlog.get_logger(module=__name__)`. Never `logger=` (reserved).
- Parameterized Cypher only. Neo4j health probes use `session.run` (no retry); catch `DriverError` + `Neo4jError`.
- Exact `==` pins in pyproject; add heavy deps only in the phase that needs them; `uv.lock` committed.
- Checks: `uv run ruff format --check . && uv run ruff check . && uv run pytest` (unit) and `uv run pytest -m integration` (live services).
- Never invent benchmark numbers; record measured values with date and versions.
