# OmniRAG — working notes for Claude sessions

Local-first multimodal GraphRAG platform, built phase by phase (see `docs/PROGRESS.md`). The user is learning: each phase gets a `docs/learning/phase-XX-*.md` covering concepts, walkthrough, real outputs, common errors, an exercise and acceptance criteria. Stop at the end of each phase for the user to verify.

## Environment (do not change without discussion)
- Source on Windows `P:\omni-rag` = `/mnt/p/omni-rag` in WSL Ubuntu 26.04. P: has only ~5 GB free: never put data, models or venvs there.
- venv: `~/.venvs/omnirag` via `UV_PROJECT_ENVIRONMENT` (set in `~/.profile`, NOT `.bashrc`). Python 3.11 (uv-managed).
- Run Python/tests through WSL: `wsl.exe -d Ubuntu -- bash -l <script.sh>`. Inline `bash -lc '...'` via wsl.exe mangles `$()` and quotes, so write a script file in the scratchpad instead. From Git Bash, set `MSYS_NO_PATHCONV=1`.
- Ollama: native Windows at `localhost:11434` (mirrored networking). Models: `qwen3:4b-instruct` (generator), `qwen3:4b`, `bge-m3`. Max 2 resident models on a 6 GB GPU.
- Neo4j: `docker compose up -d` → `neo4j:2026.10.1-community`, mem_limit 2g. Password in `.env` (`NEO4J_PASSWORD` and `OMNIRAG_NEO4J_PASSWORD`).
- WSL VM capped at 8 GB (`C:\Users\gautam\.wslconfig`); Docker shares that cap.

## Conventions
- Code lives in `src/omnirag/<module>/` (ADR-0001). Create a module's package only in its phase.
- Config only through `omnirag.core.config.Settings` (`OMNIRAG_*`). No hardcoded secrets.
- Clients are created once in the FastAPI lifespan and injected with `Depends` (see `api/dependencies.py`).
- Logging: `structlog.get_logger(module=__name__)`. Never `logger=` (reserved).
- Parameterized Cypher only. Neo4j health probes use `session.run` (no retry); catch `DriverError` + `Neo4jError`.
- Exact `==` pins in pyproject; add heavy deps only in the phase that needs them; `uv.lock` committed.
- Checks: `uv run ruff format --check . && uv run ruff check . && uv run pytest` (unit) and `uv run pytest -m integration` (live services).
- Never invent benchmark numbers; record measured values with date and versions.
