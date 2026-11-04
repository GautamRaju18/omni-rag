# Phase 1 — Backend foundation

> Status: **complete and verified on 2026-11-04**: 24 unit tests, 4 integration tests, live end-to-end checks and a negative test.

## 1. Problem being solved
Every later module (ingestion jobs, retrieval, chat, evaluation) needs the same plumbing:
1. typed configuration without hardcoded secrets,
2. one shared connection pool per external service,
3. structured logs with a request ID, so one query can be traced through many components,
4. consistent error responses,
5. a way to swap real services for fakes in tests,
6. health checks that say *what* is broken.

Building this first means Phase 2+ code is only domain logic.

## 2. Concepts
- **12-factor config.** Settings come from the environment, and code never contains secrets. `pydantic-settings` validates them at startup, so `OMNIRAG_OLLAMA_TIMEOUT_S=abc` fails right away instead of at the first request. `SecretStr` keeps the password out of `repr()`, logs and JSON dumps (tested).
- **Application factory + lifespan.** `create_app(settings)` builds a fresh app, so tests can make one with test settings. The **lifespan** is FastAPI's async context manager for startup and shutdown resources: it opens the Neo4j driver and the httpx pool once and closes them cleanly.
- **Dependency injection.** Routes declare `neo4j: Neo4jDep`, and FastAPI resolves it per request. Tests call `app.dependency_overrides[get_neo4j] = lambda: Fake()`, with no monkeypatching or real database. This is the main reason the unit tests run in about 3 s.
- **Connection pooling.** The Neo4j driver and `httpx.AsyncClient` each keep a pool. Creating them per request would mean a TCP + TLS/Bolt handshake on every call.
- **Fail-soft startup.** Creating the driver doesn't connect. If Neo4j is down, the API still starts and reports `down` instead of crash-looping, which is the right behaviour under Docker restarts.
- **Health semantics.** `ok` (fully functional), `degraded` (reachable but not fully usable, such as a missing model; HTTP 200), `down` (unreachable; HTTP **503**, so load balancers and Docker healthchecks react). The aggregate is the *worst* component.
- **Structured logging + contextvars.** Each log line is JSON. `contextvars` carry `request_id` across `await` points, so logs from deep inside the Neo4j client are tagged with the request that caused them. This is the base for the Phase 13 query-lifecycle tracing.

## 3. Architecture

```
HTTP request
  │
  ▼
request_context_middleware ── assigns/echoes X-Request-ID, binds it to structlog, logs access line
  │
  ▼
APIRouter /api/v1/health/*
  │   Depends(get_app_settings) ─┐
  │   Depends(get_neo4j) ────────┼──► app.state.*  (created once in lifespan)
  │   Depends(get_ollama) ───────┘
  ▼
Neo4jClient.health()  ‖  OllamaClient.health()       (run concurrently with asyncio.gather)
  │                         │
  ▼                         ▼
Bolt :7687 (Docker)      HTTP :11434 (Windows host)
  │
  ▼
HealthReport / ComponentHealth (Pydantic)  →  200 / 503
Exceptions → register_exception_handlers → {"error": {code, message, details, request_id}}
```

## 4. Technology choices and alternatives
| Choice | Alternatives | Why this one |
|---|---|---|
| `src/omnirag/` package | top-level `backend/`, `retrieval/` … | See [ADR-0001](../adr/0001-src-layout.md): avoids name clashes and guarantees tests import the installed package |
| `httpx` for Ollama | `ollama` Python SDK | We need fine control over timeouts, retries and streaming, plus in-memory `MockTransport` for tests. The SDK adds a layer without adding capability. |
| `structlog` | stdlib `logging` + JSON formatter | Context binding (`request_id`) is built in. JSON renderer, low overhead. |
| Neo4j async driver, `session.run` for probes | `execute_query` | `execute_query` uses managed transactions that **retry for up to 30 s** on connection errors, which is wrong for a health probe. Auto-commit `session.run` fails in about 2 ms (measured). |
| `httpx.AsyncHTTPTransport(retries=2)` | `tenacity` | Retries only failed *connections*, never a request the server already received, so it's safe and needs no dependency. Retry policies for generation come in Phase 10. |

## 5. File map
```
src/omnirag/
├── __init__.py                 __version__ from installed package metadata
├── core/config.py              Settings (OMNIRAG_*), get_settings() cached
├── core/logging.py             configure_logging(): structlog JSON + stdlib level
├── core/errors.py              OmniRAGError hierarchy + JSON error envelope handlers
├── schemas/health.py           ComponentHealth, HealthReport, worst_status()
├── llm/ollama_client.py        OllamaClient: version, list_models, loaded_models (VRAM), health
├── knowledge_base/neo4j_client.py  Neo4jClient: pooled async driver, health
└── api/
    ├── main.py                 create_app(), lifespan, router mounting, `/health` alias
    ├── dependencies.py         get_neo4j / get_ollama / get_app_settings + Annotated aliases
    ├── middleware.py           request ID + access log
    └── routers/health.py       /api/v1/health, /health/ollama, /health/neo4j
tests/
├── conftest.py                 `settings` fixture (ignores local .env)
├── unit/                       config, ollama client (MockTransport), neo4j fail-fast, API (fakes)
└── integration/                real Neo4j + Ollama + full lifespan (pytest -m integration)
```

## 6. Code walkthrough (the parts worth understanding)

**`config.py`**: `env_prefix="OMNIRAG_"` maps `OMNIRAG_NEO4J_URI` to `neo4j_uri`. `extra="ignore"` lets `.env` also hold `NEO4J_PASSWORD` for docker compose. `Field(gt=0)` makes a zero timeout a startup error. `required_ollama_models` is derived from config, so changing the generator model automatically changes what the health check requires.

**`ollama_client.py`**: `normalize_model_name()` handles Ollama storing `bge-m3` as `bge-m3:latest`. Without it, the health check would report a pulled model as missing. `loaded_models()` turns `/api/ps` `size`/`size_vram` into `gpu_percent`, the single most useful number for spotting CPU spill-over on a 6 GB card. Every network or parse failure becomes a `down` result, so a probe never raises.

**`neo4j_client.py`**: `max_transaction_retry_time` and `connection_acquisition_timeout` are kept short, and the probe is additionally wrapped in `asyncio.timeout()`, so a hung server can't hang `/health`. The query reads **all** rows of `dbms.components()`. See the bug story below.

**`api/main.py`**: the lifespan owns client lifecycles. `app.add_api_route("/health", ..., include_in_schema=False)` gives Docker a stable unversioned probe without a duplicate entry in the OpenAPI spec (a test asserts the schema has exactly three paths).

**`routers/health.py`**: `asyncio.gather` runs both probes concurrently, so the aggregate costs about `max(13 ms, 50 ms)` instead of their sum. `response.status_code = 503` keeps the typed `response_model` and still changes the HTTP status.

## 7. Bugs found during verification (real, and instructive)
1. **`structlog.get_logger(logger=...)` → `TypeError`.** `logger` is a reserved positional name inside structlog's `wrap_logger`. Fixed by binding `module=__name__`. Lesson: test imports early, because a module-level logger runs at import time.
2. **venv created on P:**: the `.bashrc` early-return issue (see the Phase 0 doc).
3. **Neo4j 2026 `dbms.components()` returns two rows** (`Neo4j Kernel` and `Cypher` with versions `["5","25"]`). The code assumed `single()`, so the driver raised `ResultNotSingleError`. That is a `DriverError`, not a `Neo4jError`, so it escaped the `except` and became an HTTP 500. Two fixes: read all rows keyed by name, and catch `DriverError` (client-side) as well as `Neo4jError` (server-side). The unit tests with fakes **could not** catch this. Only the integration test against the real server did, which is why both layers exist.
4. **Starlette `TestClient` deprecation** (`install httpx2`): the integration test now drives the real lifespan through `app.router.lifespan_context(app)` + `httpx.ASGITransport`, with no deprecated API.

## 8. How to run
In an Ubuntu (WSL) terminal:
```bash
cd /mnt/p/omni-rag
cp -n .env.example .env            # first time only; then set a real password
docker compose up -d               # Neo4j
uv sync                            # creates ~/.venvs/omnirag from uv.lock
uv run uvicorn omnirag.api.main:app --reload --no-access-log
```
Open http://localhost:8000/docs for the interactive API and http://localhost:7474 for the Neo4j Browser.

## 9. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                      # unit tests: no services needed
uv run pytest -m integration       # needs Neo4j + Ollama running
```

## 10. Expected outputs (captured 2026-11-04)
```text
24 passed, 4 deselected in 3.13s          # unit
4 passed, 24 deselected in 1.40s          # integration
```
`GET /api/v1/health` → **HTTP 200**
```json
{"status":"ok","app":"OmniRAG","version":"0.1.0","environment":"dev","components":[
 {"name":"neo4j","status":"ok","latency_ms":12.95,
  "details":{"uri":"bolt://localhost:7687","database":"neo4j","version":"2026.10.1","edition":"community","cypher_versions":["5","25"]}},
 {"name":"ollama","status":"ok","latency_ms":49.92,
  "details":{"version":"0.40.14","missing_models":[],
   "loaded_models":[{"name":"qwen3:4b-instruct","vram_gb":3.08,"gpu_percent":100.0,"context_length":8192},
                    {"name":"bge-m3:latest","vram_gb":0.64,"gpu_percent":100.0,"context_length":8192}],
   "vram_in_use_gb":3.72}}]}
```
Negative test (`docker compose stop neo4j`):
```text
GET /api/v1/health        -> HTTP 503
GET /api/v1/health/neo4j  -> {"status":"down","latency_ms":1.89,"error":"ServiceUnavailable: Couldn't connect to localhost:7687 ..."}
GET /api/v1/health/ollama -> HTTP 200   (API keeps serving)
after `docker compose start neo4j` -> HTTP 200
```
Access log line (JSON, with request ID):
```json
{"module":"omnirag.access","method":"GET","path":"/api/v1/health","status":200,"duration_ms":115.49,"event":"http_request","request_id":"96486341387445d59835b668f65aaf10","level":"info","timestamp":"2026-11-04T09:09:15.201634Z"}
```
Model sanity: bge-m3 embedding **dim 1024, L2 norm 1.0**; qwen3:4b-instruct warm **~55 tok/s, ~0.6 s** for a one-sentence answer (cold first load ~22 s).

## 11. Common errors and debugging
| Symptom | Likely cause | Debug |
|---|---|---|
| `/health/neo4j` → `AuthError` | `.env` password differs from the one the volume was **first** created with (Neo4j stores it in `/data`) | `docker compose down -v` (deletes data!) or reset via `cypher-shell` |
| `/health/ollama` → `degraded` | Required model not pulled | `ollama pull <name>` (PowerShell) |
| `--reload` doesn't reload | Watching `/mnt/p` without polling | `echo $WATCHFILES_FORCE_POLLING` should print `true` |
| `ModuleNotFoundError: omnirag` | Not running through `uv run`, or venv out of date | `uv sync`, then `uv run ...` |
| HTTP 500 with `internal_error` | Uncaught exception | grep the `request_id` from the response in the server log; the full traceback is there |

## 12. Exercise (≈45 min)
Add **`GET /api/v1/health/system`** that reports WSL memory and disk usage. Read `/proc/meminfo` for `MemTotal`/`MemAvailable` (stdlib only, no psutil) and use `shutil.disk_usage` for the repo path. It should return a `ComponentHealth` with status `degraded` when available memory is below a configurable `OMNIRAG_MIN_FREE_MEMORY_MB` (default 1024), and include it in the aggregate `/health`.
Acceptance: a unit test that feeds a fake meminfo string covers both `ok` and `degraded`; `ruff` is clean; the endpoint appears in `/docs`.
Hint: put the parsing in a pure function `parse_meminfo(text) -> dict[str,int]` so it's testable without a real `/proc`.

## 13. Acceptance criteria — Phase 1
- [x] `uv sync` reproducible from `uv.lock`; direct deps pinned with `==`
- [x] `ruff format --check` + `ruff check` clean
- [x] 24 unit tests pass without any running service
- [x] 4 integration tests pass against live Neo4j + Ollama
- [x] `/api/v1/health`, `/health/ollama`, `/health/neo4j` return correct JSON; 503 on `down`
- [x] API survives Neo4j outage and recovers without restart
- [x] Every response carries `X-Request-ID`; logs are JSON with the same ID
- [x] No secrets in code; password never appears in repr/JSON (tested)
