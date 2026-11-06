# Phase 15: Dockerization and deployment

> Decision record: [ADR-0021](../adr/0021-docker-deployment.md). Files: `Dockerfile`, `.dockerignore`, `docker-compose.yml` (profile `app`).

## 1. Problem being solved
Until now OmniRAG ran as `uv run` processes in WSL. A deployment has to start everything with one command, from pinned versions, with no secrets in the image. It also has to fit beside the GPU-owning Ollama and inside an 8 GB VM.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Multi-stage build** | A build stage with uv installs dependencies; the runtime stage copies only the virtualenv and the source. Build tools don't ship. |
| **Layer caching** | `pyproject.toml` + `uv.lock` are copied and installed *before* the source, so a code change rebuilds only the last layers. |
| **Bind mount vs named volume** | Data is a *bind mount* of the host directory (the CLI and the container share it). Neo4j uses a *named volume* (only Neo4j touches it). |
| **`host.docker.internal`** | The Docker host as seen from a container; here, Windows, where Ollama listens. |
| **Liveness vs readiness** | Liveness = "the process answers" (`/openapi.json`). `/health` also checks Neo4j and Ollama: useful to read, wrong as a restart trigger. |
| **Profiles** | `docker compose --profile app` starts the app services; plain `docker compose up -d` still starts only Neo4j (development). |

## 3. Architecture
```
Windows host ──────────────────────────────────────────────────────────────┐
  Ollama 0.40 (GPU): qwen3:4b-instruct, bge-m3, qwen3-vl:2b  :11434         │
  WSL2 VM (8 GB) / Docker Desktop                                         │
    omnirag-neo4j     neo4j:2026.10.1-community   :7474 :7687   2g        │
    omnirag-api       omnirag:0.1.0 (uvicorn)     :8000         3g ──► host.docker.internal:11434
    omnirag-dashboard omnirag:0.1.0 (streamlit)   :8501       768m ──► http://api:8000
    ~/omnirag-data ──bind──► /data      ./docs/benchmarks ──ro──► /app/docs/benchmarks
```

## 4. How to run
```bash
cd /mnt/p/omni-rag
cp .env.example .env             # set NEO4J_PASSWORD and OMNIRAG_NEO4J_PASSWORD (same value)
docker compose --profile app up -d --build     # first build ~11 min, 3.56 GB
docker compose --profile app ps                # api / dashboard / neo4j: healthy
curl -s localhost:8000/api/v1/health | jq .status
open http://localhost:8501                     # the dashboard
docker compose --profile app logs -f api       # JSON logs with request IDs
docker compose --profile app down              # stop (data stays on the host)
```
Development without containers is unchanged (`uv run uvicorn …`, `uv run streamlit run …`). Don't run both at once: they bind the same ports.

## 5. Expected outputs (measured 2026-11-06)
```
omnirag-api        Up 34 seconds (healthy)
omnirag-dashboard  Up 13 seconds (healthy)
omnirag-neo4j      Up 3 hours (healthy)
/api/v1/health: "status": "ok"; neo4j bolt://neo4j:7687 2026.10.1 community; ollama http://host.docker.internal:11434 0.40.14, missing_models []
POST /chat "Who prepared the safety review of the E-5 failure?" → "Lena Brandt prepared the safety review of the E-5 failure [1]." (1/1 verified, aurora_safety_review.pdf p1)
POST /documents/upload (an existing sample) → duplicate: true
memory: api 763 MiB / 3 GiB, dashboard 60 MiB / 768 MiB, neo4j 1.04 GiB / 2 GiB
```

## 6. Common errors
| Symptom | Cause / fix |
|---|---|
| `set NEO4J_PASSWORD in .env` | No `.env`: copy `.env.example` and set both password variables. |
| Ollama `down` in `/health` from the container | Ollama not running on Windows, or bound elsewhere. `curl localhost:11434/api/version` on the host; the container uses `host.docker.internal`. |
| `bind: address already in use` on 8000/8501 | A dev `uvicorn`/`streamlit` is running: stop it, or change the published ports. |
| `Permission denied` under `/data` | The host data dir isn't owned by uid 1000. `chown -R 1000:1000 ~/omnirag-data` (or build with your uid). |
| API killed (exit 137) | Out of memory: the reranker plus Docling on a big PDF. Raise `mem_limit`, or keep reranking off. |
| The Evaluation page is empty in containers | No `docs/benchmarks/rag-eval-*.json` yet. The folder is mounted read-only; run `rag_eval --out docs/benchmarks/...` on the host. |

## 7. Exercise (≈30 min)
1. Change one line in `src/omnirag/frontend/screens.py` and rebuild. Which build steps are cached? Why does the dependency layer survive?
2. Stop Neo4j (`docker compose stop neo4j`) and call `/api/v1/health` and `/api/v1/chat`. Which answers 200 and which 503? Why is the API container still healthy?
3. **Think.** What would have to change to run two API replicas behind a load balancer? (Hint: ADR-0007, ADR-0019, ADR-0020.)

## 8. Acceptance criteria: Phase 15
- [ ] `docker compose --profile app up -d --build` brings api, dashboard and neo4j to `healthy`.
- [ ] `/api/v1/health` from the container reports Neo4j and Ollama `ok`.
- [ ] A chat turn through `localhost:8000` is grounded with a verified citation; the dashboard at `localhost:8501` shows it under Ask.
- [ ] `docker image inspect omnirag:0.1.0` contains no `.env` and runs as a non-root user.
