# ADR-0002: Ollama runs natively on Windows; everything else in WSL2

- Status: Accepted (2026-11-04)

## Context
The GPU is an RTX 3060 Laptop with 6 GB VRAM. GPU access from Docker on WSL2 works, but it adds the NVIDIA container toolkit, another layer of memory accounting, and VRAM contention between containers. The WSL VM is also capped at 8 GB RAM, so Ollama inside it would eat into the budget for Neo4j and the parsers.

## Decision
- Ollama runs as the native Windows app with direct CUDA access. Its host RAM sits **outside** the WSL 8 GB cap.
- WSL2 uses `networkingMode=mirrored`, so the API reaches Ollama at `http://localhost:11434` with no host-IP discovery or firewall rules.
- Ollama is tuned for one user on a 6 GB card: `MAX_LOADED_MODELS=2`, `NUM_PARALLEL=1`, `FLASH_ATTENTION=1`, `KV_CACHE_TYPE=q8_0`, `CONTEXT_LENGTH=8192`, `KEEP_ALIVE=10m`.
- Containers that need Ollama later (Phase 15) will use `host.docker.internal:11434`.

## Consequences
- Measured: generator plus embedder use 3.72 GB, both 100% on GPU. The vision model must be swapped in and out (`keep_alive=0`) rather than co-resident.
- Ollama is not part of `docker compose up`, so README and health checks must make that dependency visible (`/health/ollama`).
- Ollama auto-updates on Windows (0.40.11 → 0.40.14 during setup). The version is reported in `/health/ollama` and recorded with every benchmark run.
