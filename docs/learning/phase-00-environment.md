# Phase 0 — Hardware-aware environment

> Status: **complete and verified on 2026-11-04.** Every number below was measured on this machine.

## 1. Problem being solved
A GraphRAG stack runs several heavy processes: an LLM, an embedding model, a vision model, a graph database, a cross-encoder, a document parser, an API and a UI. A 16 GB laptop with a 6 GB GPU can't hold all of them at once. Phase 0 settles **where each one runs** and **how much memory it may use** before any of them is written, so later phases don't hit OOM kills or silent CPU fallback.

## 2. Concepts

| Concept | Why it matters here |
|---|---|
| **Quantization (Q4_K_M)** | Stores weights in ~4.5 bits instead of 16, so a 4B-param model drops from ~8 GB to ~2.5 GB. Quality loss is small for RAG-style grounded answers. |
| **KV cache** | Per-token attention state. It grows linearly with context length. With `OLLAMA_KV_CACHE_TYPE=q8_0` it costs half the memory of f16, and `OLLAMA_FLASH_ATTENTION=1` is required for that to work. |
| **Partial offload** | If a model doesn't fit in VRAM, Ollama splits its layers between GPU and CPU, and generation can get 5–10× slower. Watch `gpu_percent` in `/health/ollama`: it should be 100. |
| **WSL2 VM** | Linux runs in a lightweight VM whose RAM comes from Windows. Without a cap it can use up to 50% of host RAM and keeps its page cache. `.wslconfig` caps it. |
| **Mirrored networking** | WSL shares the Windows network stack, so `localhost:11434` inside WSL is the Windows Ollama. With NAT mode you'd need host-IP discovery and firewall rules. |
| **9P file sharing** | `/mnt/p/...` is a network-like filesystem: every `stat()` crosses the VM boundary. A venv with thousands of small files there is slow, which is why the venv lives on ext4. |

## 3. Final architecture for this machine

```
Windows 11 host (16 GB RAM, RTX 3060 Laptop 6 GB)
├── Ollama 0.40.14 (native; direct CUDA, no Docker GPU passthrough)
│     ├── qwen3:4b-instruct  → 3.08 GB VRAM (measured, 8k ctx, q8 KV)
│     └── bge-m3             → 0.64 GB VRAM (measured)
└── WSL2 VM (capped 8 GB RAM, 4 GB swap, 8 vCPU, mirrored networking)
      ├── Ubuntu 26.04: uv-managed Python 3.11, venv at ~/.venvs/omnirag
      │     └── FastAPI (uvicorn) → localhost:8000
      └── Docker Desktop engine (shares the same VM, so the same 8 GB cap)
            └── neo4j:2026.10.1-community  (mem_limit 2 GB; heap 1 GB, pagecache 512 MB)
Source code: P:\omni-rag  (= /mnt/p/omni-rag in WSL)
```

### Model choices
| Role | Model | Why |
|---|---|---|
| Generation (default) | `qwen3:4b-instruct` | Non-thinking variant: no `<think>` preamble, so latency is lower and JSON/citation output is more stable. **Measured: ~55 tok/s warm.** |
| Planner / judge experiments | `qwen3:4b` | The thinking variant. Better at multi-step reasoning, slower. Only for query planning and LLM-as-judge tests later. |
| Embeddings | `bge-m3` | Multilingual, 8k context, **1024-dim, returned L2-normalized (measured norm = 1.0)**. Ollama exposes only its *dense* vectors. Sparse and multi-vector outputs need another interface (Phase 4). |
| Vision (Phase 2) | a 3–4B VL model (to be selected and measured then) | ~3–4 GB. It can't sit beside the LLM, so it is loaded only during ingestion batches and released with `keep_alive=0`. |
| Reranker (Phase 8) | BGE reranker on **CPU** | Keeps VRAM free for generation. The 5900HX's 8 vCPUs are fast enough for ~20–50 pairs per query. |

## 4. Memory budget (measured)

**VRAM (6144 MiB total).** `nvidia-smi` showed **4989 MiB used** with both models loaded. That is 3.72 GB of models plus ~1.2 GB used by the Windows desktop compositor and other apps. Headroom is ~1.1 GB, enough for longer contexts but **not** for a vision model on top. Rule: **at most two models resident** (`OLLAMA_MAX_LOADED_MODELS=2`).

**System RAM.**
| Consumer | Budget | Measured |
|---|---|---|
| Windows + apps | 4–5 GB | — |
| Ollama host process | 1–2 GB | — |
| WSL VM cap | 8 GB | `free -m` total 7941 MB |
| ↳ Neo4j container | ≤ 2 GB | **940 MiB** at idle |
| ↳ API (uvicorn) | ~0.5 GB now; +2–3 GB once Docling/torch arrive (Phase 2) | — |

⚠️ Docker Desktop also auto-started another project's Supabase containers (`supabase_*_tournament-in-a-box`, ~530 MB in total). They count against the same 8 GB, so stop them when working on OmniRAG.

## 5. What was configured

### `C:\Users\gautam\.wslconfig`
```ini
[wsl2]
memory=8GB          # hard cap for the VM (Ubuntu + Docker)
swap=4GB            # spill-over instead of OOM-kill during big parses
processors=8        # leave 8 of 16 threads for Windows + Ollama
networkingMode=mirrored   # WSL localhost == Windows localhost

[experimental]
autoMemoryReclaim=gradual # give cached memory back to Windows over time
```
Applied with `wsl --shutdown`.

### Ollama (Windows user environment variables)
| Variable | Value | Reason |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | 1 | Faster attention; required for quantized KV cache |
| `OLLAMA_KV_CACHE_TYPE` | q8_0 | Halves KV-cache VRAM with negligible quality loss |
| `OLLAMA_MAX_LOADED_MODELS` | 2 | LLM + embedder fit; a third would evict or spill to CPU |
| `OLLAMA_NUM_PARALLEL` | 1 | Each parallel slot multiplies the KV cache; this is a single-user app |
| `OLLAMA_CONTEXT_LENGTH` | 8192 | Enough for ~6–8 retrieved chunks + prompt; bounds the KV cache |
| `OLLAMA_KEEP_ALIVE` | 10m | Avoids the ~20 s cold load between questions |

These take effect only after restarting the Ollama tray app.

### WSL `~/.profile`
```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/omnirag"   # venv on ext4, source on /mnt/p
export WATCHFILES_FORCE_POLLING=true                   # inotify does not fire on /mnt/*
```
**Lesson learned:** these were first added to `~/.bashrc`, but Ubuntu's `.bashrc` begins with `case $- in *i*) ;; *) return;; esac`, so it exits early in every non-interactive shell (scripts, `wsl -- bash -lc`, IDE runners). `uv` therefore created `.venv` on P:. The fix: put environment variables in `~/.profile`.

Python: Ubuntu 26.04 ships Python 3.14. `uv python install 3.11` provides 3.11.15 without touching the system Python or adding PPAs.

## 6. Verify it yourself
Run these in an Ubuntu terminal:
```bash
free -h                                   # total ≈ 7.7Gi
nproc                                     # 8
curl -s localhost:11434/api/version       # {"version":"0.40.14"}  ← proves mirrored networking
nvidia-smi                                # GPU visible from WSL
echo $UV_PROJECT_ENVIRONMENT              # /home/gautam/.venvs/omnirag
docker info --format '{{.MemTotal}}'      # ≈ 8.3e9 → Docker is inside the 8 GB cap
ollama list                               # (in PowerShell) qwen3:4b-instruct, qwen3:4b, bge-m3
```

## 7. Common errors
| Symptom | Root cause | Fix |
|---|---|---|
| `curl localhost:11434` fails in WSL | NAT mode, or WSL not restarted after `.wslconfig` | `wsl --shutdown` from PowerShell, reopen Ubuntu |
| Ollama ignores the env vars | Tray app started before `setx` | Quit Ollama from the tray, start it again |
| `.venv/` appears in the repo | `UV_PROJECT_ENVIRONMENT` not set in that shell | Check `~/.profile`; open a new login shell |
| First chat takes 20–55 s | Cold model load from disk + CUDA init | Expected once; `keep_alive` keeps it warm |
| `gpu_percent` < 100 | VRAM full, so layers spill to CPU | Unload the extra model (`ollama stop <model>`), shorten context |
| `docker: cannot connect` in WSL | Docker Desktop not running | Start Docker Desktop; WSL integration is on for Ubuntu |
| `The command 'docker' could not be found in this WSL 2 distro` | Docker Desktop not running (its WSL integration only exists while it runs) | Start Docker Desktop, then `docker compose up -d`. Neo4j health then shows `TimeoutError` until the container is up |
| Health: *"None of the required models are on this server (Ollama 0.32.3 …)"* | **Another Ollama took port 11434.** Seen 2026-11-05: an older Ollama installed as a systemd service *inside WSL* started first at boot. Mirrored networking shares ports between Windows and WSL, so whichever server starts first wins, at random | Find it: `ss -ltnp \| grep 11434`, `systemctl status ollama`. Move it with a systemd drop-in (`Environment="OLLAMA_HOST=127.0.0.1:11435"` in `/etc/systemd/system/ollama.service.d/port.conf`, then `daemon-reload` + `restart`) or disable it. Then `curl localhost:11434/api/version` must show the Windows version |

## 8. Acceptance criteria — Phase 0
- [x] WSL capped at 8 GB / 8 vCPU (measured 7941 MB / 8)
- [x] Ollama reachable from WSL at `localhost:11434`
- [x] `qwen3:4b-instruct` + `bge-m3` both 100% on GPU (3.72 GB of models; 4989/6144 MiB on the card)
- [x] Python 3.11.15 via uv; venv on ext4
- [x] Docker engine reachable from WSL; Neo4j healthy at 940 MiB
