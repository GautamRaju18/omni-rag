# ADR-0015: Cross-encoder reranking with bge-reranker-v2-m3 (CPU today; GPU is an open decision)

**Status:** accepted (2026-11-05, Phase 8)

## Context
The brief: "Add a cross-encoder reranker. Begin with a lightweight compatible BGE reranker and provide a CPU fallback", plus retrieval mode 5, "Hybrid + reranking", and a reranking-latency metric.

Facts measured on 2026-11-05:
- **No rerank endpoint in Ollama:** Ollama 0.40.14 returns a plain 404 for `/api/rerank`.
- **CPU-only torch:** the project's torch is the CPU build (ADR-0005: the RTX 3060's 6 GB are Ollama's; CUDA wheels add several GB).
- **CPU:** WSL sees 4 cores / 8 threads of a Ryzen 9 5900HX, AVX2 only, no AVX-512 or AMX.
- **Candidate models** (pinned revisions, SHA-256 from the HF API, safetensors):

  | Model | Size | Architecture | License |
  |---|---|---|---|
  | `BAAI/bge-reranker-v2-m3` (`953dc6f6…`) | 2.27 GB | XLM-R large, 24 layers | Apache-2.0 |
  | `BAAI/bge-reranker-base` (`2cfc18c9…`) | 1.11 GB | XLM-R base, 12 layers | MIT |

- **Implementation check:** v2-m3 reproduces the model card in fp32 on CPU: logits −5.650 / −8.184 / 5.265, against the card's fp16 −5.652 / −8.188 / 5.262.

## Decision
**Reranker** (`retrieval/reranker.py`):
- `CrossEncoderReranker`: transformers `XLMRobertaForSequenceClassification` plus the model's own `tokenizer.json`.
- Input pairs are `(question, chunk context + content)`, with only the passage truncated (`max_length` 512).
- Batches are length-sorted.
- Weights load lazily on the first reranked query.
- **Device:** `auto` = CUDA if a CUDA torch is installed, else CPU; `cuda` without CUDA logs a warning and falls back to CPU. Both paths are tested.
- **Scores:** raw logits plus the sigmoid.

**Mode:** `RerankingRetriever` takes `depth` candidates from a first stage, re-scores them, and keeps k. Each hit keeps its first-stage provenance plus `sources.rerank` (rank, logit, first-stage rank).
- New modes: `dense+bm25+rerank` and `dense+bm25+graph+rerank`.

**Settings:**
- `rerank_model` = `bge-reranker-v2-m3`;
- `rerank_depth` = 20;
- `rerank_batch_size` = 4 (beat 16 on CPU: less padding);
- `rerank_threads` = 8.

**The default retrieval mode stays `dense+bm25`.** On this CPU a reranked query takes 9.5–17 s, so reranking is the explicit quality mode.

## Measured (K=5; seed set 37 questions, relational set 8; [rerank-2026-11-05.json](../benchmarks/rerank-2026-11-05.json), [rerank-relational-2026-11-05.json](../benchmarks/rerank-relational-2026-11-05.json))
| Mode | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | rel. nDCG | p50 (CPU) |
|---|---|---|---|---|---|---|---|
| dense+bm25 (default) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 19 ms |
| **+ rerank v2-m3, depth 20** | **1.000** | **0.980** | **0.948** | **1.000** | **0.938** | **0.867** | 17.0 s |
| + rerank v2-m3, depth 10 | 1.000 | 0.980 | 0.940 | 0.938 | 0.938 | 0.834 | 9.5 s |
| + rerank base, depth 10 | 0.973 | 0.887 | 0.862 | 0.812 | 0.750 | 0.694 | 2.2 s |
| + rerank base, depth 20 | 0.973 | 0.874 | 0.861 | 0.812 | 0.750 | 0.694 | 5.5 s |
| graph + rerank v2-m3, depth 20 | 1.000 | 0.980 | 0.948 | 0.938 | 0.938 | 0.834 | 16.9 s |

Per question:
- **v2-m3 against dense+bm25:** it wins q22 (seed) and r02 and r05 (relational, depth 20) and **loses nothing**.
- **base** loses q04 and r08 and wins q22 and r05. On aggregate it is worse than no reranking.
- **Graph** candidates add nothing even after reranking. On the relational set, graph + rerank at depth 20 misses r02, which rerank without graph finds.

Reading q22, the only seed-set recall win: its ground truth for the project target is the config record `target_cycles: 3000`. Reranking moves that record from #6 to #4, with a sigmoid of only 0.065. The research-notes paragraph that states the same target was already #3 without reranking, but it isn't in the ground truth (one quote per fact). So the seed recall gain is mostly a ground-truth gap; **the MRR/nDCG gains are the real effect**. Fixing the seed set (alternative evidence) is a deliberate dataset v2 re-baseline.

CPU tuning (20 real pairs, mean 100 tokens):
- **Batch size:** 4 instead of 16 cut v2-m3 from 17.5 s to 11.8 s, and base from 5.1 s to 2.4 s.
- **Threads:** 4 vs 8 made little difference.
- **Dynamic int8 quantization was rejected:**
  - v2-m3 was 2.2× faster, but its logits moved by 1.0 on average (the model card's "hi" pair went from −8.18 to −5.56);
  - base lost one of its top 5;
  - `torch.ao.quantization` is deprecated in this torch.

Weights load from disk in 2.4 s (v2-m3) and 1.0 s (base). The first reranked query in a fresh process also pays warm-up (29 s measured once for v2-m3, against 17 s warm).

**The reranker's score doesn't detect unanswerable questions:**

| Question | v2-m3 top score |
|---|---|
| q40, about the nonexistent cell AUR-C109 | 0.936 |
| q39, "Who is the CEO of Helios Energy Lab?" | 0.890 |
| q38 | 0.207 |
| lowest answerable questions: q32 / q20 | 0.071 / 0.176 |

No threshold separates the two groups.

## Consequences
- **Quality:** reranking is the single biggest improvement of Phases 6–8 (seed nDCG 0.889 → 0.948, relational 0.754 → 0.867).
- **Latency on CPU:** 9.5–17 s per query rules it out as a default for chat (Phase 10).
- **The open decision: run the reranker on the GPU.** It needs a CUDA torch in the `layout` group (several GB, reversing ADR-0005) and ~1–2 GB of VRAM next to qwen3:4b-instruct (3.3 GB) and bge-m3 (0.6 GB). GPU latency is **not measured**; it would be measured before any default changes.
- **Cheaper CPU option: depth 10.** It keeps the seed results (nDCG 0.940) and loses r02 on the relational set, at 9.5 s.
- **Phase 9 needs another "insufficient context" signal.** The reranker's top score is not one; for example, compare the question's identifiers against the retrieved text, or have the LLM judge the context.
- **Graph retrieval stays opt-in** (ADR-0014): reranking didn't make its candidates useful here.
