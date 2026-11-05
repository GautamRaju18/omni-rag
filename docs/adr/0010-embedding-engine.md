# ADR-0010: Embedding engine: versioned spaces, Ollama dense in production, BGE-M3 sparse/multi-vector as local experiments

**Status:** accepted (2026-11-05, Phase 4)

## Context
The brief: BGE-M3 first; document and query embedding; batching, caching, dimension validation, normalisation, error handling, versioning; an abstraction so models can be replaced without changing retrieval; dense through Ollama first; BGE-M3 sparse investigated separately through a compatible interface, without assuming Ollama returns it; compare models where practical.

Facts measured on 2026-11-05 (Ollama 0.40.14, `bge-m3` F16 GGUF, digest `79076464…`):
- `/api/embed` returns only `embeddings` (dense). No sparse or multi-vector output.
- **The usable input is 2,048 tokens including 2 special tokens**, not the advertised 8,192. 2,046 content tokens work; 2,047 return HTTP 413. `num_ctx: 8192` doesn't change it. With the default `truncate: true`, a ~9,000-token input returned 200 with only 2,048 tokens embedded, **silently**.
- An empty string returns a (meaningless) vector; a missing model returns 404.
- Ollama vectors vs the original fp32 weights: cosine ≥ 0.99992 on 58 chunks, and identical retrieval scores.

## Decision
- **Interfaces** (`embeddings/base.py`): `DenseEmbedder` (`embed_documents`, `embed_query`) and `SparseEmbedder`. Each exposes an `EmbeddingSpace`: provider, model, **revision** (Ollama digest / HF commit), kind, dimension, measured token limit, and query/document prefixes. Retrieval code only sees these.
- **Versioning:** `space_id = <kind>-<provider>-<model>-<revision[:12]>`, e.g. `dense-ollama-bge-m3-790764642607`. It's part of every cache key and every chunk's `embedding_ref = <space_id>/<sha256 of embedded text>`. A re-pulled model with new weights is a new space; stale vectors can't be mixed in (tested).
- **Production dense path:** `OllamaDenseEmbedder`. Batches of 32; always `truncate: false`; token counts checked before sending (bge-m3 tokenizer, limit 2,048 incl. specials); empty input rejected; dimension checked against the model on first use and on every response; NaN and zero vectors rejected; L2 normalisation; retries with exponential backoff on connection errors, timeouts, 429 and 5xx; a rejected batch is bisected to name the failing input.
- **Ingestion embeds `embedding_index_configs`** (default `multimodal-256-ctx`, from the Phase 3b benchmark). Failure is a warning; `python -m omnirag.embeddings embed` backfills and `verify` checks for stale or missing vectors.
- **BGE-M3 sparse and ColBERT**: `BGEM3Local` runs the pinned original weights on CPU (transformers, ~60 lines, no FlagEmbedding). It is **verified against the model card**: lexical weights within 0.0002, lexical score 0.19554 vs 0.195549, ColBERT 0.7797/0.4621 vs 0.7797/0.4620. It's used for experiments and benchmarks, **not at ingestion**.
- Hierarchical chunking's top level defaults to **2,000 tokens** (was 2,048) so that every chunk can be embedded.

## Measured comparison (seed set, `multimodal-256-ctx`, 58 chunks, 37 questions, K=5)
| Method | Recall@5 | MRR | Encode | Bytes/chunk |
|---|---|---|---|---|
| BM25 | 0.986 | 0.900 | — | — |
| dense, Ollama bge-m3 (GPU) | 0.932 | 0.863 | 41.6 chunks/s | 4,096 |
| dense, HF bge-m3 fp32 (CPU) | 0.932 | 0.863 | 1.9 chunks/s | 4,096 |
| dense, Ollama nomic-embed-text v1.5 | 0.919 | 0.821 | 54.8 chunks/s | 3,072 |
| sparse, BGE-M3 lexical weights | 0.905 | 0.761 | 1.9 chunks/s | 392 |
| multi-vector, BGE-M3 ColBERT | 0.959 | 0.901 | 1.9 chunks/s | 433,752 |

Per question against Ollama dense: BM25 +2/−0, ColBERT +1/−0, sparse +2/−3, nomic +1/−2. **Every difference is three questions or fewer.**

## Consequences
- The production embedding stays on the GPU at ~42 chunks/s and 0.69 GB VRAM.
- BGE-M3 sparse **did not beat BM25 here** and costs 2.7 GB RAM and 22× slower CPU encoding, so it isn't adopted. It stays available through `SparseEmbedder`.
- ColBERT was the best neural method by one question, at ~106× the storage. It's a candidate for **reranking a small top-K in Phase 8**, where storage doesn't matter, not for indexing.
- BM25 winning q04/q35 where dense fails supports hybrid retrieval (Phase 7).
- The old Phase 3b cache entries (key `bge-m3`, unversioned) remain in the cache but are never read for the current space.
- Inputs longer than 2,046 tokens can't be embedded with this Ollama version. If a later Ollama raises the limit, update `OMNIRAG_EMBEDDING_MAX_TOKENS` after re-measuring.

## Alternatives considered
| Option | Why not |
|---|---|
| sentence-transformers / FlagEmbedding for everything | CPU-only here (GPU reserved for Ollama), 22× slower; extra dependencies |
| Trusting `truncate: true` | Silent data loss: a 9k-token chunk was embedded as its first 2,048 tokens |
| Versioning by model name only | `ollama pull` can change the weights behind the same name |
| Storing vectors per document as `.npy` | The cache already is content-addressed storage; Neo4j becomes the vector store in Phase 5 |
