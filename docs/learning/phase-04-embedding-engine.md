# Phase 4: Embedding engine

> Decision record: [ADR-0010](../adr/0010-embedding-engine.md). Raw comparison: [docs/benchmarks/embeddings-2026-11-05.json](../benchmarks/embeddings-2026-11-05.json). Builds on the minimal client from [Phase 3b](phase-03b-semantic-chunking-and-benchmark.md).

## 1. Problem being solved
Dense retrieval compares a query vector with chunk vectors, so those vectors must be:

- **correct**: the right model, the right dimension, the whole text embedded;
- **comparable**: the same model *version* for queries and chunks;
- **cheap to recompute**: caching and batching;
- **replaceable**: the model can change without rewriting retrieval.

The brief also asks for an honest look at BGE-M3's two other outputs (sparse and multi-vector), which Ollama doesn't provide.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Dense embedding** | One vector per text (1024 numbers for bge-m3). Similar meaning gives a similar direction. |
| **Normalisation** | Scaling every vector to length 1, so a dot product *is* the cosine similarity. |
| **Embedding space** | Everything that determines a vector: provider, model, **exact weights revision**, kind, dimension and prefixes. Vectors from different spaces are not comparable, ever. |
| **Content digest** | Ollama's SHA-256 of the model blob. The same name can point to new weights after `ollama pull`; the digest can't. |
| **Query/document prefixes** | Some models expect `search_query:` / `search_document:` (nomic). bge-m3 needs none. The interface applies them so callers can't forget. |
| **Silent truncation** | A backend embedding only the first N tokens without saying so. Ollama does this by default (§6). |
| **Sparse (learned lexical) embedding** | BGE-M3's per-token weights: like BM25, but the model learned which tokens matter. Score = Σ wq·wd over shared tokens. |
| **Multi-vector (ColBERT)** | One vector per token. Score = for each query token, its best-matching document token, averaged ("late interaction"). Precise, but very large. |

## 3. Architecture
```
embeddings/
  base.py          EmbeddingSpace (space_id), DenseEmbedder / SparseEmbedder protocols,
                   validate_dense, sparse_dot, EmbeddingError / EmbeddingInputError
  ollama.py        OllamaDenseEmbedder (+ EmbeddingCache, SQLite keyed by space_id + text hash)
  bge_m3_local.py  BGEM3Local: pinned original weights on CPU → dense + sparse + ColBERT
  service.py       ChunkEmbedder (sets embedding_ref), load_vectors(refs)
  __main__.py      CLI: info | embed | verify
evaluation/embedding_benchmark.py   6 methods on the same chunks and questions

IngestionPipeline: loader → enrichers → chunk → embed (embedding_index_configs) → write chunks
chunk.embedding_ref = "dense-ollama-bge-m3-790764642607/<sha256 of embedded text>"
```
Default settings now follow the Phase 3b decision: `chunk_size=256`, `chunk_overlap=32`, `chunk_contextual_headers=true`, `embedding_index_configs=["multimodal-256-ctx"]`, and `chunk_parent_size=2000`, which keeps hierarchical top levels embeddable.

## 4. Error handling, in the order it happens
1. **Empty input** → `EmbeddingInputError("Input 1 is empty", index=1)`, never sent.
2. **Too long** (bge-m3 tokens + 2 > 2,048) → rejected before the request, with the index and token count.
3. **Request**: `truncate: false`. On connection errors, timeouts, 429 or 5xx: retry with exponential backoff (3 times).
4. **404** → "model not found (ollama pull bge-m3)". **400/413** on a batch → bisect to the rejected input and report its index.
5. **Response**: shape must be (n, dim), finite, and no zero vectors → normalise.
6. **First use**: `/api/tags` (digest) and `/api/show` (dimension) → `EmbeddingSpace`; a dimension mismatch with `OMNIRAG_EMBEDDING_DIM` is an error.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.embeddings info                  # space, digest, dimension, limit, cache
uv run python -m omnirag.chunking rechunk --all           # new defaults (256 + headers)
uv run python -m omnirag.embeddings embed --all           # backfill embedding refs
uv run python -m omnirag.embeddings verify                # every indexed chunk has a current vector
ollama.exe pull nomic-embed-text:v1.5                     # only for the comparison
uv run python -m omnirag.evaluation.embedding_benchmark   # ~70 s (local BGE-M3 on CPU)
```
The local BGE-M3 needs its weights once (2.27 GB, pinned revision, SHA-256 verified) in `~/omnirag-data/models/`.

## 6. Expected outputs (measured 2026-11-05)
**Ollama facts:**
```text
/api/embed keys: model, embeddings, total_duration, load_duration, prompt_eval_count   (dense only)
2046 content tokens (+2 special = 2048): HTTP 200
2047 content tokens (+2 special = 2049): HTTP 413 "the input length exceeds the context length"
~9000 tokens, truncate default:            HTTP 200, prompt_eval_count = 2048   ← silent truncation
```
**`embeddings info`:**
```text
space_id   dense-ollama-bge-m3-790764642607
revision   '7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab'
dim 1024 · max_tokens 2048 · normalized True
cache: bge-m3 (unversioned, Phase 3b) 698 · dense-ollama-bge-m3-790764642607 59 <- current
```
**Error paths (real Ollama):** "Input 1 is empty" · "Input 1 has 3603 tokens; bge-m3 accepts 2048" · bisected server rejection "Input 3: … exceeds the context length" · "Model 'nope-model:latest' not found in Ollama" · "bge-m3:latest has dimension 1024, expected 768".

**BGE-M3 local vs the model card:**
| Output | Ours (fp32) | Model card (fp16) |
|---|---|---|
| Lexical weights "What is BGE M3?" | What 0.08362, B 0.12965, GE 0.25187, 3 0.26958 | 0.08356, 0.1296, 0.252, 0.2695 |
| Lexical score q·p | 0.19554 | 0.19554901 |
| ColBERT q·p₁ / q·p₂ | 0.7797 / 0.4621 | 0.7797 / 0.4620 |
| Dense similarity matrix | [[0.6259, 0.3475], [0.3499, 0.6782]] | [[0.6265, 0.3477], [0.3499, 0.678]] |

Ollama GGUF vs these fp32 weights: **cosine ≥ 0.99992** on all 58 chunks.

**Comparison** (`multimodal-256-ctx`, 58 chunks, 37 questions, K=5):
| Method | Recall@5 | MRR | nDCG@5 | Encode | Query | Bytes/chunk |
|---|---|---|---|---|---|---|
| BM25 | 0.986 | 0.900 | 0.857 | — | 0.1 ms | — |
| dense Ollama bge-m3 | 0.932 | 0.863 | 0.829 | 1.4 s (41.6/s) | 28 ms | 4,096 |
| dense HF bge-m3 (CPU) | 0.932 | 0.863 | 0.829 | 31 s (1.9/s) | 80 ms | 4,096 |
| dense Ollama nomic-embed-text | 0.919 | 0.821 | 0.790 | 1.1 s (54.8/s) | 22 ms | 3,072 |
| sparse BGE-M3 | 0.905 | 0.761 | 0.704 | (same pass) | | 392 |
| ColBERT BGE-M3 | 0.959 | 0.901 | 0.863 | (same pass) | | 433,752 |

Local BGE-M3: 13 s to load, **2.7 GB RSS**. Ingesting all 12 samples with embeddings: 46 s. This is slower than Phase 2d's 25 s, but the extra time was in the deck (vision) and the Docling note, not in embedding (≈2 s uncached). Not investigated further.

**What it means:**
1. **Ollama's GGUF is faithful.** It gives identical retrieval scores to the original weights, 22× faster on the GPU. Production stays on Ollama.
2. **BGE-M3 sparse lost to plain BM25** (0.905 vs 0.986; per question +2/−3 against dense). Not adopted for indexing.
3. **ColBERT was the best neural method, by one question**, at ~106× the storage. That points to reranking a small top-K in Phase 8, not to indexing every chunk.
4. **nomic-embed-text plugged in with no retrieval changes** (different dimension and prefixes), and scored slightly lower.
5. **BM25 complements dense** (+2/−0 per question), which supports the hybrid design of Phase 7.

## 7. Bugs and surprises found during verification
1. **Silent truncation by Ollama** (§6), and the real 2,048-token limit. Found by probing the API before writing code. Without the probe, a long hierarchical chunk would have been indexed as its first 2,048 tokens with no warning.
2. **The hierarchical default (2,048 content tokens) could not be embedded** (limit 2,046). The default is now 2,000.
3. **Old vectors were stored under an unversioned key** (`bge-m3`, from Phase 3b). They are kept, but are never read for the versioned space. That's exactly the confusion versioning prevents.
4. **Mutation checks:** removing `truncate: false`, and dropping the revision from `space_id`, both made tests fail, as they should.
5. Test fixes: `.space` property reads written only to trigger errors (lint B018) are now assigned to `_`.

## 8. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `Input N has … tokens; bge-m3 accepts 2048` | A chunk is too long for the embedder | Lower `OMNIRAG_CHUNK_PARENT_SIZE` / chunk size, then rechunk |
| `Model 'bge-m3:latest' not found in Ollama` | Model not pulled, or another Ollama on the port | `ollama pull bge-m3`; check `/api/v1/health/ollama` (the Phase 0 doc covers the port conflict) |
| `has dimension 768, expected 1024` | `OMNIRAG_EMBEDDING_MODEL` changed without `OMNIRAG_EMBEDDING_DIM` | Set both, then `embed --all` (a new space; old vectors stay unused) |
| `verify` shows "stale" | The model was re-pulled with new weights (new digest) | `python -m omnirag.embeddings embed --all` |
| `⚠ Embedding failed: … unreachable` at ingestion | Ollama down | Chunks are stored; run `embed --all` later |
| Local BGE-M3 is very slow / RAM grows by 2.7 GB | CPU model, by design | Benchmarks only; never enable at ingestion |

## 9. Exercise (≈45 min)
Add a `prune` command to `python -m omnirag.embeddings` that **lists**, and with `--yes` deletes, cache entries from spaces other than the current one (e.g. the unversioned `bge-m3` rows).
1. Add `EmbeddingCache.delete_space(space_id) -> int`.
2. Without `--yes`, only print what would be deleted (a dry run is the default for destructive commands).
3. Never delete the current space.

Acceptance: a unit test with two spaces shows the dry run deletes nothing and `--yes` removes only the old space; `verify` still passes afterwards on your data; ruff is clean.

## 10. Acceptance criteria — Phase 4
- [x] Document and query embedding through one model-agnostic interface; a second model (nomic) swapped in without retrieval changes
- [x] Batching, caching (content-addressed, per space), dimension validation, normalisation
- [x] Error handling: empty, too long (pre-checked, never truncated), bisected server rejection, retries, missing model, wrong dimension
- [x] Versioning by exact model digest; versioned `embedding_ref` on every indexed chunk; `verify` finds stale or missing vectors
- [x] BGE-M3 sparse and multi-vector investigated through a separate interface, verified against the model card, measured, and not adopted for indexing (with reasons)
- [x] Comparison of 6 methods with quality, speed, memory and storage
- [x] 243 unit + 8 live/Docling tests pass; ruff clean
