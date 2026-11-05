# Phase 3b: Semantic chunking, contextual enrichment and the chunking benchmark

> Builds on [Phase 3a](phase-03a-chunking-strategies.md). Decision records: [ADR-0008](../adr/0008-chunking-tokens-and-element-provenance.md) and [ADR-0009](../adr/0009-evidence-based-relevance-and-early-embeddings.md). Raw results: [docs/benchmarks/chunking-2026-11-05.json](../benchmarks/chunking-2026-11-05.json).

## 1. Problem being solved
Phase 3a built seven strategies and measured *structure* (whole elements, sizes). That doesn't say which strategy **finds answers**. Phase 3b adds the eighth strategy (semantic) and an optional LLM context experiment. It then measures all of them on the same questions, with the same retrievers and the same relevance rule, as the brief requires, while being careful about what a small seed set can and can't show.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Semantic chunking** | Split prose into sentences, embed each sentence with its neighbours, and break where consecutive embeddings are unusually far apart (above the document's 90th-percentile distance). Still within sections and the size limit. |
| **Contextual retrieval** | Prepend a short "where this chunk sits in the document" text to the **embedding text**. Two variants: a deterministic header ("Document: … / Section: …") and LLM-written context (qwen3:4b-instruct). |
| **Seed evaluation set** | 40 questions (37 answerable, 3 unanswerable) of 7 types, each fact backed by **verbatim quotes** with alternatives across documents. |
| **Recall@K (fact recall)** | The share of a question's facts found in the top K returned chunks. |
| **Precision@K / MRR / nDCG@K** | How many of the top K are relevant / how high the first relevant one is / ranking quality vs the ideal. |
| **Context precision (token-weighted)** | The share of the returned *tokens* that belong to relevant chunks: what fraction of the LLM's reading is useful. |
| **Dense vs BM25** | Embedding similarity (meaning) vs term matching (exact words and IDs). |
| **Statistical caution** | With 37 questions, one question moves recall by 0.027. Compare per question (wins/losses), not just averages. |

## 3. What was added
```
chunking/semantic.py      sentence splitter (handles "Dr.", "3. Milestone"), windows, distances
chunking/strategies.py    + SEMANTIC strategy (one embedding batch per document)
chunking/enrichment.py    LLMContextualizer: one chat call per chunk → chunk.context (generated)
embeddings/ollama.py      OllamaEmbedder (batched /api/embed) + EmbeddingCache (SQLite)
evaluation/dataset.py     questions, facts, alternatives; covered_facts(); resolve() → element IDs
evaluation/metrics.py     Hit / Recall / Precision / MRR / nDCG / context precision @K
evaluation/retrieval.py   exact DenseIndex, BM25Index (keeps AUR-C103, E-7, 2026-03-19 whole)
evaluation/chunking_benchmark.py   24 configurations × 2 retrievers → tables + JSON
data/eval/chunking_seed.json       the 40 seed questions
```
New settings: semantic is opt-in, via `OMNIRAG_CHUNKING_STRATEGIES='[..., "semantic"]'` (it calls Ollama at ingestion). New dependency: `numpy==2.4.6`.

## 4. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.evaluation.chunking_benchmark                 # ~35 s (embeddings cached)
uv run python -m omnirag.evaluation.chunking_benchmark --llm-context   # + ~105 s of LLM calls
uv run python -m omnirag.chunking compare --strategy semantic --size 256
```

## 5. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                                # 236 unit tests (~13 s, no Docling, no Ollama)
uv run pytest -m "integration or docling"    # 8 tests: live services, tokenizer, real Docling
```

## 6. Results (measured 2026-11-05)
Setup: 12 sample documents, 37 answerable questions, K = 5, bge-m3 via Ollama 0.40.14, qwen3:4b-instruct for LLM context. Selected rows from the full tables in the JSON file:

**Dense (bge-m3, exact cosine)**
| Config | Indexed | Med tok | Recall@5 | MRR | nDCG@5 | Ctx prec |
|---|---|---|---|---|---|---|
| fixed-128-16 | 43 | 128 | 0.824 | 0.681 | 0.670 | 0.238 |
| recursive-256-32 | 27 | 181 | 0.919 | 0.796 | 0.781 | 0.302 |
| structure-256 | 51 | 70 | 0.905 | 0.771 | 0.756 | 0.316 |
| table-256 | 56 | 65 | 0.851 | 0.719 | 0.714 | 0.305 |
| multimodal-256 | 58 | 56 | 0.878 | 0.786 | 0.772 | 0.303 |
| semantic-256-90 | 67 | 49 | 0.892 | 0.768 | 0.752 | 0.292 |
| **multimodal-256-ctx** (headers) | 58 | 56 | **0.932** | **0.863** | **0.829** | 0.309 |
| multimodal-256-llmctx | 58 | 56 | 0.905 | 0.822 | 0.796 | 0.284 |
| parentchild-512-128-32 | 77 | 65 | 0.905 | 0.795 | 0.782 | 0.293 |
| hierarchical-2048-512-128 | 82 | 104 | 0.959 | 0.885 | 0.766 | 0.487 |
| fixed-512-64 | 16 | 294 | 1.000 | 0.798 | 0.819 | 0.284 |

**BM25**: higher recall than dense for almost every configuration (e.g. multimodal-256 0.959 vs 0.878). The best MRR is multimodal-256-llmctx at 0.938, and fixed-512-64 reaches 0.973.

Costs (one run, so a single slow value may be noise): chunking all 12 documents took 23–89 ms for most configurations; parentchild-512 took 269 ms, and semantic-128 took 692 ms (the first embedding of its sentence windows; the other semantic sizes reused the cache: 45–51 ms). LLM context: **58 chat calls in 104 s (1.8 s per chunk)**. Search: under 0.3 ms per query at this size. Index: 78–664 KB.

### What these numbers do and don't show
1. **Big-chunk "wins" are a corpus artifact.** fixed/recursive-512 reach recall 1.000 because 12 small documents become 16 chunks, so a top-5 search returns almost a third of the corpus. Their context precision (≈0.29) shows the cost: most of what the LLM would read is irrelevant. This will flip on a corpus of long documents.
2. **Hierarchical's high precision (0.43–0.45) is inflated.** Its levels repeat the same text, so one fact counts as several relevant results. Merging overlapping levels ("auto-merging retrieval") belongs in Phase 7.
3. **Most strategy differences are 0–2 questions.** Per-question comparison: table-256 vs structure-256 is **0 better / 2 worse / 35 tied**, and multimodal-256 + headers vs without is **2 better / 0 worse / 35 tied**.
4. **The one consistent improvement is contextual headers** (dense recall +0.054, MRR +0.077 on multimodal-256; better on BM25 too). They cost 4 extra tokens and no model calls.
5. **LLM-written context helped BM25 most** (MRR 0.938) but was *weaker than plain headers* for dense retrieval (0.905 vs 0.932), at 1.8 s per chunk. On this corpus it isn't worth its cost.
6. **Why table-aware lost two table questions (q04, q15):** structure-aware keeps the results table *together with* the sentence after it ("Cell AUR-C103 failed at cycle 412 …"). Table-aware isolates the table, and **a bare Markdown table embeds poorly against a natural-language question**. Headers didn't fix q04. Phase 7's hybrid retrieval (BM25 found these) is the more promising fix.
7. **BM25 is strong here** because many questions contain exact identifiers (AUR-C103, M3, NaDFOB, R2). That's a reason for hybrid retrieval, not for dropping dense.

**Choice for now:** keep all strategies available; Phase 4 indexes **multimodal-256 with contextual headers** as the default (best dense result among chunk-sized configurations, with tables and figures kept intact), and keeps structure-256 for comparison. That's a decision for *this* corpus, to be revisited after Phase 12's larger dataset. **No strategy is universally better.**

## 7. Bugs found during verification
1. **"3." became its own 1-token chunk.** The sentence splitter treated a list number at line start ("3. Milestone M2 risk") as a sentence end. List and section markers are now excluded, with a test.
2. **The evaluation set was silently git-ignored** (`data/*` excluded everything except samples). Caught before committing; `!data/eval/` added.
3. **The evidence test ran Docling inside the fast unit suite.** `pdf_engine=auto` sent the two-column note to Docling, doubling the suite time (13 → 27 s) and showing a Docling deprecation warning. It's now split: a fast PyMuPDF check of 57 quotes, and a `-m docling` check of the 5 layout-dependent quotes.
4. My own test mistakes: a chunk size below the schema minimum, a too-strict `only` filter (q07 also has quotes in other files), and a hard-coded quote count (64 guessed, 57 real; the test now computes it).

## 8. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `Semantic chunking needs an embedder` | `semantic` in strategies but no embedder | Use `build_chunker(settings)` (adds bge-m3) or pass an embedder |
| `EmbeddingError: Ollama embed call failed` | Ollama down or `bge-m3` missing | Check `/api/v1/health/ollama` |
| `LookupError: q12: '…' not found in …` | A sample or loader changed the text | Update the quote (verbatim), never weaken the check |
| Benchmark much slower than ~35 s | Embedding cache cleared, or `--llm-context` | Expected on first run; the cache lives in `<data_dir>/cache/embeddings.sqlite` |
| Dense and BM25 disagree strongly | Different strengths (meaning vs exact terms) | Expected; that's what hybrid retrieval (Phase 7) combines |

## 9. Exercise (≈60 min)
Add **5 new questions** to `data/eval/chunking_seed.json` that you expect to favour *small* chunks: precise facts buried in long paragraphs (the meeting notes and safety review are good candidates).
1. Write the quotes verbatim (short), with alternatives where the fact repeats.
2. Run `uv run pytest tests/unit/test_evaluation.py` until every quote resolves.
3. Re-run the benchmark and compare the 128- vs 512-token rows on just your questions (filter the JSON or add a `--only` flag).

Acceptance: the tests pass. In a short note, record the per-question wins and losses (with the date), and whether your expectation held.

## 10. Acceptance criteria — Phase 3b (completes Phase 3)
- [x] Semantic chunking (eighth strategy), section- and size-constrained, one embedding batch per document
- [x] Contextual enrichment: deterministic headers and an LLM-context experiment, both embedding-only and provenance-marked
- [x] Seed evaluation set with verbatim, test-verified evidence and element-level ground truth
- [x] Recall@K, Precision@K, MRR, nDCG@K, hit rate, context precision/recall, latency and index size for 24 configurations × dense + BM25
- [x] Results recorded with date, versions and an explicit list of what they *don't* show
- [x] 236 unit + 8 live/Docling tests pass; ruff clean
