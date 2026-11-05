# Phase 8: Cross-encoder reranking

> Decision record: [ADR-0015](../adr/0015-cross-encoder-reranking.md). Results: [rerank-2026-11-05.json](../benchmarks/rerank-2026-11-05.json) (seed set), [rerank-relational-2026-11-05.json](../benchmarks/rerank-relational-2026-11-05.json) (relational set). Builds on the fused retrieval of [Phase 7](phase-07-graph-retrieval-and-fusion.md).

## 1. Problem being solved
Dense retrieval embeds the question and each chunk **separately**: the chunk vector was computed before anyone asked anything. That is fast, but it can't model how *this* question relates to *this* passage.

A **cross-encoder** reads both together and scores their relevance. It is far more accurate, and far more expensive: one full transformer pass per (question, chunk) pair. So it only re-scores a short list that a cheap first stage already found ("retrieve, then rerank").

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Bi-encoder vs. cross-encoder** | Bi-encoder: `sim(embed(q), embed(d))`, so documents are embedded once. Cross-encoder: `score(q ⊕ d)`, with attention between question and passage tokens; nothing can be precomputed. |
| **Two-stage retrieval** | Stage 1 (dense + BM25 + RRF) finds 20 candidates in ~20 ms. Stage 2 (the cross-encoder) re-scores those 20 and keeps 5. |
| **Rerank depth** | How many first-stage candidates are re-scored. More depth gives more chances to rescue a missed chunk, at linear cost. |
| **Logit / sigmoid** | The model outputs one number (the logit). `sigmoid(logit)` maps it into [0, 1] (the model card's "normalize"). It is a relevance *estimate*, not a calibrated probability, as §6 shows. |
| **Padding waste** | A batch is padded to its longest pair. Sorting pairs by length and using small batches avoids computing on padding. |
| **Dynamic int8 quantization** | Storing linear weights as 8-bit integers for faster CPU matmuls. Tried; it changed the scores too much (§7). |

## 3. Architecture
```
retrieval/reranker.py
  MODELS                 pinned repo + revision + SHA-256 per file (v2-m3, base)
  CrossEncoderReranker   lazy load (ensure_hf_file -> XLMRobertaForSequenceClassification),
                         device auto|cpu|cuda (CUDA -> CPU fallback), score(query, passages)
  RerankingRetriever     base.retrieve(depth) -> score -> sort -> top k (+ sources.rerank)
retrieval/__init__.py    modes dense+bm25+rerank, dense+bm25+graph+rerank (when rerank_enabled)
evaluation/retrieval_benchmark.py   --rerank-models --rerank-bases --rerank-depths
```
```
question ─► dense+bm25 (RRF) ─► 20 candidates ─► bge-reranker-v2-m3: 20 × ( <s> question </s></s> context+chunk </s> )
                                                   └─► logits ─► sort ─► top 5
hit.sources = {dense: #4, bm25: #4, rerank: {rank: 2, score: 0.56 (logit), first_stage_rank: 4}}
```

## 4. Walkthrough: q22 "Does the Aurora S1's rated cycle life match the project's cycle target?"
The question needs two facts: the S1's rated cycle life and the project's cycle target. Its ground truth lists one quote for each:
- `| Cycle life | 3000 cycles to 80% |` from the S1 page;
- `target_cycles: 3000` from aurora_config.json.

| Chunk | dense+bm25 (fused rank) | + rerank (rank, sigmoid) |
|---|---|---|
| S1 specification table (fact 1) | #1 (dense #1, bm25 #1) | #1, 0.796 |
| research notes: "target is **80% capacity retention after 3000 cycles**" | #3 (dense #4, bm25 #4) | #2, 0.636 |
| OCR lab notebook | #5 | #3, 0.096 |
| aurora_config.json `target_cycles: 3000` (fact 2) | **#6** (dense #19, bm25 #8) | **#4, 0.065** |

So q22 is the one seed-set *recall* win: the config record moves into the top 5. Read it carefully, though:
- The chunk that states the project target best (the research notes) was already #3 *without* reranking. It isn't in the ground truth, because the seed set lists only one quote per fact.
- The config record enters the top 5 with a score of 0.065, which is close to noise.

The seed set's recall gain (0.986 → 1.000) is therefore mostly a gap in the ground truth. The MRR and nDCG gains (0.928 → 0.980, 0.889 → 0.948) are rank improvements across many questions, and that is where reranking really helps.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
# first reranked query downloads bge-reranker-v2-m3 once (2.27 GB, SHA-256 verified)
uv run python -m omnirag.retrieval search \
  "Does the Aurora S1's rated cycle life match the project's cycle target?" --mode dense+bm25+rerank
uv run pytest -m integration tests/integration/test_reranker_live.py   # model-card scores
# benchmark (~40 min on CPU for both models, both first stages, depths 10 and 20)
uv run python -m omnirag.evaluation.retrieval_benchmark --rerank-models bge-reranker-v2-m3 \
  --rerank-bases dense+bm25 --rerank-depths 10,20
uv run python -m omnirag.evaluation.retrieval_benchmark --dataset data/eval/relational_seed.json \
  --rerank-models bge-reranker-v2-m3 --rerank-depths 20
```

## 6. Expected outputs (measured 2026-11-05, CPU)
**Model card check** (bge-reranker-v2-m3, fp32 CPU vs the card's fp16):
- `("query", "passage")`: −5.650 (card −5.652).
- Panda pairs: −8.184 / 5.265 (card −8.188 / 5.262).
- The live test asserts these within 0.02.

| Mode | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | rel. nDCG | p50 |
|---|---|---|---|---|---|---|---|
| dense+bm25 (default) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 19 ms |
| **+ rerank v2-m3, depth 20** | **1.000** | **0.980** | **0.948** | **1.000** | **0.938** | **0.867** | 17.0 s |
| + rerank v2-m3, depth 10 | 1.000 | 0.980 | 0.940 | 0.938 | 0.938 | 0.834 | 9.5 s |
| + rerank base, depth 10 | 0.973 | 0.887 | 0.862 | 0.812 | 0.750 | 0.694 | 2.2 s |
| graph + rerank v2-m3, depth 20 | 1.000 | 0.980 | 0.948 | 0.938 | 0.938 | 0.834 | 16.9 s |

Per question against dense+bm25:
- **v2-m3, depth 20:** wins q22, r02 and r05; loses nothing.
- **base:** wins q22 and r05; loses q04 and r08.

**Reranker top scores (v2-m3):**

| | top score |
|---|---|
| answerable questions | median 0.973, minimum 0.071 (q32) |
| q38 (unanswerable) | 0.207 |
| q39, CEO (unanswerable) | **0.890** |
| q40, AUR-C109 (unanswerable) | **0.936** |

No threshold separates them.

**Cost:**
- Load from disk: 2.4 s (v2-m3), 1.0 s (base).
- 20 pairs on 4 cores: 11.8 s (v2-m3, batch 4) / 2.4 s (base).
- The first query in a fresh process: 29 s measured once (load + warm-up).

## 7. Surprises found during verification
- **The CPU cost is real arithmetic, not a bug.** 20 pairs × ~100 tokens ≈ 2,000 tokens through XLM-R large (~0.6 GFLOP per token) ≈ 1.2 TFLOP. AVX2 on 4 cores lands at 6–12 s.
- **Batch size 4 beat 16** (11.8 s vs 17.5 s). With length-sorted pairs, small batches waste less on padding. Thread count (4 vs 8) barely mattered: WSL's 8 vCPUs are 4 cores with hyperthreads.
- **int8 quantization looked attractive and was rejected.** It was 2.2× faster for v2-m3, but logits moved by 1.0 on average and the model card's "hi" pair went from −8.18 to −5.56. For base it changed the top 5. And the PyTorch API is deprecated.
- **The smaller model is worse than no reranker.** bge-reranker-base loses q04 (AUR-C103 retention) and r08, and its MRR (0.887 seed / 0.750 relational) is below dense+bm25 (0.928 / 0.838). "Lightweight" has to be the right lightweight model.
- **The reranker is confident about questions that have no answer.** For AUR-C109 it scores the retention table of other cells at 0.936: the passage *is* about cell retention, just not about that cell. A cross-encoder measures topical relevance, not whether the answer is present. Phase 9 needs a different "insufficient context" signal.
- **The one seed-set recall "win" is mostly a ground-truth gap** (§4). q22's ground truth has a single quote for the project target, in the config file, while the research notes state the same target and were already #3. Adding alternative evidence would change every earlier phase's numbers, so it's a dataset v2 change to re-baseline deliberately, not to slip in here.
- **Graph candidates stay useless after reranking.** The Phase 7 hypothesis (the reranker would filter graph noise) didn't hold. Graph + rerank at depth 20 even misses r02, which plain rerank finds: the graph's candidates pushed the right chunk out of the depth-20 window.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| The first reranked query takes 30+ s | One-time model download (2.27 GB) and load. Later queries take ~17 s at depth 20 on CPU. Use `OMNIRAG_RERANK_DEPTH=10` for ~9.5 s. |
| `retrieval mode 'dense+bm25+rerank' is not enabled` (HTTP 422) | `OMNIRAG_RERANK_ENABLED=false`. |
| Warning `reranker_cuda_unavailable` | `OMNIRAG_RERANK_DEVICE=cuda` with the CPU-only torch: it falls back to CPU by design. GPU needs a CUDA torch (an open decision, ADR-0015). |
| `TokenizerDownloadError … sha256 mismatch` | The download was corrupted or the file changed upstream. Pinned revisions don't change; delete the folder under `<data_dir>/models/` and retry. |
| Memory pressure in WSL | v2-m3 holds ~2.3 GB of weights. With Neo4j (2 GB) and the API it fits in the 8 GB VM, but keep the test Neo4j stopped. |

## 9. Exercise (≈45 min)
1. **Where does reranking help?** In `rerank-2026-11-05.json`, compare `per_question_recall` and per-question MRR for `neo4j:dense+bm25:k60` and `…+rerank:bge-reranker-v2-m3:d20`. For two questions whose rank improved, read the chunks. What did the cross-encoder see that the first stage missed?
2. **The depth trade-off.** Run the relational benchmark at depths 5, 10 and 20 (`--rerank-depths 5,10,20`). Plot latency against R@5. Where would you set the default for a chat UI with a 3 s budget?
3. **Confidence.** Using the `top_score_*` fields, try to find any rule (threshold, gap between #1 and #2, …) that separates q38–q40 from the answerable questions. Why does a cross-encoder fail at this, and what signal would work?
4. **Think.** Would running the reranker on the GPU change any *quality* number in §6? What would you measure before changing ADR-0005?

## 10. Acceptance criteria — Phase 8
- [ ] `pytest -m integration tests/integration/test_reranker_live.py` passes: model-card logits within 0.02, CUDA → CPU fallback.
- [ ] The q22 search in §5 ranks the S1 spec table #1 and the project-target paragraph #2, with `via … rerank #…`.
- [ ] The benchmark reproduces §6 for v2-m3 at depth 20 (seed MRR 0.980, relational R@5 1.000).
- [ ] You can state the latency/quality trade-off and the GPU decision in ADR-0015 in your own words.
- [ ] `uv run pytest` passes, and `uv run pytest -m integration` passes with `neo4j-test` running.
