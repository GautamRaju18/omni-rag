# Phase 7: Graph retrieval and Reciprocal Rank Fusion

> Decision record: [ADR-0014](../adr/0014-graph-retrieval-and-rrf.md). Results: [fusion-2026-11-05.json](../benchmarks/fusion-2026-11-05.json) (seed set), [fusion-relational-2026-11-05.json](../benchmarks/fusion-relational-2026-11-05.json) (relational set), and the variants that were tried and removed in [graph-variants-*.json](../benchmarks/). Builds on [Phase 6](phase-06-dense-and-lexical-retrieval.md) and the entity graph from [Phase 5b](phase-05b-entity-extraction.md).

## 1. Problem being solved
Dense and BM25 retrieval fail differently:
- **dense** finds paraphrases but blurs identifiers;
- **BM25** finds AUR-C103 exactly but misses paraphrases.

A graph can, in principle, connect facts that share no words: who owns the electrolyte that the cells use. This phase:
1. builds a third, independent retriever on the entity graph;
2. combines all three without adding incomparable scores;
3. measures honestly whether the graph helps.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Entity linking** | Finding which graph entities a question names. Here by dictionary: entity keys and aliases matched at word boundaries. |
| **Seed / IDF weight** | A linked entity starts the traversal. Its weight `ln(1 + N/df)` makes rare entities (in 1 of 12 documents) count more than common ones (8 of 12). |
| **Multi-hop traversal** | Following RELATED_TO edges outward: hop 1 = the seed's relations, hop 2 = their neighbours' relations. Each edge has an *evidence chunk*. |
| **Reciprocal Rank Fusion** | `RRF(d) = Σ_r w_r / (k + rank_r(d))`. Uses ranks only, so a cosine (0.84), a BM25 score (7.9) and a graph weight (2.6) never meet on one scale. |
| **RRF k** | Damping. With k=60, ranks 1 and 2 are worth 1/61 and 1/62, almost the same. With k=1, they're worth 1/2 and 1/3, so the top hit dominates. |
| **Fusion depth** | How many candidates each retriever contributes (20 here) before fusing to the final k (5). |
| **Graceful degradation** | If a member retriever fails (Ollama down → no dense), fusion still answers from the others and says so (`details.failed`). |

## 3. Architecture
```
retrieval/
  graph_search.py  link() (dictionary entity linking), GraphRetriever (mentions + RELATED_TO
                   evidence up to 2 hops, provenance paths, related documents)
  fusion.py        rrf() (ranks -> fused score, ties, per-member sources), HybridRetriever
                   (members in parallel threads, degrade on failure)
  __init__.py      RetrievalService: modes dense | bm25 | graph | dense+bm25 | dense+bm25+graph
data/eval/relational_seed.json   8 two-fact questions linked through an entity
evaluation/retrieval_benchmark.py  + graph at 0-2 hops, fused modes, k 60/10, graph weight 0.5
```
```
question ─┬─► dense  (top 20) ─┐
          ├─► bm25   (top 20) ─┼─► RRF (k=60, weights) ─► top 5 with sources {dense: #1, bm25: #2, graph: #7}
          └─► graph  (top 20) ─┘        + graph_paths: "AUR-C103 -[TESTED_BY, stated]-> Aiko Tanaka (hop 1)"
                 └─ link: "AUR-C103" ─► MENTIONS chunks + RELATED_TO evidence chunks
```

## 4. Walkthrough: "Why did cell AUR-C103 fail?" in `dense+bm25+graph`
1. **The three retrievers run in parallel threads.**
   - **dense:** research notes #1.
   - **bm25:** the swelling row #1 and the research notes #2.
   - **graph:**
     - Links `AUR-C103` (SAMPLE, in 6 documents; weight ln(1 + 12/6) = 1.10).
     - Chunks mentioning it get 1.10.
     - Chunks that hold the evidence of one of its edges add that edge's weight. In the research notes the edge is "AUR-C103 -[USES, inferred]-> E-5", hop 1, worth 1.10 × 0.5. In the CSV it is "-[TESTED_BY, stated]-> Aiko Tanaka", worth 1.10 × 1.0.
2. **RRF:**
   - The research-notes chunk gets 1/61 (dense #1) + 1/62 (bm25 #2) + 1/65 (graph #5) = 0.0479 and ranks #1.
   - The CSV row is next with 1/65 + 1/61 + 1/64 = 0.0474 (dense #5, bm25 #1, graph #4).
   - Each hit's `sources` and `graph_paths` show why.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?"            # default: dense+bm25
uv run python -m omnirag.retrieval search "Who tested cell AUR-C107?" --mode graph  # paths, related docs
uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?" --mode dense+bm25+graph
uv run python -m omnirag.evaluation.retrieval_benchmark --out /tmp/fusion.json
uv run python -m omnirag.evaluation.retrieval_benchmark --dataset data/eval/relational_seed.json
curl -s -X POST localhost:8000/api/v1/retrieval/search -H 'content-type: application/json' \
  -d '{"query": "Who supplies the salt used in E-5?", "mode": "dense+bm25+graph", "k": 3}'
```

## 6. Expected outputs (measured 2026-11-05)
K=5, 58 chunks. Seed set: 37 answerable questions. Relational set: 8 questions with 2 facts each.

| Mode | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | p50 |
|---|---|---|---|---|---|---|
| dense | 0.932 | 0.863 | 0.829 | 0.875 | **0.854** | 8 ms |
| BM25 (classic / require) | 0.986 | 0.887 | 0.861 | 0.688 | 0.656 | 5 ms |
| **dense+bm25, k=60 (default)** | **0.986** | **0.928** | **0.889** | 0.875 | 0.838 | 11 ms |
| dense+bm25, k=10 | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 11 ms |
| dense+bm25+graph, hops 0, k=60 | 0.959 | 0.817 | 0.786 | 0.625 | 0.562 | 55 ms |
| dense+bm25+graph, hops 1, k=60 | 0.959 | 0.815 | 0.783 | 0.750 | 0.588 | 59 ms |
| dense+bm25+graph, hops 2, k=10 | 0.986 | 0.825 | 0.794 | 0.875 | 0.713 | 64 ms |
| … hops 2, graph weight 0.5 | 0.959 | 0.822 | 0.787 | 0.812 | 0.719 | 58 ms |
| graph alone, hops 0 / 1 / 2 | 0.581 | 0.439 / 0.412 / 0.413 | ≈0.41 | 0.250 / 0.375 / 0.375 | ≈0.20 | 32–40 ms |

Per question:
- **dense+bm25 vs dense, seed set:** wins q04 and q35, loses nothing. vs BM25: identical recall, better ranking.
- **Adding graph at k=60** loses q26: its only linked entity, E-5, is in 7 documents, and the graph floods the candidates with E-5 chunks.
- **Relational set:** dense+bm25 and dense have identical per-question recall. No graph variant wins a question; several lose r06 or r08.
- **Graph alone:** finds something for 29 of the 37 answerable seed questions and 7 of 8 relational ones. It returns nothing for AUR-C109, as BM25/require does, but the fused modes still return dense's 5 chunks.

**Smoke outputs:**
- **`--mode graph` "Who tested cell AUR-C107?"** The top hit is cell_test_results.csv. Its paths are `MENTIONS AUR-C107` and `AUR-C107 -[TESTED_BY, stated]-> Jonas Weber (hop 1)`.
- **`details.related_documents`** for "What is the role of NaDFOB in the E-7 electrolyte?" lists the safety review, the E-7 technical note and the budget (2 shared entities each).

## 7. Surprises found during verification
- **Summing hub edges.** The first graph scorer *summed* every edge's evidence. For "Who tested AUR-C107?" its 2nd and 3rd hits were chunks about E-7, reached in 2 hops (AUR-C107 → E-7 → E-7's other edges), with scores above the direct mention. A chunk now takes its best edge only.
- **Ties are the normal case for graph retrieval.** Most questions name one entity, so every chunk mentioning it gets the same score. The first version broke ties by chunk ID: an arbitrary order that RRF then treated as a ranking. Ties now go to more seed weight, then to chunks mentioning fewer other entities.
- **Graph retrieval lowers fused quality on both sets.** Two principled fixes were tried, measured and removed:
  - **relation-intent cues:** "tested" → TESTED_BY ×2;
  - **evidence-first scoring:** mentions only break ties.

  Each made graph alone *worse* on the seed set. The best graph variant on the relational set (2 hops, evidence-first, cues) was the worst on the seed set (MRR 0.777). That is tuning noise on small sets, not a robust gain. Results: [graph-variants-*.json](../benchmarks/).
- **The relational set was written to give graph retrieval a fair test.** The seed set mostly asks single-fact questions whose wording matches the answer, and its only 2 multi-hop questions are q07 (no entity named) and q34 (hub entity). Even on two-fact questions, dense retrieval finds both facts from the question's wording: "owns", "E-7 qualification", "Marco Lindqvist" all appear in the evidence.
- **RRF k barely mattered (10 vs 60) for dense+bm25.** Recall, MRR and nDCG were identical; only context precision differed (0.333 vs 0.328), so a few lower ranks moved. With graph, k=10 helped a little: it trusts each list's top hits more, so graph's long tail matters less.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `--mode graph` returns nothing | No entity in the question was linked (`details.linked: []`). Name the entity as it appears in the documents, or check that `extract --all` ran (Phase 5b). |
| `details.failed: {"dense": "EmbeddingError…"}` | Ollama is down; the fused result came from BM25 (and graph) only. It isn't an error response, by design: check `failed` before trusting it. |
| Graph hits are all about one common entity | That's the hub problem (E-5, E-7). Expected; this is why graph isn't in the default mode. |
| The fused score looks tiny (0.03) | It's an RRF score (Σ 1/(k+rank)), not a similarity. Compare ranks and `sources`, not values. |
| `OMNIRAG_RETRIEVAL_RRF_WEIGHTS` ignored | It must be JSON: `{"graph": 0.5}`. Names outside a mode's members are ignored for that mode. |

## 9. Exercise (≈60 min)
1. **Read a fusion.** For "Who supplies the salt that the E-5 electrolyte is made with?" (r02), run each mode. Using `sources`, explain why each fused top-5 looks the way it does. Which retriever had the supplier row, and at which rank?
2. **Make graph win.** Write two questions where graph retrieval should beat dense+bm25: facts connected only through a relation, with no shared wording. Add them to a copy of `relational_seed.json` (verbatim evidence!) and run the benchmark on it. Did graph win? If not, why not?
3. **Tune RRF.** Try `OMNIRAG_RETRIEVAL_RRF_WEIGHTS='{"graph": 0.25}'` and `OMNIRAG_RETRIEVAL_RRF_K=5` for `dense+bm25+graph`. Report per-question wins and losses, not just the mean, and decide whether your change is signal or noise on 37 questions.
4. **Think.** Phase 8 adds a cross-encoder reranker that reads each candidate together with the question. Why might graph candidates become useful *after* reranking, even though they hurt in RRF?

## 10. Acceptance criteria — Phase 7
- [ ] `search "Why did cell AUR-C103 fail?"` (default mode) shows `via dense #…, bm25 #…` on each hit.
- [ ] `--mode graph "Who tested cell AUR-C107?"` shows the TESTED_BY path to Jonas Weber in the top hit.
- [ ] The benchmark reproduces §6 for dense+bm25 (seed MRR 0.928) and the graph modes. Read the per-question wins and losses, not only the means.
- [ ] Stopping Ollama: `--mode dense+bm25` still answers, with `details.failed` naming dense.
- [ ] `uv run pytest` passes, and `uv run pytest -m integration` passes with `neo4j-test` running.
