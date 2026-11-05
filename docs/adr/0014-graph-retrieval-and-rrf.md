# ADR-0014: Graph retrieval and Reciprocal Rank Fusion: dense+bm25 is the default, graph is opt-in

**Status:** accepted (2026-11-05, Phase 7)

## Context
The brief (section 9C) asks for graph retrieval:
- entity extraction and resolution;
- relationship traversal and multi-hop retrieval;
- related-document retrieval and provenance tracking.

It also asks to fuse all retrievers with RRF, `RRF(d) = Σ 1/(k + rank(d))`, with configurable parameters, never by adding raw scores. And it lists retrieval modes: dense, BM25, dense+BM25, dense+BM25+graph, with reranking and query rewriting to come later.

The graph from Phase 5b holds 82 entities, 217 mentions and 97 relations (71 stated). Every relation has an evidence chunk.

## Decision
**Graph retriever** (`retrieval/graph_search.py`):
- **Dictionary entity linking:** the query is matched against entity keys and aliases, with the same normalisation and word boundaries as extraction. There is no model call, the output is deterministic, and it is exact for identifiers.
  - The longest match counts fully; names nested inside it count half.
  - `SAME_AS` adds the aliased entity.
  - Each seed's weight is its IDF over latest documents.
- **Scoring:**
  - A chunk that mentions seeds scores Σ seed weights.
  - Evidence chunks of RELATED_TO edges up to 2 hops away add their best edge. An edge's weight: stated 1.0, inferred 0.5; ×0.5 per extra hop; ×2 when both ends are seeds. Only latest-version edges count.
  - Ties go to more seed weight, then to fewer other entities mentioned.
- **Provenance:** every hit carries its graph paths. `details` lists the linked entities and the related documents.

**Fusion** (`retrieval/fusion.py`):
- **Formula:** RRF with configurable `k` (default 60), per-retriever weights and candidate depth (default 20).
- **Parallel members:** they run in threads. A failing member degrades the result (`details.failed`); the request fails only if every member fails.
- **Ties:** best single rank, then chunk ID.
- **Provenance:** each fused hit keeps every member's rank and score (`sources`).

**Modes:** `dense`, `bm25`, `graph`, `dense+bm25`, `dense+bm25+graph`.
- `POST /retrieval/search` and the CLI take any of them.
- **The default (`retrieval_default_mode`) is `dense+bm25`**, based on the measurements below.

## Measured (2026-11-05, K=5)
Two eval sets: the seed set (37 answerable questions) and a new relational set (8 questions, each needing two facts linked through an entity; `data/eval/relational_seed.json`, quotes verified by a test).

| Mode | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | p50 |
|---|---|---|---|---|---|---|
| dense | 0.932 | 0.863 | 0.829 | 0.875 | 0.854 | 8 ms |
| BM25 (classic / require) | 0.986 | 0.887 | 0.861 | 0.688 | 0.656 | 5 ms |
| **dense+bm25, k=60 (default)** | **0.986** | **0.928** | **0.889** | 0.875 | 0.838 | 11 ms |
| dense+bm25, k=10 | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 11 ms |
| dense+bm25+graph, hops 1, k=60 | 0.959 | 0.815 | 0.783 | 0.750 | 0.588 | 59 ms |
| dense+bm25+graph, hops 2, k=10 (best with graph) | 0.986 | 0.825 | 0.794 | 0.875 | 0.713 | 64 ms |
| graph alone, hops 0–2 | 0.581 | 0.412–0.439 | ≈0.41 | 0.250–0.375 | 0.19–0.21 | 32–40 ms |

Per question:
- **On the seed set,** dense+bm25 wins q04 and q35 back over dense and loses nothing against BM25.
- **Adding graph at k=60** loses q26 ("Who prepared the safety review of the E-5 failure?"). Its only linked entity, E-5, appears in 7 documents, so chunks about E-5 crowd out the answer.
- **On the relational set,** no graph variant beats dense+bm25 on any question.

Two refinements were also measured and **removed**, because neither helped on both sets ([graph-variants-2026-11-05.json](../benchmarks/graph-variants-2026-11-05.json), [graph-variants-relational-2026-11-05.json](../benchmarks/graph-variants-relational-2026-11-05.json)):
- **Relation-intent cues:** "tested" → TESTED_BY ×2.
- **Evidence-first scoring:** mentions only break ties.

The best graph variant on one set was among the worst on the other.

## Why graph retrieval doesn't help here
- **The answer depends on the question's predicate, not only its entity.** Most questions name one entity, which the graph turns into every chunk about that entity. Whether a chunk *answers* the question depends on what is asked ("capacity retention", "who prepared"), and dense and BM25 already score that.
- **The relational questions bridge through a lookup, not through the graph.** For example: the electrolyte someone owns, then the cells using it. Their facts sit in chunks that dense retrieval finds from the question's wording ("Marco Lindqvist owns the E-7 qualification" shares words with the question). The graph's edges point at the same chunks dense already ranks high, plus many other chunks about hub entities (E-7: 16 mentions in 8 documents).
- **Questions that describe instead of naming link nothing:** q07, "the cell that failed at 45 °C". Linking descriptions needs a model; query rewriting comes in Phase 9.

## Consequences
- **What `dense+bm25` gives:** the best ranking on the seed set (MRR 0.928, +0.041 over BM25 alone) and dense-level recall on the relational set, at 11 ms p50. On the relational set its MRR is 0.016 below dense alone (one rank on one question).
- **What graph retrieval remains for:**
  - **provenance and explanation:** which relation connects a chunk to the question;
  - **related documents and entity questions:** `entity`, and the graph explorer in Phase 14;
  - **a candidate source** for later phases. Phase 8's reranker can re-score graph candidates on their text, which RRF can't; Phase 9's agentic retrieval can follow edges deliberately.
- **Re-measure on new corpora.** This corpus is 12 documents, and 8 relational questions are one question = 0.125 recall. On a corpus where entities connect many documents that share no vocabulary, graph retrieval may matter; the benchmark makes re-measuring cheap.
- **Linking cost grows with the graph:** it loads every Entity per query. Measured p50 with 82 entities: 11.3 ms to load them from Neo4j plus 0.7 ms of matching (15.7 ms for the whole linking step), within 32–40 ms of graph latency in total. A larger graph needs a cache or a full-text index on entity names. That's Phase 13's performance pass.
