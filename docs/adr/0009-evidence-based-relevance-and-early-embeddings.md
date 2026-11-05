# ADR-0009: Relevance by verbatim evidence; a minimal embedding client before Phase 4

**Status:** accepted (2026-11-05, Phase 3b)

## Context
The brief requires benchmarking every chunking strategy "against the same evaluation dataset" with Recall@K, Precision@K, MRR, nDCG, context precision/recall, latency and index size. Two things were missing:
1. A definition of "relevant chunk" that is identical for every strategy and size. Chunk IDs change per configuration (ADR-0008).
2. Embeddings: semantic chunking and dense retrieval need them, and the embedding engine is Phase 4.

## Decision
- **Evidence-based relevance.** Each question lists *facts*; each fact is a list of **verbatim alternative quotes** (the same fact appears in several documents: "AUR-C103 failed at cycle 412" is in 4 files). A retrieved chunk covers a fact if its content contains any alternative (whitespace-normalised). `resolve()` maps every quote to element IDs (table rows to their TABLE_ROW), which gives ADR-0008's element-level ground truth, and a test fails if any quote is missing from the samples.
- **Metrics on returned units.** For parent–child, the unit is the parent the LLM would read, not the child that matched. Recall@K is fact coverage, so a two-fact multi-hop question needs both facts.
- **Minimal embedding client now** (`omnirag/embeddings/`): batched `/api/embed` calls plus a SQLite cache keyed by (model, SHA-256 of the text). Phase 4 adds Neo4j storage, vector indexes and re-embedding policy on top.
- **Exact in-memory retrievers for benchmarking** (cosine over numpy; our own BM25 with identifier-preserving terms). Phase 6 builds the production versions in Neo4j.
- **Semantic chunking and LLM context are opt-in.** Both call Ollama during chunking; the default ingestion strategies stay local and fast.

## Consequences
- A quote must be specific enough to locate a fact, and short enough to fit a small chunk. Long quotes would bias the benchmark against small chunks.
- The seed set is small (12 documents, 37 answerable questions): **one question = 2.7 recall points**. Results are reported per question (wins/losses) as well as averages, and conclusions are limited accordingly (see the Phase 3b learning doc).
- Text containment can't judge paraphrased evidence. LLM-judged context precision/recall (RAGAS-style) come with Phase 12.

## Alternatives considered
| Option | Why not |
|---|---|
| Gold chunk IDs per strategy | 24 configurations × 37 questions of hand labels, invalid after any change |
| Gold element IDs only | Small strategies split long elements, so element-level containment would penalise them unfairly |
| LLM judge for relevance now | Slow, non-deterministic, and its own failure mode; Phase 12 |
| `rank_bm25` / FAISS for the benchmark | Extra dependencies for ~40 lines; exact search avoids ANN effects |
