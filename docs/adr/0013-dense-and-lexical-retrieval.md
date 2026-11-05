# ADR-0013: Dense and lexical retrieval on Neo4j: in-index filters, a safe Lucene builder, the classic analyzer

**Status:** accepted (2026-11-05, Phase 6)

## Context
The brief (section 9) requires:
- **Dense retrieval:** query embedding, the Neo4j vector index, similarity ranking, top-k and metadata filtering.
- **Sparse retrieval:** BM25 through the Neo4j full-text indexes, with exact keyword matching, technical identifier matching and lexical ranking.

Phases 3–4 measured exact in-memory baselines on the seed set: BM25 R@5 0.986 / MRR 0.900, dense R@5 0.932 / MRR 0.863. The Neo4j retrievers must reproduce or beat them.

Measured on Neo4j 2026.10.1 Community (2026-11-05):
- **Deprecated procedures:** `db.index.vector.queryNodes` and `db.index.fulltext.queryNodes` are deprecated in favour of Cypher 25 `SEARCH`.
- **Vector `SEARCH ... WHERE`:**
  - It accepts only AND-ed equality or `IN` predicates on properties declared at index creation (`WITH [...]`).
  - OR, `<>` and undeclared properties are errors.
  - `MATCH (c {document_id: …}) SEARCH …` does not pre-filter: it returned 0 rows for a document outside the global top-k.
- **Full-text `SEARCH`** accepts no `WHERE`.
- **One full-text index per label + property set.** A second `CREATE FULLTEXT INDEX … IF NOT EXISTS` on the same properties is silently a no-op.
- **Raw questions as Lucene queries:**
  - "N/P" and bare `AND OR NOT` fail to parse.
  - "Why did cell AUR-C103 fail?" returns 3 chunks without AUR-C103 (the `?` is a wildcard; the identifier is split).
- **Analyzers:** `standard*` and `english` split `AUR-C103` into `aur` + `c103`. `classic` keeps tokens containing digits whole.

## Decision
**Package.** `src/omnirag/retrieval/`:
- `DenseRetriever` and `LexicalRetriever`, with shared `RetrievalFilters` / `RetrievedChunk` / `RetrievalResult` models;
- a `RetrievalService` created once per process;
- `POST /api/v1/retrieval/search`;
- the CLI `python -m omnirag.retrieval search`.

**Dense.**
- **Same space only:** query embedding through the same `DenseEmbedder`. Before the first search, the retriever checks that every Document's `embedding_space` equals the query embedder's `space_id`, and refuses to rank across spaces (409).
- **Filters inside the index:** the vector index declares filter properties `config_id, is_latest, document_id, element_type`, so chunk set, latest version, document and element-type filters run inside the search.
- **Filters after the search:** `source_files`, `exclude_generated` and `min_confidence` over-fetch (k × 10, at most 200) and post-filter. Results report `underfilled` when fewer than k survive.
- **Scores:** the raw score is Neo4j's (1+cos)/2; `details.cosine` gives the usual value.

**Lexical.**
- **User text never reaches Lucene raw.** `lucene_query` rebuilds the query from words, escapes every special character, keeps the user's own `"quoted phrases"` (exact keyword matching), and treats technical identifiers (tokens with letters and digits) by mode:
  - `terms`;
  - `boost` (an extra exact phrase ^2);
  - `require` (`+"aur-c103"`).
- **All filters are post-filters** (no WHERE on full-text SEARCH).
- **Default:** the analyzer is **`classic`** and the identifier mode is **`require`** (measured below).

**Schema safety.**
- `ensure_schema` raises `SchemaMismatchError` when the existing vector index's dimension, quantization *or filter properties*, or the full-text index's *analyzer*, differ from the settings.
- `init --recreate-vector-index` / `--recreate-fulltext-index` rebuild them from the node properties: vectors stay on the nodes, and nothing is re-embedded.

## Measured (seed set: 58 chunks of `multimodal-256-ctx`, 37 answerable questions, 20 naming an identifier, K=5)
Full results: [retrieval-2026-11-05.json](../benchmarks/retrieval-2026-11-05.json) (main graph) and [retrieval-analyzers-2026-11-05.json](../benchmarks/retrieval-analyzers-2026-11-05.json) (sweep on the test server).

| Method | R@5 | MRR | nDCG | P@5 | ctx P | id P | q40 hits | Δ vs memory |
|---|---|---|---|---|---|---|---|---|
| memory:bm25 (exact baseline) | 0.986 | 0.900 | 0.857 | 0.276 | 0.288 | 0.570 | – | – |
| memory:dense (exact baseline) | 0.932 | 0.863 | 0.829 | 0.281 | 0.309 | 0.630 | – | – |
| **neo4j:dense** | 0.932 | 0.863 | 0.829 | 0.281 | 0.309 | 0.630 | 5 | none |
| neo4j:bm25 standard-no-stop-words / terms (Phase 5 default) | 0.932 | 0.818 | 0.779 | 0.249 | 0.271 | 0.550 | 5 | −q04, −q05 |
| … / boost | 0.986 | 0.760 | 0.749 | 0.265 | 0.273 | 0.750 | 5 | none |
| … / require | 0.986 | 0.814 | 0.782 | 0.316 | 0.320 | 0.880 | 0 | none |
| standard / require | 0.959 | 0.842 | 0.796 | 0.330 | 0.353 | 0.860 | 0 | −q35 |
| english / require | 0.959 | 0.824 | 0.794 | 0.335 | 0.370 | 0.870 | 0 | −q35 |
| classic / terms | 0.959 | 0.887 | 0.854 | 0.278 | 0.333 | 0.610 | 5 | −q05 |
| classic / boost | 0.986 | 0.874 | 0.841 | 0.284 | 0.332 | 0.720 | 5 | none |
| **classic / require (default)** | **0.986** | 0.887 | **0.861** | **0.341** | **0.377** | **0.890** | **0** | none |

How to read the columns:
- **id P:** for the 20 identifier questions, the share of top-5 chunks that contain one of the question's identifiers.
- **q40:** "AUR-C109", a cell that doesn't exist; the column is the number of chunks returned.
- **Δ vs memory:** per-question recall wins and losses against the exact baseline of the same family.

Latency on the main graph (warm, p50): dense 19.9 ms (query embedding served from the cache), BM25 classic/require 11.4 ms. I don't compare latency across analyzers: they ran one after another on one server.

## Consequences
- **Dense:** Neo4j's HNSW index (quantization NONE, m=16) returns exactly the exact-search ranking at this corpus size. Larger corpora must re-measure; the benchmark does that.
- **`require` makes BM25 return nothing when an identifier in the question isn't in the corpus.** That's good for questions about things that don't exist (q40), and it leaves the answer to dense retrieval in Phase 7's fusion. The risk: a misspelled or differently written identifier ("C103", "AUR C103") gets no lexical results. `boost` is the fallback setting.
- **`classic` removes English stop words and doesn't stem** ("fail" ≠ "failed"). On this set that cost nothing; a corpus with more inflected vocabulary may favour `english`, and the sweep re-measures it.
- **One fact is 0.027 of recall** (37 questions). The analyzer decision rests on 1–2 questions per method plus the precision columns; treat it as a baseline, not a general result.
