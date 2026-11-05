# Phase 6: Dense and lexical retrieval on Neo4j

> Decision record: [ADR-0013](../adr/0013-dense-and-lexical-retrieval.md). Results: [retrieval-2026-11-05.json](../benchmarks/retrieval-2026-11-05.json) (main graph), [retrieval-analyzers-2026-11-05.json](../benchmarks/retrieval-analyzers-2026-11-05.json) (analyzer sweep). Builds on the indexes from [Phase 5a](phase-05a-graph-structure.md) and the in-memory baselines from [Phase 4](phase-04-embedding-engine.md).

## 1. Problem being solved
Phases 3–4 measured retrieval with exact in-memory code: a numpy dot product and a 40-line BM25. A real system searches an index that persists, scales and filters. This phase answers:
- Do Neo4j's vector and full-text indexes rank as well as the exact baselines?
- How do metadata filters (one document, only tables, no model-written text, latest version) work without silently dropping results?
- How do you search for `AUR-C103`, when a standard text analyzer turns it into `aur` + `c103` and every cell matches `aur`?
- How do you put a user's question into Lucene without it breaking the query parser, or being treated as query syntax?

Graph retrieval and fusion (RRF) come in Phase 7, and reranking in Phase 8. This phase builds and measures the two independent retrievers they combine.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **HNSW** | The approximate nearest-neighbour graph behind Neo4j's vector index (`m=16`, `ef_construction=100`). "Approximate": it may miss the true nearest neighbour on large corpora, so we compare it with exact search. |
| **In-index filter** | A predicate evaluated *during* the vector search (`SEARCH … WHERE c.document_id IN $ids`), so the top-k already satisfies it. Only possible for properties declared on the index. |
| **Post-filter + over-fetch** | Fetch more candidates than needed (k × 10), filter, keep k. If too few survive, the result is "underfilled", and we report that instead of hiding it. |
| **Analyzer** | Lucene's text → terms pipeline (tokenizer + filters), applied identically to documents and queries. It decides whether `AUR-C103` is one term or two. |
| **Lucene query syntax** | `+must`, `-not`, `"phrase"`, `term^2` (boost), `?`/`*` wildcards, `/regex/`. Raw user text is full of these characters by accident. |
| **Technical identifier** | A token mixing letters and digits: AUR-C103, E-7, M3, R2, NaPF6. Users type them expecting an *exact* match. |
| **BM25** | Term-frequency × inverse-document-frequency scoring with length normalisation. Lucene's implementation scores `content` and `context` as separate fields and combines them. |

## 3. Architecture
```
retrieval/
  models.py    RetrievalFilters, RetrievedChunk (provenance), RetrievalResult (+ details)
  filters.py   predicates(): which filters go inside the vector index, which after
  dense.py     DenseRetriever: embedding-space check, embed query, vector SEARCH
  sparse.py    lucene_query() (escape, phrases, identifiers), LexicalRetriever: full-text SEARCH
  __init__.py  RetrievalService + build_retrieval(settings)
  __main__.py  CLI: search
api/routers/retrieval.py   POST /api/v1/retrieval/search
evaluation/retrieval_benchmark.py   Neo4j retrievers vs the exact baselines, analyzer sweep
knowledge_base/schema.py   vector index filter properties; analyzer / filter mismatch checks
```

```
question ──► DenseRetriever ──► embed_query (bge-m3, same space as the chunks?)
                             └► SEARCH c IN (VECTOR INDEX chunk_embedding FOR $v
                                  WHERE c.config_id = $cfg AND c.is_latest = true
                                        [AND c.document_id IN $ids] [AND c.element_type IN $t]
                                  LIMIT $fetch)  [WITH … WHERE post-filters]  LIMIT $k
question ──► lucene_query ──► 'why did cell aur\-c103 fail +"aur\-c103"'
         └─► LexicalRetriever ──► SEARCH c IN (FULLTEXT INDEX chunk_fulltext FOR $lucene LIMIT $fetch)
                                  WITH … WHERE <all filters>  LIMIT $k
```

## 4. Walkthrough: "Why did cell AUR-C103 fail?"
1. **Dense:**
   - Check once that the graph's vectors are in `dense-ollama-bge-m3-790764642607`, the same space as the query embedder.
   - Embed the question.
   - Search with `config_id = multimodal-256-ctx AND is_latest = true` inside the index.
   - Top hit: research notes, "Cell AUR-C103 failed at cycle 412 because of gas evolution…", score 0.837 (cosine 0.674).
2. **Lexical:**
   - The words become `why did cell aur\-c103 fail`, and the identifier becomes `+"aur\-c103"`.
   - The `?` is gone, and every `-` is escaped.
   - With the `classic` analyzer, `aur-c103` is one indexed term, so only chunks that literally name AUR-C103 can match. The `+` makes it required.
3. **Both** return `RetrievedChunk`s with document, section, pages, element IDs, origins and OCR confidence, plus `details` showing the exact query, the fetch size, which filters ran where, and timings.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
# existing graphs: the vector index gains filter properties and the full-text index
# switches to the classic analyzer (vectors stay on the nodes, nothing is re-embedded)
uv run python -m omnirag.knowledge_base init --recreate-vector-index --recreate-fulltext-index
uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?"
uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?" --mode bm25
uv run python -m omnirag.retrieval search "swelling" --doc doc_2b1c659c54266d54 --type table
uv run python -m omnirag.retrieval search "capacity retention chart" --type figure --no-generated
uv run python -m omnirag.evaluation.retrieval_benchmark --out /tmp/retrieval.json

# API
uv run uvicorn omnirag.api.main:app --no-access-log
curl -s -X POST localhost:8000/api/v1/retrieval/search -H 'content-type: application/json' \
  -d '{"query": "Who tested cell AUR-C107?", "mode": "bm25", "k": 3}'

# analyzer sweep: recreates the full-text index, so only against the test server
docker compose --profile test up -d neo4j-test
OMNIRAG_NEO4J_URI=bolt://localhost:7688 uv run python -m omnirag.knowledge_base load --all
OMNIRAG_NEO4J_URI=bolt://localhost:7688 uv run python -m omnirag.evaluation.retrieval_benchmark \
  --analyzers standard-no-stop-words,standard,english,classic
```

## 6. Expected outputs (measured 2026-11-05)
**Seed set:** 58 chunks, 37 answerable questions (20 naming an identifier) and 3 unanswerable ones, K=5.

| Method | R@5 | MRR | nDCG | P@5 | ctx P | id P | q40 hits | Δ vs exact baseline |
|---|---|---|---|---|---|---|---|---|
| memory:bm25 (exact) | 0.986 | 0.900 | 0.857 | 0.276 | 0.288 | 0.570 | – | – |
| memory:dense (exact) | 0.932 | 0.863 | 0.829 | 0.281 | 0.309 | 0.630 | – | – |
| **neo4j:dense** | 0.932 | 0.863 | 0.829 | 0.281 | 0.309 | 0.630 | 5 | none |
| bm25 standard-no-stop-words / terms | 0.932 | 0.818 | 0.779 | 0.249 | 0.271 | 0.550 | 5 | −q04 −q05 |
| bm25 standard-no-stop-words / require | 0.986 | 0.814 | 0.782 | 0.316 | 0.320 | 0.880 | 0 | none |
| bm25 standard / require | 0.959 | 0.842 | 0.796 | 0.330 | 0.353 | 0.860 | 0 | −q35 |
| bm25 english / require | 0.959 | 0.824 | 0.794 | 0.335 | 0.370 | 0.870 | 0 | −q35 |
| bm25 classic / terms | 0.959 | 0.887 | 0.854 | 0.278 | 0.333 | 0.610 | 5 | −q05 |
| bm25 classic / boost | 0.986 | 0.874 | 0.841 | 0.284 | 0.332 | 0.720 | 5 | none |
| **bm25 classic / require (default)** | **0.986** | 0.887 | **0.861** | **0.341** | **0.377** | **0.890** | **0** | none |

How to read the columns:
- **id P:** for the 20 identifier questions, the share of top-5 chunks containing one of the question's identifiers.
- **q40:** chunks returned for the question about the nonexistent cell AUR-C109.
- **Δ:** per-question recall losses (−) or wins (+) against the exact baseline of the same family.

The main graph and the test graph gave identical ranking numbers for every method that ran on both.

Reading it:
- **Dense through Neo4j = exact dense.** All metrics are identical and no question differs. At 58 chunks, HNSW finds the true neighbours.
- **The Phase 5 default (standard analyzer, plain terms) loses q04 and q05**, the AUR-C103 questions. "aur" matches every cell, so C105/C106 chunks crowd out C103.
- **The same analyzer with `require` fixes recall but ranks worse** (MRR 0.814). The classic analyzer fixes it at the root: AUR-C103 is one term.
- **classic / require ties the exact baseline on recall**, beats it on nDCG (0.861 vs 0.857), P@5 (0.341 vs 0.276) and context precision (0.377 vs 0.288), and is 0.013 lower on MRR. It is the only lexical setting that returns nothing for AUR-C109.

**Latency** (main graph, warm, p50):
- dense 19.9 ms (query embedding from the cache; a cold bge-m3 load measured 5.5 s once);
- BM25 classic/require 11.4 ms;
- the in-memory baselines are ≈0.1 ms (58 vectors, no network).

**API:**
- `POST /retrieval/search` `{"query": "Who tested cell AUR-C107?", "mode": "bm25"}` returned 1 hit, from cell_test_results.csv, in 12.8 ms. Only one chunk names AUR-C107.
- An empty query returns 422.

## 7. Surprises found during verification
- **Raw questions broke Lucene, or quietly ruined ranking.** "N/P" and `AND OR NOT` raise parse errors. "Why did cell AUR-C103 fail?" parses, but its top 3 contained no AUR-C103 (the `?` turned "fail" into a wildcard). Hence `lucene_query`: nothing the user types is ever interpreted as syntax.
- **A filter before SEARCH is not a pre-filter.** `MATCH (c:Chunk {document_id: X}) SEARCH c IN (VECTOR INDEX …)` returned **0 rows**: it filters the global top-k afterwards. Document and element-type filters are now declared on the vector index and run inside it.
- **Vector SEARCH WHERE is narrow:** AND of `=` / `IN` on declared properties only. `OR` and `<>` are syntax errors; undeclared properties are errors.
- **A second full-text index on the same properties is silently a no-op.** The analyzer probe "created" three indexes that then didn't exist. Changing the analyzer requires drop + recreate, and `ensure_schema` now reports a changed analyzer (or changed vector filter properties) instead of silently keeping the old index. That old behaviour would have hidden this whole phase's main change.
- **`standard` and `english` lose q35** ("When is the next technical review meeting?"). Both remove stop words, but so does `classic` (verified: "the", "is", "by" and "and" match nothing, "next" matches 4 chunks), and `classic` keeps q35. So stop words alone don't explain it; I haven't isolated the cause. `english` stemming never beat `classic` on this set. With 37 questions this is one question.
- **Exact dense == HNSW dense** on 58 chunks: expected, but now measured instead of assumed.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `SchemaMismatchError: … filter properties (…'config_id', 'is_latest'), requested (…'document_id', 'element_type')` | A Phase 5 graph: run `init --recreate-vector-index` (re-indexes the stored vectors; no re-embedding). |
| `SchemaMismatchError: Full-text index chunk_fulltext uses analyzer 'standard-no-stop-words'` | The analyzer default changed to `classic`: run `init --recreate-fulltext-index`. |
| HTTP 409 "Graph vectors are in … queries in …" | The embedding model (digest) changed since the graph was loaded: `python -m omnirag.embeddings embed --all`, then `knowledge_base load --all`. |
| HTTP 503 "Embedding model unavailable" | Ollama is down or bge-m3 is missing; BM25 still works (`"mode": "bm25"`). |
| BM25 returns nothing for a question with an identifier | `require` mode: the identifier isn't in the corpus as written (e.g. "C103" vs "AUR-C103"). Use `--identifiers boost`, or wait for Phase 7, where dense results fill the gap. |
| `details.underfilled: true` | Post-filters removed too many of the k × overfetch candidates: raise `OMNIRAG_RETRIEVAL_OVERFETCH` or make the filter in-index (declared property). |
| `--analyzers … only against port 7688` | The sweep drops and recreates the full-text index; it refuses to touch the main graph. |

## 9. Exercise (≈45 min)
1. **Break the query builder.** Try 10 nasty queries with `search --mode bm25`: quotes, backslashes, `title:foo`, `NOT`, emoji, a 2,000-character paste. Check `details.lucene` for each. Can you make Lucene raise an error? If you can, add the case to `test_lucene_query_escapes_everything` and fix it.
2. **Find require's weakness.** Write three questions that name an identifier *differently* from the corpus ("C103", "AUR C103", "e7"). Compare `--identifiers require` and `boost`, and dense. Which mode would you choose for a corpus of product codes users type from memory?
3. **Filters.** Using only filters, find the swelling *table* of the safety review, then every chunk *without* model-written text that mentions retention. Which filters ran inside the index (`details.in_index_filters`)?
4. **Think.** Why is it wrong to add a BM25 score (e.g. 7.9) to a cosine score (e.g. 0.84)? What would Phase 7 have to know about each score's distribution to combine them, and why does RRF avoid that question?

## 10. Acceptance criteria — Phase 6
- [ ] `init` on your graph reports `recreated ['chunk_embedding', 'chunk_fulltext']` once, then everything `existing`.
- [ ] `retrieval_benchmark` reproduces §6 for `neo4j:dense` (= memory:dense) and `neo4j:bm25:classic:require` (R@5 0.986, MRR 0.887).
- [ ] `search "What capacity retention did cell AUR-C109 reach?" --mode bm25` returns 0 hits; dense returns 5 (and Phase 7 must handle that).
- [ ] `POST /api/v1/retrieval/search` works for both modes and returns 422 for an empty query.
- [ ] `uv run pytest` passes, and `uv run pytest -m integration` passes with `neo4j-test` running.
