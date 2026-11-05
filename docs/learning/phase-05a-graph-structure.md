# Phase 5a: Neo4j knowledge graph: document structure, vectors, versions

> Decision record: [ADR-0011](../adr/0011-graph-schema-and-versioning.md). Builds on the chunks and embedding refs from [Phase 4](phase-04-embedding-engine.md). Entities and relationships extracted by the LLM (MENTIONS, RELATED_TO) come in Phase 5b.

## 1. Problem being solved
Phases 2–4 left everything as files: elements, chunk sets and a vector cache. Retrieval (Phases 6–7) needs three things from them:

- **search**: vector search and keyword (BM25) search over chunks, in one store;
- **structure**: for any chunk, its section, page, neighbours, parent chunk, and the tables and figures it talks about, so context can be expanded and citations shown;
- **lifecycle**: when a file is re-uploaded with new content, the old version stays traceable but is no longer searched by default. Deleting a document removes it everywhere.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Property graph** | Nodes with labels (`:Chunk`) and properties, linked by typed, directed relationships (`-[:NEXT_CHUNK]->`). |
| **Uniqueness constraint** | Neo4j refuses a second node with the same key, e.g. `chunk_id`. It also creates an index, so lookups by key are fast. |
| **Vector index** | An approximate nearest-neighbour index (HNSW) over a list property. We use cosine similarity, 1024 dimensions, and **no quantization** (§6). |
| **Filtered vector search** | Filters on properties declared in the index (`WITH [c.config_id, c.is_latest]`) are applied *inside* the search. All k results then pass the filter, instead of filtering k results down to fewer. |
| **Full-text index** | A Lucene index: BM25 scoring and Lucene query syntax (`"phrase"`, `AND`, `term*`). |
| **Managed transaction** | `session.execute_write(fn)`: the driver retries `fn` on transient errors. Everything inside commits or rolls back together. |
| **Idempotent write** | Running it twice gives the same graph as running it once. Here: delete what the document owns, then recreate it, in one transaction. |
| **Source vs. version** | A *Source* is the logical document, the file name. Each distinct content hash is a *Document* version of it. |
| **Parameterised Cypher** | Values are passed as `$params`, never pasted into the query text. This prevents injection and lets Neo4j reuse query plans. |

## 3. Architecture
```
knowledge_base/
  graph_store.py    GraphStore: sync driver, write/read (managed tx), query, schema (auto-commit)
  schema.py         ensure_schema: 7 constraints, 7 lookup indexes, vector + full-text index
  graph_builder.py  build_payload (pure: elements + chunks -> rows), GraphWriter (Cypher),
                    cached_vectors (embedding cache -> vectors), GraphIndexer (pipeline step)
  traversal.py      stats, chunk_context, document_outline, versions, vector/fulltext search
  __main__.py       CLI: init | load | stats | outline | show-chunk | versions |
                         search-vector | search-text | unload

IngestionPipeline: load -> enrich -> chunk -> embed -> write chunks -> graph (warning on failure)
DELETE /documents/{id}: graph first, then files (503 and files kept if Neo4j is down)
```

```
(:Source)<-[:VERSION_OF]-(:Document)-[:PREVIOUS_VERSION]->(:Document)
(:Document)-[:HAS_PAGE]->(:Page)-[:CONTAINS]->(:Chunk)
(:Document)-[:HAS_SECTION]->(:Section)-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(:Chunk)
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT_CHUNK]->(:Chunk)      (:Chunk)-[:CHILD_OF]->(:Chunk)
(:Document)-[:HAS_TABLE]->(:Table)-[:DISCUSSED_IN]->(:Chunk)
(:Document)-[:HAS_FIGURE]->(:Figure)-[:DISCUSSED_IN]->(:Chunk)   (:Figure)-[:HAS_TABLE]->(:Table)
```

Design points:
- **The graph is an index of the files, not the primary store.** Dropping it loses nothing: `load --all` rebuilds it from disk.
- **Pure payload, thin Cypher.** `build_payload` turns elements and chunks into plain rows, with no Neo4j involved. That's where the logic lives, and it's unit-tested. The Cypher only does UNWIND, CREATE and MATCH.
- **Vectors come from the cache by `embedding_ref`**, so loading the graph never calls Ollama.
- **Original vs. generated stays visible.** Figure nodes keep `alt_text`, `caption`, `ocr_text` and `description` (with `description_model`) as separate properties. Chunks keep `origins` and `min_confidence`.
- **Which chunk sets become Chunk nodes:** only `graph_chunk_configs`, by default the embedded set `multimodal-256-ctx`. Other strategies stay on disk for benchmarks.

## 4. What one write does (one transaction per document)
1. `MERGE` the Source (`source_key` = lowercased file name) and the Document. `SET d += $doc`.
2. `DETACH DELETE` every Chunk, Page, Section, Table and Figure with this `document_id` (lookup index).
3. `UNWIND` the rows to create pages, the section tree, tables, figures and chunks. `db.create.setNodeVectorProperty` stores each chunk's vector.
4. Create the links: CONTAINS, NEXT_CHUNK, CHILD_OF, DISCUSSED_IN.
5. **Recompute the versions of this Source**:
   - delete the old PREVIOUS_VERSION chain;
   - order the versions by `created_at, document_id`;
   - set `version = 1..n` and `is_latest` on the last one;
   - chain each version to the one before it;
   - copy `is_latest` onto the chunks.

If any step fails, the transaction rolls back and the previous graph stays as it was.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
docker compose up -d                                       # main Neo4j (7687)
uv run python -m omnirag.knowledge_base init               # constraints + indexes (idempotent)
uv run python -m omnirag.knowledge_base load --all         # 12 samples from disk
uv run python -m omnirag.knowledge_base stats
uv run python -m omnirag.knowledge_base search-vector "Why did cell AUR-C103 fail?" --k 3
uv run python -m omnirag.knowledge_base search-text '"AUR-C103"' --k 3
uv run python -m omnirag.knowledge_base show-chunk doc_865ac481a2392f72-multimodal-256-ctx-00003
uv run python -m omnirag.knowledge_base outline doc_2b1c659c54266d54

# integration tests use a throwaway Neo4j (port 7688, data in RAM, wiped by the tests)
docker compose --profile test up -d neo4j-test
uv run pytest -m integration tests/integration/test_graph_live.py
docker compose --profile test stop neo4j-test              # frees ~0.8 GB
```
Browse the graph at http://localhost:7474, for example:
`MATCH p=(:Document {source_file:'aurora_q1_review.pptx'})-[*1..2]->() RETURN p`.

## 6. Expected outputs (measured 2026-11-05, Neo4j 2026.10.1 Community)
`init` on an empty database creates 7 constraints, 7 lookup indexes, `chunk_embedding` and `chunk_fulltext`. A second run lists them all under `existing`.

`load --all` (12 samples): every document's `vectors` equals its `chunks`, with `missing=0`. Each document takes 75–310 ms, except the first one (~2.8 s: driver connection and query planning).

`stats`:
| Nodes | | Relationships | |
|---|---|---|---|
| Chunk | 58 | HAS_CHUNK | 58 |
| Document / Source | 12 / 12 | NEXT_CHUNK | 46 (= 58 − 12 documents) |
| Section | 40 | HAS_SECTION | 40 |
| Page | 11 | CONTAINS | 70 |
| Table | 11 | HAS_TABLE | 12 (11 + 1 chart table under its figure) |
| Figure | 8 | DISCUSSED_IN | 22 |

`chunks_with_vectors: 58`, `vector_dim: 1024`. There are no CHILD_OF links, because `multimodal-256-ctx` is flat.

`search-vector "Why did cell AUR-C103 fail?"`:
| # | Chunk | Score |
|---|---|---|
| 1 | aurora_research_notes.md, "Results": *Cell AUR-C103 failed at cycle 412 because of gas evolution…* | 0.837 |
| 2 | aurora_safety_review.pdf, "1. Incident Summary" | 0.816 |
| 3 | meeting_notes_2026-03-12.txt | 0.815 |

Neo4j reports cosine as (1 + cos) / 2, so 0.837 corresponds to cos ≈ 0.67.

`search-text '"AUR-C103"'` returns three chunks, all about C103. The top one is *All cells except AUR-C103 meet the 5% swelling limit…*.

`show-chunk …pptx…-00003` (the E-7 retention chart slide):
- `previous` / `next` chunk IDs;
- `section ["E-7 Retention at 45 °C"]`, `pages [3]`;
- `figures [e00006]` (the chart) and `tables [e00007]` (its data table).

Tests: 262 unit tests (19 new) and 12 integration tests (6 new graph tests) pass. The new graph tests cover idempotent writes, versioning out of order, re-chaining after a delete, in-index filtering, the dimension-mismatch guard, and full-text search.

## 7. Surprises found during verification
- **New vector indexes default to BINARY quantization** in Neo4j 2026.x: 1 bit per dimension for the candidate stage. We set `NONE` explicitly, so comparisons in Phases 6–8 use exact vectors. A test asserts it.
- **The `queryNodes` procedures are deprecated.** The first version used `db.index.vector.queryNodes`, and the server warned. Now we use Cypher 25 `SEARCH`. Index names can't be parameters there, so they're validated as identifiers before being put in the query.
- **Full-text `SEARCH` accepts no WHERE**, unlike vector `SEARCH`. Full-text results are over-fetched (k×10) and then filtered.
- **The native `VECTOR` property type** is rejected (error 51N77) on Community 2026.10.1. A float list set via `db.create.setNodeVectorProperty` works with the index.
- **`AUR-C103` without quotes also matches AUR-C105**: the standard analyzer splits on the hyphen. Use phrase queries for now; Phase 6 evaluates analyzers.
- **Dropping a constraint's backing index fails** (`IndexDropFailed`): drop constraints first, then indexes. This was found when the test fixture cleaned the schema.
- **A "relationship type CHILD_OF does not exist" warning** appears on graphs without hierarchical chunks. It's harmless, and that notification class is now disabled; deprecation warnings stay visible.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `GraphUnavailableError: ... Unable to retrieve routing information` / connection refused | Neo4j isn't running: start Docker Desktop, then `docker compose up -d`. |
| `... authentication failure` | `OMNIRAG_NEO4J_PASSWORD` in `.env` doesn't match `NEO4J_PASSWORD`, the one the container was created with. |
| `SchemaMismatchError: Vector index chunk_embedding has dimension 1024 ... requested 768` | The embedding model changed. Re-embed, then `init --recreate-vector-index`, then `load --all`. |
| Ingestion warning `Graph write failed: ...` | Neo4j was down during ingestion. The files are fine: run `load --all` (or `load DOC_ID`) later. |
| `DELETE /documents/{id}` returns 503 | Neo4j is down. Nothing was deleted, on purpose, so the graph can't keep orphans. |
| Integration tests: `Refusing to wipe bolt://localhost:7687` | The graph tests only run against the test server on 7688: `docker compose --profile test up -d neo4j-test`. |
| `search-vector` returns `[]` | No vectors yet (`missing>0` in `load`): run `python -m omnirag.embeddings embed --all`, then `load --all`. |
| Old content still appears in search | Pass `--all-versions` only if you want old versions. Otherwise check `versions FILE`: the newest version must be `is_latest`. |

## 9. Exercise (≈30 min)
1. **Versioning by hand.**
   - Copy `data/samples/meeting_notes_2026-03-12.txt` to the scratch directory and change one sentence.
   - Upload it through `POST /api/v1/documents` under the **same file name**.
   - Run `versions meeting_notes_2026-03-12.txt`: you should see v1 (not latest) and v2 (latest, `previous` = v1).
   - `search-text` for the changed sentence should find only v2; with `--all-versions`, both versions' chunks can appear.
   - `DELETE` v2 through the API and check that v1 is latest again.
   - Finally, delete v1 too, and re-ingest the original sample so the corpus is back to 12 documents.
2. **Cypher.** In the Neo4j Browser, write a query that returns, for every Figure, its document, its caption and the IDs of the chunks it is `DISCUSSED_IN`.
   - Measured: all 8 figures have at least one link, and the PPTX chart has two. Why can one figure be discussed in two chunks? Hint: compare its element IDs with the `element_ids` of both chunks.
   - Then do the same for tables: which document has a table discussed in two chunks?
3. **Think.** Why does the writer delete and recreate a document's nodes instead of `MERGE`-ing each one? What would MERGE get wrong when a re-ingested document has *fewer* chunks than before?

## 10. Acceptance criteria — Phase 5a
- [ ] `init` reports 16 schema objects, and `SHOW VECTOR INDEXES` shows 1024 dimensions, cosine, quantization NONE.
- [ ] `load --all` loads 12 documents and 58 chunks, every one with a vector (`missing=0`).
- [ ] `stats` matches §6, and a second `load --all` leaves every count unchanged.
- [ ] `search-vector` and `search-text` return the C103 chunks shown in §6.
- [ ] Exercise 1: two versions chained, only the latest searched by default, deletion re-chains.
- [ ] `uv run pytest` (262 passed) and `uv run pytest -m integration` with `neo4j-test` running (12 passed).
