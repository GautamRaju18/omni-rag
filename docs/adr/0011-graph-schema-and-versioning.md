# ADR-0011: Graph schema, idempotent per-document writes, and source-level versioning

**Status:** accepted (2026-11-05, Phase 5a)

## Context
The brief (section 7) requires:
- **Nodes:** Document, Page, Section, Chunk, Entity, Table, Figure, Source.
- **Relationships:** HAS_PAGE, HAS_SECTION, HAS_CHUNK, CONTAINS, MENTIONS, RELATED_TO, CITES, HAS_TABLE, HAS_FIGURE, CHILD_OF, NEXT_CHUNK, DISCUSSED_IN.
- **Behaviour:** chunk vectors stored on nodes with the right dimension and similarity, vector and full-text indexes, document versioning, deletion, and duplicate prevention.

Phase 5a covers the document structure. Entities (MENTIONS, RELATED_TO, CITES) come in Phase 5b, with validation and provenance.

Facts measured on Neo4j 2026.10.1 Community (2026-11-05):
- **Vector quantization:** new vector indexes default to `vector.quantization.type: BINARY`. Valid values are NONE, SCALAR and BINARY.
- **Native vector type:** the native `VECTOR` property type is rejected with error 51N77 on this edition and version. `db.create.setNodeVectorProperty` stores a float list that the index accepts.
- **Deprecated procedures:** `db.index.vector.queryNodes` and `db.index.fulltext.queryNodes` are deprecated in favour of the Cypher 25 `SEARCH` clause.
- **Vector `SEARCH`:**
  - It can filter inside the index on properties declared `WITH [...]` at index creation.
  - The index name can't be a parameter.
- **Full-text `SEARCH`** accepts no `WHERE`.
- **Full-text tokenisation:** the `standard-no-stop-words` analyzer splits `AUR-C103` into `aur` and `c103`. An unquoted query therefore also matches AUR-C105; a phrase query (`"AUR-C103"`) is exact.
- **Single database:** Community has one user database, so tests can't use a separate database on the same server.
- **Schema DDL** can't share a transaction with data writes.

## Decision
**Schema.** One uniqueness constraint per node key (`source_key`, `document_id`, `page_id`, `section_id`, `chunk_id`, `table_id`, `figure_id`) and lookup indexes on `document_id` for all owned labels.
- The vector index `chunk_embedding` is on `Chunk.embedding`:
  - 1024 dimensions, cosine similarity.
  - Quantization is explicitly **NONE**: exact vectors, so model and strategy comparisons aren't blurred by a lossy candidate stage.
  - Filter properties `config_id` and `is_latest` are declared `WITH [...]`.
- The full-text index `chunk_fulltext` covers `content` and `context`.
- `ensure_schema` is idempotent. It raises `SchemaMismatchError` when the existing vector index has a different dimension or quantization; it recreates the index only when asked.

**Structure.**
- Documents and pages: `Document-HAS_PAGE->Page`.
- Sections: `Document-HAS_SECTION->Section-HAS_SECTION->Section`. The tree comes from heading trails, and `section_id` is a hash of the path.
- Chunks:
  - `Document-HAS_CHUNK->Chunk`.
  - `Section/Page-CONTAINS->Chunk`.
  - `NEXT_CHUNK` follows reading order within one chunk set and level.
  - `CHILD_OF` links a child chunk to its parent chunk.
- Tables and figures:
  - `HAS_TABLE` / `HAS_FIGURE`; a chart's data table hangs off its figure (`Figure-HAS_TABLE->Table`).
  - `Table/Figure-DISCUSSED_IN->Chunk` is derived from chunk `element_ids`. A row or a caption is enough to make the link.
  - Figures keep the alt text, caption, OCR text and generated description **as separate properties**, plus the model that wrote the description.
- Only chunk sets in `graph_chunk_configs` become Chunk nodes; by default, the embedded sets.

**Writes.** Each document is written in one managed transaction:
1. MERGE the Source and the Document.
2. Delete every node the document owns.
3. Recreate them with UNWIND batches.
4. Recompute versions.

The result:
- Loading the same document twice changes nothing (tested).
- A failure rolls back to the previous state.
- Duplicates are impossible, because every ID is content-addressed or deterministic.

Vectors are read from the embedding cache by `embedding_ref`, so loading the graph never calls the model.

**Versioning.**
- **Sources and versions:** a Source is the logical document, keyed by file name (case-insensitive). Each distinct content hash is a Document version.
- **Order and chain:** versions are ordered by first-ingest time and then document ID, numbered 1..n, and chained by `PREVIOUS_VERSION` (newer to older).
- **Latest flag:** only the newest has `is_latest`, copied onto its chunks so search can filter inside the index.
- **Recompute on every write:** the chain is rebuilt each time, so re-loading an old version can't create a cycle or a second chain (tested).

**Deletion.**
- The API deletes from the graph first, then the files.
- If Neo4j is down, it returns 503 and keeps the files, so the graph never holds orphans.
- Deleting a version re-chains the remaining versions. The Source goes with its last version.

**Pipeline.** The graph write runs after chunking and embedding. Its failure is a warning, and `python -m omnirag.knowledge_base load` rebuilds the graph from disk. The files on disk stay the source of truth.

**Tests.** A separate `neo4j-test` container (compose profile `test`, port 7688, data on tmpfs, 1 GB). The integration tests refuse to run against any other port, because they wipe the database.

## Consequences
- The graph is a rebuildable index of the files, not primary storage. Dropping it loses nothing.
- Versions follow the file name. A renamed file starts a new Source, and two different documents that share a name become versions of each other. That's acceptable for the corpus; an explicit `source_key` field on upload can come later if needed.
- Full-text filtering happens after the search (over-fetch k×10). Phase 6 measures whether results are lost.
- Identifier search through the standard analyzer needs phrase queries. Phase 6 evaluates analyzers and query forms.
