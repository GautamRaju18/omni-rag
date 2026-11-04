# ADR-0003: Neo4j as the single retrieval store (graph + vector + full-text)

- Status: Accepted (2026-11-04)

## Context
GraphRAG needs graph traversal, dense vector search and lexical (BM25) search. Common setups add a separate vector DB (Qdrant/Chroma) and a separate search engine, which means three stores to keep consistent and three containers on a machine with an 8 GB VM budget.

## Decision
Neo4j Community (`neo4j:2026.10.1-community`) holds documents, chunks, entities and relationships, plus **vector indexes** on chunk embeddings and **full-text (Lucene) indexes** for BM25-style lexical search. PostgreSQL arrives later only for operational metadata (jobs, chat history, eval runs), not for retrieval.

Container budget: `mem_limit: 2g`, heap 512m–1G, pagecache 512m. Measured idle: 940 MiB.

## Consequences
- One write path keeps chunk text, its embedding and its graph neighbourhood consistent, and deleting a document cascades in one transaction.
- Hybrid retrieval can be a single Cypher query that combines vector hits with graph expansion.
- Full-text scoring is Lucene BM25 with Neo4j's analyzer choices. Exact-identifier matching needs a suitable analyzer, to be evaluated in Phase 6.
- The Community edition has no online backups or clustering. That's fine for a local-first project.
- The pagecache will need to grow as the corpus grows. Revisit after Phase 5 with actual index sizes.
