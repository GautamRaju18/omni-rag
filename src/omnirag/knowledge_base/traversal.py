"""Read-side graph queries: context around a chunk, outlines, versions, and sanity searches.

The searches here are *checks* that the indexes work, not the retriever: Phase 6 builds and
evaluates the real dense / full-text retrievers on top of the same indexes.
"""

from __future__ import annotations

import re
from typing import Any

from omnirag.knowledge_base.graph_store import GraphStore


def stats(store: GraphStore) -> dict[str, Any]:
    nodes = store.query("MATCH (n) UNWIND labels(n) AS label "
                        "RETURN label, count(*) AS n ORDER BY label")  # fmt: skip
    rels = store.query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")
    vectors = store.query("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
                          "RETURN count(c) AS n, max(size(c.embedding)) AS dim")[0]  # fmt: skip
    return {
        "nodes": {r["label"]: r["n"] for r in nodes},
        "relationships": {r["type"]: r["n"] for r in rels},
        "chunks_with_vectors": vectors["n"],
        "vector_dim": vectors["dim"],
    }


def chunk_context(store: GraphStore, chunk_id: str) -> dict[str, Any] | None:
    """A chunk with everything a citation or context expansion needs."""
    rows = store.query(
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk {chunk_id: $id})
        OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
        OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
        OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Chunk)
        OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c)
        OPTIONAL MATCH (p:Page)-[:CONTAINS]->(c)
        OPTIONAL MATCH (x)-[:DISCUSSED_IN]->(c)
        RETURN c {.*, embedding: null} AS chunk,
               d {.document_id, .source_file, .title, .version, .is_latest} AS document,
               prev.chunk_id AS previous, next.chunk_id AS next, parent.chunk_id AS parent,
               s.path AS section, collect(DISTINCT p.number) AS pages,
               collect(DISTINCT CASE WHEN x:Table THEN x.table_id END) AS tables,
               collect(DISTINCT CASE WHEN x:Figure THEN x.figure_id END) AS figures
        """,
        id=chunk_id,
    )
    if not rows:
        return None
    row = rows[0]
    row["chunk"] = {k: v for k, v in row["chunk"].items() if v is not None}
    row["pages"] = sorted(row["pages"])
    return row


def document_outline(store: GraphStore, document_id: str) -> dict[str, Any] | None:
    doc = store.query("MATCH (d:Document {document_id: $id})-[:VERSION_OF]->(s:Source) "
                      "RETURN d {.*} AS document, s.name AS source", id=document_id)  # fmt: skip
    if not doc:
        return None
    sections = store.query(
        """
        MATCH (s:Section {document_id: $id})
        OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
        RETURN s.path AS path, s.level AS level, count(c) AS chunks
        ORDER BY s.path
        """,
        id=document_id,
    )
    counts = store.query(
        """
        MATCH (d:Document {document_id: $id})
        RETURN COUNT { (d)-[:HAS_PAGE]->() } AS pages,
               COUNT { (d)-[:HAS_CHUNK]->() } AS chunks,
               COUNT { (d)-[:HAS_TABLE]->() } AS tables,
               COUNT { (d)-[:HAS_FIGURE]->() } AS figures
        """,
        id=document_id,
    )[0]
    return {**doc[0], "counts": counts, "sections": sections}


def versions(store: GraphStore, source_file: str) -> list[dict[str, Any]]:
    return store.query(
        """
        MATCH (s:Source {source_key: $key})<-[:VERSION_OF]-(d:Document)
        OPTIONAL MATCH (d)-[:PREVIOUS_VERSION]->(prev:Document)
        RETURN d.document_id AS document_id, d.version AS version, d.is_latest AS is_latest,
               d.sha256 AS sha256, d.created_at AS created_at, prev.document_id AS previous
        ORDER BY d.version
        """,
        key=source_file.strip().lower(),
    )


def _index_name(name: str) -> str:
    """Index names can't be parameters in a SEARCH clause: allow plain identifiers only."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid index name {name!r}")
    return name


def vector_search(
    store: GraphStore,
    vector: list[float],
    *,
    k: int = 5,
    index: str = "chunk_embedding",
    config_id: str | None = None,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Top-k chunks by cosine similarity (Neo4j reports cosine as (1 + cos) / 2, in [0, 1]).

    Cypher 25 ``SEARCH`` filters *inside* the index on the properties the index was created
    ``WITH`` (config_id, is_latest), so the k results all pass the filter. Measured
    2026-11-05: ``db.index.vector.queryNodes`` is deprecated in Neo4j 2026.10 and the index
    name can't be a parameter here.
    """
    where = []
    if config_id is not None:
        where.append("c.config_id = $config")
    if latest_only:
        where.append("c.is_latest = true")
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    return store.query(
        f"CYPHER 25 MATCH (c:Chunk) SEARCH c IN (VECTOR INDEX {_index_name(index)} "
        f"FOR $vector{clause} LIMIT $k) SCORE AS score "
        "RETURN c.chunk_id AS chunk_id, c.source_file AS source_file, "
        "c.section_path AS section, score, left(c.content, 160) AS preview "
        "ORDER BY score DESC",
        vector=vector, config=config_id, k=k,
    )  # fmt: skip


def fulltext_search(
    store: GraphStore,
    query: str,
    *,
    k: int = 5,
    index: str = "chunk_fulltext",
    config_id: str | None = None,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Top-k chunks by Lucene BM25 over content + context; ``query`` is Lucene syntax.

    Full-text ``SEARCH`` accepts no WHERE (measured 2026-11-05), so this over-fetches and
    filters afterwards: fine for checks, and Phase 6 measures whether it loses results.
    """
    return store.query(
        f"CYPHER 25 MATCH (c:Chunk) SEARCH c IN (FULLTEXT INDEX {_index_name(index)} "
        "FOR $query LIMIT $fetch) SCORE AS score "
        "WITH c, score WHERE ($config IS NULL OR c.config_id = $config) "
        "AND (NOT $latest OR c.is_latest) "
        "RETURN c.chunk_id AS chunk_id, c.source_file AS source_file, "
        "c.section_path AS section, score, left(c.content, 160) AS preview "
        "ORDER BY score DESC LIMIT $k",
        query=query, fetch=k * 10, config=config_id, latest=latest_only, k=k,
    )  # fmt: skip
