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


def entities(
    store: GraphStore, *, entity_type: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    return store.query(
        """
        MATCH (e:Entity) WHERE $type IS NULL OR e.type = $type
        RETURN e.name AS name, e.type AS type, e.mention_count AS mentions,
               e.document_count AS documents, e.aliases AS aliases,
               COUNT { (e)-[:RELATED_TO]-() } AS relations
        ORDER BY mentions DESC, name LIMIT $limit
        """,
        type=entity_type, limit=limit,
    )  # fmt: skip


def entity_detail(store: GraphStore, name: str, *, latest_only: bool = True) -> dict | None:
    """An entity (by name or alias key) with its relations, mentions and SAME_AS links."""
    from omnirag.knowledge_base.entities import entity_key

    rows = store.query(
        """
        MATCH (e:Entity) WHERE e.key = $key OR $name IN e.aliases
        WITH e ORDER BY e.key = $key DESC, e.mention_count DESC LIMIT 1  // exact key first
        OPTIONAL MATCH (e)-[s:SAME_AS]-(same:Entity)
        WITH e, collect(DISTINCT {name: same.name, rule: s.rule}) AS same_as
        OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
        WHERE NOT $latest OR r.is_latest
        WITH e, same_as, collect(DISTINCT {
            direction: CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END,
            type: r.type, other: other.name, support: r.support, evidence: r.evidence,
            evidence_origin: r.evidence_origin, direction_fixed: r.direction_fixed,
            chunk_id: r.chunk_id}) AS relations
        OPTIONAL MATCH (c:Chunk)-[m:MENTIONS]->(e) WHERE NOT $latest OR c.is_latest
        RETURN e {.*} AS entity, same_as, relations,
               collect({chunk_id: c.chunk_id, surface: m.surface, origin: m.origin,
                        confidence: m.confidence, resolution: m.resolution}) AS mentions
        """,
        key=entity_key(name), name=name, latest=latest_only,
    )  # fmt: skip
    if not rows:
        return None
    row = rows[0]
    row["same_as"] = [s for s in row["same_as"] if s["name"]]
    row["relations"] = [r for r in row["relations"] if r["type"]]
    row["mentions"] = [m for m in row["mentions"] if m["chunk_id"]]
    return row


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


def neighborhood(store: GraphStore, *, entity: str | None = None,
                 document_id: str | None = None, limit: int = 60) -> dict[str, Any]:  # fmt: skip
    """Nodes and edges for the graph explorer (Phase 14), latest versions only.

    * ``entity``: the entity, its RELATED_TO neighbours, the chunks that mention it and their
      documents;
    * ``document_id``: the document, its chunks (with sections) and the entities they mention;
    * neither: documents linked to their most-mentioned entities (mention counts on the edges).
    """
    from omnirag.knowledge_base.entities import entity_key

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(nid: str, label: str, kind: str, **props: Any) -> str:
        nodes.setdefault(nid, {"id": nid, "label": label, "kind": kind, **props})
        return nid

    if entity:
        rows = store.query(
            """
            MATCH (e:Entity) WHERE e.key = $key OR $name IN e.aliases
            WITH e ORDER BY e.key = $key DESC, e.mention_count DESC LIMIT 1
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(o:Entity) WHERE r.is_latest
            WITH e, collect(DISTINCT {other: o.name, otype: o.type, type: r.type,
                 out: startNode(r) = e, support: r.support})[..$limit] AS rels
            OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e)
            WHERE c.is_latest
            RETURN e.name AS name, e.type AS type, rels,
                   collect(DISTINCT {chunk: c.chunk_id, file: d.source_file,
                                     document_id: d.document_id})[..$limit] AS mentions
            """,
            key=entity_key(entity), name=entity, limit=limit,
        )  # fmt: skip
        if rows:
            r = rows[0]
            center = node(f"e:{r['name']}", r["name"], "entity", type=r["type"], center=True)
            for rel in r["rels"]:
                if not rel["other"]:
                    continue
                other = node(f"e:{rel['other']}", rel["other"], "entity", type=rel["otype"])
                src, dst = (center, other) if rel["out"] else (other, center)
                edges.append({"source": src, "target": dst, "type": rel["type"],
                              "support": rel["support"]})  # fmt: skip
            for m in r["mentions"]:
                if not m["chunk"]:
                    continue
                c = node(f"c:{m['chunk']}", m["chunk"].rsplit("-", 1)[-1], "chunk",
                         chunk_id=m["chunk"])  # fmt: skip
                d = node(f"d:{m['document_id']}", m["file"], "document",
                         document_id=m["document_id"])  # fmt: skip
                edges += [
                    {"source": c, "target": center, "type": "MENTIONS"},
                    {"source": d, "target": c, "type": "HAS_CHUNK"},
                ]
    elif document_id:
        rows = store.query(
            """
            MATCH (d:Document {document_id: $id})-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH d, c, collect(DISTINCT {name: e.name, type: e.type}) AS ents
            ORDER BY c.chunk_index
            RETURN d.source_file AS file, c.chunk_id AS chunk, c.section_path AS section,
                   c.config_id AS config, ents LIMIT $limit
            """,
            id=document_id, limit=limit,
        )  # fmt: skip
        for r in rows:
            d = node(f"d:{document_id}", r["file"], "document", document_id=document_id,
                     center=True)  # fmt: skip
            label = " > ".join(r["section"] or []) or r["chunk"].rsplit("-", 1)[-1]
            c = node(f"c:{r['chunk']}", label[:40], "chunk", chunk_id=r["chunk"],
                     config=r["config"])  # fmt: skip
            edges.append({"source": d, "target": c, "type": "HAS_CHUNK"})
            for e in r["ents"]:
                if e["name"]:
                    en = node(f"e:{e['name']}", e["name"], "entity", type=e["type"])
                    edges.append({"source": c, "target": en, "type": "MENTIONS"})
    else:
        rows = store.query(
            """
            MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE d.is_latest
            WITH d, e, count(DISTINCT c) AS n ORDER BY n DESC, e.name
            RETURN d.document_id AS document_id, d.source_file AS file, e.name AS entity,
                   e.type AS type, n LIMIT $limit
            """,
            limit=limit,
        )  # fmt: skip
        for r in rows:
            d = node(f"d:{r['document_id']}", r["file"], "document", document_id=r["document_id"])
            e = node(f"e:{r['entity']}", r["entity"], "entity", type=r["type"])
            edges.append({"source": d, "target": e, "type": "MENTIONS", "count": r["n"]})
    return {"nodes": list(nodes.values()), "edges": edges}
