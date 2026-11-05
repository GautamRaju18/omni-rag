"""Build and maintain each document's subgraph in Neo4j.

    (:Source {source_key})<-[:VERSION_OF]-(:Document {document_id, version, is_latest})
    (:Document)-[:PREVIOUS_VERSION]->(:Document)            older version of the same source
    (:Document)-[:HAS_PAGE]->(:Page)-[:CONTAINS]->(:Chunk)
    (:Document)-[:HAS_SECTION]->(:Section)-[:HAS_SECTION]->(:Section)-[:CONTAINS]->(:Chunk)
    (:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT_CHUNK]->(:Chunk)   reading order, per set + level
    (:Chunk)-[:CHILD_OF]->(:Chunk)                          hierarchical / parent-child
    (:Document)-[:HAS_TABLE]->(:Table)-[:DISCUSSED_IN]->(:Chunk)
    (:Document)-[:HAS_FIGURE]->(:Figure)-[:DISCUSSED_IN]->(:Chunk)
    (:Figure)-[:HAS_TABLE]->(:Table)                         a chart's data table

**Writes are idempotent replacements.** One transaction per document merges the Source and
Document, deletes every node the document owned, recreates them, and recomputes the version
chain. Loading twice never duplicates anything, and a failure leaves the previous state intact.

**Versioning.** A *source* is the logical document (keyed by file name, case-insensitive); each
distinct content hash is a *version*. Versions are ordered by first-ingest time, numbered
1..n and chained with PREVIOUS_VERSION. Only the newest has ``is_latest = true``, copied onto
its chunks for search filtering. Recomputing the chain on every write means re-loading an old
version can't create a cycle.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from neo4j import ManagedTransaction

from omnirag.chunking.models import Chunk
from omnirag.embeddings.ollama import EmbeddingCache
from omnirag.ingestion.models import ContentOrigin, DocumentElement, DocumentRecord, ElementType
from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.knowledge_base.schema import SchemaReport, ensure_schema

OWNED_LABELS = ("Chunk", "Page", "Section", "Table", "Figure")  # deleted with their document


def page_id(document_id: str, number: int) -> str:
    return f"{document_id}-p{number:04d}"


def section_id(document_id: str, path: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(path).encode("utf-8")).hexdigest()[:12]
    return f"{document_id}-s{digest}"


def source_key(source_file: str) -> str:
    return source_file.strip().lower()


def build_payload(
    record: DocumentRecord,
    elements: list[DocumentElement],
    chunk_sets: dict[str, list[Chunk]],
    vectors: dict[str, list[float]],
    embedding_space: str | None,
) -> dict[str, Any]:
    """Plain rows (primitives and lists only) for the Cypher statements."""
    doc_id = record.document_id
    meta = record.metadata
    chunks = [c for cs in chunk_sets.values() for c in cs]
    by_id = {e.element_id: e for e in elements}
    children: dict[str, list[DocumentElement]] = defaultdict(list)
    for e in elements:
        if e.parent_element_id:
            children[e.parent_element_id].append(e)

    document = {
        "document_id": doc_id,
        "source_file": record.source_file,
        "file_kind": record.file_kind.value,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "title": str(meta.get("title") or record.source_file),
        "author": meta.get("author"),
        "page_count": meta.get("page_count"),
        "element_count": record.element_count,
        "pdf_engine": meta.get("engine"),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "graph_loaded_at": datetime.now(UTC).isoformat(),
        "embedding_space": embedding_space,
    }

    numbers = {e.page_number for e in elements if e.page_number}
    numbers |= {p for c in chunks for p in c.page_numbers}
    pages = [{"page_id": page_id(doc_id, n), "document_id": doc_id, "number": n}
             for n in sorted(numbers)]  # fmt: skip

    paths: set[tuple[str, ...]] = set()
    for e in elements:
        path = list(e.section_path)
        heading = e.element_type in (ElementType.HEADING, ElementType.TITLE) and e.content
        if heading and (not path or path[-1] != e.content):  # a heading's path = its parents
            path.append(e.content)
        for i in range(1, len(path) + 1):
            paths.add(tuple(path[:i]))
    for c in chunks:
        for i in range(1, len(c.section_path) + 1):
            paths.add(tuple(c.section_path[:i]))
    sections = [
        {"section_id": section_id(doc_id, p), "document_id": doc_id, "title": p[-1],
         "path": list(p), "level": len(p),
         "parent_id": section_id(doc_id, p[:-1]) if len(p) > 1 else None}
        for p in sorted(paths, key=lambda p: (len(p), p))
    ]  # fmt: skip

    row_table: dict[str, str] = {}  # TABLE_ROW element -> its TABLE
    tables = []
    for e in elements:
        if e.element_type is not ElementType.TABLE:
            continue
        parent = by_id.get(e.parent_element_id or "")
        for r in children[e.element_id]:
            row_table[r.element_id] = e.element_id
        tables.append({
            "table_id": e.element_id, "document_id": doc_id, "page_number": e.page_number,
            "section_path": e.section_path, "content": e.content,
            "columns": [str(c) for c in e.metadata.get("columns", [])],
            "n_rows": e.metadata.get("n_rows"), "n_cols": e.metadata.get("n_cols"),
            "caption": e.metadata.get("caption"), "confidence": e.confidence,
            "figure_id": parent.element_id if parent is not None
            and parent.element_type is ElementType.FIGURE else None,
        })  # fmt: skip

    member_figure: dict[str, str] = {}  # element inside a figure group -> the figure
    figures = []
    for e in elements:
        if e.element_type not in (ElementType.IMAGE, ElementType.FIGURE):
            continue
        group, stack = [], list(children[e.element_id])
        while stack:
            g = stack.pop()
            group.append(g)
            stack.extend(children[g.element_id])
        for g in group:
            member_figure[g.element_id] = e.element_id
        description = next((g for g in group if g.element_type is ElementType.DESCRIPTION), None)
        figures.append({
            "figure_id": e.element_id, "document_id": doc_id, "kind": e.element_type.value,
            "page_number": e.page_number, "section_path": e.section_path,
            "alt_text": e.content if e.element_type is ElementType.IMAGE and e.content else None,
            "title": e.content if e.element_type is ElementType.FIGURE else None,
            "caption": " ".join(g.content for g in group
                                if g.element_type is ElementType.CAPTION) or None,
            "ocr_text": "\n".join(g.content for g in group if g.origin is ContentOrigin.OCR)
            or None,
            "description": description.content if description else None,
            "description_model": description.metadata.get("model") if description else None,
            "asset_ref": e.asset_ref,
        })  # fmt: skip

    chunk_rows, section_links, page_links, discussed = [], [], [], []
    for c in chunks:
        chunk_rows.append({
            "props": {
                "chunk_id": c.chunk_id, "document_id": doc_id, "source_file": c.source_file,
                "config_id": c.config_id, "strategy": c.chunking_strategy.value,
                "chunk_index": c.chunk_index, "level": c.level, "retrievable": c.retrievable,
                "content": c.content, "context": c.context, "token_count": c.token_count,
                "page_number": c.page_number, "page_numbers": c.page_numbers,
                "section": c.section, "section_path": c.section_path,
                "element_type": c.element_type, "element_ids": c.element_ids,
                "partial_element_ids": c.partial_element_ids, "origins": c.origins,
                "min_confidence": c.min_confidence, "embedding_ref": c.embedding_ref,
                "parent_chunk_id": c.parent_chunk_id, "is_latest": False,
            },
            "vector": vectors.get(c.chunk_id),
        })  # fmt: skip
        if c.section_path:
            section_links.append({"section_id": section_id(doc_id, c.section_path),
                                  "chunk_id": c.chunk_id})  # fmt: skip
        page_links += [{"page_id": page_id(doc_id, p), "chunk_id": c.chunk_id}
                       for p in c.page_numbers]  # fmt: skip
        ids = set(c.element_ids)
        hit_tables = {t["table_id"] for t in tables if t["table_id"] in ids}
        hit_tables |= {row_table[x] for x in ids if x in row_table}
        hit_figures = {f["figure_id"] for f in figures if f["figure_id"] in ids}
        hit_figures |= {member_figure[x] for x in ids if x in member_figure}
        discussed += [{"label": "Table", "id": t, "chunk_id": c.chunk_id} for t in hit_tables]
        discussed += [{"label": "Figure", "id": f, "chunk_id": c.chunk_id} for f in hit_figures]

    next_links = []
    groups: dict[tuple[str, int], list[Chunk]] = defaultdict(list)
    for c in chunks:
        groups[(c.config_id, c.level)].append(c)
    for group in groups.values():
        group.sort(key=lambda c: c.chunk_index)
        next_links += [{"from": a.chunk_id, "to": b.chunk_id} for a, b in pairwise(group)]
    child_links = [{"child": c.chunk_id, "parent": c.parent_chunk_id}
                   for c in chunks if c.parent_chunk_id]  # fmt: skip

    return {
        "document": document,
        "source": {"source_key": source_key(record.source_file), "name": record.source_file},
        "pages": pages, "sections": sections, "tables": tables, "figures": figures,
        "chunks": chunk_rows, "section_links": section_links, "page_links": page_links,
        "next_links": next_links, "child_links": child_links,
        "table_links": [d for d in discussed if d["label"] == "Table"],
        "figure_links": [d for d in discussed if d["label"] == "Figure"],
    }  # fmt: skip


# ------------------------------------------------------------- transactions
def _write(tx: ManagedTransaction, p: dict[str, Any]) -> dict[str, int]:
    doc_id = p["document"]["document_id"]
    tx.run("MERGE (s:Source {source_key: $s.source_key}) "
           "ON CREATE SET s.created_at = $now SET s.name = $s.name",
           s=p["source"], now=datetime.now(UTC).isoformat())  # fmt: skip
    tx.run("MERGE (d:Document {document_id: $doc.document_id}) SET d += $doc "
           "WITH d MATCH (s:Source {source_key: $key}) MERGE (d)-[:VERSION_OF]->(s)",
           doc=p["document"], key=p["source"]["source_key"])  # fmt: skip
    for label in OWNED_LABELS:  # constants, not user input
        tx.run(f"MATCH (n:{label} {{document_id: $id}}) DETACH DELETE n", id=doc_id)

    tx.run("UNWIND $rows AS r MATCH (d:Document {document_id: r.document_id}) "
           "CREATE (d)-[:HAS_PAGE]->(:Page {page_id: r.page_id, document_id: r.document_id, "
           "number: r.number})", rows=p["pages"])  # fmt: skip
    tx.run("UNWIND $rows AS r CREATE (s:Section) SET s = r", rows=p["sections"])
    tx.run("UNWIND $rows AS r MATCH (s:Section {section_id: r.section_id}) "
           "MATCH (d:Document {document_id: r.document_id}) "
           "OPTIONAL MATCH (parent:Section {section_id: r.parent_id}) "
           "FOREACH (_ IN CASE WHEN parent IS NULL THEN [1] ELSE [] END | "
           "  CREATE (d)-[:HAS_SECTION]->(s)) "
           "FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END | "
           "  CREATE (parent)-[:HAS_SECTION]->(s))", rows=p["sections"])  # fmt: skip
    tx.run("UNWIND $rows AS r MATCH (d:Document {document_id: r.document_id}) "
           "CREATE (d)-[:HAS_TABLE]->(t:Table) SET t = r", rows=p["tables"])  # fmt: skip
    tx.run("UNWIND $rows AS r MATCH (d:Document {document_id: r.document_id}) "
           "CREATE (d)-[:HAS_FIGURE]->(f:Figure) SET f = r", rows=p["figures"])  # fmt: skip
    tx.run(
        "MATCH (t:Table {document_id: $id}) WHERE t.figure_id IS NOT NULL "
        "MATCH (f:Figure {figure_id: t.figure_id}) CREATE (f)-[:HAS_TABLE]->(t)",
        id=doc_id,
    )
    tx.run("UNWIND $rows AS r MATCH (d:Document {document_id: r.props.document_id}) "
           "CREATE (d)-[:HAS_CHUNK]->(c:Chunk) SET c = r.props "
           "WITH c, r WHERE r.vector IS NOT NULL "
           "CALL db.create.setNodeVectorProperty(c, 'embedding', r.vector)",
           rows=p["chunks"])  # fmt: skip
    for key, query in (
        ("section_links", "MATCH (a:Section {section_id: r.section_id}) "
                          "MATCH (b:Chunk {chunk_id: r.chunk_id}) CREATE (a)-[:CONTAINS]->(b)"),
        ("page_links", "MATCH (a:Page {page_id: r.page_id}) "
                       "MATCH (b:Chunk {chunk_id: r.chunk_id}) CREATE (a)-[:CONTAINS]->(b)"),
        ("next_links", "MATCH (a:Chunk {chunk_id: r.from}) MATCH (b:Chunk {chunk_id: r.to}) "
                       "CREATE (a)-[:NEXT_CHUNK]->(b)"),
        ("child_links", "MATCH (a:Chunk {chunk_id: r.child}) "
                        "MATCH (b:Chunk {chunk_id: r.parent}) CREATE (a)-[:CHILD_OF]->(b)"),
        ("table_links", "MATCH (a:Table {table_id: r.id}) MATCH (b:Chunk {chunk_id: r.chunk_id}) "
                        "CREATE (a)-[:DISCUSSED_IN]->(b)"),
        ("figure_links", "MATCH (a:Figure {figure_id: r.id}) "
                         "MATCH (b:Chunk {chunk_id: r.chunk_id}) CREATE (a)-[:DISCUSSED_IN]->(b)"),
    ):  # fmt: skip
        tx.run("UNWIND $rows AS r " + query, rows=p[key])
    _recompute_versions(tx, p["source"]["source_key"])
    return {k: len(v) for k, v in p.items() if isinstance(v, list)}


def _recompute_versions(tx: ManagedTransaction, key: str) -> None:
    tx.run("MATCH (:Source {source_key: $key})<-[:VERSION_OF]-(:Document)"
           "-[r:PREVIOUS_VERSION]->() DELETE r", key=key)  # fmt: skip
    tx.run("MATCH (:Source {source_key: $key})<-[:VERSION_OF]-(d:Document) "
           "WITH d ORDER BY d.created_at, d.document_id WITH collect(d) AS docs "
           "UNWIND range(0, size(docs) - 1) AS i WITH docs, i, docs[i] AS d "
           "SET d.version = i + 1, d.is_latest = (i = size(docs) - 1) "
           "WITH docs, i, d WHERE i > 0 WITH docs[i - 1] AS previous, d "
           "MERGE (d)-[:PREVIOUS_VERSION]->(previous)", key=key)  # fmt: skip
    tx.run("MATCH (:Source {source_key: $key})<-[:VERSION_OF]-(d:Document)-[:HAS_CHUNK]->(c) "
           "SET c.is_latest = d.is_latest", key=key)  # fmt: skip


def _delete(tx: ManagedTransaction, document_id: str) -> bool:
    found = tx.run("MATCH (d:Document {document_id: $id})-[:VERSION_OF]->(s:Source) "
                   "RETURN s.source_key AS key", id=document_id).single()  # fmt: skip
    if found is None:
        return False
    for label in OWNED_LABELS:
        tx.run(f"MATCH (n:{label} {{document_id: $id}}) DETACH DELETE n", id=document_id)
    tx.run("MATCH (d:Document {document_id: $id}) DETACH DELETE d", id=document_id)
    _recompute_versions(tx, found["key"])
    tx.run("MATCH (s:Source {source_key: $key}) WHERE NOT (s)<-[:VERSION_OF]-() DELETE s",
           key=found["key"])  # fmt: skip
    return True


class GraphWriter:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def write_document(
        self,
        record: DocumentRecord,
        elements: list[DocumentElement],
        chunk_sets: dict[str, list[Chunk]],
        vectors: dict[str, list[float]],
        embedding_space: str | None = None,
    ) -> dict[str, int]:
        payload = build_payload(record, elements, chunk_sets, vectors, embedding_space)
        return self.store.write(_write, p=payload)

    def delete_document(self, document_id: str) -> bool:
        return self.store.write(_delete, document_id=document_id)


def cached_vectors(
    cache: EmbeddingCache, chunks: Sequence[Chunk]
) -> tuple[str | None, dict[str, list[float]], int]:
    """(space_id, {chunk_id: vector}, missing) for chunks with an ``embedding_ref``.

    Vectors come from the embedding cache by content-addressed ref, so loading the graph never
    calls the model. All refs must share one space: mixed spaces are incomparable vectors.
    """
    refs = {c.chunk_id: c.embedding_ref for c in chunks if c.embedding_ref}
    spaces = {r.split("/", 1)[0] for r in refs.values()}
    if not spaces:
        return None, {}, 0
    if len(spaces) > 1:
        raise ValueError(f"Chunks reference {len(spaces)} embedding spaces: {sorted(spaces)}")
    space_id = spaces.pop()
    keys = {cid: r.split("/", 1)[1] for cid, r in refs.items()}
    found = cache.get_many(space_id, list(set(keys.values())))
    vectors = {cid: found[k].tolist() for cid, k in keys.items() if k in found}
    return space_id, vectors, len(keys) - len(vectors)


class GraphIndexer:
    """Pipeline step: ensure the schema once, then write each document's subgraph."""

    def __init__(
        self,
        store: GraphStore,
        cache: EmbeddingCache,
        config_ids: Sequence[str],
        *,
        vector_dim: int,
        schema_options: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.writer = GraphWriter(store)
        self.cache = cache
        self.config_ids = list(config_ids)
        self.vector_dim = vector_dim
        self.schema_options = schema_options or {}
        self._schema_ready = False
        self._lock = threading.Lock()

    def ensure_schema(self) -> SchemaReport:
        with self._lock:
            report = ensure_schema(self.store, vector_dim=self.vector_dim, **self.schema_options)
            self._schema_ready = True
            return report

    def index(
        self, record: DocumentRecord, elements: list[DocumentElement], sets: dict[str, list[Chunk]]
    ) -> dict[str, Any]:
        if not self._schema_ready:
            self.ensure_schema()
        chosen = {cid: sets[cid] for cid in self.config_ids if cid in sets}
        space_id, vectors, missing = cached_vectors(
            self.cache, [c for cs in chosen.values() for c in cs]
        )
        started = time.perf_counter()
        counts = self.writer.write_document(record, elements, chosen, vectors, space_id)
        return {
            "configs": list(chosen),
            "nodes": {k: v for k, v in counts.items() if not k.endswith("_links")},
            "vectors": len(vectors),
            "vectors_missing": missing,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def delete(self, document_id: str) -> bool:
        return self.writer.delete_document(document_id)

    def close(self) -> None:
        self.store.close()
