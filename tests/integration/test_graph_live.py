"""Graph writer, versioning, deletion, indexes and traversal against a real Neo4j.

Runs against the throwaway test server only, because each test wipes the database:

    docker compose --profile test up -d neo4j-test     # bolt://localhost:7688, data in RAM
    uv run pytest -m integration tests/integration/test_graph_live.py

Override the target with OMNIRAG_TEST_NEO4J_URI. The URI must use port 7688, so a
misconfiguration can't wipe the real graph on 7687.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlparse

import numpy as np
import pytest

from omnirag.core.config import Settings
from omnirag.knowledge_base import traversal
from omnirag.knowledge_base.graph_builder import GraphWriter
from omnirag.knowledge_base.graph_store import GraphStore, GraphUnavailableError
from omnirag.knowledge_base.schema import SchemaMismatchError, ensure_schema

pytestmark = pytest.mark.integration

DIM = 1024
MM = "multimodal-256-ctx"
V1, V2 = "doc_00000000000000a1", "doc_00000000000000a2"


@pytest.fixture(scope="module")
def store() -> Iterator[GraphStore]:
    uri = os.environ.get("OMNIRAG_TEST_NEO4J_URI", "bolt://localhost:7688")
    if urlparse(uri).port != 7688:
        pytest.fail(f"Refusing to wipe {uri}: graph tests only run against port 7688")
    password = Settings().neo4j_password.get_secret_value()  # reads .env
    graph = GraphStore(uri, "neo4j", password)
    try:
        graph.verify()
    except GraphUnavailableError as exc:
        pytest.fail(f"Test Neo4j not reachable ({exc}); run: docker compose --profile test "
                    "up -d neo4j-test")  # fmt: skip
    # Start from an empty schema. Constraints first: they own their backing indexes.
    for constraint in graph.schema("SHOW CONSTRAINTS YIELD name"):
        graph.schema(f"DROP CONSTRAINT {constraint['name']} IF EXISTS")  # names from server
    for index in graph.schema("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'"):
        graph.schema(f"DROP INDEX {index['name']} IF EXISTS")
    ensure_schema(graph, vector_dim=DIM)
    yield graph
    graph.close()


@pytest.fixture(autouse=True)
def empty(store: GraphStore) -> None:
    store.schema("MATCH (n) DETACH DELETE n")


def _vectors(sets, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    out = {}
    for c in (c for cs in sets.values() for c in cs if c.retrievable):
        v = rng.normal(size=DIM)
        out[c.chunk_id] = (v / np.linalg.norm(v)).tolist()
    return out


def _counts(store: GraphStore) -> dict[str, int]:
    return {**traversal.stats(store)["nodes"], **traversal.stats(store)["relationships"]}


def test_schema_is_idempotent_and_detects_dimension_changes(store: GraphStore) -> None:
    report = ensure_schema(store, vector_dim=DIM)
    assert report.created == [] and "chunk_embedding" in report.existing
    with pytest.raises(SchemaMismatchError, match=r"\(1024, 'NONE'.*requested \(8, "):
        ensure_schema(store, vector_dim=8)
    assert ensure_schema(store, vector_dim=8, recreate_vector_index=True).recreated == [
        "chunk_embedding"
    ]
    ensure_schema(store, vector_dim=DIM, recreate_vector_index=True)
    options = store.schema("SHOW INDEXES YIELD name, options WHERE name = 'chunk_embedding'")
    config = options[0]["options"]["indexConfig"]
    assert config["vector.dimensions"] == DIM
    assert config["vector.quantization.type"] == "NONE"  # not the BINARY default


def test_write_is_idempotent(store: GraphStore, graph_doc) -> None:
    record, elements, sets = graph_doc(V1)
    writer = GraphWriter(store)
    writer.write_document(record, elements, sets, _vectors(sets), "space-test")
    first = _counts(store)
    writer.write_document(record, elements, sets, _vectors(sets), "space-test")
    assert _counts(store) == first
    assert first["Chunk"] == 5 and first["Section"] == 2 and first["Page"] == 2
    assert first["Table"] == 1 and first["Figure"] == 1 and first["Source"] == 1
    assert first["NEXT_CHUNK"] == 2 and first["CHILD_OF"] == 1
    assert first["DISCUSSED_IN"] == 3


def test_chunk_context_and_outline(store: GraphStore, graph_doc) -> None:
    record, elements, sets = graph_doc(V1)
    GraphWriter(store).write_document(record, elements, sets, _vectors(sets), "space-test")

    ctx = traversal.chunk_context(store, f"{V1}-{MM}-00001")
    assert ctx["previous"] == f"{V1}-{MM}-00000" and ctx["next"] == f"{V1}-{MM}-00002"
    assert ctx["tables"] == [f"{V1}-e00003"] and ctx["figures"] == []
    assert ctx["section"] == ["Aurora Notes", "Results"] and ctx["pages"] == [1]
    assert "embedding" not in ctx["chunk"]  # 1,024 floats are never returned for display
    assert ctx["document"]["version"] == 1 and ctx["document"]["is_latest"] is True

    child = traversal.chunk_context(store, f"{V1}-parentchild-256-128-32-00001")
    assert child["parent"] == f"{V1}-parentchild-256-128-32-00000"

    outline = traversal.document_outline(store, V1)
    assert outline["counts"] == {"pages": 2, "chunks": 5, "tables": 1, "figures": 1}
    assert [s["path"] for s in outline["sections"]] == [["Aurora Notes"],
                                                       ["Aurora Notes", "Results"]]  # fmt: skip
    assert traversal.chunk_context(store, "nope") is None
    assert traversal.document_outline(store, "nope") is None


def test_versioning_and_deletion(store: GraphStore, graph_doc) -> None:
    writer = GraphWriter(store)
    old = graph_doc(V1, "Notes.md", datetime(2026, 1, 1, tzinfo=UTC))
    new = graph_doc(V2, "notes.md", datetime(2026, 2, 1, tzinfo=UTC), text="Cell AUR-C103 "
                    "failed at cycle 415.")  # fmt: skip
    writer.write_document(*new, _vectors(new[2], 2), "space-test")
    writer.write_document(*old, _vectors(old[2], 1), "space-test")  # loaded out of order

    versions = traversal.versions(store, "NOTES.MD")  # source key is case-insensitive
    assert [(v["document_id"], v["version"], v["is_latest"], v["previous"]) for v in versions] == [
        (V1, 1, False, None),
        (V2, 2, True, V1),
    ]
    latest = store.query("MATCH (c:Chunk) RETURN c.document_id AS d, "
                         "collect(DISTINCT c.is_latest) AS flags ORDER BY d")  # fmt: skip
    assert latest == [{"d": V1, "flags": [False]}, {"d": V2, "flags": [True]}]

    writer.write_document(*old, _vectors(old[2], 1), "space-test")  # reload: still one chain
    assert len(store.query("MATCH ()-[r:PREVIOUS_VERSION]->() RETURN r")) == 1

    assert writer.delete_document(V2) is True
    versions = traversal.versions(store, "notes.md")
    assert [(v["document_id"], v["version"], v["is_latest"]) for v in versions] == [(V1, 1, True)]
    assert store.query("MATCH (n {document_id: $id}) RETURN n", id=V2) == []
    assert writer.delete_document(V2) is False
    assert writer.delete_document(V1) is True
    assert _counts(store) == {}  # Source removed with its last version


def test_vector_search_filters_inside_the_index(store: GraphStore, graph_doc) -> None:
    writer = GraphWriter(store)
    old = graph_doc(V1, "notes.md", datetime(2026, 1, 1, tzinfo=UTC))
    new = graph_doc(V2, "notes.md", datetime(2026, 2, 1, tzinfo=UTC))
    old_vectors = _vectors(old[2], 1)
    writer.write_document(*old, old_vectors, "space-test")
    writer.write_document(*new, _vectors(new[2], 2), "space-test")
    store.schema("CALL db.awaitIndexes(60)")
    query = old_vectors[f"{V1}-{MM}-00000"]

    hits = traversal.vector_search(store, query, k=3, latest_only=False)
    assert hits[0]["chunk_id"] == f"{V1}-{MM}-00000" and hits[0]["score"] > 0.999

    hits = traversal.vector_search(store, query, k=3)  # latest only: V1 is superseded
    assert len(hits) == 3 and all(h["chunk_id"].startswith(V2) for h in hits)
    hits = traversal.vector_search(store, query, k=10, config_id="parentchild-256-128-32")
    assert [h["chunk_id"] for h in hits] == [f"{V2}-parentchild-256-128-32-00001"]


def test_fulltext_search(store: GraphStore, graph_doc) -> None:
    record, elements, sets = graph_doc(V1)
    GraphWriter(store).write_document(record, elements, sets, {}, None)
    store.schema("CALL db.awaitIndexes(60)")
    hits = traversal.fulltext_search(store, '"cycle 412"', k=5, config_id=MM)
    assert [h["chunk_id"] for h in hits] == [f"{V1}-{MM}-00000"]
    assert traversal.fulltext_search(store, "nonexistentword") == []


# ---------------------------------------------------------------- Phase 5b: entities
def _extraction(doc_id: str, person: str, *, support: str = "stated"):
    from omnirag.ingestion.models import ContentOrigin
    from omnirag.knowledge_base.entities import (
        ChunkExtraction,
        DocumentExtraction,
        EntityType,
        Mention,
        Relation,
        RelationType,
    )

    def mention(key, surface, etype):
        return Mention(key=key, surface=surface, type=etype, origin=ContentOrigin.ORIGINAL,
                       confidence=1.0)  # fmt: skip

    chunk = ChunkExtraction(
        chunk_id=f"{doc_id}-{MM}-00000", input_sha256="x",
        mentions=[mention("aur-c103", "AUR-C103", EntityType.SAMPLE),
                  mention(person, person.title(), EntityType.PERSON)],
        relations=[Relation(source="aur-c103", target=person, type=RelationType.TESTED_BY,
                            support=support, evidence="Cell AUR-C103 failed at cycle 412.",
                            evidence_verified=True, evidence_origin=ContentOrigin.ORIGINAL)],
    )  # fmt: skip
    return DocumentExtraction(document_id=doc_id, config_id=MM, model="m", model_digest="d",
                              prompt_version="kg-test", extracted_at="2026-11-05",
                              chunks=[chunk])  # fmt: skip


V3 = "doc_00000000000000a3"


def test_entities_are_shared_refreshed_and_cleaned_up(store: GraphStore, graph_doc) -> None:
    writer = GraphWriter(store)
    a = graph_doc(V1, "notes.md")
    b = graph_doc(V3, "other.md")
    writer.write_document(*a, {}, None, _extraction(V1, "aiko tanaka"))
    writer.write_document(*a, {}, None, _extraction(V1, "aiko tanaka"))  # idempotent
    writer.write_document(*b, {}, None, _extraction(V3, "aiko"))

    rows = store.query("MATCH (e:Entity) RETURN e.key AS key, e.type AS type, "
                       "e.mention_count AS m, e.document_count AS d ORDER BY key")  # fmt: skip
    assert rows == [{"key": "aiko", "type": "PERSON", "m": 1, "d": 1},
                    {"key": "aiko tanaka", "type": "PERSON", "m": 1, "d": 1},
                    {"key": "aur-c103", "type": "SAMPLE", "m": 2, "d": 2}]  # fmt: skip
    assert _counts(store)["RELATED_TO"] == 2 and _counts(store)["MENTIONS"] == 4
    # "Aiko" in one document, "Aiko Tanaka" in another: linked, not merged.
    same = store.query("MATCH (a:Entity)-[s:SAME_AS]->(b:Entity) "
                       "RETURN a.key AS a, b.key AS b, s.rule AS rule")  # fmt: skip
    assert same == [{"a": "aiko", "b": "aiko tanaka", "rule": "alias:first_name"}]

    detail = traversal.entity_detail(store, "AUR-C103")
    assert {(r["direction"], r["type"], r["other"], r["support"]) for r in detail["relations"]} == {
        ("out", "TESTED_BY", "Aiko Tanaka", "stated"),
        ("out", "TESTED_BY", "Aiko", "stated"),
    }

    assert writer.delete_document(V3) is True
    keys = [r["key"] for r in store.query("MATCH (e:Entity) RETURN e.key AS key ORDER BY key")]
    assert keys == ["aiko tanaka", "aur-c103"]  # orphan "aiko" removed, shared entity kept
    assert store.query("MATCH (e:Entity {key: 'aur-c103'}) RETURN e.mention_count AS m") == [
        {"m": 1}
    ]
    assert _counts(store).get("SAME_AS", 0) == 0 and _counts(store)["RELATED_TO"] == 1


def test_old_version_relations_are_not_latest(store: GraphStore, graph_doc) -> None:
    writer = GraphWriter(store)
    old = graph_doc(V1, "notes.md", datetime(2026, 1, 1, tzinfo=UTC))
    new = graph_doc(V2, "notes.md", datetime(2026, 2, 1, tzinfo=UTC))
    writer.write_document(*old, {}, None, _extraction(V1, "aiko tanaka"))
    writer.write_document(*new, {}, None, _extraction(V2, "aiko tanaka", support="inferred"))
    rows = store.query("MATCH ()-[r:RELATED_TO]->() RETURN r.document_id AS d, "
                       "r.is_latest AS latest, r.support AS support ORDER BY d")  # fmt: skip
    assert rows == [{"d": V1, "latest": False, "support": "stated"},
                    {"d": V2, "latest": True, "support": "inferred"}]  # fmt: skip
    latest = traversal.entity_detail(store, "aur-c103")["relations"]
    assert [r["support"] for r in latest] == ["inferred"]
    everything = traversal.entity_detail(store, "aur-c103", latest_only=False)["relations"]
    assert sorted(r["support"] for r in everything) == ["inferred", "stated"]
    assert len(store.query("MATCH ()-[r:RELATED_TO]->() RETURN r")) == 2


def test_cites_links_a_chunk_to_a_titled_document(store: GraphStore, graph_doc) -> None:
    writer = GraphWriter(store)
    cited = graph_doc(V1, "notes.md")  # title "Aurora Notes"
    citing_record, elements, sets = graph_doc(V3, "summary.md")
    citing_record.metadata["title"] = "Quarterly Summary 2026"  # its chunks say "Aurora Notes"
    writer.write_document(*cited, {}, None)
    counts = writer.write_document(citing_record, elements, sets, {}, None)
    assert counts["cites"] == 1
    rows = store.query("MATCH (c:Chunk)-[:CITES]->(d:Document) "
                       "RETURN c.chunk_id AS chunk, d.document_id AS doc")  # fmt: skip
    assert rows == [{"chunk": f"{V3}-{MM}-00000", "doc": V1}]
