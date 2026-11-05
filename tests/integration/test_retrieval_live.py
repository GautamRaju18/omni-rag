"""Dense and lexical retrieval against the throwaway test Neo4j (port 7688, wiped per test).

docker compose --profile test up -d neo4j-test
uv run pytest -m integration tests/integration/test_retrieval_live.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pytest

from omnirag.core.config import Settings
from omnirag.knowledge_base.graph_builder import GraphWriter
from omnirag.knowledge_base.graph_store import GraphStore, GraphUnavailableError
from omnirag.knowledge_base.schema import SchemaMismatchError, ensure_schema
from omnirag.retrieval.dense import DenseRetriever
from omnirag.retrieval.models import RetrievalFilters
from omnirag.retrieval.sparse import LexicalRetriever

pytestmark = pytest.mark.integration

DIM, MM = 1024, "multimodal-256-ctx"
A, A2, B = "doc_00000000000000b1", "doc_00000000000000b2", "doc_00000000000000b3"


@pytest.fixture(scope="module")
def store() -> Iterator[GraphStore]:
    uri = os.environ.get("OMNIRAG_TEST_NEO4J_URI", "bolt://localhost:7688")
    if urlparse(uri).port != 7688:
        pytest.fail(f"Refusing to wipe {uri}: retrieval tests only run against port 7688")
    graph = GraphStore(uri, "neo4j", Settings().neo4j_password.get_secret_value())
    try:
        graph.verify()
    except GraphUnavailableError as exc:
        pytest.fail(f"Test Neo4j not reachable ({exc})")
    for c in graph.schema("SHOW CONSTRAINTS YIELD name"):
        graph.schema(f"DROP CONSTRAINT {c['name']} IF EXISTS")
    for i in graph.schema("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'"):
        graph.schema(f"DROP INDEX {i['name']} IF EXISTS")
    ensure_schema(graph, vector_dim=DIM)
    yield graph
    graph.close()


def _vectors(sets, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    out = {}
    for c in (c for cs in sets.values() for c in cs if c.retrievable):
        v = rng.normal(size=DIM)
        out[c.chunk_id] = (v / np.linalg.norm(v)).tolist()
    return out


@pytest.fixture
def corpus(store: GraphStore, graph_doc):
    """A (old + new version of notes.md), B (other.md, about AUR-C105, a figure chunk)."""
    store.schema("MATCH (n) DETACH DELETE n")
    writer = GraphWriter(store)
    vectors = {}
    docs = [graph_doc(A, "notes.md", datetime(2026, 1, 1, tzinfo=UTC)),
            graph_doc(A2, "notes.md", datetime(2026, 2, 1, tzinfo=UTC)),
            graph_doc(B, "other.md", text="Cell AUR-C105 failed at cycle 600.")]  # fmt: skip
    record, elements, sets = docs[2]
    figure = sets[MM][2].model_copy(update={"element_type": "figure",
                                            "origins": ["generated", "original"]})  # fmt: skip
    sets[MM][2] = figure
    for i, (record, elements, sets) in enumerate(docs):
        v = _vectors(sets, i)
        vectors.update(v)
        writer.write_document(record, elements, sets, v, "space-test")
    store.schema("CALL db.awaitIndexes(60)")
    return vectors


class FixedEmbedder:
    """Returns a stored chunk vector as the "query", so the expected top hit is known."""

    space = SimpleNamespace(space_id="space-test")

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.array(vector, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.vector


def _ids(result) -> list[str]:
    return [h.chunk_id for h in result.hits]


def test_dense_ranks_and_filters_inside_the_index(store: GraphStore, corpus) -> None:
    query = corpus[f"{B}-{MM}-00000"]
    dense = DenseRetriever(store, FixedEmbedder(query), config_id=MM)
    result = dense.retrieve("q", 3)
    assert result.hits[0].chunk_id == f"{B}-{MM}-00000" and result.hits[0].score > 0.999
    assert result.details["post_filters"] == []

    # Latest only: A (superseded by A2) never appears; with all versions it can.
    assert not any(c.startswith(A + "-") for c in _ids(dense.retrieve("q", 20)))
    every = dense.retrieve("q", 20, RetrievalFilters(latest_only=False))
    assert any(c.startswith(A + "-") for c in _ids(every))

    only_a2 = dense.retrieve("q", 3, RetrievalFilters(document_ids=[A2]))
    assert len(only_a2.hits) == 3 and all(h.document_id == A2 for h in only_a2.hits)
    figures = dense.retrieve("q", 5, RetrievalFilters(element_types=["figure"]))
    assert _ids(figures) == [f"{B}-{MM}-00002"]
    assert figures.details["post_filters"] == []  # still inside the index


def test_dense_post_filters_overfetch(store: GraphStore, corpus) -> None:
    dense = DenseRetriever(store, FixedEmbedder(corpus[f"{B}-{MM}-00002"]), config_id=MM)
    result = dense.retrieve("q", 3, RetrievalFilters(exclude_generated=True))
    assert f"{B}-{MM}-00002" not in _ids(result) and len(result.hits) == 3
    assert result.details["fetch"] == 30
    by_file = dense.retrieve("q", 5, RetrievalFilters(source_files=["other.md"]))
    assert {h.source_file for h in by_file.hits} == {"other.md"}


def test_bm25_identifier_modes(store: GraphStore, corpus) -> None:
    bm25 = LexicalRetriever(store, config_id=MM)
    question = "Which cell AUR-C103 failed?"
    boosted = bm25.retrieve(question, 3)
    assert all("AUR-C103" in h.content for h in boosted.hits)
    required = bm25.retrieve(question, 10, identifier_mode="require")
    assert required.hits and all("AUR-C103" in h.content for h in required.hits)
    assert bm25.retrieve("Which cell AUR-C109 failed?", 5, identifier_mode="require").hits == []
    # The classic analyzer keeps "aur-c103" whole: its parts match nothing, and AUR-C105
    # chunks never match AUR-C103 (the standard analyzers split it into "aur" + "c103").
    assert bm25.retrieve("aur", 10, identifier_mode="terms").hits == []
    assert bm25.retrieve("c103", 10, identifier_mode="terms").hits == []
    plain = bm25.retrieve("AUR-C103", 10, identifier_mode="terms")
    assert plain.hits and all("AUR-C103" in h.content for h in plain.hits)


@pytest.mark.parametrize(
    "query", ["What is the N/P ratio?", "AND OR NOT", 'x:y ( "unbalanced', "a/b/c\\", "+-&&||!"]
)
def test_bm25_never_passes_raw_lucene(store: GraphStore, corpus, query: str) -> None:
    LexicalRetriever(store, config_id=MM).retrieve(query, 3)  # must not raise


def test_bm25_filters_are_post_filters(store: GraphStore, corpus) -> None:
    bm25 = LexicalRetriever(store, config_id=MM)
    result = bm25.retrieve("swelling by cell", 5,
                           RetrievalFilters(element_types=["figure"]))  # fmt: skip
    assert _ids(result) == [f"{B}-{MM}-00002"]
    assert "c.element_type IN $element_types" in result.details["post_filters"]


def test_schema_reports_changed_filters_and_analyzer(store: GraphStore) -> None:
    store.schema("DROP INDEX chunk_embedding IF EXISTS")
    store.schema("CYPHER 25 CREATE VECTOR INDEX chunk_embedding FOR (c:Chunk) ON (c.embedding) "
                 "WITH [c.config_id, c.is_latest] OPTIONS {indexConfig: {`vector.dimensions`: "
                 "1024, `vector.similarity_function`: 'cosine', "
                 "`vector.quantization.type`: 'NONE'}}")  # the Phase 5a index  # fmt: skip
    with pytest.raises(SchemaMismatchError, match="filter properties"):
        ensure_schema(store, vector_dim=DIM)
    assert ensure_schema(store, vector_dim=DIM, recreate_vector_index=True).recreated == [
        "chunk_embedding"
    ]
    with pytest.raises(SchemaMismatchError, match="analyzer 'classic'"):
        ensure_schema(store, vector_dim=DIM, analyzer="standard")
    report = ensure_schema(store, vector_dim=DIM, analyzer="standard", recreate_fulltext_index=True)
    assert report.recreated == ["chunk_fulltext"]
    ensure_schema(store, vector_dim=DIM, recreate_fulltext_index=True)  # back to the default
