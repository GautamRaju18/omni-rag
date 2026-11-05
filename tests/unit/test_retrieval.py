"""Lucene query building, filter placement, retrievers against a fake store, the API route."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from omnirag.api.dependencies import get_retrieval
from omnirag.api.main import create_app
from omnirag.core.config import Settings
from omnirag.embeddings.base import EmbeddingError
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.knowledge_base.schema import VECTOR_FILTER_PROPERTIES
from omnirag.retrieval.dense import DenseRetriever, index_name
from omnirag.retrieval.filters import VECTOR_INDEX_FILTERS, fetch_size, predicates
from omnirag.retrieval.models import RetrievalError, RetrievalFilters, RetrievalResult
from omnirag.retrieval.sparse import LexicalRetriever, identifiers, lucene_query

CFG = "multimodal-256-ctx"


# ---------------------------------------------------------------- Lucene query building
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is the N/P ratio of the anode?", "what is the n\\/p ratio of anode"),
        ("AND OR NOT", "and or not"),  # lowercase: not operators any more
        ("fail? (E-5) [x] ~y ^z", 'fail e\\-5 x y z "e\\-5"^2'),
        ('the "solid-electrolyte interphase" forms', '"solid\\-electrolyte interphase" the forms'),
        ("???", ""),
    ],
)
def test_lucene_query_escapes_everything(text: str, expected: str) -> None:
    assert lucene_query(text) == expected


def test_identifier_modes() -> None:
    q = "Why did cell AUR-C103 fail?"
    assert lucene_query(q, "terms") == "why did cell aur\\-c103 fail"
    assert lucene_query(q, "boost").endswith(' "aur\\-c103"^2')
    assert lucene_query(q, "require").endswith(' +"aur\\-c103"')


def test_identifiers() -> None:
    text = "Did AUR-C103, E-7, NaPF6 or M3 (R2) beat 45 °C on 2026-03-19? e-7 again"
    assert identifiers(text) == ["AUR-C103", "E-7", "NaPF6", "M3", "R2"]  # no dates, numbers


@pytest.mark.parametrize("name", ["", "1x", "a-b", "idx) RETURN 1 //"])
def test_index_name_rejects_non_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid index name"):
        index_name(name)


# ---------------------------------------------------------------- filters
def test_vector_filter_properties_match_the_schema() -> None:
    assert set(VECTOR_FILTER_PROPERTIES) == VECTOR_INDEX_FILTERS


def test_filters_split_between_index_and_post() -> None:
    f = RetrievalFilters(document_ids=["d1"], element_types=["table"], source_files=["a.pdf"],
                         exclude_generated=True, min_confidence=0.8)  # fmt: skip
    inside, after, params = predicates(f, CFG, VECTOR_INDEX_FILTERS)
    assert inside == [
        "c.config_id = $config_id",
        "c.is_latest = true",
        "c.document_id IN $document_ids",
        "c.element_type IN $element_types",
    ]
    assert after == ["c.source_file IN $source_files", "NOT 'generated' IN c.origins",
                     "c.min_confidence >= $min_confidence"]  # fmt: skip
    assert params == {"config_id": CFG, "document_ids": ["d1"], "element_types": ["table"],
                      "source_files": ["a.pdf"], "min_confidence": 0.8}  # fmt: skip
    # Full-text search has no in-index filters: everything is a post-filter.
    inside, after, _ = predicates(RetrievalFilters(latest_only=False), CFG)
    assert inside == [] and after == ["c.config_id = $config_id"]
    assert fetch_size(5, [], 10, 200) == 5
    assert fetch_size(5, ["x"], 10, 200) == 50 and fetch_size(50, ["x"], 10, 200) == 200


# ---------------------------------------------------------------- retrievers (fake store)
ROW = {"chunk_id": "c1", "document_id": "d1", "source_file": "a.md", "content": "AUR-C103",
       "context": None, "section_path": ["S"], "page_numbers": [1], "element_type": "paragraph",
       "element_ids": ["d1-e00000"], "origins": ["original"], "min_confidence": 1.0,
       "config_id": CFG, "score": 0.9}  # fmt: skip


class FakeStore:
    def __init__(self, spaces=("space-a",), rows=(ROW,)) -> None:
        self.spaces, self.rows, self.calls = spaces, list(rows), []

    def query(self, cypher: str, **params):
        self.calls.append((cypher, params))
        if "embedding_space" in cypher:
            return [{"space": s, "n": 1} for s in self.spaces]
        return self.rows


class FakeEmbedder:
    space = SimpleNamespace(space_id="space-a")

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([0.6, 0.8], dtype=np.float32)


def test_dense_retriever_filters_in_index_and_after() -> None:
    store = FakeStore()
    retriever = DenseRetriever(store, FakeEmbedder(), config_id=CFG)
    result = retriever.retrieve("q", 3, RetrievalFilters(document_ids=["d1"],
                                                         exclude_generated=True))  # fmt: skip
    cypher, params = store.calls[-1]
    assert (
        "VECTOR INDEX chunk_embedding FOR $vector WHERE c.config_id = $config_id AND "
        "c.is_latest = true AND c.document_id IN $document_ids LIMIT $fetch" in cypher
    )
    assert "WITH c, score WHERE NOT 'generated' IN c.origins" in cypher
    assert params["fetch"] == 30 and params["k"] == 3
    assert params["vector"] == pytest.approx([0.6, 0.8])
    assert result.hits[0].rank == 1 and result.hits[0].retriever == "dense"
    assert result.details["cosine"] == pytest.approx(0.8)
    assert result.details["underfilled"] is True  # 1 hit < k after a post-filter


def test_dense_retriever_refuses_a_different_embedding_space() -> None:
    retriever = DenseRetriever(FakeStore(spaces=("space-old",)), FakeEmbedder(), config_id=CFG)
    with pytest.raises(RetrievalError, match="space-old"):
        retriever.retrieve("q")


def test_lexical_retriever_passes_lucene_as_a_parameter() -> None:
    store = FakeStore()
    result = LexicalRetriever(store, config_id=CFG).retrieve("AUR-C103 OR 1=1 //", 2)
    cypher, params = store.calls[-1]
    assert "$lucene" in cypher and "AUR" not in cypher
    assert params["lucene"] == 'aur\\-c103 or 1 +"aur\\-c103"'  # default mode: require
    assert params["fetch"] == 20 and result.hits[0].retriever == "bm25"


def test_lexical_retriever_skips_empty_queries() -> None:
    store = FakeStore()
    result = LexicalRetriever(store, config_id=CFG).retrieve("?!", 2)
    assert result.hits == [] and store.calls == []


# ---------------------------------------------------------------- API
class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def search(self, query, mode, k, filters) -> RetrievalResult:
        if self.error:
            raise self.error
        return RetrievalResult(query=query, retriever=mode, k=k, hits=[], latency_ms=1.0,
                               details={"filters": filters.model_dump()})  # fmt: skip


async def _client(service: FakeService) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(Settings(_env_file=None, env="test", log_json=False, neo4j_password="x"))
    app.dependency_overrides[get_retrieval] = lambda: service
    transport = httpx.ASGITransport(app=app)  # no lifespan: no real clients
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_search_endpoint() -> None:
    async for client in _client(FakeService()):
        r = await client.post("/api/v1/retrieval/search",
                              json={"query": "AUR-C103", "mode": "bm25", "k": 3,
                                    "filters": {"element_types": ["table"]}})  # fmt: skip
        assert r.status_code == 200
        body = r.json()
        assert body["retriever"] == "bm25" and body["k"] == 3
        assert body["details"]["filters"]["element_types"] == ["table"]
        assert (
            await client.post("/api/v1/retrieval/search", json={"query": ""})
        ).status_code == 422
        bad_mode = await client.post("/api/v1/retrieval/search", json={"query": "x", "mode": "x"})
        assert bad_mode.status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [(GraphUnavailableError("down"), 503), (EmbeddingError("ollama down"), 503),
     (RetrievalError("spaces differ"), 409)],
)  # fmt: skip
async def test_search_endpoint_errors(error: Exception, status: int) -> None:
    async for client in _client(FakeService(error)):
        r = await client.post("/api/v1/retrieval/search", json={"query": "x"})
        assert r.status_code == status
