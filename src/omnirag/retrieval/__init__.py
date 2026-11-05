"""Retrieval: dense (Neo4j vector index) and lexical (Neo4j full-text BM25), Phase 6.

Graph retrieval and Reciprocal Rank Fusion join in Phase 7, reranking in Phase 8.
"""

from __future__ import annotations

from omnirag.core.config import Settings
from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.retrieval.dense import DenseRetriever
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult, RetrieverName
from omnirag.retrieval.sparse import LexicalRetriever


class RetrievalService:
    """The configured retrievers over one graph connection (created once per process)."""

    def __init__(self, store: GraphStore, dense: DenseRetriever, bm25: LexicalRetriever) -> None:
        self.store = store
        self.retrievers = {"dense": dense, "bm25": bm25}

    def search(
        self,
        query: str,
        mode: RetrieverName = "dense",
        k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        return self.retrievers[mode].retrieve(query, k, filters)

    def close(self) -> None:
        self.store.close()


def build_retrieval(settings: Settings) -> RetrievalService:
    from omnirag.chunking import build_embedder  # local: chunking imports are heavier

    store = GraphStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
        connection_timeout_s=settings.neo4j_connection_timeout_s,
    )
    common = {
        "config_id": settings.retrieval_config_id,
        "overfetch": settings.retrieval_overfetch,
        "max_fetch": settings.retrieval_max_fetch,
    }
    # No token counter for queries: the pre-check is skipped, and an over-long query is still
    # rejected by Ollama (truncate=false), never silently cut.
    dense =DenseRetriever(store, build_embedder(settings), index=settings.graph_vector_index,
                           **common)  # fmt: skip
    bm25 = LexicalRetriever(store, index=settings.graph_fulltext_index,
                            identifier_mode=settings.retrieval_identifier_mode,
                            **common)  # fmt: skip
    return RetrievalService(store, dense, bm25)
