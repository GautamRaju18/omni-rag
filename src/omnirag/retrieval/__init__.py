"""Retrieval: dense (vector index), lexical (BM25 full-text), graph (entities and relations), and
their Reciprocal Rank Fusion (Phases 6–7). Reranking joins in Phase 8.
"""

from __future__ import annotations

from typing import Any

from omnirag.core.config import Settings
from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.retrieval.dense import DenseRetriever
from omnirag.retrieval.fusion import HybridRetriever
from omnirag.retrieval.graph_search import GraphRetriever
from omnirag.retrieval.models import RetrievalFilters, RetrievalMode, RetrievalResult
from omnirag.retrieval.sparse import LexicalRetriever

FUSED_MODES: dict[str, tuple[str, ...]] = {
    "dense+bm25": ("dense", "bm25"),
    "dense+bm25+graph": ("dense", "bm25", "graph"),
}


class RetrievalService:
    """The configured retrievers over one graph connection (created once per process)."""

    def __init__(
        self,
        store: GraphStore,
        retrievers: dict[str, Any],
        *,
        rrf_k: int = 60,
        depth: int = 20,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.store = store
        self.retrievers = dict(retrievers)
        for mode, names in FUSED_MODES.items():
            if all(n in retrievers for n in names):
                self.retrievers[mode] = HybridRetriever(
                    {n: retrievers[n] for n in names}, mode=mode, rrf_k=rrf_k, depth=depth,
                    weights={n: w for n, w in (weights or {}).items() if n in names},
                )  # fmt: skip

    def search(
        self,
        query: str,
        mode: RetrievalMode = "dense",
        k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        return self.retrievers[mode].retrieve(query, k, filters)

    def close(self) -> None:
        for r in self.retrievers.values():
            if isinstance(r, HybridRetriever):
                r.close()
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
    retrievers = {
        "dense": DenseRetriever(store, build_embedder(settings),
                                index=settings.graph_vector_index, **common),
        "bm25": LexicalRetriever(store, index=settings.graph_fulltext_index,
                                 identifier_mode=settings.retrieval_identifier_mode, **common),
        "graph": GraphRetriever(store, config_id=settings.retrieval_config_id,
                                max_hops=settings.retrieval_graph_max_hops),
    }  # fmt: skip
    return RetrievalService(store, retrievers, rrf_k=settings.retrieval_rrf_k,
                            depth=settings.retrieval_fusion_depth,
                            weights=settings.retrieval_rrf_weights)  # fmt: skip
