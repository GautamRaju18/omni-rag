"""Dense retrieval: embed the query, then Cypher 25 vector ``SEARCH`` on the chunk index.

* **Same space or nothing.** A query vector is only comparable to chunk vectors from the same
  embedding space (model + exact weights, ADR-0010). Before the first search the retriever
  checks that every Document in the graph was loaded with the query embedder's ``space_id``,
  and refuses to rank otherwise.
* **Scores.** Neo4j reports cosine similarity as ``(1 + cos) / 2`` in [0, 1]. The raw score is
  kept, and ``details.cosine`` gives the familiar [-1, 1] value of the top hit.
* **Filters** on declared properties (chunk set, latest version, document, element type) run
  inside the index; the rest over-fetch and post-filter (``filters.py``).
"""

from __future__ import annotations

import re
import time

from omnirag.embeddings.base import DenseEmbedder
from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.retrieval.filters import VECTOR_INDEX_FILTERS, fetch_size, predicates
from omnirag.retrieval.models import (
    CHUNK_FIELDS,
    RetrievalError,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)

_INDEX_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def index_name(name: str) -> str:
    """Index names can't be parameters in a SEARCH clause: plain identifiers only."""
    if not _INDEX_NAME.fullmatch(name):
        raise ValueError(f"Invalid index name {name!r}")
    return name


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        store: GraphStore,
        embedder: DenseEmbedder,
        *,
        config_id: str,
        index: str = "chunk_embedding",
        overfetch: int = 10,
        max_fetch: int = 200,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config_id = config_id
        self.index = index_name(index)
        self.overfetch = overfetch
        self.max_fetch = max_fetch
        self._space_checked = False

    def check_space(self) -> str:
        """The graph's vectors and the query embedder must share one embedding space."""
        space_id = self.embedder.space.space_id
        rows = self.store.query(
            "MATCH (d:Document) WHERE d.embedding_space IS NOT NULL "
            "RETURN d.embedding_space AS space, count(*) AS n"
        )
        spaces = {r["space"] for r in rows}
        if spaces and spaces != {space_id}:
            raise RetrievalError(
                f"Graph vectors are in {sorted(spaces)}, queries in {space_id}: re-embed "
                "(python -m omnirag.embeddings embed --all) and reload the graph"
            )
        self._space_checked = True
        return space_id

    def retrieve(
        self, query: str, k: int = 5, filters: RetrievalFilters | None = None
    ) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        if not self._space_checked:
            self.check_space()
        started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - started) * 1000
        inside, after, params = predicates(filters, self.config_id, VECTOR_INDEX_FILTERS)
        fetch = fetch_size(k, after, self.overfetch, self.max_fetch)
        where = f" WHERE {' AND '.join(inside)}" if inside else ""
        post = f"WITH c, score WHERE {' AND '.join(after)} " if after else ""
        cypher = (
            f"CYPHER 25 MATCH (c:Chunk) SEARCH c IN (VECTOR INDEX {self.index} FOR $vector"
            f"{where} LIMIT $fetch) SCORE AS score {post}"
            f"RETURN {CHUNK_FIELDS}, score ORDER BY score DESC, chunk_id LIMIT $k"
        )
        search_started = time.perf_counter()
        rows = self.store.query(cypher, vector=vector.tolist(), fetch=fetch, k=k, **params)
        search_ms = (time.perf_counter() - search_started) * 1000
        hits = [RetrievedChunk(rank=i + 1, retriever="dense", **r) for i, r in enumerate(rows)]
        return RetrievalResult(
            query=query, retriever="dense", k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={
                "embed_ms": round(embed_ms, 2), "search_ms": round(search_ms, 2),
                "space_id": self.embedder.space.space_id, "fetch": fetch,
                "in_index_filters": inside, "post_filters": after,
                "underfilled": len(hits) < k and bool(after),
                "cosine": round(2 * hits[0].score - 1, 4) if hits else None,
            },
        )  # fmt: skip
