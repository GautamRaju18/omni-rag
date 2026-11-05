"""Retrieval endpoint (the retrieval laboratory's backend).

* ``POST /retrieval/search``  one retriever (``dense`` or ``bm25``), top-k chunks with
  provenance, latency and the exact query that ran (e.g. the Lucene string).

Retrieval runs in a worker thread: the Neo4j driver used here is synchronous.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel, Field

from omnirag.api.dependencies import RetrievalDep
from omnirag.core.errors import ConflictError, DependencyUnavailableError, ValidationFailedError
from omnirag.embeddings.base import EmbeddingError, EmbeddingInputError
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.retrieval.models import (
    RetrievalError,
    RetrievalFilters,
    RetrievalResult,
    RetrieverName,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: RetrieverName = "dense"
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


@router.post("/search", response_model=RetrievalResult, summary="Retrieve chunks for a query")
async def search(request: SearchRequest, retrieval: RetrievalDep) -> RetrievalResult:
    try:
        return await asyncio.to_thread(
            retrieval.search, request.query, request.mode, request.k, request.filters
        )
    except EmbeddingInputError as exc:  # e.g. an empty or over-long query
        raise ValidationFailedError(str(exc)) from exc
    except RetrievalError as exc:  # query and index embeddings differ
        raise ConflictError(str(exc)) from exc
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc
    except EmbeddingError as exc:
        raise DependencyUnavailableError(f"Embedding model unavailable: {exc}") from exc
