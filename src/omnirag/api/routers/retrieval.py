"""Retrieval endpoint (the retrieval laboratory's backend).

* ``POST /retrieval/advanced``  the LangGraph workflow (query transforms, context check,
  bounded corrective retries): hits plus verdict, queries, checks and a per-node trace.
* ``POST /retrieval/search``  one retrieval mode (``dense``, ``bm25``, ``graph``, or the RRF
  fusions ``dense+bm25`` / ``dense+bm25+graph``): top-k chunks with provenance (graph paths,
  per-retriever ranks), latency and what actually ran (Lucene string, linked entities).

Retrieval runs in a worker thread: the Neo4j driver used here is synchronous.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel, Field

from omnirag.api.dependencies import RetrievalDep, SettingsDep, WorkflowDep
from omnirag.core.errors import ConflictError, DependencyUnavailableError, ValidationFailedError
from omnirag.embeddings.base import EmbeddingError, EmbeddingInputError
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.llm.chat import LLMError
from omnirag.retrieval.models import (
    RetrievalError,
    RetrievalFilters,
    RetrievalMode,
    RetrievalResult,
)
from omnirag.workflows.retrieval_graph import Strategy

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: RetrievalMode | None = None  # None = settings.retrieval_default_mode
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


@router.post("/search", response_model=RetrievalResult, summary="Retrieve chunks for a query")
async def search(
    request: SearchRequest, retrieval: RetrievalDep, settings: SettingsDep
) -> RetrievalResult:
    mode = request.mode or settings.retrieval_default_mode
    try:
        return await asyncio.to_thread(
            retrieval.search, request.query, mode, request.k, request.filters
        )
    except EmbeddingInputError as exc:  # e.g. an empty or over-long query
        raise ValidationFailedError(str(exc)) from exc
    except LookupError as exc:  # a mode that this deployment didn't enable
        raise ValidationFailedError(str(exc)) from exc
    except RetrievalError as exc:  # query and index embeddings differ
        raise ConflictError(str(exc)) from exc
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc
    except EmbeddingError as exc:
        raise DependencyUnavailableError(f"Embedding model unavailable: {exc}") from exc


class AdvancedRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    strategy: Strategy | None = None  # None = the settings' workflow defaults


@router.post("/advanced", response_model=RetrievalResult,
             summary="Corrective retrieval workflow (LangGraph)")  # fmt: skip
async def advanced(request: AdvancedRequest, workflow: WorkflowDep) -> RetrievalResult:
    try:
        return await asyncio.to_thread(workflow.retrieve, request.query, request.k,
                                       request.filters, request.strategy)  # fmt: skip
    except EmbeddingInputError as exc:
        raise ValidationFailedError(str(exc)) from exc
    except RetrievalError as exc:
        raise ConflictError(str(exc)) from exc
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc
    except (EmbeddingError, LLMError) as exc:
        raise DependencyUnavailableError(f"Model server unavailable: {exc}") from exc
