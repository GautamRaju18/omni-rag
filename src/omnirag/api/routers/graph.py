"""Knowledge-graph read endpoints (the dashboard's corpus overview; the explorer comes in Phase 14).

* ``GET /graph/stats``  node counts per label, relationship counts per type, vectors.
* ``GET /graph/entities``  the most-mentioned entities (optionally one type).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from omnirag.api.dependencies import RetrievalDep
from omnirag.core.errors import DependencyUnavailableError
from omnirag.knowledge_base import traversal
from omnirag.knowledge_base.graph_store import GraphUnavailableError

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats", summary="Graph size: nodes, relationships, vectors")
async def graph_stats(retrieval: RetrievalDep) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(traversal.stats, retrieval.store)
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc


@router.get("/entities", summary="Most-mentioned entities")
async def graph_entities(
    retrieval: RetrievalDep,
    entity_type: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(traversal.entities, retrieval.store,
                                       entity_type=entity_type, limit=limit)  # fmt: skip
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc
