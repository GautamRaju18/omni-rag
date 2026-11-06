"""Knowledge-graph read endpoints (corpus overview and the Phase 14 graph explorer).

* ``GET /graph/stats``  node counts per label, relationship counts per type, vectors.
* ``GET /graph/entities``  the most-mentioned entities (optionally one type).
* ``GET /graph/neighborhood``  nodes + edges around an entity, a document, or the overview.
* ``GET /graph/entity?name=``  one entity with relations, evidence and mentions.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from omnirag.api.dependencies import RetrievalDep
from omnirag.core.errors import DependencyUnavailableError, NotFoundError
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


@router.get("/neighborhood", summary="Nodes and edges for the graph explorer")
async def graph_neighborhood(
    retrieval: RetrievalDep,
    entity: str | None = Query(default=None, max_length=200),
    document_id: str | None = Query(default=None, pattern=r"^doc_[0-9a-f]{16}$"),
    limit: int = Query(default=60, ge=1, le=300),
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(traversal.neighborhood, retrieval.store, entity=entity,
                                       document_id=document_id, limit=limit)  # fmt: skip
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc


@router.get("/entity", summary="One entity: relations (with evidence), mentions, SAME_AS")
async def graph_entity(retrieval: RetrievalDep, name: str = Query(min_length=1, max_length=200)
                       ) -> dict[str, Any]:  # fmt: skip
    try:
        detail = await asyncio.to_thread(traversal.entity_detail, retrieval.store, name)
    except GraphUnavailableError as exc:
        raise DependencyUnavailableError(f"Neo4j unavailable: {exc}") from exc
    if detail is None:
        raise NotFoundError(f"entity {name!r} not found")
    return detail
