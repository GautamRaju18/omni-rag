"""Health endpoints.

* ``GET /health``         – aggregate; HTTP 200 unless a component is ``down`` (then 503)
* ``GET /health/ollama``  – Ollama reachability, version, required/loaded models, VRAM
* ``GET /health/neo4j``   – Neo4j connectivity, version, edition

Probes run concurrently, so ``/health`` costs max(latencies), not their sum.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status

from omnirag import __version__
from omnirag.api.dependencies import Neo4jDep, OllamaDep, SettingsDep
from omnirag.schemas.health import ComponentHealth, HealthReport, worst_status

router = APIRouter(prefix="/health", tags=["health"])


def _set_status_code(response: Response, health_status: str) -> None:
    if health_status == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


@router.get("", response_model=HealthReport, summary="Aggregate health of API and dependencies")
async def health(
    response: Response, settings: SettingsDep, neo4j: Neo4jDep, ollama: OllamaDep
) -> HealthReport:
    components = await asyncio.gather(
        neo4j.health(timeout_s=settings.neo4j_health_timeout_s),
        ollama.health(settings.required_ollama_models, timeout_s=settings.ollama_health_timeout_s),
    )
    overall = worst_status([c.status for c in components])
    _set_status_code(response, overall)
    return HealthReport(
        status=overall,
        app=settings.app_name,
        version=__version__,
        environment=settings.env,
        components=list(components),
    )


@router.get("/ollama", response_model=ComponentHealth, summary="Ollama health and loaded models")
async def health_ollama(
    response: Response, settings: SettingsDep, ollama: OllamaDep
) -> ComponentHealth:
    result = await ollama.health(
        settings.required_ollama_models, timeout_s=settings.ollama_health_timeout_s
    )
    _set_status_code(response, result.status)
    return result


@router.get("/neo4j", response_model=ComponentHealth, summary="Neo4j connectivity and version")
async def health_neo4j(
    response: Response, settings: SettingsDep, neo4j: Neo4jDep
) -> ComponentHealth:
    result = await neo4j.health(timeout_s=settings.neo4j_health_timeout_s)
    _set_status_code(response, result.status)
    return result
