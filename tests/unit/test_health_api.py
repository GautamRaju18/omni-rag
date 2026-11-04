"""API-level tests with fake dependencies injected via ``dependency_overrides``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from omnirag.api.dependencies import get_neo4j, get_ollama
from omnirag.api.main import create_app
from omnirag.core.config import Settings
from omnirag.core.errors import DependencyUnavailableError
from omnirag.schemas.health import ComponentHealth, HealthStatus, worst_status


class FakeComponent:
    """Stands in for Neo4jClient / OllamaClient: only ``health`` is needed."""

    def __init__(self, name: str, status: HealthStatus) -> None:
        self.name, self.status = name, status

    async def health(self, *args: object, **kwargs: object) -> ComponentHealth:
        return ComponentHealth(name=self.name, status=self.status, latency_ms=1.0)


def _app(settings: Settings, neo4j: HealthStatus, ollama: HealthStatus) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_neo4j] = lambda: FakeComponent("neo4j", neo4j)
    app.dependency_overrides[get_ollama] = lambda: FakeComponent("ollama", ollama)
    return app


async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # ASGITransport does not run the lifespan, so no real clients are created.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def healthy_client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client(_app(settings, "ok", "ok")):
        yield c


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], "ok"),
        (["ok", "ok"], "ok"),
        (["ok", "degraded"], "degraded"),
        (["degraded", "down", "ok"], "down"),
    ],
)
def test_worst_status(statuses: list[HealthStatus], expected: HealthStatus) -> None:
    assert worst_status(statuses) == expected


async def test_health_all_ok(healthy_client: httpx.AsyncClient) -> None:
    response = await healthy_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert {c["name"] for c in body["components"]} == {"neo4j", "ollama"}


@pytest.mark.parametrize(
    ("neo4j", "ollama", "http_status", "overall"),
    [
        ("down", "ok", 503, "down"),
        ("ok", "down", 503, "down"),
        ("ok", "degraded", 200, "degraded"),
    ],
)
async def test_health_aggregation(
    settings: Settings,
    neo4j: HealthStatus,
    ollama: HealthStatus,
    http_status: int,
    overall: str,
) -> None:
    async for client in _client(_app(settings, neo4j, ollama)):
        response = await client.get("/api/v1/health")
        assert response.status_code == http_status
        assert response.json()["status"] == overall


async def test_component_endpoints(settings: Settings) -> None:
    async for client in _client(_app(settings, "down", "ok")):
        neo = await client.get("/api/v1/health/neo4j")
        oll = await client.get("/api/v1/health/ollama")
        assert (neo.status_code, neo.json()["status"]) == (503, "down")
        assert (oll.status_code, oll.json()["status"]) == (200, "ok")


async def test_unversioned_alias_and_request_id(healthy_client: httpx.AsyncClient) -> None:
    response = await healthy_client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc123"

    generated = await healthy_client.get("/health")
    assert len(generated.headers["X-Request-ID"]) == 32  # uuid4 hex


async def test_domain_errors_use_json_envelope(settings: Settings) -> None:
    app = _app(settings, "ok", "ok")

    @app.get("/boom")
    async def boom() -> None:
        raise DependencyUnavailableError("Neo4j is unreachable", details={"uri": "bolt://x"})

    async for client in _client(app):
        response = await client.get("/boom", headers={"X-Request-ID": "req-1"})
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "dependency_unavailable",
            "message": "Neo4j is unreachable",
            "details": {"uri": "bolt://x"},
            "request_id": "req-1",
        }
    }


async def test_openapi_lists_versioned_routes_only(healthy_client: httpx.AsyncClient) -> None:
    paths = (await healthy_client.get("/openapi.json")).json()["paths"]
    assert set(paths) == {"/api/v1/health", "/api/v1/health/ollama", "/api/v1/health/neo4j"}
