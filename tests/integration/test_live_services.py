"""Integration tests against the real services.

Prerequisites: ``docker compose up -d`` (Neo4j) and Ollama running on Windows with the
required models pulled. Run with:  ``uv run pytest -m integration``
"""

from __future__ import annotations

import httpx
import pytest

from omnirag.api.main import create_app
from omnirag.core.config import Settings
from omnirag.knowledge_base.neo4j_client import Neo4jClient
from omnirag.llm.ollama_client import OllamaClient

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings() -> Settings:
    return Settings(env="test", log_json=False)  # reads the real .env


async def test_neo4j_is_reachable(live_settings: Settings) -> None:
    client = Neo4jClient(
        live_settings.neo4j_uri,
        live_settings.neo4j_user,
        live_settings.neo4j_password.get_secret_value(),
        database=live_settings.neo4j_database,
    )
    try:
        result = await client.health()
    finally:
        await client.close()
    assert result.status == "ok", result.error
    assert result.details["edition"] == "community"


async def test_neo4j_wrong_password_reports_down(live_settings: Settings) -> None:
    client = Neo4jClient(live_settings.neo4j_uri, "neo4j", "definitely-wrong")
    try:
        result = await client.health()
    finally:
        await client.close()
    assert result.status == "down"
    assert "Auth" in (result.error or "") or "Unauthorized" in (result.error or "")


async def test_ollama_has_required_models(live_settings: Settings) -> None:
    client = OllamaClient(live_settings.ollama_base_url)
    try:
        result = await client.health(live_settings.required_ollama_models)
    finally:
        await client.aclose()
    assert result.status == "ok", result.error


async def test_full_app_lifespan_and_health(live_settings: Settings) -> None:
    app = create_app(live_settings)
    # Run the real lifespan (creates real clients), then call the app in-process.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ok"
