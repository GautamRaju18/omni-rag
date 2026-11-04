"""FastAPI application factory.

Run (from the repo root, inside WSL):

    uv run uvicorn omnirag.api.main:app --reload --no-access-log
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from omnirag import __version__
from omnirag.api.middleware import request_context_middleware
from omnirag.api.routers import health
from omnirag.core.config import Settings, get_settings
from omnirag.core.errors import register_exception_handlers
from omnirag.core.logging import configure_logging
from omnirag.knowledge_base.neo4j_client import Neo4jClient
from omnirag.llm.ollama_client import OllamaClient

log = structlog.get_logger(module=__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients at startup and close them at shutdown.

    Startup never blocks on dependencies: if Neo4j or Ollama is down, the API still starts
    and reports it via ``/health`` instead of crash-looping.
    """
    settings: Settings = app.state.settings
    app.state.neo4j = Neo4jClient(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
        connection_timeout_s=settings.neo4j_connection_timeout_s,
    )
    app.state.ollama = OllamaClient(
        settings.ollama_base_url,
        timeout_s=settings.ollama_timeout_s,
        connect_retries=settings.ollama_connect_retries,
    )
    log.info("startup", env=settings.env, neo4j=settings.neo4j_uri, ollama=settings.ollama_base_url)
    try:
        yield
    finally:
        await app.state.ollama.aclose()
        await app.state.neo4j.close()
        log.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured app. Tests pass their own ``Settings``."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Multimodal graph-based retrieval augmented generation platform.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)

    app.include_router(health.router, prefix=settings.api_prefix)
    # Unversioned liveness alias for container/orchestrator probes; hidden from OpenAPI.
    app.add_api_route("/health", health.health, methods=["GET"], include_in_schema=False)
    return app


app = create_app()
