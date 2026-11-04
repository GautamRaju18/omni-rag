"""FastAPI dependency providers.

Routes never construct clients themselves; they declare what they need and receive the
process-wide instances created in the lifespan. Tests replace these providers through
``app.dependency_overrides`` to inject fakes without touching real services.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from omnirag.core.config import Settings
from omnirag.knowledge_base.neo4j_client import Neo4jClient
from omnirag.llm.ollama_client import OllamaClient


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
