"""FastAPI dependency providers.

Routes never construct clients themselves; they declare what they need and receive the
process-wide instances created in the lifespan. Tests replace these providers through
``app.dependency_overrides`` to inject fakes without touching real services.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from omnirag.core.config import Settings
from omnirag.evaluation.runs import EvaluationRuns
from omnirag.generation.chat import ChatService
from omnirag.ingestion.jobs import JobManager
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base.neo4j_client import Neo4jClient
from omnirag.llm.ollama_client import OllamaClient
from omnirag.retrieval import RetrievalService
from omnirag.workflows.retrieval_graph import RetrievalWorkflow


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_store(request: Request) -> DocumentStore:
    return request.app.state.store


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_workflow(request: Request) -> RetrievalWorkflow:
    return request.app.state.workflow


def get_chat(request: Request) -> ChatService:
    return request.app.state.chat


def get_evaluations(request: Request) -> EvaluationRuns:
    return request.app.state.evaluations


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
StoreDep = Annotated[DocumentStore, Depends(get_store)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]
RetrievalDep = Annotated[RetrievalService, Depends(get_retrieval)]
WorkflowDep = Annotated[RetrievalWorkflow, Depends(get_workflow)]
ChatDep = Annotated[ChatService, Depends(get_chat)]
EvaluationsDep = Annotated[EvaluationRuns, Depends(get_evaluations)]
