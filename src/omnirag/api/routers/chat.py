"""Chat endpoints (Phases 10-11): grounded, cited answers and their history.

* ``POST /chat``                 one turn: retrieval workflow → answer → validated citations.
* ``POST /chat/stream``          the same turn as Server-Sent Events: ``start``, ``retrieval``,
  ``claim`` (each validated claim as soon as the model finishes it), ``answer``, ``done``.
* ``GET /chat/history``          conversations, newest first.
* ``GET /chat/history/{id}``     one conversation with every turn's full record (the trace).
* ``DELETE /chat/history/{id}``  forget a conversation.
* ``GET /chat/turns/{turn_id}``  one turn's record.

Model failures don't fail a turn: the answer says ``refused_by="error"`` and the turn is kept.
Neo4j or embedding failures during retrieval do fail it (503), like ``/retrieval/advanced``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omnirag.api.dependencies import ChatDep
from omnirag.core.errors import (
    ConflictError,
    DependencyUnavailableError,
    NotFoundError,
    ValidationFailedError,
)
from omnirag.embeddings.base import EmbeddingError, EmbeddingInputError
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.llm.chat import LLMError
from omnirag.retrieval.models import RetrievalError, RetrievalFilters
from omnirag.workflows.retrieval_graph import Strategy

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None  # None = start a new conversation
    k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    strategy: Strategy | None = None  # None = the settings' workflow defaults
    respect_check: bool | None = None  # None = settings.generation_respect_check


def _translate(exc: Exception) -> Exception:
    if isinstance(exc, LookupError):
        return NotFoundError(str(exc))
    if isinstance(exc, EmbeddingInputError):
        return ValidationFailedError(str(exc))
    if isinstance(exc, RetrievalError):
        return ConflictError(str(exc))
    if isinstance(exc, GraphUnavailableError):
        return DependencyUnavailableError(f"Neo4j unavailable: {exc}")
    if isinstance(exc, EmbeddingError | LLMError):
        return DependencyUnavailableError(f"Model server unavailable: {exc}")
    return exc


def _kwargs(request: ChatRequest) -> dict[str, Any]:
    return {"conversation_id": request.conversation_id, "k": request.k,
            "filters": request.filters, "strategy": request.strategy,
            "respect_check": request.respect_check}  # fmt: skip


@router.post("", summary="Ask a question: grounded answer with verified citations")
async def chat(request: ChatRequest, service: ChatDep) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(service.ask, request.question, **_kwargs(request))
    except Exception as exc:  # noqa: BLE001 - translated or re-raised
        raise _translate(exc) from exc


@router.post("/stream", summary="Ask a question, streamed as Server-Sent Events")
async def chat_stream(request: ChatRequest, service: ChatDep) -> StreamingResponse:
    def events() -> Iterator[str]:
        try:
            for event in service.stream(request.question, **_kwargs(request)):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - reported in-band: headers are already sent
            err = _translate(exc)
            body = {"type": "error", "message": str(getattr(err, "message", err))[:500]}
            yield f"event: error\ndata: {json.dumps(body)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})  # fmt: skip


@router.get("/history", summary="Conversations, newest first")
async def history(service: ChatDep, limit: int = Query(default=50, ge=1, le=500)
                  ) -> list[dict[str, Any]]:  # fmt: skip
    return await asyncio.to_thread(service.store.conversations, limit)


@router.get("/history/{conversation_id}", summary="One conversation with every turn's trace")
async def conversation(conversation_id: str, service: ChatDep) -> dict[str, Any]:
    conv = await asyncio.to_thread(service.store.conversation, conversation_id)
    if conv is None:
        raise NotFoundError(f"conversation {conversation_id} not found")
    return conv


@router.delete("/history/{conversation_id}", status_code=204, summary="Delete a conversation")
async def delete_conversation(conversation_id: str, service: ChatDep) -> None:
    if not await asyncio.to_thread(service.store.delete_conversation, conversation_id):
        raise NotFoundError(f"conversation {conversation_id} not found")


@router.get("/turns/{turn_id}", summary="One turn's full record")
async def turn(turn_id: str, service: ChatDep) -> dict[str, Any]:
    record = await asyncio.to_thread(service.store.turn, turn_id)
    if record is None:
        raise NotFoundError(f"turn {turn_id} not found")
    return record
