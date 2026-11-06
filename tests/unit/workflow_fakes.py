"""Fakes shared by the workflow tests: an LLM that answers by system prompt, ranked lists."""

from __future__ import annotations

import json

from omnirag.llm.chat import ChatResult
from omnirag.retrieval.agent import AGENT_SYSTEM
from omnirag.retrieval.models import RetrievalResult, RetrievedChunk
from omnirag.retrieval.router import ROUTE_SYSTEM
from omnirag.retrieval.self_query import SELF_QUERY_SYSTEM
from omnirag.retrieval.transforms import (
    CORRECT_SYSTEM,
    GRADE_SYSTEM,
    HYDE_SYSTEM,
    MULTI_SYSTEM,
    REWRITE_SYSTEM,
)

CFG = "multimodal-256-ctx"
PROMPTS = {REWRITE_SYSTEM: "rewrite", MULTI_SYSTEM: "multi", HYDE_SYSTEM: "hyde",
           GRADE_SYSTEM: "grade", CORRECT_SYSTEM: "correct", ROUTE_SYSTEM: "route",
           AGENT_SYSTEM: "agent", SELF_QUERY_SYSTEM: "self_query"}  # fmt: skip


class FakeLLM:
    """Replies by system prompt; each reply list is consumed in order (last one repeats)."""

    def __init__(self, **replies) -> None:
        self.replies = {k: list(v) if isinstance(v, list) else [v] for k, v in replies.items()}
        self.calls: list[tuple[str, str]] = []
        self.schemas: list[dict | None] = []
        self.stats = {"calls": 0}

    def chat(self, system, user, *, schema=None, **kw) -> ChatResult:
        name = PROMPTS[system]
        self.calls.append((name, user))
        self.schemas.append(schema)
        self.stats["calls"] += 1
        queue = self.replies[name]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        if schema is None:
            return ChatResult(reply, None, 1, 1, 1.0)
        return ChatResult(json.dumps(reply), reply, 1, 1, 1.0)


def _hit(cid: str, content: str, rank: int = 1, doc: str = "d") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=doc, source_file=f"{doc}.md", rank=rank,
                          score=0.5, retriever="dense", content=content,
                          element_type="paragraph", config_id=CFG)  # fmt: skip


class Retriever:
    """Returns chunk IDs per query; records every (query, filters) it was called with."""

    def __init__(self, by_query: dict[str, list[str]], content: str = "text",
                 texts: dict[str, str] | None = None) -> None:  # fmt: skip
        self.by_query, self.content, self.texts = by_query, content, texts or {}
        self.calls: list[str] = []
        self.filters: list = []

    def retrieve(self, query, k, filters=None, **kw) -> RetrievalResult:
        self.calls.append(query)
        self.kwargs = kw
        self.filters.append(filters)
        ids = self.by_query.get(query, [])
        hits = [_hit(c, self.texts.get(c, f"{self.content} {c}"), i + 1) for i, c in
                enumerate(ids[:k])]  # fmt: skip
        return RetrievalResult(query=query, retriever="dense", k=k, hits=hits, latency_ms=1.0)
