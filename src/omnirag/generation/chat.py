"""The chat service: retrieval workflow → grounded answer → stored trace (Phases 10-13).

``ask`` runs one turn and returns (and stores) its full record; ``stream`` runs the same turn
as events (``retrieval``, ``claim`` ×n, ``answer``, ``done``) for the streaming endpoint.

The record is the query lifecycle the brief's observability section asks for: the question,
strategy, rewritten queries, route, agent steps, every ranked list, rerank scores, the
selected context, the answer, its citations and the latency of every component. Turns are
independent retrievals: a follow-up question is answered from its own retrieval (history is
stored and shown, not fed back into the prompt).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from omnirag.generation.generator import AnswerGenerator
from omnirag.generation.history import ChatStore
from omnirag.generation.models import Answer
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult


def retrieval_summary(result: RetrievalResult) -> dict[str, Any]:
    """What the trace keeps from the workflow (everything except the bulky rendered text)."""
    d = result.details
    ctx = d.get("context") or {}
    return {
        "verdict": d.get("verdict"), "attempts": d.get("attempts"), "queries": d.get("queries"),
        "route": d.get("route"), "self_query": d.get("self_query"), "agent": d.get("agent"),
        "checks": d.get("checks"), "strategy": d.get("strategy"), "trace": d.get("trace"),
        "lists": d.get("lists"), "latency_ms": result.latency_ms,
        "hits": [{"chunk_id": h.chunk_id, "rank": h.rank, "score": round(h.score, 6),
                  "source_file": h.source_file, "page_numbers": h.page_numbers,
                  "rerank": h.sources.get("rerank")} for h in result.hits],
        "context": {k: v for k, v in ctx.items() if k != "rendered"},
    }  # fmt: skip


def timings(result: RetrievalResult, answer: Answer, total_ms: float) -> dict[str, float]:
    """Per-component latency (ms). Retrieval nodes come from the workflow trace."""
    out: dict[str, float] = {}
    for step in result.details.get("trace", []):
        out[step["node"]] = round(out.get(step["node"], 0.0) + step["ms"], 1)
    lists = result.details.get("lists", {})
    out["embedding"] = round(sum(v.get("embed_ms") or 0 for v in lists.values()), 1)
    out["retrieval_total"] = result.latency_ms
    out["generation"] = answer.latency_ms
    out["total_ms"] = round(total_ms, 1)
    return out


class ChatService:
    def __init__(self, workflow: Any, generator: AnswerGenerator, store: ChatStore) -> None:
        self.workflow = workflow
        self.generator = generator
        self.store = store

    def _retrieve(self, question: str, k: int, filters: RetrievalFilters | None,
                  strategy: Any) -> RetrievalResult:  # fmt: skip
        return self.workflow.retrieve(question, k, filters, strategy)

    def _record(self, question: str, result: RetrievalResult, answer: Answer,
                started: float) -> dict[str, Any]:  # fmt: skip
        return {
            "question": question,
            "answer": answer.model_dump(mode="json"),
            "retrieval": retrieval_summary(result),
            "timings": timings(result, answer, (time.perf_counter() - started) * 1000),
        }

    def ask(self, question: str, *, conversation_id: str | None = None, k: int = 5,
            filters: RetrievalFilters | None = None, strategy: Any = None,
            respect_check: bool | None = None) -> dict[str, Any]:  # fmt: skip
        started = time.perf_counter()
        conversation_id = self.store.ensure_conversation(conversation_id, question)
        result = self._retrieve(question, k, filters, strategy)
        d = result.details
        answer = self.generator.generate(
            question,
            d.get("context"),
            d.get("verdict"),
            d.get("checks"),
            respect_check=respect_check,
        )
        return self.store.add_turn(conversation_id, self._record(question, result, answer, started))

    def stream(self, question: str, *, conversation_id: str | None = None, k: int = 5,
               filters: RetrievalFilters | None = None, strategy: Any = None,
               respect_check: bool | None = None) -> Iterator[dict[str, Any]]:  # fmt: skip
        started = time.perf_counter()
        conversation_id = self.store.ensure_conversation(conversation_id, question)
        yield {"type": "start", "conversation_id": conversation_id}
        result = self._retrieve(question, k, filters, strategy)
        d = result.details
        ctx = d.get("context") or {}
        yield {"type": "retrieval", "verdict": d.get("verdict"), "route": d.get("route"),
               "latency_ms": result.latency_ms,
               "blocks": [{"number": i, "source_file": b["source_file"],
                           "section_path": b["section_path"], "page_numbers": b["page_numbers"],
                           "tokens": b["tokens"]}
                          for i, b in enumerate(ctx.get("blocks", []), start=1)]}  # fmt: skip
        answer: Answer | None = None
        for event in self.generator.stream(question, ctx, d.get("verdict"), d.get("checks"),
                                           respect_check=respect_check):  # fmt: skip
            if event["type"] == "answer":
                answer = Answer.model_validate(event["answer"])
            yield event
        if answer is None:  # pragma: no cover - the generator always ends with an answer
            raise RuntimeError("generator ended without an answer event")
        turn = self.store.add_turn(conversation_id, self._record(question, result, answer, started))
        yield {
            "type": "done",
            "turn_id": turn["id"],
            "conversation_id": conversation_id,
            "timings": turn["timings"],
        }


def build_chat(settings: Any, workflow: Any) -> ChatService:
    """The configured generator (its own client: generation timeouts differ from the
    workflow's short structured calls) over the workflow, storing turns in SQLite."""
    from omnirag.llm.chat import OllamaChat

    llm = OllamaChat(settings.ollama_base_url, settings.generation_model,
                     timeout_s=settings.generation_timeout_s)  # fmt: skip
    generator = AnswerGenerator(llm, num_predict=settings.generation_num_predict,
                                respect_check=settings.generation_respect_check)  # fmt: skip
    return ChatService(workflow, generator, ChatStore(settings.chat_db))
