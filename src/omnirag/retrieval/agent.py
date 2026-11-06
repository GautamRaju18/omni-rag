"""Agentic retrieval (Phase 9b): the model reads what was found and chooses the next search.

The 9a corrective step wrote its new query from the grader's one-line "missing" note,
without seeing the passages, so it could only reword the question. Multi-hop questions
need a *bridge*: "Who supplies the salt that the E-5 electrolyte is made with?" can't be
answered until a passage says the salt is NaPF6, and then "NaPF6 supplier" finds the
budget row. The planner sees the question, the searches already run and the current top
passages, and picks one tool:

* ``search``: a new dense + BM25 query (typically: a name the passages gave + what is asked);
* ``graph``: everything the knowledge graph links to one named entity (Phase 7 retriever);
* ``finish``: enough is there, or nothing more will be found.

**Guards (code decides):** every identifier in a new query or entity name must occur in the
question or in the passages shown (after ``norm``: NaPF₆ = NaPF6). The model may follow
names it has *read*, never invent one. A repeated query is refused. A refused step ends the
agent's turn and is kept in the trace. The number of steps is bounded by the strategy
(0-3) and by LangGraph's recursion limit.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal

from omnirag.knowledge_base.entities import norm
from omnirag.llm.chat import LLMError, OllamaChat
from omnirag.retrieval.models import RetrievedChunk
from omnirag.retrieval.sparse import identifiers

Action = Literal["search", "graph", "finish"]

AGENT_SYSTEM = (
    "You plan searches over technical project documents to answer a question. You see the "
    "question, the searches already run and the passages found so far. Choose one action. "
    "'search': one new search query for information that is still missing. Questions often "
    "need a chain of facts: when a passage names the thing that the question only describes "
    "(a material, a person, a part, a document), search for that name together with what "
    "the question still asks. 'graph': look up one entity in the knowledge graph to find "
    "every passage connected to it; 'query' is the entity's name copied from a passage. "
    "'finish': the passages already contain everything needed, or nothing more can be "
    "found. Copy names, identifiers and numbers exactly from the question or the passages; "
    "never invent one. Do not answer the question. 'reason': at most 15 words."
)
AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "graph", "finish"]},
        "query": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "query", "reason"],
}


@dataclass
class AgentStep:
    action: Action
    query: str = ""
    reason: str = ""
    refused: str | None = None  # why the code didn't run the step
    ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def ungrounded(query: str, question: str, hits: list[RetrievedChunk]) -> str | None:
    """Why a model-written query can't run (an identifier nobody wrote), or None."""
    known = norm(" ".join([question, *(h.content for h in hits)]))
    invented = [i for i in identifiers(query) if norm(i) not in known]
    return f"identifier not in the question or passages: {', '.join(invented)}" if invented \
        else None  # fmt: skip


class AgentPlanner:
    def __init__(self, llm: OllamaChat, *, max_chars: int = 700) -> None:
        self.llm = llm
        self.max_chars = max_chars

    def next_step(self, question: str, hits: list[RetrievedChunk], tried: list[str]) -> AgentStep:
        started = time.perf_counter()
        passages = "\n\n".join(f"[{i}] ({h.source_file}) {h.content[: self.max_chars]}"
                               for i, h in enumerate(hits, start=1))  # fmt: skip
        user = (f"Question: {question}\nSearches already run: {'; '.join(tried)}\n\n"
                f"Passages:\n{passages or '(none)'}")  # fmt: skip
        try:
            r = self.llm.chat(AGENT_SYSTEM, user, schema=AGENT_SCHEMA, num_predict=160)
        except LLMError as exc:  # no plan: stop searching, keep what was found
            return AgentStep("finish", refused=f"llm_error: {exc}"[:200])
        ms = round((time.perf_counter() - started) * 1000, 1)
        action = r.data.get("action")
        query = " ".join(str(r.data.get("query", "")).split())[:300]
        step = AgentStep(action if action in ("search", "graph", "finish") else "finish", query,
                         str(r.data.get("reason", ""))[:200], ms=ms)  # fmt: skip
        if step.action == "finish":
            return step
        if len(query) < 2:
            step.refused = "empty query"
        elif query.lower() in {t.lower().removeprefix("graph: ") for t in tried}:
            step.refused = "repeats an earlier search"
        elif step.action == "graph" and norm(query) not in norm(" ".join(
                [question, *(h.content for h in hits)])):  # fmt: skip
            step.refused = "entity not named in the question or passages"
        else:
            step.refused = ungrounded(query, question, hits)
        return step
