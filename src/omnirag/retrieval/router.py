"""Query routing (Phase 9b): decide what kind of question this is before retrieving.

Intents:
* ``lookup``: one fact stated in text;
* ``table``: a value read from structured data (an amount, count, share, ranking, rows);
* ``image``: about visual content (a photo, chart, scan, whiteboard, drawing);
* ``multi_hop``: several facts combined: two questions in one, a thing *described* instead
  of named ("the cell that failed at 45 °C" has to be identified first), or a comparison.

What a route changes (``workflows/retrieval_graph.py``):
* ``table`` / ``image``: one extra ranked list restricted to the matching element types joins
  RRF. That is a *soft* preference: nothing is filtered out, so a wrong route costs rank
  positions, never recall;
* ``multi_hop``: the agent may search again with what the first passages name (``agent="auto"``).

Two routers, one interface:
* ``RuleRouter``: regular expressions, free and deterministic. The cues were written from the
  seed set's questions (so accuracy there is optimistic) and checked on
  ``data/eval/routing_holdout.json``, which was labelled before the rules existed;
* ``LLMRouter``: one structured call with the same definitions, no example questions (the
  model copies examples: Phase 9a).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from omnirag.llm.chat import LLMError, OllamaChat

Intent = Literal["lookup", "table", "image", "multi_hop"]
INTENTS: tuple[Intent, ...] = ("lookup", "table", "image", "multi_hop")
# Element types a route prefers (the chunk types in the graph: paragraph, list, table,
# record, figure, code). Figures carry image descriptions, OCR text and chart data.
PREFERRED_TYPES: dict[str, tuple[str, ...]] = {"table": ("table", "record"), "image": ("figure",)}


@dataclass
class Route:
    intent: Intent
    method: Literal["rules", "llm", "llm_error", "off"]
    cues: list[str] = field(default_factory=list)  # what matched (rules) or the reason (llm)
    ms: float = 0.0

    @property
    def element_types(self) -> tuple[str, ...]:
        return PREFERRED_TYPES.get(self.intent, ())

    def as_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "method": self.method, "cues": self.cues,
                "element_types": list(self.element_types), "ms": self.ms}  # fmt: skip


_I = re.IGNORECASE
_WH = r"(?:who|what|which|when|where|why|how|is|was|are|were|does|did|do)"
# Checked in this order: the first intent with a cue wins (multi-hop questions often also
# contain table words: "how much of the budget of the workstream owned by ...").
_CUES: list[tuple[Intent, str, re.Pattern[str]]] = [
    # two questions in one: "..., and who owns its qualification?"
    ("multi_hop", "two-part", re.compile(rf",?\s+and\s+{_WH}\b", _I)),
    # a described thing: "the electrolyte whose ...", "the salt that ...", "the cells that use"
    ("multi_hop", "described", re.compile(r"\bthe\s+(?:[\w-]+\s+){0,3}?(?:that|who|whose)\s", _I)),
    # "the workstream owned by", "the electrolyte used in", "the limit set in"
    ("multi_hop", "described",
     re.compile(r"\bthe\s+(?:[\w-]+\s+){0,2}?(?:\w+ed|set|made|run|held|written|drawn)"
                r"\s+(?:by|in|for|with)\s+the\b", _I)),
    ("multi_hop", "comparison",
     re.compile(r"\b(?:match(?:es)?|compare[ds]?|comparison|differ(?:s|ence)?|versus|vs\.?|"
                r"same as|(?:higher|lower|more|less|better|worse) than)\b", _I)),
    ("image", "visual",
     re.compile(r"\b(?:photo(?:graph)?s?|pictures?|images?|charts?|graphs?|plots?|diagrams?|"
                r"figures?|whiteboards?|drawings?|drawn|sketch(?:es|ed)?|scan(?:s|ned)?|"
                r"handwritten|screenshots?)\b", _I)),
    ("table", "quantity",
     re.compile(r"\b(?:how (?:much|many)|what (?:share|percentage|fraction|proportion)|"
                r"percent(?:age)?|budgets?|spent|costs?|totals?|average|highest|lowest|"
                r"largest|smallest)\b", _I)),
]  # fmt: skip


class RuleRouter:
    def route(self, question: str) -> Route:
        started = time.perf_counter()
        for intent, cue, pattern in _CUES:
            m = pattern.search(question)
            if m:
                ms = round((time.perf_counter() - started) * 1000, 3)
                return Route(intent, "rules", [f"{cue}: {m.group(0).strip()}"], ms)
        return Route("lookup", "rules", [], round((time.perf_counter() - started) * 1000, 3))


ROUTE_SYSTEM = (
    "You classify a question about a collection of technical project documents by what the "
    "search has to do. 'lookup': one fact stated in text. 'table': a value read from "
    "structured data: an amount, count, share, ranking or list of rows from a table, "
    "spreadsheet or data record. 'image': the question is about visual content: a photo, "
    "chart, scan, whiteboard, drawing or diagram. 'multi_hop': answering needs several facts "
    "combined: the question has two parts, describes a thing instead of naming it (so it "
    "must be identified first), or compares two things. Pick the single best intent; "
    "'reason': at most 10 words."
)
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": list(INTENTS)},
                   "reason": {"type": "string"}},
    "required": ["intent", "reason"],
}  # fmt: skip


class LLMRouter:
    def __init__(self, llm: OllamaChat) -> None:
        self.llm = llm

    def route(self, question: str) -> Route:
        try:
            r = self.llm.chat(ROUTE_SYSTEM, f"Question: {question}", schema=ROUTE_SCHEMA,
                              num_predict=64)  # fmt: skip
        except LLMError as exc:  # no route is a plain lookup, not a failed request
            return Route("lookup", "llm_error", [str(exc)[:200]])
        intent = r.data.get("intent")
        if intent not in INTENTS:
            return Route("lookup", "llm_error", [f"unknown intent {intent!r}"], round(r.ms, 1))
        return Route(intent, "llm", [str(r.data.get("reason", ""))[:120]], round(r.ms, 1))
