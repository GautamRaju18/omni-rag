"""LLM query transforms and the context quality check (Phase 9).

Transforms (one structured LLM call each; identifiers and numbers must survive verbatim):
* **rewrite**: the question as one concise search query;
* **multi-query**: N differently worded queries; for multi-part questions, one per part;
* **HyDE** (Gao et al. 2022): a short *hypothetical* passage that would answer the
  question. Its embedding sits closer to real answer passages than the question's does.
  Used for dense search only, never shown and never treated as evidence;
* **corrective rewrite**: a new query aimed at what the context check says is missing.

The **context check** answers "do these passages let us answer the question?" in two steps:
1. *identifier coverage* (deterministic, free): every technical identifier in the question
   (AUR-C109, M2) must appear in at least one passage. If one doesn't, the context is
   ``insufficient``, with no model call. This catches questions about things that don't exist,
   which the reranker scores as relevant (Phase 8: AUR-C109 scored 0.936).
2. *LLM grader*: ``sufficient`` / ``partial`` / ``insufficient`` plus what is missing,
   judged only from the numbered passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from omnirag.llm.chat import LLMError, OllamaChat
from omnirag.retrieval.models import RetrievedChunk
from omnirag.retrieval.sparse import identifiers

Verdict = Literal["sufficient", "partial", "insufficient"]

# No example identifiers in prompts: the model copied them into queries and HyDE passages
# ("AUR-C103, E-7, M2" from the corpus, then "XB-12, QZ-4"), measured 2026-11-05.
_KEEP = (
    "Keep every name, identifier, number, unit and date exactly as written in the "
    "question. Never add an identifier that is not in the question."
)

REWRITE_SYSTEM = (
    "You turn a user's question into one concise search query for a search engine over "
    f"technical project documents. {_KEEP} Do not answer the question."
)
MULTI_SYSTEM = (
    "You write different search queries that together find all the information needed to "
    "answer a question over technical project documents. Each query targets a different "
    "aspect or wording; if the question has several parts or steps, write one query per "
    f"part. {_KEEP} Do not answer the question."
)
HYDE_SYSTEM = (
    "You write a short passage (2-4 sentences) in the style of a technical project document "
    "that would answer the question. Plausible details are fine: the passage is only used to "
    f"find similar real passages and is never shown to anyone. {_KEEP}"
)
GRADE_SYSTEM = (
    "You check whether numbered context passages contain the information needed to answer a "
    "question. Use only the passages, never outside knowledge. 'sufficient': every part of "
    "the question can be answered from the passages. 'partial': some parts can. "
    "'insufficient': none can, or the question asks about a thing (person, identifier, "
    "value) that the passages do not mention. List the numbers of the passages that support "
    "the answer, and say in at most 15 words what is missing ('' if nothing)."
)
CORRECT_SYSTEM = (
    "A search for a question did not find everything needed. Write one new search query "
    f"that targets what is missing. {_KEEP} Do not answer the question."
)

REWRITE_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"]}  # fmt: skip
MULTI_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array",
                "items": {"type": "string"}}}, "required": ["queries"]}  # fmt: skip
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["sufficient", "partial", "insufficient"]},
        "supporting": {"type": "array", "items": {"type": "integer"}},
        "missing": {"type": "string"},
    },
    "required": ["verdict", "supporting", "missing"],
}  # fmt: skip


@dataclass
class ContextCheck:
    verdict: Verdict
    method: Literal["identifiers", "llm", "llm_error", "none"]
    missing: str = ""
    missing_identifiers: list[str] = field(default_factory=list)
    supporting: list[str] = field(default_factory=list)  # chunk IDs the grader cited
    ms: float = 0.0


def _clean(query: str, question: str) -> str:
    """Model output as a query; falls back to the question if empty or junk."""
    q = " ".join(str(query).split())
    return q if 3 <= len(q) <= 500 else question


def _keeps_identifiers(query: str, question: str) -> bool:
    """Every identifier of the question survives, and none is invented."""
    kept = all(re.search(re.escape(i), query, re.I) for i in identifiers(question))
    known = {i.lower() for i in identifiers(question)}
    return kept and all(i.lower() in known for i in identifiers(query))


class QueryTransformer:
    def __init__(self, llm: OllamaChat) -> None:
        self.llm = llm

    def rewrite(self, question: str) -> str:
        r = self.llm.chat(REWRITE_SYSTEM, f"Question: {question}", schema=REWRITE_SCHEMA,
                          num_predict=128)  # fmt: skip
        query = _clean(r.data.get("query", ""), question)
        # A rewrite that drops an identifier is worse than none (BM25 requires them).
        return query if _keeps_identifiers(query, question) else question

    def multi_query(self, question: str, n: int = 3) -> list[str]:
        r = self.llm.chat(MULTI_SYSTEM, f"Write {n} queries.\nQuestion: {question}",
                          schema=MULTI_SCHEMA, num_predict=256)  # fmt: skip
        out = [question]  # the original always stays in the set
        for q in r.data.get("queries", [])[:n]:
            q = _clean(q, question)
            if _keeps_identifiers(q, question) and q.lower() not in (o.lower() for o in out):
                out.append(q)
        return out

    def hyde(self, question: str) -> str:
        r = self.llm.chat(HYDE_SYSTEM, f"Question: {question}", num_predict=200)
        return r.text if len(r.text) >= 20 else question

    def corrective(self, question: str, missing: str, tried: list[str]) -> str:
        user = (f"Question: {question}\nAlready searched: {'; '.join(tried)}\n"
                f"Missing: {missing or 'unknown'}")  # fmt: skip
        r = self.llm.chat(CORRECT_SYSTEM, user, schema=REWRITE_SCHEMA, num_predict=128)
        query = _clean(r.data.get("query", ""), question)
        return query if _keeps_identifiers(query, question) else question

    def check(self, question: str, hits: list[RetrievedChunk], *, use_llm: bool = True,
              max_chars: int = 1200) -> ContextCheck:  # fmt: skip
        if not hits:
            return ContextCheck("insufficient", "identifiers", "no passages retrieved")
        text = "\n".join(h.content for h in hits).lower()
        absent = [i for i in identifiers(question) if i.lower() not in text]
        if absent:
            return ContextCheck(
                "insufficient",
                "identifiers",
                f"not mentioned in any passage: {', '.join(absent)}",
                absent,
            )
        if not use_llm:
            return ContextCheck("sufficient", "none")
        passages = "\n\n".join(f"[{i}] ({h.source_file}) {h.content[:max_chars]}"
                               for i, h in enumerate(hits, start=1))  # fmt: skip
        try:
            r = self.llm.chat(GRADE_SYSTEM, f"Question: {question}\n\nPassages:\n{passages}",
                              schema=GRADE_SCHEMA, num_predict=400)  # fmt: skip
        except LLMError as exc:  # the check failed, not the context: say so, don't guess
            return ContextCheck("partial", "llm_error", f"grader failed: {exc}"[:300])
        data: dict[str, Any] = r.data
        cited = [hits[n - 1].chunk_id for n in data.get("supporting", [])
                 if isinstance(n, int) and 1 <= n <= len(hits)]  # fmt: skip
        return ContextCheck(
            data.get("verdict", "partial"),
            "llm",
            str(data.get("missing", ""))[:300],
            [],
            cited,
            round(r.ms, 1),
        )
