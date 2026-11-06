"""LLM-as-a-judge for the semantic generation metrics (Phase 12).

Deterministic metrics come first (``rag_eval``): answer terms (``must_include``), quote-verified
faithfulness, citation accuracy, refusals. The judge adds what text matching can't see:

* **correctness** against the expert reference answer: correct / partial / incorrect / refused;
* **relevance** (1-5): does the answer address the question that was asked;
* **supported**: is every statement backed by the sources the system was given.

Design choices (ADR-0020): temperature 0 and a JSON schema; the reference answer is shown, so
the judge compares rather than knows; the rubric names the failure modes. The judge is the
same local model as the generator (a second model would not fit beside it on a 6 GB GPU), so
it may favour its own phrasing: its agreement with the deterministic answer check is reported
with every run, and a disagreement is a reason to read the answer, not a score to trust.
"""

from __future__ import annotations

from typing import Any

from omnirag.llm.chat import LLMError

JUDGE_VERSION = "judge-v2"  # v2: short replies (v1 reasons ran to ~300 characters)

JUDGE_SYSTEM = (
    "You grade the answer of a question-answering system over technical project documents. "
    "You get the question, a reference answer written by an expert, the system's answer and "
    "the numbered sources the system was given.\n"
    "Grade:\n"
    '- correctness: "correct" if the system answer states the key facts of the reference '
    '(extra correct detail is fine); "partial" if it states some of them but misses or gets '
    'others wrong; "incorrect" if it misses them or contradicts the reference; "refused" if it '
    "says the information is not available. If the reference says the documents don't contain "
    'the answer, a refusal is "correct" and any concrete answer is "incorrect".\n'
    "- relevance: 1 to 5, how directly the answer addresses the question asked.\n"
    "- supported: true if every statement in the answer is backed by the sources, false if any "
    "statement is not; list at most 3 unsupported statements, shortened.\n"
    "- reason: one sentence of at most 15 words.\n"
    "Judge only from the texts given. Numbers and identifiers must match exactly."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect", "refused"]},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "supported": {"type": "boolean"},
        "unsupported_statements": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "relevance", "supported", "unsupported_statements", "reason"],
}

NO_ANSWER_REFERENCE = "The documents do not contain this information."


def judge(llm: Any, question: str, reference: str | None, answer_text: str, missing: str | None,
          sources: str) -> dict[str, Any]:  # fmt: skip
    """One structured judge call. Never raises: an ``LLMError`` becomes ``{"error": ...}``."""
    shown = answer_text or f"(no answer) {missing or 'The system said it cannot answer.'}"
    user = (
        f"Question: {question}\n\nReference answer: {reference or NO_ANSWER_REFERENCE}\n\n"
        f"System answer: {shown}\n\nSources given to the system:\n{sources or '(none)'}"
    )
    try:
        r = llm.chat(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA, num_predict=200)
    except LLMError as exc:
        return {"error": str(exc)[:200]}
    data = r.data if isinstance(r.data, dict) else {}
    relevance = data.get("relevance")
    return {
        "correctness": data.get("correctness"),
        "relevance": relevance if isinstance(relevance, int) and 1 <= relevance <= 5 else None,
        "supported": data.get("supported") if isinstance(data.get("supported"), bool) else None,
        "unsupported_statements": [str(s) for s in data.get("unsupported_statements") or []][:5],
        "reason": str(data.get("reason", ""))[:300],
        "ms": round(r.ms, 1),
    }
