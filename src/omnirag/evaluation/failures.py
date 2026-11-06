"""Failure analysis (Phase 13): which stage of the pipeline lost a question?

Each answerable question's facts are followed through the pipeline in order; the first stage
where a fact is lost is the **root cause** (later stages can't recover what an earlier one
dropped, so counting it again would double-blame):

=========================  ==========================================================
parsing / OCR failure      the evidence quote is in no element of its file (OCR for
                           images and scans)
chunking failure           the quote is in an element but in no single chunk (split)
graph construction         no chunk that covers the fact exists in Neo4j
embedding failure          covering chunks exist but none has a vector (dense modes)
retrieval failure          no covering chunk among the ranked candidates
reranking failure          the first stage had it in its top k, the reranked list
                           pushed it out of the context
context construction       a candidate covers it, the assembled context doesn't
generation failure         the context had every fact; the answer misses terms or refuses
hallucination              an unsupported claim, or an answer to an unanswerable question
citation failure           the answer is right but a claim cites the wrong source
=========================  ==========================================================

Live chat turns have no ground truth; ``live_flags`` classifies what can be seen without it
(refusals, unsupported claims, relocated citations, empty retrieval).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from omnirag.evaluation.dataset import EvalQuestion, norm
from omnirag.knowledge_base.entities import norm as norm_text

CATEGORIES = ("parsing_failure", "ocr_failure", "chunking_failure", "graph_construction_failure",
              "embedding_failure", "retrieval_failure", "reranking_failure",
              "context_construction_failure", "generation_failure", "hallucination",
              "citation_failure")  # fmt: skip
OCR_FILES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def answer_includes(text: str, must_include: list[str | list[str]]) -> tuple[bool, float]:
    """(all terms present, share present). An inner list is a set of alternatives."""
    if not must_include:
        return True, 1.0
    t = norm_text(text)
    hits = sum(any(norm_text(alt) in t for alt in ([m] if isinstance(m, str) else m))
               for m in must_include)  # fmt: skip
    return hits == len(must_include), round(hits / len(must_include), 3)


def _covers(text: str, fact: list[Any]) -> bool:
    t = norm(text)
    return any(norm(a.text) in t for a in fact)


class FailureClassifier:
    """Built once per run from the corpus: elements by file, chunks, and what the graph has."""

    def __init__(self, elements_by_file: dict[str, list[Any]], chunks: list[Any],
                 graph_chunk_ids: set[str] | None = None,
                 embedded_chunk_ids: set[str] | None = None) -> None:  # fmt: skip
        self.elements = elements_by_file
        self.chunks = chunks
        self.graph = graph_chunk_ids
        self.embedded = embedded_chunk_ids

    def _stage_of_source(self, fact: list[Any]) -> tuple[str | None, list[Any]]:
        """Before retrieval: parsing → chunking → graph → embedding. Returns the failure (or
        None) and the chunks that cover the fact."""
        in_elements = any(_covers(e.content, [a]) for a in fact
                          for e in self.elements.get(a.file, []))  # fmt: skip
        if not in_elements:
            ocr = any(a.file.lower().endswith(OCR_FILES) or "scan" in a.file.lower() for a in fact)
            return ("ocr_failure" if ocr else "parsing_failure"), []
        covering = [c for c in self.chunks if _covers(c.content, fact)]
        if not covering:
            return "chunking_failure", []
        if self.graph is not None and not any(c.chunk_id in self.graph for c in covering):
            return "graph_construction_failure", covering
        return None, covering

    def classify(self, q: EvalQuestion, *, candidates: list[Any], context_text: str,
                 answer: dict[str, Any], dense: bool = False,
                 reranked: bool = False, k: int = 5) -> list[str]:  # fmt: skip
        """Failure categories for one question in one experiment ([] = success), root cause
        first. ``candidates`` are the ranked hits the context was built from."""
        claims = answer.get("claims", [])
        unsupported = any(c.get("status") == "unsupported" for c in claims)
        if not q.answerable:
            return ["hallucination"] if answer.get("answerable") else []
        out: list[str] = []
        for fact in q.facts:
            stage, covering = self._stage_of_source(fact)
            if stage is None and dense and self.embedded is not None and not any(
                    c.chunk_id in self.embedded for c in covering):  # fmt: skip
                stage = "embedding_failure"
            if stage is None:
                ids = {c.chunk_id for c in covering}
                found = [h for h in candidates if h.chunk_id in ids or _covers(h.content, fact)]
                if not found:
                    stage = "retrieval_failure"
                elif not _covers(context_text, fact):
                    first = [(h.sources.get("rerank") or {}).get("first_stage_rank") for h in found]
                    stage = ("reranking_failure" if reranked and any(r and r <= k for r in first)
                             else "context_construction_failure")  # fmt: skip
            if stage:
                out.append(stage)
        if out:
            out = [min(out, key=CATEGORIES.index)]  # the earliest stage is the root cause
        else:
            ok, _ = answer_includes(answer.get("text", ""), q.must_include)
            if not answer.get("answerable") or not ok:
                out.append("generation_failure")
        if unsupported:
            out.append("hallucination")
        if not out and any(c.get("status") == "relocated" for c in claims):
            out.append("citation_failure")
        return out


def live_flags(turn: dict[str, Any]) -> list[str]:
    """What a live chat turn shows without ground truth."""
    answer = turn.get("answer", {})
    flags = []
    if answer.get("refused_by") == "empty_context":
        flags.append("retrieval_failure")
    if answer.get("refused_by") == "error":
        flags.append("generation_failure")
    claims = answer.get("claims", [])
    if any(c.get("status") == "unsupported" for c in claims):
        flags.append("hallucination")
    if any(c.get("status") == "relocated" for c in claims):
        flags.append("citation_failure")
    return flags


def summarize(categories: dict[str, list[str]]) -> dict[str, Any]:
    """Root-cause counts and the questions behind each category."""
    primary = Counter(c[0] for c in categories.values() if c)
    by_cat: dict[str, list[str]] = {}
    for qid, cats in sorted(categories.items()):
        for c in cats:
            by_cat.setdefault(c, []).append(qid)
    return {
        "root_causes": dict(primary.most_common()),
        "questions": by_cat,
        "failed": sum(1 for c in categories.values() if c),
        "total": len(categories),
    }
