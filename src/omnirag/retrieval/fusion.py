"""Reciprocal Rank Fusion and the hybrid retriever (dense + BM25 [+ graph]).

    RRF(d) = Σ_r  w_r / (k + rank_r(d))

Only ranks enter the formula. A cosine of 0.84, a BM25 score of 7.9 and a graph weight of
2.6 are not on one scale, and adding them would let whichever retriever produces the
largest numbers dominate. ``k`` (default 60, Cormack et al. 2009) damps the head of each
list: with k=60, ranks 1 and 2 are worth 1/61 and 1/62, almost equal. A small k trusts
each retriever's top hit more. Each member retrieves ``depth`` candidates; a chunk that a
member didn't return contributes nothing for that member.

Members run in parallel threads (the Neo4j driver is thread-safe; one session per call). If
a member fails, for example dense while Ollama is down, the others still answer, and the
failure is listed in ``details.failed``. Only when every member fails does the error propagate.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from omnirag.retrieval.models import (
    RetrievalFilters,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
)

log = structlog.get_logger(module=__name__)


def rrf(
    results: dict[str, RetrievalResult],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top: int = 5,
    mode: RetrievalMode = "dense+bm25",
) -> list[RetrievedChunk]:
    """Fuse ranked lists by reciprocal rank. Ties: best single rank, then chunk ID."""
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    weights = weights or {}
    score: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    payload: dict[str, RetrievedChunk] = {}
    sources: dict[str, dict[str, dict[str, float]]] = {}
    for name, result in results.items():
        w = weights.get(name, 1.0)
        for hit in result.hits:
            cid = hit.chunk_id
            score[cid] = score.get(cid, 0.0) + w / (k + hit.rank)
            best_rank[cid] = min(best_rank.get(cid, hit.rank), hit.rank)
            sources.setdefault(cid, {})[name] = {"rank": hit.rank, "score": hit.score}
            if cid not in payload or hit.graph_paths:  # keep the graph's provenance
                payload[cid] = hit
    order = sorted(score, key=lambda c: (-score[c], best_rank[c], c))[:top]
    return [
        payload[c].model_copy(update={"rank": i + 1, "score": round(score[c], 6),
                                      "retriever": mode, "sources": sources[c]})
        for i, c in enumerate(order)
    ]  # fmt: skip


def diversify(
    hits: list[RetrievedChunk], k: int, max_per_document: int | None = None
) -> list[RetrievedChunk]:
    """The top k, with at most ``max_per_document`` chunks from any one document (Phase 9b).

    Greedy in rank order: a chunk over its document's cap is skipped, and skipped chunks
    fill the remaining slots when fewer documents than ``k / cap`` were found. ``None``
    keeps the ranking unchanged. Multi-document questions need facts from several sources;
    one document's near-duplicates can push the second source out of the top k.
    """
    if not max_per_document:
        return hits[:k]
    picked: list[RetrievedChunk] = []
    skipped: list[RetrievedChunk] = []
    per_doc: dict[str, int] = {}
    for hit in hits:
        if len(picked) == k:
            break
        if per_doc.get(hit.document_id, 0) < max_per_document:
            picked.append(hit)
            per_doc[hit.document_id] = per_doc.get(hit.document_id, 0) + 1
        else:
            skipped.append(hit)
    return picked + skipped[: k - len(picked)]


class HybridRetriever:
    def __init__(
        self,
        members: dict[str, Any],
        *,
        mode: RetrievalMode,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        depth: int = 20,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.members = members
        self.mode = mode
        self.rrf_k = rrf_k
        self.weights = weights or {}
        self.depth = depth
        self._pool = ThreadPoolExecutor(max_workers=len(members), thread_name_prefix="retrieve")

    def retrieve(
        self, query: str, k: int = 5, filters: RetrievalFilters | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        depth = max(self.depth, k)
        futures = {name: self._pool.submit(r.retrieve, query, depth, filters)
                   for name, r in self.members.items()}  # fmt: skip
        results: dict[str, RetrievalResult] = {}
        failed: dict[str, str] = {}
        first_error: Exception | None = None
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:  # degrade: the other retrievers still answer
                log.warning("retriever_failed", retriever=name, error=str(exc))
                failed[name] = f"{type(exc).__name__}: {exc}"
                first_error = first_error or exc
        if not results and first_error is not None:
            raise first_error
        hits = rrf(results, k=self.rrf_k, weights=self.weights, top=k, mode=self.mode)
        return RetrievalResult(
            query=query, retriever=self.mode, k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={
                "rrf_k": self.rrf_k, "depth": depth, "weights": self.weights,
                "failed": failed,
                "members": {n: {"hits": len(r.hits), "latency_ms": r.latency_ms, **r.details}
                            for n, r in results.items()},
            },
        )  # fmt: skip

    def close(self) -> None:
        self._pool.shutdown(wait=False)
