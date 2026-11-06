"""Corrective retrieval workflow in LangGraph (Phase 9a).

    START → transform → retrieve → [rerank] → check ─┬─ sufficient / out of retries ─→ END
                           ↑                         └─ partial / insufficient → correct ┘

* **transform**: ``none`` | ``rewrite`` | ``multi_query`` | ``hyde`` | ``multi_query+hyde``.
  The original question always stays among the queries.
* **retrieve**: dense and BM25 for every query (and dense for the HyDE passage), in parallel,
  fused with RRF together with every list from earlier attempts. A retry only adds evidence.
* **rerank** (optional): cross-encoder on the fused candidates (slow on CPU, ADR-0015).
* **check**: identifier coverage, then (optionally) the LLM grader (``transforms.py``).
* **correct**: one new query aimed at what the check says is missing, then retrieve again.

**Bounded by construction:** ``max_retries`` (0–3) is counted in the state, and the graph runs
with a ``recursion_limit`` derived from it, so a loop beyond the plan raises instead of
spinning. Every node appends to ``trace`` (what ran, how long, what it produced). The result
says whether the context was judged sufficient; ``insufficient`` is a signal for generation
(Phase 10) to say "not in the documents", not an error.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from omnirag.llm.chat import LLMError
from omnirag.retrieval.fusion import rrf
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult, RetrievedChunk
from omnirag.retrieval.reranker import CrossEncoderReranker, passage_text, sigmoid
from omnirag.retrieval.transforms import QueryTransformer

Transform = Literal["none", "rewrite", "multi_query", "hyde", "multi_query+hyde"]


class Strategy(BaseModel):
    transform: Transform = "none"
    multi_query_n: int = Field(default=3, ge=1, le=6)
    grade: bool = True  # LLM grader after the identifier check
    max_retries: int = Field(default=1, ge=0, le=3)
    rerank: bool = False
    depth: int = Field(default=20, ge=1, le=100)  # candidates per list / reranked
    rrf_k: int = Field(default=60, ge=1)


class State(TypedDict, total=False):
    question: str
    k: int
    filters: RetrievalFilters
    strategy: Strategy
    queries: list[str]  # every query issued so far (original first)
    hyde: str | None
    lists: dict[str, RetrievalResult]  # every ranked list so far, fused together
    hits: list[RetrievedChunk]
    attempts: int  # corrective retries done
    checks: list[dict[str, Any]]
    verdict: str
    trace: list[dict[str, Any]]


def _step(state: State, node: str, started: float, **info: Any) -> list[dict[str, Any]]:
    return [
        *state.get("trace", []),
        {"node": node, "ms": round((time.perf_counter() - started) * 1000, 1), **info},
    ]


class RetrievalWorkflow:
    def __init__(
        self,
        dense: Any,
        bm25: Any,
        transformer: QueryTransformer,
        reranker: CrossEncoderReranker | None = None,
        default: Strategy | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.tx = transformer
        self.reranker = reranker
        self.default = default or Strategy()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="workflow")
        graph = StateGraph(State)
        graph.add_node("transform", self._transform)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("check", self._check)
        graph.add_node("correct", self._correct)
        graph.add_edge(START, "transform")
        graph.add_edge("transform", "retrieve")
        graph.add_conditional_edges("retrieve", self._after_retrieve, ["rerank", "check"])
        graph.add_edge("rerank", "check")
        graph.add_conditional_edges("check", self._after_check, ["correct", END])
        graph.add_edge("correct", "retrieve")
        self.graph = graph.compile()

    # ---------------------------------------------------------------- nodes
    def _transform(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        q, s = state["question"], state["strategy"]
        queries, hyde, errors = [q], None, []
        # A failed transform degrades to the original question; it never fails the request
        # (measured: qwen3:4b-instruct once returned an empty reply twice for a rewrite).
        try:
            if s.transform == "rewrite":
                rewritten = self.tx.rewrite(q)
                queries += [rewritten] if rewritten != q else []
            if s.transform.startswith("multi_query"):
                queries = self.tx.multi_query(q, s.multi_query_n)
        except LLMError as exc:
            errors.append(f"{s.transform}: {exc}"[:200])
        try:
            if s.transform.endswith("hyde"):
                hyde = self.tx.hyde(q)
        except LLMError as exc:
            errors.append(f"hyde: {exc}"[:200])
        return {"queries": queries, "hyde": hyde, "lists": {}, "attempts": 0, "checks": [],
                "trace": _step(state, "transform", started, transform=s.transform,
                               queries=queries[1:], hyde=(hyde or "")[:200],
                               errors=errors)}  # fmt: skip

    def _retrieve(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        s, lists = state["strategy"], dict(state.get("lists", {}))
        jobs = {}
        for q in state["queries"]:
            for name, retriever in (("dense", self.dense), ("bm25", self.bm25)):
                if f"{name}:{q}" not in lists:
                    jobs[f"{name}:{q}"] = self._pool.submit(retriever.retrieve, q, s.depth,
                                                            state["filters"])  # fmt: skip
        if state.get("hyde") and "dense:hyde" not in lists:
            jobs["dense:hyde"] = self._pool.submit(self.dense.retrieve, state["hyde"], s.depth,
                                                   state["filters"])  # fmt: skip
        for key, future in jobs.items():
            lists[key] = future.result()
        fused = rrf(lists, k=s.rrf_k, top=max(s.depth, state["k"]), mode="dense+bm25")
        return {"lists": lists, "hits": fused,
                "trace": _step(state, "retrieve", started, lists=len(lists),
                               new_lists=len(jobs), candidates=len(fused))}  # fmt: skip

    def _rerank(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        hits = state["hits"]
        logits = self.reranker.score(state["question"], [passage_text(h) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-logits[i], hits[i].rank))
        reranked = [hits[i].model_copy(update={"rank": r, "score": round(sigmoid(logits[i]), 6)})
                    for r, i in enumerate(order, start=1)]  # fmt: skip
        return {"hits": reranked, "trace": _step(state, "rerank", started, pairs=len(hits))}

    def _check(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        top = state["hits"][: state["k"]]
        result = self.tx.check(state["question"], top, use_llm=state["strategy"].grade)
        check = {"attempt": state.get("attempts", 0), "verdict": result.verdict,
                 "method": result.method, "missing": result.missing,
                 "missing_identifiers": result.missing_identifiers,
                 "supporting": result.supporting}  # fmt: skip
        return {"checks": [*state.get("checks", []), check], "verdict": result.verdict,
                "trace": _step(state, "check", started, verdict=result.verdict,
                               method=result.method)}  # fmt: skip

    def _correct(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        missing = state["checks"][-1]["missing"]
        try:
            query, error = self.tx.corrective(state["question"], missing, state["queries"]), None
        except LLMError as exc:  # no new query: the retry re-checks the same evidence
            query, error = state["question"], str(exc)[:200]
        queries = state["queries"] + ([query] if query not in state["queries"] else [])
        return {"queries": queries, "attempts": state.get("attempts", 0) + 1,
                "trace": _step(state, "correct", started, query=query, error=error)}  # fmt: skip

    # ---------------------------------------------------------------- routing
    def _after_retrieve(self, state: State) -> str:
        return "rerank" if state["strategy"].rerank and self.reranker else "check"

    def _after_check(self, state: State) -> str:
        if state["verdict"] == "sufficient":
            return END
        if state.get("attempts", 0) >= state["strategy"].max_retries:
            return END
        # The identifier check found an identifier that no passage mentions: rewording
        # the query won't make it exist, so stop instead of spending retries.
        if state["checks"][-1]["method"] == "identifiers" and state["hits"]:
            return END
        return "correct"

    # ---------------------------------------------------------------- run
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: RetrievalFilters | None = None,
        strategy: Strategy | None = None,
    ) -> RetrievalResult:
        strategy = strategy or self.default
        started = time.perf_counter()
        # 4 nodes per attempt + transform: a loop beyond max_retries raises, never spins.
        limit = 4 * (strategy.max_retries + 1) + 4
        out: State = self.graph.invoke(
            {"question": query, "k": k, "filters": filters or RetrievalFilters(),
             "strategy": strategy},
            config={"recursion_limit": limit},
        )  # fmt: skip
        hits = [h.model_copy(update={"rank": i + 1, "retriever": "advanced"})
                for i, h in enumerate(out["hits"][:k])]  # fmt: skip
        return RetrievalResult(
            query=query, retriever="advanced", k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={"verdict": out["verdict"], "attempts": out["attempts"],
                     "queries": out["queries"], "hyde": out.get("hyde"),
                     "checks": out["checks"], "strategy": strategy.model_dump(),
                     "trace": out["trace"], "recursion_limit": limit},
        )  # fmt: skip

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def build_workflow(settings: Any, service: Any) -> RetrievalWorkflow:
    """Workflow over the service's dense and BM25 retrievers and its reranker."""
    from omnirag.llm.chat import OllamaChat

    llm = OllamaChat(settings.ollama_base_url, settings.generation_model,
                     timeout_s=settings.workflow_llm_timeout_s)  # fmt: skip
    default = Strategy(
        transform=settings.workflow_transform, multi_query_n=settings.workflow_multi_query_n,
        grade=settings.workflow_grade, max_retries=settings.workflow_max_retries,
        rerank=settings.workflow_rerank, depth=settings.retrieval_fusion_depth,
        rrf_k=settings.retrieval_rrf_k,
    )  # fmt: skip
    return RetrievalWorkflow(service.retrievers["dense"], service.retrievers["bm25"],
                             QueryTransformer(llm), service.reranker, default)  # fmt: skip
