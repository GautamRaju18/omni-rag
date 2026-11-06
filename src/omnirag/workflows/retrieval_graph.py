"""Retrieval workflow in LangGraph: routing, transforms, agentic search, corrective retries.

    START → route → transform → retrieve ──(agent on, steps left)──→ plan
                                    ↑   ↑                                │ search / graph
                                    │   └────────────────────────────────┘
                                    │   retrieve / plan(finish) → [rerank] → check
                                    │                                         │ sufficient or
                                    │                                         │ no retry left → END
                                    └──────────── correct ←───────────────────┘ partial/insufficient

* **route** (Phase 9b): the router's intent (``retrieval/router.py``) and the self-query's
  documents (``retrieval/self_query.py``) become *preferences*: extra ranked lists for the
  original question, restricted to the preferred element types or documents, that join RRF
  (``self_query="hard"`` filters instead). ``multi_hop`` turns the agent on (``agent="auto"``).
* **transform** (9a): ``none`` | ``rewrite`` | ``multi_query`` | ``hyde`` | ``multi_query+hyde``.
* **retrieve**: dense and BM25 for every query (dense for the HyDE passage, graph for entities
  the agent looked up), in parallel, fused with RRF together with every list from earlier
  steps. A later step only adds evidence.
* **plan** (9b): the agent reads the current top passages and picks ``search`` / ``graph`` /
  ``finish`` (``retrieval/agent.py``); its queries may only use names it has read.
* **rerank** (optional), **check** (identifier coverage, then the LLM grader) and **correct**
  (one new query aimed at what is missing) are the 9a corrective loop.
* The returned top k may cap chunks per document (``max_per_document``, ``fusion.diversify``);
  the check and the agent see that same top k.

**Bounded by construction:** ``agent_steps`` (0–3) and ``max_retries`` (0–3) are counted in
the state, and the graph runs with a ``recursion_limit`` derived from both
(``recursion_limit()``), so a loop beyond the plan raises instead of spinning. Every node
appends to ``trace``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from omnirag.llm.chat import LLMError
from omnirag.retrieval.agent import AgentPlanner
from omnirag.retrieval.context import Compress, ContextAssembler, ContextOptions, Expand
from omnirag.retrieval.fusion import diversify, rrf
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult, RetrievedChunk
from omnirag.retrieval.reranker import CrossEncoderReranker, passage_text, sigmoid
from omnirag.retrieval.router import LLMRouter, Route, RuleRouter
from omnirag.retrieval.self_query import SelfQuery, SelfQueryResult
from omnirag.retrieval.sparse import identifiers
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
    # Phase 9b (defaults measured, ADR-0017): rule router + agent on multi_hop routes
    route: Literal["off", "rules", "llm"] = "rules"
    self_query: Literal["off", "soft", "hard"] = "off"
    agent: Literal["off", "auto", "always"] = "auto"  # auto: only for multi_hop routes
    agent_steps: int = Field(default=2, ge=0, le=3)
    max_per_document: int | None = Field(default=None, ge=1, le=20)
    # Phase 9c: the context handed to generation (ADR-0018)
    context_budget: int = Field(default=1024, ge=64, le=8192)  # bge-m3 tokens
    expand: Expand = "none"
    compress: Compress = "none"


def recursion_limit(s: Strategy) -> int:
    """route + transform, 2 nodes per agent step, 4 per attempt (retrieve, rerank, check,
    correct), assemble, plus slack. Adding a node to a loop means changing this."""
    steps = s.agent_steps if s.agent != "off" else 0
    return 4 * (s.max_retries + 1) + 2 * steps + 6


class State(TypedDict, total=False):
    question: str
    k: int
    filters: RetrievalFilters
    strategy: Strategy
    route: dict[str, Any]
    self_query: dict[str, Any] | None
    prefer: list[tuple[str, RetrievalFilters]]  # extra lists for the original question
    queries: list[str]  # every dense + BM25 query issued so far (original first)
    graph_queries: list[str]  # entities the agent looked up
    hyde: str | None
    lists: dict[str, RetrievalResult]  # every ranked list so far, fused together
    hits: list[RetrievedChunk]
    use_agent: bool
    agent_steps: int
    agent_done: bool
    steps: list[dict[str, Any]]  # the agent's decisions
    attempts: int  # corrective retries done
    checks: list[dict[str, Any]]
    verdict: str
    context: dict[str, Any] | None  # the assembled context (Phase 9c)
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
        *,
        graph: Any = None,
        self_query: SelfQuery | None = None,
        router: RuleRouter | None = None,
        llm_router: LLMRouter | None = None,
        agent: AgentPlanner | None = None,
        assembler: ContextAssembler | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.graph = graph
        self.tx = transformer
        self.reranker = reranker
        self.default = default or Strategy()
        self.rules = router or RuleRouter()
        self.llm_router = llm_router or LLMRouter(transformer.llm)
        self.self_query = self_query
        self.agent = agent or AgentPlanner(transformer.llm)
        self.assembler = assembler
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="workflow")
        graph_ = StateGraph(State)
        for name in ("route", "transform", "retrieve", "plan", "rerank", "check", "correct",
                     "assemble"):  # fmt: skip
            graph_.add_node(name, getattr(self, f"_{name}"))
        graph_.add_edge(START, "route")
        graph_.add_edge("route", "transform")
        graph_.add_edge("transform", "retrieve")
        graph_.add_conditional_edges("retrieve", self._after_retrieve, ["plan", "rerank", "check"])
        graph_.add_conditional_edges("plan", self._after_plan, ["retrieve", "rerank", "check"])
        graph_.add_edge("rerank", "check")
        graph_.add_conditional_edges("check", self._after_check, ["correct", "assemble"])
        graph_.add_edge("correct", "retrieve")
        graph_.add_edge("assemble", END)
        self.graph_app = graph_.compile()

    def _top(self, state: State) -> list[RetrievedChunk]:
        return diversify(state["hits"], state["k"], state["strategy"].max_per_document)

    # ---------------------------------------------------------------- nodes
    def _route(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        q, s, filters = state["question"], state["strategy"], state["filters"]
        if s.route == "rules":
            route = self.rules.route(q)
        elif s.route == "llm":
            route = self.llm_router.route(q)
        else:
            route = Route("lookup", "off")
        sq: SelfQueryResult | None = None
        prefer: list[tuple[str, RetrievalFilters]] = []
        if s.self_query != "off":
            sq = self.self_query.run(q) if self.self_query else SelfQueryResult(method="off")
            ids = [
                d["document_id"]
                for d in sq.documents
                if not filters.document_ids or d["document_id"] in filters.document_ids
            ]
            if ids and s.self_query == "hard":
                filters = filters.model_copy(update={"document_ids": ids})
            elif ids:
                prefer.append(("documents", filters.model_copy(update={"document_ids": ids})))
        if route.element_types and not filters.element_types:  # the caller's filter wins
            prefer.append((route.intent, filters.model_copy(
                update={"element_types": list(route.element_types)})))  # fmt: skip
        use_agent = s.agent_steps > 0 and (
            s.agent == "always" or (s.agent == "auto" and route.intent == "multi_hop")
        )
        return {"route": route.as_dict(), "self_query": sq.as_dict() if sq else None,
                "filters": filters, "prefer": prefer, "use_agent": use_agent,
                "agent_steps": 0, "agent_done": not use_agent, "graph_queries": [],
                "steps": [],
                "trace": _step(state, "route", started, intent=route.intent,
                               method=route.method, prefer=[n for n, _ in prefer],
                               documents=sq.as_dict()["documents"] if sq else [],
                               agent=use_agent)}  # fmt: skip

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
        s, lists, filters = state["strategy"], dict(state.get("lists", {})), state["filters"]
        variants = [("", filters), *state.get("prefer", [])]
        # BM25 requires the user's identifiers only; others in LLM-written queries are boosts.
        required = identifiers(state["question"])
        jobs = {}
        for q in state["queries"]:
            # Preferences apply to the original question's lists only.
            for variant, f in variants if q == state["question"] else variants[:1]:
                for name, retriever in (("dense", self.dense), ("bm25", self.bm25)):
                    key = f"{name}:{variant}:{q}" if variant else f"{name}:{q}"
                    if key in lists:
                        continue
                    kw = {"required": required} if name == "bm25" else {}
                    jobs[key] = self._pool.submit(retriever.retrieve, q, s.depth, f, **kw)
        if state.get("hyde") and "dense:hyde" not in lists:
            jobs["dense:hyde"] = self._pool.submit(self.dense.retrieve, state["hyde"], s.depth,
                                                   filters)  # fmt: skip
        if self.graph is not None:
            for entity in state.get("graph_queries", []):
                if f"graph:{entity}" not in lists:
                    jobs[f"graph:{entity}"] = self._pool.submit(self.graph.retrieve, entity,
                                                                s.depth, filters)  # fmt: skip
        for key, future in jobs.items():
            lists[key] = future.result()
        fused = rrf(lists, k=s.rrf_k, top=max(s.depth, state["k"]), mode="dense+bm25")
        return {"lists": lists, "hits": fused,
                "trace": _step(state, "retrieve", started, lists=len(lists),
                               new_lists=len(jobs), candidates=len(fused))}  # fmt: skip

    def _plan(self, state: State) -> dict[str, Any]:
        started = time.perf_counter()
        tried = [*state["queries"], *(f"graph: {e}" for e in state.get("graph_queries", []))]
        step = self.agent.next_step(state["question"], self._top(state), tried)
        if step.action == "graph" and self.graph is None and not step.refused:
            step.refused = "graph retrieval is not available"
        out: dict[str, Any] = {"agent_steps": state["agent_steps"] + 1,
                               "steps": [*state.get("steps", []), step.as_dict()]}  # fmt: skip
        if step.refused or step.action == "finish":
            out["agent_done"] = True
        elif step.action == "search":
            out["queries"] = [*state["queries"], step.query]
        else:
            out["graph_queries"] = [*state.get("graph_queries", []), step.query]
        out["trace"] = _step(state, "plan", started, action=step.action, query=step.query,
                             refused=step.refused)  # fmt: skip
        return out

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
        result = self.tx.check(state["question"], self._top(state),
                               use_llm=state["strategy"].grade)  # fmt: skip
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

    def _assemble(self, state: State) -> dict[str, Any]:
        """The context for generation: budgeted, optionally expanded / compressed (9c)."""
        started = time.perf_counter()
        s = state["strategy"]
        if self.assembler is None:
            return {"context": None, "trace": _step(state, "assemble", started, skipped=True)}
        options = ContextOptions(budget_tokens=s.context_budget, expand=s.expand,
                                 compress=s.compress, max_candidates=s.depth)  # fmt: skip
        # The diversified ranking first (the order the check judged), then the other candidates.
        top = self._top(state)
        seen = {t.chunk_id for t in top}
        ranked = [
            h.model_copy(update={"rank": i})
            for i, h in enumerate([*top, *(h for h in state["hits"] if h.chunk_id not in seen)], 1)
        ]
        ctx = self.assembler.assemble(state["question"], ranked, options)
        out = {**ctx.model_dump(mode="json"), "rendered": ctx.render()}
        return {"context": out,
                "trace": _step(state, "assemble", started, blocks=len(ctx.blocks),
                               tokens=ctx.tokens, budget=ctx.budget_tokens,
                               expand=s.expand, compress=s.compress)}  # fmt: skip

    # ---------------------------------------------------------------- routing
    def _after_retrieve(self, state: State) -> str:
        if not state.get("agent_done", True) and \
                state.get("agent_steps", 0) < state["strategy"].agent_steps:  # fmt: skip
            return "plan"
        return "rerank" if state["strategy"].rerank and self.reranker else "check"

    def _after_plan(self, state: State) -> str:
        if state["agent_done"]:
            return "rerank" if state["strategy"].rerank and self.reranker else "check"
        return "retrieve"

    def _after_check(self, state: State) -> str:
        if state["verdict"] == "sufficient":
            return "assemble"
        if state.get("attempts", 0) >= state["strategy"].max_retries:
            return "assemble"
        # The agent already searched with what the passages say; the corrective rewrite (which
        # doesn't read them) only dilutes RRF. Measured 2026-11-06: after the agent found r02's
        # supplier row, a corrective retry pushed it back out of the top 5 (ADR-0017).
        if state.get("use_agent"):
            return "assemble"
        # The identifier check found an identifier that no passage mentions: rewording
        # the query won't make it exist, so stop instead of spending retries.
        if state["checks"][-1]["method"] == "identifiers" and state["hits"]:
            return "assemble"
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
        limit = recursion_limit(strategy)
        out: State = self.graph_app.invoke(
            {"question": query, "k": k, "filters": filters or RetrievalFilters(),
             "strategy": strategy},
            config={"recursion_limit": limit},
        )  # fmt: skip
        hits = [h.model_copy(update={"rank": i + 1, "retriever": "advanced"})
                for i, h in enumerate(self._top(out))]  # fmt: skip
        return RetrievalResult(
            query=query, retriever="advanced", k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={"verdict": out["verdict"], "attempts": out["attempts"],
                     "queries": out["queries"], "hyde": out.get("hyde"),
                     "route": out["route"], "self_query": out["self_query"],
                     "agent": {"used": out["use_agent"], "steps": out["steps"],
                               "graph_queries": out["graph_queries"]},
                     "checks": out["checks"], "strategy": strategy.model_dump(),
                     "context": out.get("context"),
                     "trace": out["trace"], "recursion_limit": limit,
                     # what each search found before fusion (top 10 per list)
                     "lists": {key: {"latency_ms": r.latency_ms,
                                     "embed_ms": r.details.get("embed_ms"),
                                     "hits": [[h.chunk_id, h.rank, round(h.score, 4),
                                               h.source_file] for h in r.hits[:10]]}
                               for key, r in out["lists"].items()}},
        )  # fmt: skip

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def build_workflow(settings: Any, service: Any) -> RetrievalWorkflow:
    """Workflow over the service's dense, BM25 and graph retrievers and its reranker."""
    from omnirag.llm.chat import OllamaChat
    from omnirag.retrieval.compression import Compressor
    from omnirag.retrieval.self_query import catalog_from

    llm = OllamaChat(settings.ollama_base_url, settings.generation_model,
                     timeout_s=settings.workflow_llm_timeout_s)  # fmt: skip
    default = Strategy(
        transform=settings.workflow_transform, multi_query_n=settings.workflow_multi_query_n,
        grade=settings.workflow_grade, max_retries=settings.workflow_max_retries,
        rerank=settings.workflow_rerank, depth=settings.retrieval_fusion_depth,
        rrf_k=settings.retrieval_rrf_k, route=settings.workflow_route,
        self_query=settings.workflow_self_query, agent=settings.workflow_agent,
        agent_steps=settings.workflow_agent_steps,
        max_per_document=settings.workflow_max_per_document,
        context_budget=settings.context_budget_tokens, expand=settings.context_expand,
        compress=settings.context_compress,
    )  # fmt: skip
    assembler = ContextAssembler(_LazyCount(settings), service.store,
                                 Compressor(embedder=service.retrievers["dense"].embedder,
                                            llm=llm))  # fmt: skip
    return RetrievalWorkflow(
        service.retrievers["dense"], service.retrievers["bm25"], QueryTransformer(llm),
        service.reranker, default, graph=service.retrievers.get("graph"),
        self_query=SelfQuery(llm, catalog_from(service.store)), assembler=assembler,
    )  # fmt: skip


class _LazyCount:
    """bge-m3 token counter, loaded on the first assembled context (not at API start-up)."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._count: Any = None

    def __call__(self, text: str) -> int:
        if self._count is None:
            from omnirag.chunking import load_tokenizer

            self._count = load_tokenizer(self.settings).count
        return self._count(text)
