"""Phase 9b: router, self-query, agent planner, per-document cap and the extended workflow."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from workflow_fakes import FakeLLM, Retriever, _hit

from omnirag.llm.chat import LLMError
from omnirag.retrieval.agent import AgentPlanner, ungrounded
from omnirag.retrieval.fusion import diversify
from omnirag.retrieval.models import RetrievalFilters
from omnirag.retrieval.router import LLMRouter, RuleRouter
from omnirag.retrieval.self_query import SelfQuery, descriptors, grounded
from omnirag.retrieval.transforms import QueryTransformer
from omnirag.workflows.retrieval_graph import RetrievalWorkflow, Strategy, recursion_limit

CATALOG = [
    {"document_id": "doc_s", "source_file": "aurora_safety_review.pdf",
     "title": "Aurora E-5 Failure Safety Review"},
    {"document_id": "doc_b", "source_file": "aurora_budget_2026.xlsx",
     "title": "Project Aurora Budget 2026"},
    {"document_id": "doc_w", "source_file": "whiteboard_2026-03-20.jpg",
     "title": "whiteboard_2026-03-20.jpg"},
    {"document_id": "doc_h", "source_file": "helios_aurora_s1.html",
     "title": "Aurora S1 Sodium-Ion Cell | Helios Energy Lab"},
    {"document_id": "doc_n", "source_file": "aurora_research_notes.md",
     "title": "Project Aurora: Research Notes"},
]  # fmt: skip


# ---------------------------------------------------------------- router
@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Which electrolyte replaced it, and who owns its qualification?", "multi_hop"),
        ("Who supplies the salt that the E-5 electrolyte is made with?", "multi_hop"),
        ("What is the budget of the workstream owned by the chemist?", "multi_hop"),  # first
        ("Does the rated cycle life match the target?", "multi_hop"),
        ("What does the photo on the product page show?", "image"),
        ("What share of the cathode budget has been spent?", "table"),
        ("Why did cell AUR-C103 fail?", "lookup"),
    ],
)
def test_rule_router(question: str, intent: str) -> None:
    route = RuleRouter().route(question)
    assert (route.intent, route.method) == (intent, "rules")
    assert bool(route.cues) == (intent != "lookup")
    assert route.element_types == {"table": ("table", "record"), "image": ("figure",)}.get(
        intent, ()
    )


def test_llm_router_degrades_to_lookup() -> None:
    ok = LLMRouter(FakeLLM(route={"intent": "image", "reason": "a chart"})).route("Q?")
    assert (ok.intent, ok.method) == ("image", "llm")
    odd = LLMRouter(FakeLLM(route={"intent": "poetry", "reason": ""})).route("Q?")
    down = LLMRouter(FakeLLM(route=LLMError("down"))).route("Q?")
    assert (odd.intent, odd.method) == (down.intent, down.method) == ("lookup", "llm_error")


# ---------------------------------------------------------------- self-query
def test_descriptors_are_distinctive_words() -> None:
    words = descriptors(CATALOG)
    assert words["aurora_safety_review.pdf"] == ({"e-5", "failure", "safety"}, {"review"})
    assert words["aurora_budget_2026.xlsx"][0] == {"budget"}  # "aurora" is in most titles
    assert words["whiteboard_2026-03-20.jpg"] == ({"whiteboard"}, {"photo", "image", "picture"})
    assert "page" in words["helios_aurora_s1.html"][1]  # the format names it too


def test_grounding_needs_the_document_named() -> None:
    words = descriptors(CATALOG)
    assert grounded("What does the safety review recommend?", words["aurora_safety_review.pdf"])[0]
    assert not grounded("Who reviewed it?", words["aurora_safety_review.pdf"])[0]  # generic only
    assert grounded("What is on the whiteboard?", words["whiteboard_2026-03-20.jpg"])[0]
    assert grounded("What does the photo on the S1 product page show?",
                    words["helios_aurora_s1.html"])[0]  # fmt: skip
    assert not grounded("What is the voltage of the S1?", words["helios_aurora_s1.html"])[0]
    assert not grounded("What do the notes say?", words["aurora_research_notes.md"])[0]


def test_self_query_keeps_only_named_documents() -> None:
    llm = FakeLLM(
        self_query={
            "documents": ["aurora_safety_review.pdf", "aurora_budget_2026.xlsx", "invented.pdf"]
        }
    )
    out = SelfQuery(llm, lambda: CATALOG).run("What does the safety review recommend?")
    assert [d["document_id"] for d in out.documents] == ["doc_s"]
    assert out.proposed == ["aurora_safety_review.pdf", "aurora_budget_2026.xlsx"]  # no invented
    assert out.rejected[0]["file"] == "aurora_budget_2026.xlsx"
    enum = llm.schemas[0]["properties"]["documents"]["items"]["enum"]
    assert enum == [d["source_file"] for d in CATALOG]  # the model can only name real files
    down = SelfQuery(FakeLLM(self_query=LLMError("down")), lambda: CATALOG).run("Q?")
    assert down.documents == [] and down.method == "llm_error"


# ---------------------------------------------------------------- agent
def test_agent_follows_names_it_read_never_invented_ones() -> None:
    hits = [_hit("c1", "E-5 is 1.0 M NaPF₆ in EC:DEC (1:1)")]
    q = "Who supplies the salt that the E-5 electrolyte is made with?"
    assert ungrounded("NaPF6 supplier", q, hits) is None  # NFKC: NaPF₆ = NaPF6
    assert "LiPF6" in ungrounded("LiPF6 supplier", q, hits)

    def step(reply, tried=(q,)):
        return AgentPlanner(FakeLLM(agent=reply)).next_step(q, hits, list(tried))

    ok = step({"action": "search", "query": "NaPF6 salt supplier", "reason": "bridge"})
    assert (ok.action, ok.refused) == ("search", None)
    assert step({"action": "search", "query": "LiPF6 supplier", "reason": ""}).refused
    assert step({"action": "search", "query": q.upper(), "reason": ""}).refused  # repeat
    assert step({"action": "graph", "query": "NaPF6", "reason": ""}, [q, "graph: NaPF6"]).refused
    assert step({"action": "graph", "query": "Nordic Salts", "reason": ""}).refused  # unread
    assert step({"action": "finish", "query": "", "reason": "done"}).refused is None
    assert step(LLMError("down")).action == "finish"


# ---------------------------------------------------------------- diversity
def test_diversify_caps_documents_and_backfills() -> None:
    hits = [_hit("a1", "", 1, "A"), _hit("a2", "", 2, "A"), _hit("a3", "", 3, "A"),
            _hit("b1", "", 4, "B"), _hit("a4", "", 5, "A")]  # fmt: skip
    assert [h.chunk_id for h in diversify(hits, 3, 2)] == ["a1", "a2", "b1"]
    assert [h.chunk_id for h in diversify(hits, 4, 1)] == ["a1", "b1", "a2", "a3"]  # backfill
    assert diversify(hits, 3) == hits[:3]


# ---------------------------------------------------------------- workflow
def _wf(llm, dense, bm25, graph=None, catalog=()) -> RetrievalWorkflow:
    return RetrievalWorkflow(dense, bm25, QueryTransformer(llm), graph=graph,
                             self_query=SelfQuery(llm, lambda: list(catalog)))  # fmt: skip


def test_route_and_self_query_add_preference_lists() -> None:
    llm = FakeLLM(self_query={"documents": ["whiteboard_2026-03-20.jpg"]})
    dense, bm25 = Retriever({"What is drawn on the whiteboard?": ["a"]}), Retriever({})
    wf = _wf(llm, dense, bm25, catalog=CATALOG)
    q = "What is drawn on the whiteboard?"
    soft = wf.retrieve(q, strategy=Strategy(route="rules", self_query="soft", grade=False))
    assert soft.details["route"]["intent"] == "image"
    assert soft.details["self_query"]["documents"] == ["whiteboard_2026-03-20.jpg"]
    used = sorted(repr((f.document_ids, f.element_types)) for f in dense.filters)  # threads
    assert used == sorted(repr(u) for u in [(None, None), (["doc_w"], None), (None, ["figure"])])
    dense.filters.clear()
    wf.retrieve(q, strategy=Strategy(route="off", self_query="hard", grade=False))
    assert [f.document_ids for f in dense.filters] == [["doc_w"]]  # only that document
    dense.filters.clear()
    user = RetrievalFilters(document_ids=["doc_s"])  # the caller's filter wins
    wf.retrieve(q, filters=user, strategy=Strategy(route="off", self_query="hard", grade=False))
    assert [f.document_ids for f in dense.filters] == [["doc_s"]]
    wf.close()


def test_agent_bridges_to_a_second_fact() -> None:
    q = "Who supplies the salt that the E-5 electrolyte is made with?"
    llm = FakeLLM(agent=[{"action": "search", "query": "NaPF6 supplier", "reason": "bridge"},
                         {"action": "finish", "query": "", "reason": "both facts found"}],
                  grade={"verdict": "sufficient", "supporting": [1, 2], "missing": ""})  # fmt: skip
    dense = Retriever(
        {q: ["salt"], "NaPF6 supplier": ["row"]},
        texts={"salt": "E-5: 1.0 M NaPF₆ in EC:DEC", "row": "NaPF6 | Nordic Salts"},
    )
    wf = _wf(llm, dense, Retriever({}))
    result = wf.retrieve(q, strategy=Strategy(route="rules", agent="auto", max_retries=0))
    wf.close()
    d = result.details
    assert d["agent"]["used"] and [s["action"] for s in d["agent"]["steps"]] == ["search", "finish"]
    assert d["queries"] == [q, "NaPF6 supplier"]
    assert {h.chunk_id for h in result.hits} == {"salt", "row"}
    assert [t["node"] for t in d["trace"]] == ["route", "transform", "retrieve", "plan",
                                               "retrieve", "plan", "check",
                                               "assemble"]  # fmt: skip
    assert "[1] (d.md) E-5" in [u for n, u in llm.calls if n == "agent"][0]


def test_agent_auto_skips_lookups_and_graph_tool_runs() -> None:
    llm = FakeLLM(agent={"action": "graph", "query": "E-7", "reason": ""})
    graph = Retriever({"E-7": ["g"]})
    wf = _wf(llm, Retriever({"Who owns E-7?": ["a"]}, content="E-7 note"), Retriever({}), graph)
    plain = wf.retrieve("Who owns E-7?", strategy=Strategy(route="rules", agent="auto",
                                                           grade=False))  # fmt: skip
    assert not plain.details["agent"]["used"] and llm.calls == []
    always = wf.retrieve("Who owns E-7?", strategy=Strategy(agent="always", agent_steps=1,
                                                            grade=False))  # fmt: skip
    wf.close()
    assert always.details["agent"]["graph_queries"] == ["E-7"] and graph.calls == ["E-7"]
    assert "g" in {h.chunk_id for h in always.hits}


def test_worst_case_is_bounded() -> None:
    """Agent always searches, grader never satisfied, every loop at its maximum."""
    queries = iter(f"query {i}" for i in range(100))

    class Searching(FakeLLM):
        def chat(self, system, user, *, schema=None, **kw):
            if "plan searches" in system:
                self.replies["agent"] = [{"action": "search", "query": next(queries),
                                          "reason": ""}]  # fmt: skip
            return super().chat(system, user, schema=schema, **kw)

    llm = Searching(agent={}, grade={"verdict": "partial", "supporting": [], "missing": "x"},
                    correct={"query": "q"})  # fmt: skip

    class Reranker:
        def score(self, query, passages):
            return [0.0] * len(passages)

    wf = RetrievalWorkflow(Retriever({}, content="x"), Retriever({}), QueryTransformer(llm),
                           Reranker())  # fmt: skip
    s = Strategy(agent="always", agent_steps=3, max_retries=3, rerank=True)
    agent = wf.retrieve("Q?", strategy=s).details
    retries = wf.retrieve("Q?", strategy=s.model_copy(update={"agent": "off"})).details
    wf.close()
    # The agent replaces corrective retries (ADR-0017); each path stays under the limit.
    assert len(agent["agent"]["steps"]) == 3 and agent["attempts"] == 0
    assert retries["attempts"] == 3 and not retries["agent"]["steps"]
    assert agent["recursion_limit"] == recursion_limit(s) == 4 * 4 + 2 * 3 + 6
    assert max(len(agent["trace"]), len(retries["trace"])) < recursion_limit(s)


def test_bm25_requires_only_the_users_identifiers() -> None:
    from omnirag.retrieval.sparse import lucene_query

    q = lucene_query("hard carbon supplier Aurora S1", "require", required=[])
    assert '"s1"^2' in q and "+" not in q  # copied from a passage: a boost, not a filter
    assert r'+"e\-7" "s1"^2' in lucene_query("E-7 owner S1", "require", required=["E-7"])
    assert '+"s1"' in lucene_query("S1 owner", "require")  # no list: all required (Phase 6)
    bm25 = Retriever({})
    wf = RetrievalWorkflow(Retriever({}), bm25, QueryTransformer(FakeLLM()))
    wf.retrieve("Who owns E-7?", strategy=Strategy(grade=False, max_retries=0))
    wf.close()
    assert bm25.kwargs == {"required": ["E-7"]}


def test_strategy_bounds() -> None:
    for bad in ({"agent_steps": 4}, {"max_per_document": 0}, {"route": "magic"}):
        with pytest.raises(ValidationError):
            Strategy(**bad)
