"""Query transforms, the context check, the LangGraph workflow and the chat client (all fakes)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from workflow_fakes import FakeLLM, Retriever, _hit

from omnirag.api.dependencies import get_workflow
from omnirag.api.main import create_app
from omnirag.core.config import Settings
from omnirag.llm.chat import LLMError, OllamaChat
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult
from omnirag.retrieval.transforms import QueryTransformer
from omnirag.workflows.retrieval_graph import RetrievalWorkflow, Strategy


# ---------------------------------------------------------------- transforms
def test_rewrite_keeps_and_never_invents_identifiers() -> None:
    q = "Why did cell AUR-C103 fail?"
    assert QueryTransformer(FakeLLM(rewrite={"query": "AUR-C103 failure cause"})).rewrite(q) \
        == "AUR-C103 failure cause"  # fmt: skip
    for bad in ("cell failure cause", "AUR-C103 failure E-7 M2", ""):  # dropped / invented / empty
        assert QueryTransformer(FakeLLM(rewrite={"query": bad})).rewrite(q) == q


def test_multi_query_keeps_the_original_first() -> None:
    llm = FakeLLM(multi={"queries": ["E-7 owner", "e-7 owner", "Who owns XB-12?", "E-7 cells"]})
    out = QueryTransformer(llm).multi_query("Who owns E-7?", n=4)
    assert out == ["Who owns E-7?", "E-7 owner", "E-7 cells"]  # dup + invented id dropped
    assert "Write 4 queries" in llm.calls[0][1]


def test_hyde_falls_back_on_empty_text() -> None:
    assert QueryTransformer(FakeLLM(hyde="ok")).hyde("Question?") == "Question?"
    passage = "E-7 replaces E-5 at elevated temperature."
    assert QueryTransformer(FakeLLM(hyde=passage)).hyde("Q?") == passage


def test_check_identifiers_first_then_llm() -> None:
    hits = [_hit("c1", "AUR-C105 reached 89.6%"), _hit("c2", "AUR-C106 reached 90.2%")]
    llm = FakeLLM(grade={"verdict": "sufficient", "supporting": [2, 7, "x"], "missing": ""})
    tx = QueryTransformer(llm)
    missing = tx.check("Retention of AUR-C109?", hits)
    assert (missing.verdict, missing.method) == ("insufficient", "identifiers")
    assert missing.missing_identifiers == ["AUR-C109"] and llm.calls == []  # no model call
    assert tx.check("Anything?", []).verdict == "insufficient"
    ok = tx.check("Retention of AUR-C105?", hits)
    assert (ok.verdict, ok.method, ok.supporting) == ("sufficient", "llm", ["c2"])
    assert "[1] (d.md) AUR-C105" in llm.calls[0][1]
    assert tx.check("Retention of AUR-C105?", hits, use_llm=False).method == "none"


def test_check_reports_a_failed_grader() -> None:
    tx = QueryTransformer(FakeLLM(grade=LLMError("unusable JSON")))
    result = tx.check("Q?", [_hit("c1", "text")])
    assert (result.verdict, result.method) == ("partial", "llm_error")


# ---------------------------------------------------------------- workflow
def _workflow(llm, dense, bm25, reranker=None) -> RetrievalWorkflow:
    return RetrievalWorkflow(dense, bm25, QueryTransformer(llm), reranker)


def test_sufficient_first_try_stops() -> None:
    llm = FakeLLM(grade={"verdict": "sufficient", "supporting": [1], "missing": ""})
    wf = _workflow(llm, Retriever({"Q?": ["a", "b"]}), Retriever({"Q?": ["b"]}))
    result = wf.retrieve("Q?", k=2, strategy=Strategy(max_retries=2))
    wf.close()
    assert [h.chunk_id for h in result.hits] == ["b", "a"]  # RRF: b is in both lists
    assert result.retriever == "advanced" and result.details["attempts"] == 0
    assert [t["node"] for t in result.details["trace"]] == ["route", "transform", "retrieve",
                                                            "check"]  # fmt: skip
    assert result.details["verdict"] == "sufficient"


def test_corrective_retry_adds_evidence_and_is_bounded() -> None:
    llm = FakeLLM(
        grade={"verdict": "partial", "supporting": [], "missing": "the owner"},
        correct=[{"query": "owner of E-7"}, {"query": "E-7 qualification owner"}],
    )
    dense = Retriever({"Who owns E-7?": ["a"], "owner of E-7": ["owner"],
                       "E-7 qualification owner": ["x"]}, content="E-7 note")  # fmt: skip
    wf = _workflow(llm, dense, Retriever({}))
    result = wf.retrieve("Who owns E-7?", k=3, strategy=Strategy(max_retries=2))
    wf.close()
    d = result.details
    assert d["attempts"] == 2 and d["verdict"] == "partial"  # gave up after 2 retries
    assert d["queries"] == ["Who owns E-7?", "owner of E-7", "E-7 qualification owner"]
    assert {h.chunk_id for h in result.hits} == {"a", "owner", "x"}  # retries only add lists
    assert dense.calls.count("Who owns E-7?") == 1  # earlier lists are reused, not re-run
    nodes = [t["node"] for t in d["trace"]]
    assert nodes.count("check") == 3 and nodes.count("correct") == 2
    assert d["recursion_limit"] == 4 * 3 + 2 * 2 + 5  # 2 agent steps could run (auto)
    assert "Missing: the owner" in [u for n, u in llm.calls if n == "correct"][0]


def test_missing_identifier_stops_without_retrying() -> None:
    llm = FakeLLM(grade={"verdict": "sufficient", "supporting": [], "missing": ""})
    wf = _workflow(llm, Retriever({"AUR-C109 retention?": ["a"]}), Retriever({}))
    result = wf.retrieve("AUR-C109 retention?", strategy=Strategy(max_retries=3))
    wf.close()
    assert result.details["verdict"] == "insufficient" and result.details["attempts"] == 0
    assert llm.calls == []


def test_transforms_and_rerank_nodes() -> None:
    llm = FakeLLM(multi={"queries": ["alt query"]}, hyde="hypothetical passage text",
                  grade={"verdict": "sufficient", "supporting": [1], "missing": ""})  # fmt: skip
    dense = Retriever({"Q?": ["a"], "alt query": ["b"], "hypothetical passage text": ["c"]})

    class Reranker:
        def score(self, query, passages):
            return [3.0 if p.endswith(" c") else 0.0 for p in passages]

    wf = _workflow(llm, dense, Retriever({}), Reranker())
    result = wf.retrieve("Q?", k=3, strategy=Strategy(transform="multi_query+hyde", rerank=True,
                                                      grade=False))  # fmt: skip
    wf.close()
    assert result.details["queries"] == ["Q?", "alt query"]
    assert result.details["hyde"] == "hypothetical passage text"
    assert "hypothetical passage text" in dense.calls  # dense only, never BM25
    assert result.hits[0].chunk_id == "c"  # the reranker's choice
    assert [t["node"] for t in result.details["trace"]] == ["route", "transform", "retrieve",
                                                            "rerank", "check"]  # fmt: skip


# ---------------------------------------------------------------- chat client
def _chat(handler) -> OllamaChat:
    client = httpx.Client(base_url="http://o", transport=httpx.MockTransport(handler))
    return OllamaChat("http://o", "m", client=client, backoff_s=0)


def test_chat_structured_retry_and_errors() -> None:
    replies = iter(
        [
            httpx.Response(503),
            httpx.Response(
                200,
                json={
                    "message": {"content": '{"query": "x"}'},
                    "eval_count": 3,
                    "prompt_eval_count": 9,
                },
            ),
        ]
    )
    result = _chat(lambda r: next(replies)).chat("s", "u", schema={"type": "object"})
    assert result.data == {"query": "x"} and result.eval_tokens == 3
    sent = []

    def broken(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"query": '}})

    with pytest.raises(LLMError, match="unusable JSON twice"):
        _chat(broken).chat("s", "u", schema={"type": "object"})
    assert len(sent) == 2 and sent[0]["options"]["temperature"] == 0 and "format" in sent[0]
    with pytest.raises(LLMError, match="not found"):
        _chat(lambda r: httpx.Response(404)).chat("s", "u")


# ---------------------------------------------------------------- API
class FakeWorkflow:
    def retrieve(self, query, k, filters, strategy) -> RetrievalResult:
        if query == "boom":
            raise LLMError("ollama down")
        return RetrievalResult(query=query, retriever="advanced", k=k, hits=[], latency_ms=1.0,
                               details={"strategy": strategy.model_dump() if strategy else None,
                                        "filters": filters.model_dump()})  # fmt: skip


async def _client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(Settings(_env_file=None, env="test", log_json=False, neo4j_password="x"))
    app.dependency_overrides[get_workflow] = lambda: FakeWorkflow()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:  # fmt: skip
        yield client


async def test_advanced_endpoint() -> None:
    async for client in _client():
        r = await client.post(
            "/api/v1/retrieval/advanced",
            json={
                "query": "Who owns E-7?",
                "strategy": {"transform": "multi_query", "max_retries": 2},
            },
        )
        assert r.status_code == 200 and r.json()["retriever"] == "advanced"
        assert r.json()["details"]["strategy"]["transform"] == "multi_query"
        bad = await client.post(
            "/api/v1/retrieval/advanced", json={"query": "x", "strategy": {"max_retries": 9}}
        )
        assert bad.status_code == 422  # retries are bounded by the API too
        down = await client.post("/api/v1/retrieval/advanced", json={"query": "boom"})
        assert down.status_code == 503
        default = await client.post("/api/v1/retrieval/advanced", json={"query": "x"})
        assert default.json()["details"]["strategy"] is None  # workflow uses its defaults
        assert RetrievalFilters().latest_only


def test_failed_transforms_degrade_to_the_original_question() -> None:
    llm = FakeLLM(multi=LLMError("empty reply"), hyde=LLMError("empty reply"),
                  grade={"verdict": "partial", "supporting": [], "missing": "x"},
                  correct=LLMError("empty reply"))  # fmt: skip
    wf = _workflow(llm, Retriever({"Q?": ["a"]}), Retriever({}))
    result = wf.retrieve("Q?", strategy=Strategy(transform="multi_query+hyde", max_retries=1))
    wf.close()
    d = result.details
    assert d["queries"] == ["Q?"] and d["hyde"] is None and d["attempts"] == 1
    assert [h.chunk_id for h in result.hits] == ["a"]
    errors = d["trace"][1]["errors"]
    assert len(errors) == 2 and "empty reply" in errors[0]
    assert "empty reply" in [t for t in d["trace"] if t["node"] == "correct"][0]["error"]
