"""Dashboard: view helpers, the API client, and every page rendered headless (AppTest)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from workflow_fakes import FakeLLM, Retriever

from omnirag.frontend import views
from omnirag.frontend.api_client import ApiClient, ApiError
from omnirag.retrieval.transforms import QueryTransformer
from omnirag.workflows.retrieval_graph import RetrievalWorkflow, Strategy

ROOT = Path(__file__).resolve().parents[2]
Q = "Who supplies the salt that the E-5 electrolyte is made with?"


def _agent_run() -> dict:
    """A real workflow run (fakes underneath): the exact JSON shape the API returns."""
    llm = FakeLLM(agent=[{"action": "search", "query": "NaPF6 supplier", "reason": "bridge"},
                         {"action": "finish", "query": "", "reason": "done"}],
                  grade={"verdict": "sufficient", "supporting": [1], "missing": ""})  # fmt: skip
    dense = Retriever(
        {Q: ["salt"], "NaPF6 supplier": ["row"]},
        texts={"salt": "E-5: 1.0 M NaPF₆ in EC:DEC", "row": "NaPF6 | Nordic Salts"},
    )
    wf = RetrievalWorkflow(dense, Retriever({Q: ["salt"]}), QueryTransformer(llm))
    result = wf.retrieve(Q, strategy=Strategy(route="rules", agent="auto", max_retries=0))
    wf.close()
    return result.model_dump(mode="json")


# ---------------------------------------------------------------- views
def test_pipeline_dot_marks_the_path_taken() -> None:
    trace = _agent_run()["details"]["trace"]
    dot = views.pipeline_dot(trace)
    assert 'retrieve [label="retrieve\\n×2' in dot and 'plan [label="plan\\n×2' in dot
    assert 'rerank [label="rerank" style="rounded,dashed"]' in dot  # didn't run
    assert f'plan -> retrieve [color="{views.RAN}" penwidth=2]' in dot  # the agent's loop
    assert 'correct -> retrieve [style="dashed"]' in dot


def test_timeline_and_list_views() -> None:
    d = _agent_run()["details"]
    rows = views.timeline_rows(d["trace"])
    assert [r["node"] for r in rows] == ["route", "transform", "retrieve", "plan", "retrieve",
                                         "plan", "check"]  # fmt: skip
    assert all(a["end_ms"] == b["start_ms"] for a, b in zip(rows, rows[1:], strict=False))
    assert "intent multi_hop via rules" in rows[0]["what happened"]
    assert rows[3]["what happened"] == "search NaPF6 supplier"
    lists = views.list_rows(d["lists"], Q)
    assert {r["query"] for r in lists} == {"(the question)", "NaPF6 supplier"}
    assert {r["top 3"] for r in lists if r["hits"]} == {"d.md"}  # file names, not chunk IDs
    assert views.parse_list_key("bm25:image:Q?") == {"retriever": "bm25", "variant": "image",
                                                     "query": "Q?"}  # fmt: skip
    assert views.parse_list_key("dense:hyde")["variant"] == "hyde"
    assert views.parse_list_key("graph:E-7") == {"retriever": "graph", "variant": "entity",
                                                 "query": "E-7"}  # fmt: skip


def test_contributions_explain_rrf() -> None:
    hits = _agent_run()["hits"]
    rows = {r["chunk"]: r for r in views.contribution_rows(hits)}
    assert rows["salt"]["in lists"] == 2  # dense and BM25 of the question
    assert rows["salt"]["rrf"] == round(2 / 61, 4)
    assert rows["row"]["rrf"] == round(1 / 61, 4) and rows["row"]["rank"] == 2


def test_compare_overlap_and_benchmarks() -> None:
    def res(*ids):
        return {"hits": [{"chunk_id": c, "rank": i + 1, "source_file": "f", "content": c,
                          "element_type": "table"} for i, c in enumerate(ids)]}  # fmt: skip

    results = {"dense": res("a", "b"), "bm25": res("b", "c")}
    rows = views.compare_rows(results)
    assert [(r["chunk"], r.get("dense"), r.get("bm25")) for r in rows] == [
        ("a", 1, None), ("b", 2, 1), ("c", None, 2)]  # fmt: skip
    assert views.overlap(results)[0] == {"mode": "dense", "dense": 2, "bm25": 1}
    data = views.load_json(ROOT / "docs" / "benchmarks" / "workflow-2026-11-05.json")
    base = next(r for r in views.benchmark_rows(data) if r["config"] == "base")
    assert base["mrr"] == 0.928  # read from the measured file, not recomputed
    assert len(views.example_questions(ROOT / "data" / "eval")) == 48


def test_api_client_reports_the_error_envelope() -> None:
    def handler(request):
        return httpx.Response(422, json={"error": {"code": "validation", "message": "bad k"}})

    client = ApiClient(
        "http://api",
        client=httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ApiError, match="HTTP 422: bad k"):
        client.search("q", "dense", 5, {})


# ---------------------------------------------------------------- pages, headless
@pytest.fixture
def fake_api(monkeypatch):
    run = _agent_run()
    calls = []

    def search(self, query, mode, k, filters):
        calls.append(("search", mode))
        return {**run, "retriever": mode, "details": {"lucene": "x"}}

    def advanced(self, query, k, filters, strategy):
        calls.append(("advanced", strategy))
        return run

    health = {"status": "ok", "app": "omnirag", "version": "0", "environment": "test",
              "components": [{"name": "neo4j", "status": "ok", "latency_ms": 3.0},
                             {"name": "ollama", "status": "ok", "latency_ms": 9.0}]}  # fmt: skip
    monkeypatch.setattr(ApiClient, "health", lambda self: health)
    monkeypatch.setattr(ApiClient, "graph_stats", lambda self: {
        "nodes": {"Document": 12, "Chunk": 58}, "relationships": {"RELATED_TO": 40},
        "chunks_with_vectors": 58, "vector_dim": 1024})  # fmt: skip
    monkeypatch.setattr(ApiClient, "documents", lambda self, limit=100: {"items": []})
    monkeypatch.setattr(ApiClient, "search", search)
    monkeypatch.setattr(ApiClient, "advanced", advanced)
    return calls


def _page(name: str) -> AppTest:
    return AppTest.from_string(f"from omnirag.frontend.pages import {name}\n{name}()",
                               default_timeout=30)  # fmt: skip


def test_overview_page(fake_api) -> None:
    at = _page("overview").run()
    assert not at.exception
    assert [m.label for m in at.metric][:3] == ["Overall", "neo4j · 3 ms", "ollama · 9 ms"]
    assert any(m.label == "Documents" and m.value == "12" for m in at.metric)


def test_workflow_inspector_runs_and_explains_every_node(fake_api) -> None:
    at = _page("workflow_inspector").run()
    at.button(key="preset_No LLM calls").click().run()
    at.button(key="preset_Default").click().run()
    at.button(key="wf_run").click().run()
    assert not at.exception
    strategy = fake_api[-1][1]
    assert (strategy["route"], strategy["agent"], strategy["max_retries"]) == ("rules", "auto", 1)
    steps = [e.label for e in at.expander if e.label[:2] in {f"{i}." for i in range(1, 10)}]
    assert len(steps) == 7 and "search NaPF6 supplier" in steps[3]
    assert any(m.label == "Verdict" and m.value == "sufficient" for m in at.metric)


def test_retrieval_lab_runs_each_mode(fake_api) -> None:
    at = _page("retrieval_lab").run()
    at.button(key="lab_run").click().run()
    assert not at.exception
    assert [mode for kind, mode in fake_api if kind == "search"] == ["dense", "bm25", "graph",
                                                                     "dense+bm25"]  # fmt: skip


def test_benchmarks_page_reads_measured_files() -> None:
    at = _page("benchmarks").run()
    assert not at.exception and at.dataframe


def test_dot_node_names_are_not_dot_keywords() -> None:
    """A node called ``graph`` made the whole system diagram render blank (found in the browser)."""
    import re

    from omnirag.frontend.pages import SYSTEM_DOT

    for dot in (SYSTEM_DOT, views.pipeline_dot(_agent_run()["details"]["trace"])):
        body = dot.split("\n", 1)[1]
        assert not re.search(r"(^|;|->)\s*(graph|subgraph|digraph|strict)\b", body, re.M)
