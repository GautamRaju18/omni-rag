"""The retrieval workflow against the real graph and Ollama (read-only on the main graph).

Needs: Neo4j with the sample corpus loaded, Ollama with qwen3:4b-instruct and bge-m3.
"""

from __future__ import annotations

import pytest

from omnirag.core.config import Settings
from omnirag.retrieval import build_retrieval
from omnirag.workflows.retrieval_graph import Strategy, build_workflow

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workflow():
    settings = Settings(rerank_enabled=False)
    service = build_retrieval(settings)
    wf = build_workflow(settings, service)
    yield wf
    wf.close()
    service.close()


def test_unknown_identifier_is_insufficient_without_a_model_call(workflow) -> None:
    calls = workflow.tx.llm.stats["calls"]
    result = workflow.retrieve("What capacity retention did cell AUR-C109 reach?",
                               strategy=Strategy(grade=True, max_retries=2))  # fmt: skip
    d = result.details
    assert d["verdict"] == "insufficient" and d["attempts"] == 0
    assert d["checks"][0]["missing_identifiers"] == ["AUR-C109"]
    assert workflow.tx.llm.stats["calls"] == calls  # identifier check only
    assert len(result.hits) == 5  # retrieval still returns its best guesses


def test_agent_bridges_through_a_name_it_read(workflow) -> None:
    """r02: the salt is only named in the first passage (NaPF₆); its supplier is a budget row."""
    result = workflow.retrieve(
        "Who supplies the salt that the E-5 electrolyte is made with?",
        strategy=Strategy(route="rules", agent="auto", grade=False, max_retries=0),
    )
    d = result.details
    assert d["route"]["intent"] == "multi_hop" and d["agent"]["used"]
    searches = [s["query"] for s in d["agent"]["steps"] if s["action"] == "search"]
    assert any("napf" in q.lower() for q in searches)  # a name from a passage, not the question
    assert any("Nordic Salts AB" in h.content for h in result.hits)
    assert any("NaPF₆ in EC:DEC" in h.content for h in result.hits)


def test_answerable_question_is_graded_by_the_model(workflow) -> None:
    result = workflow.retrieve("Why did cell AUR-C103 fail?",
                               strategy=Strategy(grade=True, max_retries=1))  # fmt: skip
    d = result.details
    assert d["checks"][0]["method"] == "llm"
    assert d["verdict"] == "sufficient" and d["attempts"] == 0
    assert any("failed at cycle 412" in h.content for h in result.hits)
    assert [t["node"] for t in d["trace"]] == ["route", "transform", "retrieve", "check",
                                               "assemble"]  # fmt: skip
    ctx = d["context"]
    assert ctx["tokens"] <= ctx["budget_tokens"] and ctx["blocks"]
    assert any("failed at cycle 412" in b["text"] for b in ctx["blocks"])  # evidence kept
    assert ctx["rendered"].startswith("[1] ")
