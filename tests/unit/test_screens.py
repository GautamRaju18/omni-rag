"""Phase 14 screens rendered headless (AppTest) against the real JSON shapes the API returns:
a chat turn from the real ChatService (fake model underneath) and an aggregated eval report."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_generation import _service
from test_rag_evaluation import GraphStore, _rec

from omnirag.evaluation.rag_eval import aggregate
from omnirag.frontend import views
from omnirag.frontend.api_client import ApiClient, parse_sse
from omnirag.knowledge_base import traversal

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def turn(tmp_path: Path) -> dict:
    return _service(tmp_path).ask("Why did AUR-C103 fail?")


def _report() -> dict:
    recs = [_rec("q05", True, True, False), _rec("e56", False, True, True)]
    recs[0]["failures"] = []
    bad = _rec("q15", True, False, False, "partial")
    bad["failures"] = ["retrieval_failure"]
    e3 = {"label": "Dense + BM25", **aggregate([*recs, bad])}
    e7 = {"label": "Full pipeline (default /chat)", **aggregate(recs)}
    return {
        "created": "2026-11-06T12:00:00+00:00",
        "dataset": "omnirag-eval-v1",
        "questions": 3,
        "generation_model": "qwen3:4b-instruct",
        "judge_model": "qwen3:4b-instruct",
        "prompt_version": "answer-v1",
        "wall_s": 600,
        "experiments": {"E3": e3, "E7": e7},
        "memory": {"vm_used_mb": 4597},
        "indexing": {
            "documents": 12,
            "chunks": 58,
            "ingestion_ms_total": 9000.0,
            "graph_load_ms_total": 1500.0,
        },
    }


# ---------------------------------------------------------------- pure helpers
def test_citation_cards_claims_and_badge(turn: dict) -> None:
    cards = views.citation_cards(turn)
    assert [c["number"] for c in cards] == [1] and cards[0][
        "source_file"
    ] == "aurora_safety_review.pdf"
    assert "failed at cycle 412" in cards[0]["text"] and len(cards[0]["claims"]) == 2
    assert views.claim_rows(turn["answer"])[0]["check"] == "✅ verified"
    assert views.answer_badge(turn["answer"]).startswith("✅ grounded · 2/2")
    assert views.answer_badge({"answerable": False, "refused_by": "check"}).startswith("🛑")


def test_eval_and_failure_rows_come_straight_from_the_report() -> None:
    r = _report()
    rows = views.eval_rows(r)
    assert [x["exp"] for x in rows] == ["E3", "E7"]
    assert rows[0]["answer correct"] == r["experiments"]["E3"]["generation"]["answer_correctness"]
    assert rows[0]["unanswerable refused"] == "1/1"
    assert views.failure_rows(r) == [
        {"experiment": "E3", "category": "retrieval failure", "questions": 1}
    ]
    pq = views.per_question_rows(r, "E3", {"q15": {"type": "table", "question": "Q?"}})
    assert {x["id"]: x["root cause"] for x in pq}["q15"] == "retrieval failure"
    assert len(views.eval_questions(ROOT / "data" / "eval")) == 110


def test_graph_dot_uses_generated_ids_and_escapes_labels() -> None:
    g = traversal.neighborhood(GraphStore(), entity="E-7")
    g["nodes"][0]["label"] = 'graph "quoted"'
    dot = views.graph_dot(g)
    assert 'n0 [label="graph \\"quoted\\"' in dot and "n1 -> n0" in dot
    assert "graph [" not in dot.split("\n", 1)[1]  # user text never becomes a node ID


def test_parse_sse() -> None:
    lines = [
        "event: start",
        'data: {"type": "start"}',
        "",
        "event: claim",
        'data: {"type": "claim", "n": 1}',
        "",
    ]
    assert [e["type"] for e in parse_sse(lines)] == ["start", "claim"]


# ---------------------------------------------------------------- pages
@pytest.fixture
def fake(monkeypatch, turn: dict):
    state = {"asked": []}
    conv = {"id": turn["conversation_id"], "title": turn["question"], "turns": [turn]}

    def chat_stream(self, question, **kw):
        state["asked"].append((question, kw))
        yield {"type": "start", "conversation_id": turn["conversation_id"]}
        yield {
            "type": "retrieval",
            "verdict": "sufficient",
            "route": {},
            "latency_ms": 900.0,
            "blocks": [{"number": 1, "source_file": "aurora_safety_review.pdf"}],
        }
        for c in turn["answer"]["claims"]:
            yield {"type": "claim", "claim": c}
        yield {"type": "answer", "answer": turn["answer"]}
        yield {
            "type": "done",
            "turn_id": turn["id"],
            "conversation_id": turn["conversation_id"],
            "timings": turn["timings"],
        }

    report = _report()
    doc = {"document_id": "doc_2b1c659c54266d54", "source_file": "notes.md", "file_kind": "md",
           "status": "completed", "element_count": 2, "warning_count": 0, "size_bytes": 2048,
           "duration_ms": 120.0, "warnings": []}  # fmt: skip
    patches = {
        "history": lambda self, limit=50: [{"id": conv["id"], "title": conv["title"], "turns": 1}],
        "conversation": lambda self, cid: conv,
        "chat_stream": chat_stream,
        "documents": lambda self, limit=100: {"items": [doc]},
        "document": lambda self, did: doc,
        "original": lambda self, did: b"# Notes\nCell AUR-C103 failed at cycle 412.",
        "elements": lambda self, did, limit=200: {"items": [
            {"element_id": f"{did}-e00001", "element_type": "paragraph", "origin": "original",
             "page_number": None, "confidence": 1.0, "content": "Cell AUR-C103 failed."}]},
        "ingestion_status": lambda self: {"workers": 1, "workers_alive": 1,
                                          "counts": {"queued": 0, "running": 0, "completed": 12}},
        "jobs": lambda self, limit=50: {"items": [{"job_id": "job_000000000001",
                                                  "source_file": "notes.md", "status": "completed",
                                                  "document_id": doc["document_id"],
                                                  "duration_ms": 120.0, "error": None}]},
        "graph_entities": lambda self, limit=100: [{"name": "E-7", "type": "MATERIAL",
                                                    "mentions": 9}],
        "graph_neighborhood": lambda self, entity=None, document_id=None, limit=60:
            traversal.neighborhood(GraphStore(), entity=entity, limit=limit),
        "graph_entity": lambda self, name: {"entity": {"name": name}, "relations": [
            {"type": "OWNS", "direction": "in", "other": "Marco Lindqvist", "support": "stated",
             "evidence": "Marco Lindqvist owns the E-7 qualification"}], "mentions": [1, 2],
            "same_as": []},
        "eval_runs": lambda self: [{"run_id": "rag-eval-2026-11-06", "status": "completed",
                                    "created": report["created"], "experiments": ["E3", "E7"]}],
        "eval_run": lambda self, run_id, full=False: {"run_id": run_id, "status": "completed",
                                                      "report": report},
        "analytics": lambda self: {"chat": {
            "turns": 1, "answered": 1, "grounded": 1, "refused_by": {},
            "citation_accuracy_mean": 1.0, "failure_flags": {}, "verdicts": {"sufficient": 1},
            "latency_ms": {"total_ms": {"p50": 3000.0, "p95": 3000.0, "n": 1}},
            "recent": [{"turn_id": turn["id"], "question": turn["question"]}]},
            "evaluation": {"questions": 3, "experiments": {
                k: {"label": e["label"], "performance": e["performance"]}
                for k, e in report["experiments"].items()}}},
    }  # fmt: skip
    for name, fn in patches.items():
        monkeypatch.setattr(ApiClient, name, fn)
    return state


def _screen(name: str) -> AppTest:
    return AppTest.from_string(f"from omnirag.frontend.screens import {name}\n{name}()",
                               default_timeout=30)  # fmt: skip


def test_chat_streams_claims_then_shows_the_cited_answer(fake) -> None:
    at = _screen("chat_page").run()
    assert not at.exception
    at.chat_input(key="chat_q").set_value("Why did AUR-C103 fail?").run()
    assert not at.exception
    assert fake["asked"][0][0] == "Why did AUR-C103 fail?"
    assert fake["asked"][0][1]["strategy"] is None  # the Default preset = server defaults
    assert at.session_state["chat_conv"].startswith("conv_")
    texts = " ".join(m.value for m in at.markdown)
    assert "AUR-C103 failed at cycle 412 at 45 °C [1]." in texts
    assert any(e.label.startswith("[1] aurora_safety_review.pdf") for e in at.expander)
    assert any("✅ grounded" in c.value for c in at.caption)


def test_library_graph_evaluation_and_performance_pages(fake) -> None:
    for name in ("library_page", "graph_page", "evaluation_page", "performance_page"):
        at = _screen(name).run()
        assert not at.exception, (name, at.exception)
    at = _screen("graph_page").run()
    at.segmented_control(key="g_mode").set_value("Entity").run()
    assert not at.exception and at.dataframe  # the relations table
    at = _screen("evaluation_page").run()
    assert any(s.value == "Experiments compared" for s in at.subheader)
    assert len(at.dataframe[0].value) == 2  # E3, E7
