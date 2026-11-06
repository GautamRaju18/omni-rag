"""Phase 12-13: the 110-question dataset, failure analysis, aggregation, judge, LangSmith
redaction, tracing switch, background evaluation runs, analytics and their API."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from omnirag.api.dependencies import get_chat, get_evaluations, get_retrieval
from omnirag.api.main import create_app
from omnirag.core.config import Settings
from omnirag.evaluation import langsmith_sync
from omnirag.evaluation.dataset import EvalDataset, EvalQuestion, resolve
from omnirag.evaluation.failures import FailureClassifier, answer_includes, live_flags, summarize
from omnirag.evaluation.judge import JUDGE_SCHEMA, judge
from omnirag.evaluation.rag_eval import EXPERIMENTS, aggregate, rescore
from omnirag.evaluation.runs import EvaluationRuns, RunConflictError
from omnirag.knowledge_base import traversal
from omnirag.llm.chat import ChatResult, LLMError
from omnirag.observability.analytics import evaluation_summary, turn_analytics
from omnirag.observability.tracing import configure_tracing

ROOT = Path(__file__).parents[2]
V1 = EvalDataset.load(ROOT / "data" / "eval" / "omnirag_eval_v1.json")


# ---------------------------------------------------------------- dataset
def test_v1_dataset_has_100_plus_questions_of_every_kind() -> None:
    ids = [q.id for q in V1.questions]
    assert len(ids) == len(set(ids)) == 110
    assert {q.type for q in V1.questions} == {
        "factual",
        "multi_hop",
        "cross_document",
        "table",
        "image",
        "comparative",
        "keyword",
        "unanswerable",
    }
    for q in V1.questions:
        if q.answerable:
            assert q.answer and q.must_include and q.expected_sources, q.id
            assert answer_includes(q.answer, q.must_include)[0], q.id  # terms match the answer
        else:
            assert q.answer is None and not q.facts and not q.must_include, q.id
    assert sum(not q.answerable for q in V1.questions) == 10


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="OCR evidence needs tesseract")
def test_v1_quotes_exist_in_the_samples(tmp_path: Path) -> None:
    """Every original quote outside the Docling-only PDF resolves (e7 and generated image
    descriptions are checked against the live store: tests/integration)."""
    from test_evaluation import LAYOUT_FILES, _ingest_samples

    by_file = _ingest_samples(tmp_path, "pymupdf")
    checked = 0
    for q in V1.questions:
        if q.evidence_origin == "generated":
            continue
        facts = [[a for a in fact if a.file not in LAYOUT_FILES] for fact in q.facts]
        facts = [f for f in facts if f]
        if facts:
            assert all(resolve(q.model_copy(update={"facts": facts}), by_file)), q.id
            checked += sum(len(f) for f in facts)
    assert checked >= 150


def test_answer_terms_are_normalised_and_alternatives_count() -> None:
    assert answer_includes("The pouch swelled by 14 %; it vented WITHOUT flame.",
                           ["14", ["no flame", "without flame"]]) == (True, 1.0)  # fmt: skip
    assert answer_includes("E-7 at 2.1%", ["E-7", "3.2"]) == (False, 0.5)


# ---------------------------------------------------------------- failure analysis
Q = EvalQuestion.model_validate({"id": "x", "type": "factual", "question": "?", "answer": "a",
                                 "facts": [[{"file": "a.md", "text": "failed at cycle 412"}]],
                                 "must_include": ["412"]})  # fmt: skip
EL = {"a.md": [SimpleNamespace(content="Cell AUR-C103 failed at cycle 412.")],
      "scan.jpg": []}  # fmt: skip
CH = [SimpleNamespace(chunk_id="c1", content="Cell AUR-C103 failed at cycle 412."),
      SimpleNamespace(chunk_id="c2", content="Unrelated.")]  # fmt: skip
GOOD = {"answerable": True, "text": "It failed at cycle 412 [1].",
        "claims": [{"status": "verified"}]}  # fmt: skip


def _hit(cid: str, first_stage: int | None = None) -> SimpleNamespace:
    content = next(c.content for c in CH if c.chunk_id == cid)
    sources = {"rerank": {"first_stage_rank": first_stage}} if first_stage else {}
    return SimpleNamespace(chunk_id=cid, content=content, sources=sources)


def test_failures_follow_the_pipeline_and_blame_the_first_stage() -> None:
    fc = FailureClassifier(EL, CH, graph_chunk_ids={"c1", "c2"}, embedded_chunk_ids={"c1"})
    ctx = "[1] a.md\nCell AUR-C103 failed at cycle 412."
    assert fc.classify(Q, candidates=[_hit("c1")], context_text=ctx, answer=GOOD) == []
    assert fc.classify(Q, candidates=[_hit("c2")], context_text="", answer=GOOD) == [
        "retrieval_failure"]  # fmt: skip
    assert fc.classify(Q, candidates=[_hit("c1")], context_text="other", answer=GOOD) == [
        "context_construction_failure"]  # fmt: skip
    assert fc.classify(Q, candidates=[_hit("c1", first_stage=2)], context_text="", answer=GOOD,
                       reranked=True) == ["reranking_failure"]  # fmt: skip
    wrong = {"answerable": True, "text": "It failed at cycle 400.",
             "claims": [{"status": "unsupported"}]}  # fmt: skip
    assert fc.classify(Q, candidates=[_hit("c1")], context_text=ctx, answer=wrong) == [
        "generation_failure", "hallucination"]  # fmt: skip
    moved = {**GOOD, "claims": [{"status": "relocated"}]}
    assert fc.classify(Q, candidates=[_hit("c1")], context_text=ctx, answer=moved) == [
        "citation_failure"]  # fmt: skip
    no_graph = FailureClassifier(EL, CH, graph_chunk_ids={"c2"})
    assert no_graph.classify(Q, candidates=[], context_text="", answer=GOOD) == [
        "graph_construction_failure"]  # fmt: skip
    no_vec = FailureClassifier(EL, CH, graph_chunk_ids={"c1"}, embedded_chunk_ids=set())
    assert no_vec.classify(Q, candidates=[], context_text="", answer=GOOD, dense=True) == [
        "embedding_failure"]  # fmt: skip
    split = FailureClassifier(EL, [SimpleNamespace(chunk_id="c9", content="failed at cycle")])
    assert split.classify(Q, candidates=[], context_text="", answer=GOOD) == ["chunking_failure"]
    unparsed = Q.model_copy(update={"facts": [[Q.facts[0][0].model_copy(
        update={"file": "scan.jpg"})]]})  # fmt: skip
    assert fc.classify(unparsed, candidates=[], context_text="", answer=GOOD) == ["ocr_failure"]
    nothing = EvalQuestion.model_validate({"id": "u", "type": "unanswerable", "question": "?",
                                           "answer": None})  # fmt: skip
    assert fc.classify(nothing, candidates=[], context_text="", answer=GOOD) == ["hallucination"]
    assert fc.classify(nothing, candidates=[], context_text="",
                       answer={"answerable": False}) == []  # fmt: skip
    s = summarize(
        {"q1": ["retrieval_failure"], "q2": [], "q3": ["generation_failure", "hallucination"]}
    )
    assert s["root_causes"] == {"retrieval_failure": 1, "generation_failure": 1}
    assert s["questions"]["hallucination"] == ["q3"] and (s["failed"], s["total"]) == (2, 3)


def test_live_flags_need_no_ground_truth() -> None:
    turn = {
        "answer": {
            "refused_by": None,
            "claims": [{"status": "unsupported"}, {"status": "relocated"}],
        }
    }
    assert live_flags(turn) == ["hallucination", "citation_failure"]
    assert live_flags({"answer": {"refused_by": "empty_context", "claims": []}}) == [
        "retrieval_failure"]  # fmt: skip


# ---------------------------------------------------------------- aggregation
def _rec(qid: str, answerable: bool, correct: bool, refused: bool, judge_c: str = "correct",
         halluc: bool = False) -> dict:  # fmt: skip
    m = {"claims": 2, "verified": 2, "relocated": 0, "partial": 0, "unsupported": 0,
         "faithfulness": 1.0, "citation_accuracy": 1.0}  # fmt: skip
    rec = {"experiment": "E3", "qid": qid, "type": "factual" if answerable else "unanswerable",
           "answerable": answerable, "correct": correct, "refused": refused,
           "completeness": 1.0 if correct else 0.0, "hallucination": halluc, "failures": [],
           "judge": {"correctness": judge_c, "relevance": 5, "supported": True, "ms": 10},
           "answer": {"metrics": m, "refused_by": "model" if refused else None,
                      "prompt_tokens": 1000, "eval_tokens": 90},
           "timings": {"end_to_end_ms": 2000.0, "retrieval_ms": 50.0, "embed_ms": 5.0,
                       "rerank_ms": 0.0, "generation_ms": 1900.0, "context_ms": 3.0,
                       "judge_ms": 10.0}}  # fmt: skip
    if answerable:
        rec["retrieval"] = {
            k: 1.0
            for k in (
                "recall",
                "precision",
                "mrr",
                "ndcg",
                "hit",
                "context_precision",
                "context_recall",
            )
        }
    return rec


def test_aggregate_reports_every_metric_family() -> None:
    a = aggregate([_rec("q1", True, True, False), _rec("q2", True, False, True, "refused"),
                   _rec("u1", False, True, True, "correct")], wall_s=60)  # fmt: skip
    g = a["generation"]
    assert (g["answer_correctness"], g["false_refusals"], g["unanswerable_refused"]) == (0.5, 1, 1)
    assert a["judge"]["correctness"] == {"correct": 2, "refused": 1}
    assert a["judge"]["agreement_with_answer_check"] == 1.0
    assert a["performance"]["throughput_q_per_min"] == 30.0  # 60 s / 2 s end to end
    assert a["performance"]["generation_ms_p50"] == 1900.0
    assert a["by_type"]["factual"]["n"] == 2 and a["retrieval"]["recall"] == 1.0
    assert set(EXPERIMENTS) == {f"E{i}" for i in range(1, 8)}


# ---------------------------------------------------------------- judge
class JudgeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, system, user, *, schema=None, **kw):
        assert schema == JUDGE_SCHEMA and "Reference answer: The documents do not" in user
        if isinstance(self.reply, Exception):
            raise self.reply
        return ChatResult(json.dumps(self.reply), self.reply, 10, 100, 5.0)


def test_judge_parses_and_never_raises() -> None:
    out = judge(JudgeLLM({"correctness": "correct", "relevance": 9, "supported": True,
                          "unsupported_statements": [], "reason": "refused"}),
                "q", None, "", "No source mentions it.", "")  # fmt: skip
    assert out["correctness"] == "correct" and out["relevance"] is None  # 9 is out of range
    assert judge(JudgeLLM(LLMError("down")), "q", None, "", None, "") == {"error": "down"}


# ---------------------------------------------------------------- LangSmith + tracing
def test_langsmith_payloads_carry_no_document_text(tmp_path: Path, monkeypatch) -> None:
    examples = langsmith_sync.dataset_examples(V1)
    assert len(examples) == 110 and set(examples[0]["inputs"]) == {"question", "qid"}
    assert "facts" not in json.dumps(examples)  # evidence quotes stay local
    rec = {
        **_rec("q1", True, True, False),
        "answer": {**_rec("q1", True, True, False)["answer"], "text": "secret text [1]"},
    }
    assert langsmith_sync.run_outputs(rec)["answer"] == "[redacted]"
    assert langsmith_sync.run_outputs(rec, include_answers=True)["answer"] == "secret text [1]"

    partial = tmp_path / "r.partial.jsonl"
    partial.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    calls = []

    class FakeClient:
        def has_dataset(self, dataset_name):
            return False

        def create_dataset(self, name, description=None):
            calls.append(("dataset", name))
            return SimpleNamespace(id="ds1")

        def create_examples(self, dataset_id, examples):
            calls.append(("examples", len(examples)))

    import langsmith

    monkeypatch.setattr(langsmith, "evaluate", lambda target, **kw: calls.append(
        ("evaluate", kw["experiment_prefix"], target({"qid": "q1"})["answer"])))  # fmt: skip
    report = {"experiments": {"E3": {"label": "Dense + BM25"}}, "prompt_version": "p",
              "generation_model": "m", "chunk_config": "c"}  # fmt: skip
    msg = langsmith_sync.upload_report(report, V1, partial, client=FakeClient())
    assert calls == [
        ("dataset", "omnirag-eval-v1"),
        ("examples", 110),
        ("evaluate", "E3-Dense + BM25", "[redacted]"),
    ]
    assert "experiments ['E3']" in msg
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert "skipped" in langsmith_sync.upload_report(report, V1, partial)


def test_tracing_needs_the_flag_and_a_key_and_redacts_by_default(settings: Settings) -> None:
    env: dict[str, str] = {}
    assert not configure_tracing(settings, env) and env == {}
    on = settings.model_copy(update={"langsmith_tracing": True})
    assert not configure_tracing(on, env) and "LANGSMITH_TRACING" not in env
    env["LANGSMITH_API_KEY"] = "lsv2-test"
    assert configure_tracing(on, env)
    assert env["LANGSMITH_TRACING"] == env["LANGSMITH_HIDE_INPUTS"] == "true"
    assert env["LANGSMITH_PROJECT"] == "omnirag"


# ---------------------------------------------------------------- runs + analytics + API
def _runner(report: dict):
    def run(settings, dataset, experiments, **kw):
        kw["progress"]("  E3 1/1")
        time.sleep(0.05)
        return report

    return run


REPORT = {
    "created": "2026-11-06T12:00:00+00:00",
    "dataset": "omnirag-eval-v1",
    "questions": 1,
    "experiments": {"E3": {"label": "Dense + BM25", **aggregate([_rec("q1", True, True, False)])}},
}


def _wait(runs: EvaluationRuns, run_id: str) -> dict:
    for _ in range(100):
        state = runs.get(run_id)
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_evaluation_runs_one_at_a_time(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"data_dir": tmp_path, "benchmarks_dir": tmp_path / "none"})
    runs = EvaluationRuns(s, runner=_runner(REPORT))
    state = runs.start(["E3"], limit=1)
    with pytest.raises(RunConflictError):
        runs.start(["E3"])
    done = _wait(runs, state["run_id"])
    assert done["status"] == "completed" and done["progress"] == ["  E3 1/1"]
    assert runs.latest_report()["questions"] == 1
    assert [r["run_id"] for r in runs.list()] == [state["run_id"]]
    failing = EvaluationRuns(s, runner=lambda *a, **k: 1 / 0)
    assert _wait(failing, failing.start(["E1"])["run_id"])["error"].startswith("ZeroDivision")


def test_turn_analytics_and_evaluation_summary() -> None:
    turns = [
        {
            "id": "t1",
            "conversation_id": "c1",
            "question": "q",
            "created_at": "x",
            "answer": {
                "answerable": True,
                "grounded": True,
                "refused_by": None,
                "metrics": {"citation_accuracy": 1.0},
                "claims": [{"status": "verified"}],
            },
            "retrieval": {
                "verdict": "sufficient",
                "route": {"intent": "lookup"},
                "context": {"tokens": 900},
            },
            "timings": {"total_ms": 3000.0, "generation": 2500.0},
        },
        {
            "id": "t2",
            "conversation_id": "c1",
            "question": "q2",
            "created_at": "y",
            "answer": {
                "answerable": False,
                "grounded": False,
                "refused_by": "check",
                "metrics": {},
                "claims": [],
            },
            "retrieval": {"verdict": "insufficient"},
            "timings": {"total_ms": 30.0},
        },
    ]
    a = turn_analytics(turns)
    assert (a["turns"], a["conversations"], a["answered"], a["grounded"]) == (2, 1, 1, 1)
    assert a["refused_by"] == {"check": 1} and a["latency_ms"]["total_ms"]["p50"] == 30.0
    assert a["verdicts"] == {"sufficient": 1, "insufficient": 1}
    assert evaluation_summary(REPORT)["experiments"]["E3"]["root_causes"] == {}
    assert evaluation_summary(None) is None


class GraphStore:
    def query(self, cypher, **params):
        if "RELATED_TO" in cypher:
            return [
                {
                    "name": "E-7",
                    "type": "MATERIAL",
                    "rels": [
                        {
                            "other": "Marco Lindqvist",
                            "otype": "PERSON",
                            "type": "OWNS",
                            "out": False,
                            "support": "stated",
                        }
                    ],
                    "mentions": [
                        {"chunk": "doc_a-cfg-00001", "file": "notes.md", "document_id": "doc_a"}
                    ],
                }
            ]
        return [
            {
                "document_id": "doc_a",
                "file": "notes.md",
                "entity": "E-7",
                "type": "MATERIAL",
                "n": 3,
            }
        ]


def test_graph_neighborhood_builds_nodes_and_edges() -> None:
    g = traversal.neighborhood(GraphStore(), entity="E-7")
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    assert kinds == {
        "e:E-7": "entity",
        "e:Marco Lindqvist": "entity",
        "c:doc_a-cfg-00001": "chunk",
        "d:doc_a": "document",
    }
    assert {
        "source": "e:Marco Lindqvist",
        "target": "e:E-7",
        "type": "OWNS",
        "support": "stated",
    } in g["edges"]
    overview = traversal.neighborhood(GraphStore())
    assert overview["edges"] == [
        {"source": "d:doc_a", "target": "e:E-7", "type": "MENTIONS", "count": 3}
    ]


async def test_evaluate_analytics_and_graph_api(settings: Settings, tmp_path: Path) -> None:
    from test_generation import _service

    s = settings.model_copy(update={"data_dir": tmp_path, "benchmarks_dir": tmp_path / "none"})
    app = create_app(s)
    runs = EvaluationRuns(s, runner=_runner(REPORT))
    service = _service(tmp_path)
    app.dependency_overrides[get_evaluations] = lambda: runs
    app.dependency_overrides[get_chat] = lambda: service
    app.dependency_overrides[get_retrieval] = lambda: SimpleNamespace(store=GraphStore())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bad = await client.post("/api/v1/evaluate", json={"experiments": ["E9"]})
        assert bad.status_code == 422
        r = await client.post("/api/v1/evaluate", json={"experiments": ["E3"], "limit": 1})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        _wait(runs, run_id)
        got = (await client.get(f"/api/v1/evaluate/{run_id}")).json()
        assert (
            got["status"] == "completed"
            and "per_question" not in got["report"]["experiments"]["E3"]
        )
        assert (await client.get("/api/v1/evaluate/eval_000000000000")).status_code == 404
        await client.post("/api/v1/chat", json={"question": "Why did AUR-C103 fail?"})
        a = (await client.get("/api/v1/analytics")).json()
        assert a["chat"]["turns"] == 1 and a["evaluation"]["questions"] == 1
        g = (await client.get("/api/v1/graph/neighborhood", params={"entity": "E-7"})).json()
        assert len(g["nodes"]) == 4


def test_rescore_applies_current_terms_and_settles_refusals_without_the_judge() -> None:
    q = EvalQuestion.model_validate({"id": "e27", "type": "comparative", "question": "Which?",
                                     "answer": "AUR-C106", "must_include": ["AUR-C106"],
                                     "facts": [[{"file": "a.md", "text": "x"}]]})  # fmt: skip
    rec = _rec("e27", True, False, False)
    rec["answer"] = {**rec["answer"], "text": "AUR-C106 retained more [1].", "answerable": True}
    rec["failures"] = ["generation_failure"]
    out = rescore(rec, q)
    assert out["correct"] and out["failures"] == [] and out["completeness"] == 1.0
    refused = _rec("e27", True, False, True, judge_c="correct")
    refused["answer"] = {**refused["answer"], "text": "", "answerable": False}
    out = rescore(refused, q)
    assert out["judge"]["correctness"] == "refused" and out["judge"]["raw"] == "correct"
    assert out["failures"] == ["generation_failure"]
    assert rescore(out, q)["judge"]["raw"] == "correct"  # idempotent
    a = aggregate([out])
    assert a["judge"]["refusal_overrides"] == {"refusals_judged": 1, "raw_said_correct": 1}
