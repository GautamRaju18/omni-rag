"""Generation against the real graph and Ollama (read-only on the main graph), plus the
110-question dataset's quotes against the live store (incl. Docling and generated text).

Needs: Neo4j with the sample corpus loaded, Ollama with qwen3:4b-instruct and bge-m3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnirag.core.config import Settings
from omnirag.evaluation.dataset import EvalDataset, resolve
from omnirag.generation.chat import ChatService, build_chat
from omnirag.generation.history import ChatStore
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.retrieval import build_retrieval
from omnirag.workflows.retrieval_graph import build_workflow

pytestmark = pytest.mark.integration
ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def chat(tmp_path_factory):
    settings = Settings(rerank_enabled=False)
    service = build_retrieval(settings)
    wf = build_workflow(settings, service)
    built = build_chat(settings, wf)
    yield ChatService(wf, built.generator, ChatStore(tmp_path_factory.mktemp("chat") / "c.db"))
    wf.close()
    service.close()


def test_answer_is_grounded_with_verified_citations(chat) -> None:
    turn = chat.ask("Why did cell AUR-C103 fail?")
    a = turn["answer"]
    assert a["answerable"] and a["grounded"] and a["metrics"]["unsupported"] == 0
    assert a["metrics"]["verified"] >= 1
    assert "412" in a["text"] or "E-5" in a["text"]
    for c in a["citations"]:
        assert c["chunk_ids"] and c["source_file"]
    assert turn["timings"]["generation"] > 0 and turn["retrieval"]["verdict"] == "sufficient"
    assert chat.store.turn(turn["id"])["answer"]["text"] == a["text"]


def test_unknown_identifier_is_refused_without_a_model_call(chat) -> None:
    calls = chat.generator.llm.stats["calls"]
    a = chat.ask("What capacity retention did cell AUR-C109 reach?")["answer"]
    assert (a["answerable"], a["refused_by"]) == (False, "check")
    assert "AUR-C109" in a["missing"] and chat.generator.llm.stats["calls"] == calls


def test_stream_yields_claims_before_the_answer(chat) -> None:
    kinds = [e["type"] for e in chat.stream("Who prepared the safety review of the E-5 failure?")]
    assert kinds[:2] == ["start", "retrieval"] and kinds[-2:] == ["answer", "done"]
    assert "claim" in kinds


def test_every_v1_quote_exists_in_the_live_store() -> None:
    settings = Settings()
    store = DocumentStore(settings.data_dir)
    by_file = {r.source_file: store.read_elements(r.document_id) for r in store.list_records()
               if r.status is IngestionStatus.COMPLETED}  # fmt: skip
    dataset = EvalDataset.load(ROOT / "data" / "eval" / "omnirag_eval_v1.json")
    for q in dataset.questions:
        for fact in q.facts:
            for alt in fact:
                assert resolve(q.model_copy(update={"facts": [[alt]]}), by_file), (q.id, alt)
