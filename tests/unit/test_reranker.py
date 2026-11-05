"""Reranking logic with a fake cross-encoder (no model weights needed)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from omnirag.retrieval import RetrievalService
from omnirag.retrieval.models import RetrievalResult, RetrievedChunk
from omnirag.retrieval.reranker import (
    MODELS,
    CrossEncoderReranker,
    RerankingRetriever,
    passage_text,
    sigmoid,
)

CFG = "multimodal-256-ctx"


def _hit(cid: str, rank: int, content: str, context: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id="d", source_file="f", rank=rank, score=0.01,
                          retriever="dense+bm25", content=content, context=context,
                          element_type="paragraph", config_id=CFG,
                          sources={"dense": {"rank": rank, "score": 0.8}})  # fmt: skip


class Base:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits, self.calls = hits, []

    def retrieve(self, query, k, filters=None) -> RetrievalResult:
        self.calls.append(k)
        return RetrievalResult(query=query, retriever="dense+bm25", k=k, hits=self.hits[:k],
                               latency_ms=3.0, details={"rrf_k": 60})  # fmt: skip


class FakeReranker:
    """Scores a passage by how often it contains the word 'answer'."""

    name, device = "fake", "cpu"

    def __init__(self) -> None:
        self.seen: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.seen.append((query, list(passages)))
        return [4.0 * p.count("answer") - 2.0 for p in passages]


def test_sigmoid_is_stable() -> None:
    assert sigmoid(0) == 0.5
    assert sigmoid(-5.65234375) == pytest.approx(0.003497, abs=1e-6)  # model card value
    assert sigmoid(-1000) == pytest.approx(0.0) and sigmoid(1000) == pytest.approx(1.0)


def test_passage_text_includes_the_contextual_header() -> None:
    assert passage_text(_hit("a", 1, "body", "Document: X")) == "Document: X\n\nbody"
    assert passage_text(_hit("a", 1, "body")) == "body"


def test_reranking_reorders_and_keeps_provenance() -> None:
    hits = [_hit("noise", 1, "nothing here"), _hit("weak", 2, "an answer"),
            _hit("best", 3, "answer answer")]  # fmt: skip
    base, reranker = Base(hits), FakeReranker()
    retriever = RerankingRetriever(base, reranker, mode="dense+bm25+rerank", depth=20)
    result = retriever.retrieve("q", k=2)
    assert base.calls == [20]  # fetches depth, returns k
    assert [h.chunk_id for h in result.hits] == ["best", "weak"]
    best = result.hits[0]
    assert best.rank == 1 and best.retriever == "dense+bm25+rerank"
    assert best.score == pytest.approx(round(sigmoid(6.0), 6))
    assert best.sources["dense"] == {"rank": 3, "score": 0.8}  # first-stage provenance kept
    assert best.sources["rerank"] == {"rank": 1, "score": 6.0, "first_stage_rank": 3}
    assert result.details["candidates"] == 3 and result.details["reranker"] == "fake"
    assert result.details["top_score"] == best.score
    assert reranker.seen[0][1] == ["nothing here", "an answer", "answer answer"]


def test_ties_keep_the_first_stage_order() -> None:
    hits = [_hit("a", 1, "x"), _hit("b", 2, "y")]
    result = RerankingRetriever(Base(hits), FakeReranker(), mode="dense+bm25+rerank").retrieve("q")
    assert [h.chunk_id for h in result.hits] == ["a", "b"]


def test_empty_first_stage() -> None:
    result = RerankingRetriever(Base([]), FakeReranker(), mode="dense+bm25+rerank").retrieve("q")
    assert result.hits == [] and result.details["top_score"] is None


def test_service_adds_rerank_modes_only_with_a_reranker() -> None:
    members = {"dense": Base([_hit("a", 1, "answer")]), "bm25": Base([]), "graph": Base([])}
    plain = RetrievalService(store=None, retrievers=members)
    assert "dense+bm25+rerank" not in plain.retrievers
    with pytest.raises(LookupError, match="not enabled"):
        plain.search("q", "dense+bm25+rerank")
    reranked = RetrievalService(store=None, retrievers=members, reranker=FakeReranker(),
                                rerank_depth=7)  # fmt: skip
    assert {"dense+bm25+rerank", "dense+bm25+graph+rerank"} <= set(reranked.retrievers)
    assert reranked.retrievers["dense+bm25+rerank"].depth == 7
    result = reranked.search("q", "dense+bm25+rerank", 1)
    assert result.hits[0].chunk_id == "a" and "rerank" in result.hits[0].sources
    for service in (plain, reranked):
        for r in service.retrievers.values():
            if hasattr(r, "_pool"):
                r.close()


def test_model_specs_are_pinned(tmp_path: Path) -> None:
    for spec in MODELS.values():
        assert len(spec.revision) == 40
        assert set(spec.files) == {"model.safetensors", "tokenizer.json", "config.json"}
        assert all(len(sha) == 64 for sha in spec.files.values())
    with pytest.raises(ValueError, match="Unknown reranker"):
        CrossEncoderReranker(tmp_path, "bge-reranker-large")
    base = CrossEncoderReranker(tmp_path, "bge-reranker-base", max_length=4096)
    assert base.max_length == 512  # capped at the model's position limit
    assert not math.isnan(sigmoid(3.0))
