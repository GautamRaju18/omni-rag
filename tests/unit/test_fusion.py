"""RRF fusion, the hybrid retriever, dictionary entity linking, mode wiring."""

from __future__ import annotations

import math

import pytest

from omnirag.retrieval import FUSED_MODES, RetrievalService
from omnirag.retrieval.fusion import HybridRetriever, rrf
from omnirag.retrieval.graph_search import link
from omnirag.retrieval.models import RetrievalFilters, RetrievalResult, RetrievedChunk

CFG = "multimodal-256-ctx"


def _hit(cid: str, rank: int, retriever: str = "dense", score: float = 1.0, **kw):
    return RetrievedChunk(chunk_id=cid, document_id="d", source_file="f", rank=rank,
                          score=score, retriever=retriever, content=cid, element_type="paragraph",
                          config_id=CFG, **kw)  # fmt: skip


def _result(name: str, ids: list[str], **kw) -> RetrievalResult:
    hits = [_hit(c, i + 1, name, **kw) for i, c in enumerate(ids)]
    return RetrievalResult(query="q", retriever=name, k=len(ids), latency_ms=1.0, hits=hits)


# ---------------------------------------------------------------- RRF
def test_rrf_scores_and_order() -> None:
    fused = rrf({"dense": _result("dense", ["a", "b", "c"]),
                 "bm25": _result("bm25", ["b", "d"])}, k=60, top=4)  # fmt: skip
    # b = 1/62 + 1/61 (both lists), a = 1/61, d = 1/62 (bm25 #2), c = 1/63 (dense #3)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    b = fused[0]
    assert b.score == pytest.approx(round(1 / 62 + 1 / 61, 6))
    assert b.sources == {"dense": {"rank": 2, "score": 1.0}, "bm25": {"rank": 1, "score": 1.0}}
    assert b.rank == 1 and b.retriever == "dense+bm25"


def test_rrf_small_k_trusts_top_ranks_more() -> None:
    lists = {"dense": _result("dense", ["a", "x", "y"]), "bm25": _result("bm25", ["z", "w", "a"])}
    # a: ranks 1 and 3. z: rank 1 once. k=60: a wins (two votes); k=1: still a (1/2+1/4 > 1/2).
    assert rrf(lists, k=60, top=1)[0].chunk_id == "a"
    assert rrf(lists, k=1, top=1)[0].chunk_id == "a"


def test_rrf_ties_break_by_best_rank_then_id() -> None:
    fused = rrf({"dense": _result("dense", ["b"]), "bm25": _result("bm25", ["a"])}, top=2)
    assert [h.chunk_id for h in fused] == ["a", "b"]  # same score and best rank: ID order


def test_rrf_weights_and_graph_provenance() -> None:
    graph = RetrievalResult(query="q", retriever="graph", k=1, latency_ms=1.0, hits=[
        _hit("g", 1, "graph", graph_paths=["MENTIONS AUR-C103"])])  # fmt: skip
    lists = {"dense": _result("dense", ["a"]), "graph": graph}
    weighted = rrf(lists, weights={"graph": 0.5}, top=2, mode="dense+bm25+graph")
    assert [h.chunk_id for h in weighted] == ["a", "g"]
    assert weighted[1].score == pytest.approx(round(0.5 / 61, 6))
    assert weighted[1].graph_paths == ["MENTIONS AUR-C103"]
    with pytest.raises(ValueError, match="k must be"):
        rrf(lists, k=0)


# ---------------------------------------------------------------- hybrid retriever
class Member:
    def __init__(self, ids: list[str], error: Exception | None = None) -> None:
        self.ids, self.error, self.calls = ids, error, []

    def retrieve(self, query: str, k: int, filters=None) -> RetrievalResult:
        self.calls.append(k)
        if self.error:
            raise self.error
        return _result("dense", self.ids[:k])


def test_hybrid_fuses_and_degrades() -> None:
    dense, bm25 = Member(["a", "b"]), Member([], error=RuntimeError("Neo4j down"))
    hybrid = HybridRetriever({"dense": dense, "bm25": bm25}, mode="dense+bm25", depth=20)
    result = hybrid.retrieve("q", k=3, filters=RetrievalFilters())
    assert [h.chunk_id for h in result.hits] == ["a", "b"]
    assert result.details["failed"] == {"bm25": "RuntimeError: Neo4j down"}
    assert dense.calls == [20] and result.details["members"]["dense"]["hits"] == 2
    assert hybrid.retrieve("q", k=30).details["depth"] == 30  # never fetch fewer than k
    hybrid.close()


def test_hybrid_raises_when_every_member_fails() -> None:
    hybrid = HybridRetriever({"a": Member([], ValueError("x")), "b": Member([], KeyError("y"))},
                             mode="dense+bm25")  # fmt: skip
    with pytest.raises(ValueError, match="x"):
        hybrid.retrieve("q")
    hybrid.close()


def test_service_builds_fused_modes() -> None:
    members = {"dense": Member(["a"]), "bm25": Member(["b"]), "graph": Member(["c"])}
    service = RetrievalService(store=None, retrievers=members, weights={"graph": 0.5})
    assert set(FUSED_MODES) <= set(service.retrievers)
    fused = service.retrievers["dense+bm25+graph"]
    assert fused.weights == {"graph": 0.5} and service.retrievers["dense+bm25"].weights == {}
    assert [h.chunk_id for h in service.search("q", "dense+bm25", 3).hits] == ["a", "b"]
    for r in service.retrievers.values():
        if isinstance(r, HybridRetriever):
            r.close()


# ---------------------------------------------------------------- entity linking
ENTITIES = [
    {"entity_id": "e1", "key": "aurora", "name": "Project Aurora", "type": "PROJECT",
     "aliases": ["Project Aurora", "Aurora"], "document_count": 6},
    {"entity_id": "e2", "key": "aurora s1", "name": "Aurora S1", "type": "PRODUCT",
     "aliases": ["Aurora S1"], "document_count": 1},
    {"entity_id": "e3", "key": "ec", "name": "EC", "type": "MATERIAL", "aliases": ["EC"],
     "document_count": 3},
    {"entity_id": "e4", "key": "aiko tanaka", "name": "Aiko Tanaka", "type": "PERSON",
     "aliases": ["Aiko Tanaka", "Aiko"], "document_count": 4},
]  # fmt: skip


def test_link_longest_match_nested_and_idf() -> None:
    linked = {f["entity_id"]: f for f in link("What is the energy density of Aurora S1?",
                                              ENTITIES, n_docs=12)}  # fmt: skip
    assert set(linked) == {"e1", "e2"}
    assert linked["e2"]["weight"] == pytest.approx(round(math.log(1 + 12 / 1), 4))
    assert linked["e1"]["nested"] is True  # "aurora" inside "aurora s1": half weight
    assert linked["e1"]["weight"] == pytest.approx(round(0.5 * math.log(1 + 12 / 6), 4))


def test_link_aliases_and_word_boundaries() -> None:
    assert [f["name"] for f in link("What did Aiko report?", ENTITIES, 12)] == ["Aiko Tanaka"]
    assert link("The DEC solvent", ENTITIES, 12) == []  # "ec" is not inside "dec"
    assert link("Who is the CEO?", ENTITIES, 12) == []
