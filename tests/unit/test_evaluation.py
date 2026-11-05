"""Seed dataset integrity, retrieval metrics, BM25."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from omnirag.core.config import Settings
from omnirag.evaluation.dataset import EvalDataset, EvalQuestion, covered_facts, resolve
from omnirag.evaluation.metrics import Unit, ranking_metrics
from omnirag.evaluation.retrieval import BM25Index, DenseIndex, terms
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import ElementType
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore

ROOT = Path(__file__).parents[2]
DATASET = EvalDataset.load(ROOT / "data" / "eval" / "chunking_seed.json")


def test_dataset_shape() -> None:
    ids = [q.id for q in DATASET.questions]
    assert len(ids) == len(set(ids)) == 40
    types = {q.type for q in DATASET.questions}
    assert types >= {"factual", "table", "image", "multi_hop", "keyword", "comparative",
                     "unanswerable"}  # fmt: skip
    assert all(q.answer is None for q in DATASET.questions if not q.answerable)


LAYOUT_FILES = {"e7_technical_note.pdf"}  # two columns + borderless table: needs Docling


def _ingest_samples(tmp_path: Path, engine: str, files: set[str] | None = None) -> dict:
    settings = Settings(_env_file=None, neo4j_password="x", pdf_engine=engine)
    store = DocumentStore(tmp_path)
    pipeline = IngestionPipeline(store, default_loaders(settings), max_bytes=10 << 20)
    by_file = {}
    for path in sorted((ROOT / "data" / "samples").iterdir()):
        if path.name.startswith("README") or (files is not None and path.name not in files):
            continue
        record = pipeline.ingest(path).record
        by_file[path.name] = store.read_elements(record.document_id)
    return by_file


def _check(by_file: dict, files: set[str]) -> int:
    """Resolve every evidence quote that comes from ``files``; return how many were checked."""
    checked = 0
    for q in DATASET.questions:
        if q.evidence_origin == "generated":
            continue  # vision descriptions are not produced without the VLM
        facts = [[a for a in fact if a.file in files] for fact in q.facts]
        facts = [fact for fact in facts if fact]
        if facts:
            assert all(resolve(q.model_copy(update={"facts": facts}), by_file)), q.id
            checked += sum(len(fact) for fact in facts)
    return checked


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="OCR evidence needs tesseract")
def test_every_evidence_quote_exists_in_the_samples(tmp_path: Path) -> None:
    """Fails if a sample or loader change breaks the ground truth (PyMuPDF, no Docling)."""
    by_file = _ingest_samples(tmp_path, "pymupdf")
    files = set(by_file) - LAYOUT_FILES
    expected = sum(1 for q in DATASET.questions if q.evidence_origin != "generated"
                   for fact in q.facts for a in fact if a.file in files)  # fmt: skip
    assert _check(by_file, files) == expected == 57  # every non-layout, non-generated quote
    # Table-row quotes resolve to the row element, not just the table.
    gold = resolve(next(q for q in DATASET.questions if q.id == "q10"), by_file)
    row_ids = {e.element_id for e in by_file["aurora_budget_2026.xlsx"]
               if e.element_type is ElementType.TABLE_ROW}  # fmt: skip
    assert gold[0] <= row_ids


@pytest.mark.docling
def test_layout_evidence_needs_docling(tmp_path: Path) -> None:
    by_file = _ingest_samples(tmp_path, "auto", LAYOUT_FILES)
    assert _check(by_file, LAYOUT_FILES) == 5  # q07, q23 (2 quotes), q24, q25


def test_covered_facts_normalises_whitespace() -> None:
    q = EvalQuestion.model_validate({"id": "x", "type": "multi_hop", "question": "?", "answer": "a",
                                     "facts": [[{"file": "f", "text": "after 800 cycles"}],
                                               [{"file": "f", "text": "E-7"}]]})  # fmt: skip
    assert covered_facts("at 83.1% after 800\ncycles on AUR-C104", q) == {0}
    assert covered_facts("E-7 at  83.1% after 800 cycles", q) == {0, 1}


def test_ranking_metrics() -> None:
    ranked = [Unit("a", 100, frozenset()), Unit("b", 50, frozenset({0})),
              Unit("c", 50, frozenset({1})), Unit("d", 10, frozenset())]  # fmt: skip
    m = ranking_metrics(ranked, n_facts=2, n_relevant_in_index=2, k=3)
    assert m["hit"] == 1.0 and m["recall"] == 1.0
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["mrr"] == 0.5
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / ideal)
    assert m["context_precision"] == pytest.approx(100 / 200)
    assert ranking_metrics(ranked[:1], 2, 2, 3)["mrr"] == 0.0


def test_bm25_keeps_identifiers_whole_and_ranks_exact_matches() -> None:
    assert terms("Cell AUR-C103 at 45 °C, 89.6% on 2026-03-19") == [
        "cell", "aur-c103", "at", "45", "c", "89.6", "on", "2026-03-19"]  # fmt: skip
    index = BM25Index(["cell AUR-C105 retained 89.6", "cell AUR-C103 failed", "budget"])
    assert index.search("What happened to AUR-C103?", 2)[0] == 1


def test_dense_index_is_exact_cosine() -> None:
    vectors = np.eye(3, dtype=np.float32)
    assert DenseIndex(vectors).search(np.array([0.1, 0.9, 0.2], dtype=np.float32), 2) == [1, 2]


RELATIONAL = EvalDataset.load(ROOT / "data" / "eval" / "relational_seed.json")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="OCR evidence needs tesseract")
def test_relational_quotes_exist_in_the_samples(tmp_path: Path) -> None:
    """Every quote of the relational set resolves (layout-only files excepted, as above)."""
    by_file = _ingest_samples(tmp_path, "pymupdf")
    files = set(by_file) - LAYOUT_FILES
    assert len(RELATIONAL.questions) == 8 and all(len(q.facts) == 2 for q in RELATIONAL.questions)
    for q in RELATIONAL.questions:
        facts = [[a for a in fact if a.file in files] for fact in q.facts]
        assert all(facts), f"{q.id}: a fact has no non-layout evidence"
        assert all(resolve(q.model_copy(update={"facts": facts}), by_file)), q.id
