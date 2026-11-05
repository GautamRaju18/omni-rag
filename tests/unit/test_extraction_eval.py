"""Gold entity set integrity (checked against the real samples) and the extraction metrics."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from omnirag.core.config import Settings
from omnirag.evaluation.extraction_eval import GoldSet, evaluate
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import ContentOrigin
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base.entities import (
    ChunkExtraction,
    DocumentExtraction,
    EntityType,
    Mention,
    Relation,
    RelationType,
    contains,
    norm,
)

ROOT = Path(__file__).parents[2]
GOLD = GoldSet.load()


def test_gold_shape() -> None:
    names = [e.name for e in GOLD.entities]
    assert len(names) == len(set(names)) == 37
    keys = [k for e in GOLD.entities for k in e.keys()]
    assert len(keys) == len(set(keys)), "two gold entities share a key"
    assert all(r.source in names and r.target in names for r in GOLD.relations)
    assert len(GOLD.relations) == 14


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="OCR text needs tesseract")
def test_gold_is_grounded_in_the_samples(tmp_path: Path) -> None:
    """Every gold name (or alias) and every evidence quote exists in the ingested samples."""
    settings = Settings(_env_file=None, neo4j_password="x", pdf_engine="pymupdf")
    store = DocumentStore(tmp_path)
    pipeline = IngestionPipeline(store, default_loaders(settings), max_bytes=10 << 20)
    text = {}
    for path in sorted((ROOT / "data" / "samples").iterdir()):
        if not path.name.startswith("README"):
            record = pipeline.ingest(path).record
            text[path.name] = norm("\n".join(e.content for e in
                                             store.read_elements(record.document_id)))  # fmt: skip
    corpus = "\n".join(text.values())
    missing = [e.name for e in GOLD.entities
               if not any(contains(corpus, norm(n)) for n in [e.name, *e.aliases])]  # fmt: skip
    assert missing == []
    bad = [r.evidence for r in GOLD.relations if norm(r.evidence) not in text[r.file]]
    assert bad == []


def _x(doc: str, mentions: list[tuple[str, EntityType]], relations=()) -> DocumentExtraction:
    chunk = ChunkExtraction(
        chunk_id=f"{doc}-c0", input_sha256="x",
        mentions=[Mention(key=k, surface=k, type=t, origin=ContentOrigin.ORIGINAL,
                          confidence=1.0) for k, t in mentions],
        relations=[Relation(source=s, type=RelationType(t), target=g, support=sup,
                            evidence="e", evidence_verified=True)
                   for s, t, g, sup in relations],
    )  # fmt: skip
    return DocumentExtraction(document_id=doc, config_id="c", model="m", model_digest="d",
                              prompt_version="p", extracted_at="t", chunks=[chunk])  # fmt: skip


def test_metrics_on_a_synthetic_extraction() -> None:
    P, M, S = EntityType.PERSON, EntityType.MATERIAL, EntityType.SAMPLE  # noqa: N806
    extractions = [
        _x("a", [("marco lindqvist", P), ("e-7", M), ("aur-c105", S),
                 ("aurora", EntityType.PROJECT)],
           [("marco lindqvist", "RESPONSIBLE_FOR", "e-7", "stated"),
            ("e-7", "USES", "aur-c105", "inferred")]),  # reversed direction
        _x("b", [("marco", P), ("project aurora", EntityType.ORGANIZATION), ("widget", M)]),
    ]  # fmt: skip
    result = evaluate(GOLD, extractions)
    s = result["summary"]
    assert s["entity_recall"] == round(4 / 37, 3)
    assert s["same_as_links"] == 1  # "marco" -> "marco lindqvist" across documents
    # Typed PROJECT, "Project Aurora" keys to "aurora"; typed ORGANIZATION it stays "project
    # aurora": two nodes for one project. "marco" is joined to its full name by SAME_AS.
    assert s["fragmented"] == ["Project Aurora"]
    rows = {r["name"]: r for r in result["entities"]}
    assert rows["Project Aurora"]["keys"] == ["aurora", "project aurora"]
    assert rows["Marco Lindqvist"]["keys"] == ["marco", "marco lindqvist"]
    assert not rows["Marco Lindqvist"]["fragmented"]
    rel = {(r["source"], r["target"]): r for r in result["relations"]}
    assert rel[("Marco Lindqvist", "E-7")]["typed"] and rel[("Marco Lindqvist", "E-7")]["stated"]
    assert rel[("AUR-C105", "E-7")]["pair"] and not rel[("AUR-C105", "E-7")]["typed"]
    assert [u["key"] for u in result["unlisted"]] == ["widget"]
