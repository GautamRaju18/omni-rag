"""Graph payload building, vector lookup, settings and pipeline wiring (no Neo4j needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import Chunker
from omnirag.chunking.tokenizer import WhitespaceTokenizer
from omnirag.core.config import Settings
from omnirag.embeddings.ollama import EmbeddingCache
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import ElementType, IngestionStatus
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base.graph_builder import (
    build_payload,
    cached_vectors,
    page_id,
    section_id,
    source_key,
)
from omnirag.knowledge_base.graph_store import GraphUnavailableError
from omnirag.knowledge_base.schema import ensure_schema
from omnirag.knowledge_base.traversal import _index_name

DOC = "doc_00000000000000aa"
MM = "multimodal-256-ctx"


def test_ids_are_deterministic() -> None:
    assert page_id(DOC, 3) == f"{DOC}-p0003"
    assert section_id(DOC, ["A", "B"]) == section_id(DOC, ["A", "B"])
    # The separator keeps ["A B"] and ["A", "B"] apart.
    assert section_id(DOC, ["A", "B"]) != section_id(DOC, ["A B"])
    assert source_key("  Report.PDF ") == "report.pdf"


def test_payload_structure(graph_doc) -> None:
    record, elements, sets = graph_doc()
    p = build_payload(record, elements, sets, {f"{DOC}-{MM}-00000": [0.1, 0.2]}, "space-test")

    assert p["source"] == {"source_key": "notes.md", "name": "notes.md"}
    assert p["document"]["title"] == "Aurora Notes"
    assert p["document"]["embedding_space"] == "space-test"
    assert [x["number"] for x in p["pages"]] == [1, 2]

    # Section tree from heading trails: the title, then "Results" under it.
    paths = [s["path"] for s in p["sections"]]
    assert paths == [["Aurora Notes"], ["Aurora Notes", "Results"]]
    assert p["sections"][0]["parent_id"] is None
    assert p["sections"][1]["parent_id"] == p["sections"][0]["section_id"]

    (table,) = p["tables"]
    assert table["table_id"] == f"{DOC}-e00003"
    assert table["columns"] == ["Cell", "Swelling"] and table["figure_id"] is None
    (figure,) = p["figures"]
    assert figure["figure_id"] == f"{DOC}-e00005" and figure["asset_ref"] == "assets/img.png"
    assert figure["caption"] == "Figure 1: swelling by cell"
    assert figure["ocr_text"] == "14%"  # OCR text kept apart from generated text
    assert figure["description"] == "A bar chart of swelling."
    assert figure["description_model"] == "qwen3-vl:2b-instruct"

    rows = {r["props"]["chunk_id"]: r for r in p["chunks"]}
    assert len(rows) == 5
    assert rows[f"{DOC}-{MM}-00000"]["vector"] == [0.1, 0.2]
    assert rows[f"{DOC}-{MM}-00001"]["vector"] is None
    assert all(r["props"]["is_latest"] is False for r in rows.values())  # set by versioning

    # A chunk holding only a TABLE_ROW still discusses its table; figure members map to it.
    assert p["table_links"] == [{"label": "Table", "id": f"{DOC}-e00003",
                                 "chunk_id": f"{DOC}-{MM}-00001"},
                                {"label": "Table", "id": f"{DOC}-e00003",
                                 "chunk_id": f"{DOC}-parentchild-256-128-32-00000"}]  # fmt: skip
    assert p["figure_links"] == [{"label": "Figure", "id": f"{DOC}-e00005",
                                  "chunk_id": f"{DOC}-{MM}-00002"}]  # fmt: skip


def test_payload_chunk_links(graph_doc) -> None:
    record, elements, sets = graph_doc()
    p = build_payload(record, elements, sets, {}, None)
    # Reading order within one chunk set and level only: never across sets or levels.
    assert p["next_links"] == [
        {"from": f"{DOC}-{MM}-00000", "to": f"{DOC}-{MM}-00001"},
        {"from": f"{DOC}-{MM}-00001", "to": f"{DOC}-{MM}-00002"},
    ]
    pc = "parentchild-256-128-32"
    assert p["child_links"] == [{"child": f"{DOC}-{pc}-00001", "parent": f"{DOC}-{pc}-00000"}]
    assert len(p["page_links"]) == 5 and len(p["section_links"]) == 5


def test_heading_already_in_its_path_is_not_doubled(graph_doc) -> None:
    record, elements, sets = graph_doc()
    # A loader that puts the heading itself at the end of its section_path.
    elements[1] = elements[1].model_copy(update={"section_path": ["Aurora Notes", "Results"]})
    paths = [s["path"] for s in build_payload(record, elements, sets, {}, None)["sections"]]
    assert ["Aurora Notes", "Results", "Results"] not in paths


def test_figure_with_a_chart_table(graph_doc) -> None:
    record, elements, sets = graph_doc()
    figure = elements[5].model_copy(update={"element_type": ElementType.FIGURE,
                                            "content": "Chart: swelling"})  # fmt: skip
    table = elements[3].model_copy(update={"parent_element_id": figure.element_id})
    elements[5], elements[3] = figure, table
    p = build_payload(record, elements, sets, {}, None)
    assert p["tables"][0]["figure_id"] == figure.element_id
    assert p["figures"][0]["title"] == "Chart: swelling" and p["figures"][0]["alt_text"] is None


def test_cached_vectors(tmp_path: Path, graph_doc) -> None:
    _, _, sets = graph_doc()
    chunks = sets[MM]
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    key = chunks[0].embedding_ref.split("/", 1)[1]
    cache.put_many("space-test", {key: np.array([0.6, 0.8])})
    space, vectors, missing = cached_vectors(cache, chunks)
    assert space == "space-test"
    assert vectors == {chunks[0].chunk_id: pytest.approx([0.6, 0.8])}
    assert missing == 2

    assert cached_vectors(cache, [c.model_copy(update={"embedding_ref": None})
                                  for c in chunks]) == (None, {}, 0)  # fmt: skip
    mixed = [chunks[0], chunks[1].model_copy(update={"embedding_ref": "other/abc"})]
    with pytest.raises(ValueError, match="2 embedding spaces"):
        cached_vectors(cache, mixed)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vector_dim": 0}, "dimension"),
        ({"vector_dim": "1024"}, "dimension"),
        ({"vector_dim": 8, "quantization": "FP16"}, "quantization"),
    ],
)
def test_schema_rejects_invalid_options(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ensure_schema(None, **kwargs)  # validated before any query


@pytest.mark.parametrize("name", ["chunk_embedding", "idx2"])
def test_index_name_allowed(name: str) -> None:
    assert _index_name(name) == name


@pytest.mark.parametrize("name", ["", "1abc", "a-b", "x) RETURN 1 //", "a b"])
def test_index_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid index name"):
        _index_name(name)


def test_graph_configs_default_to_embedded_sets() -> None:
    s = Settings(_env_file=None, neo4j_password="x")
    assert s.graph_configs == s.embedding_index_configs
    s = Settings(_env_file=None, neo4j_password="x", graph_chunk_configs=["structure-256"])
    assert s.graph_configs == ["structure-256"]


class FakeGraph:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.indexed: list[str] = []
        self.deleted: list[str] = []

    def index(self, record, elements, sets, extraction=None) -> dict:
        if self.fail:
            raise GraphUnavailableError("Neo4j down")
        self.indexed.append(record.document_id)
        return {"configs": list(sets)}

    def delete(self, document_id: str) -> bool:
        if self.fail:
            raise GraphUnavailableError("Neo4j down")
        self.deleted.append(document_id)
        return True


def _pipeline(tmp_path: Path, graph: FakeGraph) -> tuple[IngestionPipeline, Path]:
    configs = [ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, chunk_size=64)]
    pipeline = IngestionPipeline(
        DocumentStore(tmp_path / "data"), default_loaders(), max_bytes=1 << 20,
        chunker=Chunker(WhitespaceTokenizer, configs), graph=graph,
    )  # fmt: skip
    path = tmp_path / "n.txt"
    path.write_text("Alpha beta gamma.\n\nDelta epsilon.")
    return pipeline, path


def test_pipeline_writes_graph_and_deletes_graph_first(tmp_path: Path) -> None:
    graph = FakeGraph()
    pipeline, path = _pipeline(tmp_path, graph)
    record = pipeline.ingest(path).record
    assert graph.indexed == [record.document_id]
    assert record.metadata["graph"] == {"configs": ["structure-64"]}
    assert pipeline.delete(record.document_id) is True
    assert graph.deleted == [record.document_id]
    assert pipeline.store.read_record(record.document_id) is None


def test_graph_failure_is_a_warning_and_blocks_delete(tmp_path: Path) -> None:
    pipeline, path = _pipeline(tmp_path, FakeGraph(fail=True))
    record = pipeline.ingest(path).record
    assert record.status is IngestionStatus.COMPLETED
    assert "Graph write failed: GraphUnavailableError: Neo4j down" in record.warnings
    with pytest.raises(GraphUnavailableError):
        pipeline.delete(record.document_id)
    # Files stay when the graph can't be cleaned, so the graph never holds orphans.
    assert pipeline.store.read_record(record.document_id) is not None
