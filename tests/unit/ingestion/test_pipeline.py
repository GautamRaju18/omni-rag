from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omnirag.core.errors import LoaderNotAvailableError, UnsupportedFileTypeError
from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import FileKind, IngestionStatus
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(tmp_path / "data")


@pytest.fixture
def pipeline(store: DocumentStore) -> IngestionPipeline:
    return IngestionPipeline(store, default_loaders(), max_bytes=1024 * 1024)


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nBody text.\n", encoding="utf-8")
    return path


def test_ingest_persists_record_elements_and_original(
    pipeline: IngestionPipeline, store: DocumentStore, md_file: Path
) -> None:
    result = pipeline.ingest(md_file)
    record = result.record
    assert not result.duplicate
    assert record.status is IngestionStatus.COMPLETED
    assert record.element_count == 2
    assert record.element_counts_by_type == {"heading": 1, "paragraph": 1}
    assert record.metadata["title"] == "Title"
    assert record.duration_ms is not None

    assert store.read_record(record.document_id) == record
    elements = store.read_elements(record.document_id)
    assert [e.content for e in elements] == ["Title", "Body text."]
    original = store.original_path(record.document_id)
    assert original is not None and original.read_bytes() == md_file.read_bytes()


def test_duplicate_is_detected_by_content_not_name(
    pipeline: IngestionPipeline, md_file: Path, tmp_path: Path
) -> None:
    first = pipeline.ingest(md_file)
    copy = tmp_path / "renamed.md"
    copy.write_bytes(md_file.read_bytes())
    second = pipeline.ingest(copy)
    assert second.duplicate
    assert second.record.document_id == first.record.document_id
    assert second.record.source_file == "notes.md"  # the stored record is not overwritten


def test_force_reingest_gives_identical_element_ids(
    pipeline: IngestionPipeline, store: DocumentStore, md_file: Path
) -> None:
    first = pipeline.ingest(md_file)
    ids_before = [e.element_id for e in store.read_elements(first.record.document_id)]
    second = pipeline.ingest(md_file, force=True)
    ids_after = [e.element_id for e in store.read_elements(second.record.document_id)]
    assert not second.duplicate
    assert ids_before == ids_after
    assert second.record.created_at == first.record.created_at


class _ExplodingLoader:
    name = "exploding"
    kinds = (FileKind.TXT,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        ctx.warn("about to fail")
        raise RuntimeError("parser crashed")


def test_loader_crash_marks_failed_and_keeps_original(store: DocumentStore, tmp_path: Path) -> None:
    pipeline = IngestionPipeline(store, {FileKind.TXT: _ExplodingLoader()}, max_bytes=1024)
    path = tmp_path / "x.txt"
    path.write_text("content")
    result = pipeline.ingest(path)
    assert result.record.status is IngestionStatus.FAILED
    assert result.record.error == "RuntimeError: parser crashed"
    assert result.record.warnings == ["about to fail"]
    assert store.read_record(result.record.document_id).status is IngestionStatus.FAILED
    assert store.original_path(result.record.document_id) is not None

    # A failed document is retried (not treated as a duplicate) once a loader works.
    retry = IngestionPipeline(store, default_loaders(), max_bytes=1024).ingest(path)
    assert not retry.duplicate and retry.record.status is IngestionStatus.COMPLETED


def test_known_kind_without_loader_raises_before_storing(
    store: DocumentStore, tmp_path: Path
) -> None:
    loaders = {k: v for k, v in default_loaders().items() if k is not FileKind.PDF}
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7 minimal")
    with pytest.raises(LoaderNotAvailableError):
        IngestionPipeline(store, loaders, max_bytes=1024).ingest(path)
    assert store.list_records() == []


def test_corrupt_pdf_fails_but_keeps_original(
    pipeline: IngestionPipeline, store: DocumentStore, tmp_path: Path
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7 minimal")  # right signature, no PDF structure
    record = pipeline.ingest(path).record
    assert record.status is IngestionStatus.FAILED and record.error
    assert store.original_path(record.document_id).read_bytes() == b"%PDF-1.7 minimal"


def test_unsupported_type_raises(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    path = tmp_path / "tool.exe"
    path.write_bytes(b"MZ")
    with pytest.raises(UnsupportedFileTypeError):
        pipeline.ingest(path)


def test_store_rejects_path_traversal(store: DocumentStore) -> None:
    with pytest.raises(ValueError, match="Invalid document id"):
        store.doc_dir("../../etc")
    with pytest.raises(ValueError, match="escapes"):
        store.resolve_asset("../../etc/passwd")


def test_assets_are_saved_under_document(store: DocumentStore) -> None:
    ref = store.save_asset("doc_1", "../sneaky/fig.png", b"\x89PNG")
    assert ref == "documents/doc_1/assets/fig.png"
    assert store.resolve_asset(ref).read_bytes() == b"\x89PNG"


def test_delete_removes_everything(
    pipeline: IngestionPipeline, store: DocumentStore, md_file: Path
) -> None:
    doc_id = pipeline.ingest(md_file).record.document_id
    assert store.delete(doc_id) is True
    assert store.read_record(doc_id) is None
    assert store.delete(doc_id) is False
