"""Chunk storage, the ingestion hook, and the tokenizer download (all offline)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import Chunker
from omnirag.chunking.tokenizer import TokenizerDownloadError, WhitespaceTokenizer, ensure_tokenizer
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore

CONFIGS = [
    ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, chunk_size=64),
    ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=32, chunk_overlap=8),
]


def _pipeline(tmp_path: Path, chunker) -> tuple[IngestionPipeline, DocumentStore]:
    store = DocumentStore(tmp_path / "data")
    return IngestionPipeline(store, default_loaders(), max_bytes=1 << 20, chunker=chunker), store


def _text_file(tmp_path: Path) -> Path:
    path = tmp_path / "n.txt"
    path.write_text("\n\n".join(" ".join(f"w{p}x{i}" for i in range(25)) for p in range(4)))
    return path


def test_ingestion_writes_chunk_sets(tmp_path: Path) -> None:
    pipeline, store = _pipeline(tmp_path, Chunker(WhitespaceTokenizer, CONFIGS))
    record = pipeline.ingest(_text_file(tmp_path)).record
    # 100 words: structure keeps 2 paragraphs per chunk; fixed windows start at 0, 24, 48, 72
    assert record.metadata["chunk_sets"] == {"structure-64": 2, "fixed-32-8": 4}
    assert store.chunk_config_ids(record.document_id) == ["fixed-32-8", "structure-64"]
    chunks = store.read_chunks(record.document_id, "structure-64")
    assert chunks[0].chunk_id == f"{record.document_id}-structure-64-00000"
    assert chunks[0].element_ids == [f"{record.document_id}-e00000", f"{record.document_id}-e00001"]


def test_reingest_clears_stale_chunks(tmp_path: Path) -> None:
    pipeline, store = _pipeline(tmp_path, Chunker(WhitespaceTokenizer, CONFIGS))
    path = _text_file(tmp_path)
    doc_id = pipeline.ingest(path).record.document_id
    pipeline.chunker = Chunker(WhitespaceTokenizer, CONFIGS[:1])
    pipeline.ingest(path, force=True)
    assert store.chunk_config_ids(doc_id) == ["structure-64"]  # fixed-32-8 removed


def test_chunking_failure_is_a_warning(tmp_path: Path) -> None:
    def broken():
        raise TokenizerDownloadError("offline")

    pipeline, store = _pipeline(tmp_path, Chunker(broken, CONFIGS))
    record = pipeline.ingest(_text_file(tmp_path)).record
    assert record.status is IngestionStatus.COMPLETED and record.element_count == 4
    assert "Chunking failed: TokenizerDownloadError: offline" in record.warnings
    assert store.chunk_config_ids(record.document_id) == []


@pytest.mark.parametrize("bad", ["../x", "structure", "Fixed-1", "a-1/b"])
def test_store_rejects_unsafe_config_ids(tmp_path: Path, bad: str) -> None:
    store = DocumentStore(tmp_path)
    with pytest.raises(ValueError, match="config id"):
        store.write_chunks("doc_00000000000000aa", bad, [])


def _client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, content=body))
    )


def test_tokenizer_download_verifies_checksum(tmp_path: Path) -> None:
    body = b'{"fake": "tokenizer"}'
    good = hashlib.sha256(body).hexdigest()
    path = ensure_tokenizer(tmp_path, "org/model", "a" * 40, good, client=_client(body))
    assert path.read_bytes() == body and path.parent.name == "org--model-aaaaaaaaaaaa"
    # Second call: no network needed (a client that would fail is never used).
    assert ensure_tokenizer(tmp_path, "org/model", "a" * 40, good, client=_client(b"", 500)) == path


def test_tokenizer_download_rejects_wrong_checksum(tmp_path: Path) -> None:
    with pytest.raises(TokenizerDownloadError, match="Checksum mismatch"):
        ensure_tokenizer(tmp_path, "org/model", "b" * 40, "0" * 64, client=_client(b"tampered"))
    assert list(tmp_path.rglob("*")) == [tmp_path / "org--model-bbbbbbbbbbbb"]  # no file left


def test_tokenizer_download_http_error(tmp_path: Path) -> None:
    with pytest.raises(TokenizerDownloadError, match="Could not download"):
        ensure_tokenizer(tmp_path, "org/model", "c" * 40, "0" * 64, client=_client(b"", 404))
