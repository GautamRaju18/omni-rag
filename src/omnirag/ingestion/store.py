"""File-based document store.

Layout under ``<data_dir>/documents/<document_id>/``::

    original.<ext>     the uploaded file, byte-for-byte (never modified or deleted by parsing)
    document.json      DocumentRecord: status, hash, metadata, warnings, errors
    elements.jsonl     one DocumentElement per line, in reading order
    assets/            extracted originals (embedded images, page renders, ...)
    chunks/<config>.jsonl   one Chunk per line per chunking configuration (Phase 3)

This is the file-level source of truth. Neo4j (graph, vectors, full-text) is built from it in
Phase 5; originals and assets always stay here.

Writes are atomic (write a temp file, then ``os.replace``), so a crash never leaves a
half-written ``document.json`` that would make the document unreadable.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from omnirag.ingestion.models import DocumentElement, DocumentRecord

_CONFIG_ID = re.compile(r"[a-z_]+(-[0-9]+)+(-ctx)?")  # e.g. "recursive-512-64", file-name safe


class DocumentStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.documents_dir = data_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def doc_dir(self, document_id: str) -> Path:
        path = (self.documents_dir / document_id).resolve()
        if path.parent != self.documents_dir.resolve():  # blocks "../" style IDs
            raise ValueError(f"Invalid document id: {document_id!r}")
        return path

    def original_path(self, document_id: str) -> Path | None:
        matches = sorted(self.doc_dir(document_id).glob("original.*"))
        return matches[0] if matches else None

    def resolve_asset(self, asset_ref: str) -> Path:
        path = (self.data_dir / asset_ref).resolve()
        if not path.is_relative_to(self.documents_dir.resolve()):
            raise ValueError(f"Asset reference escapes the data dir: {asset_ref!r}")
        return path

    # ----------------------------------------------------------------- writes
    def save_original(self, document_id: str, source: Path) -> Path:
        target = self.doc_dir(document_id) / f"original{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def save_asset(self, document_id: str, name: str, data: bytes) -> str:
        safe_name = Path(name).name  # drop any directory components
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError(f"Invalid asset name: {name!r}")
        target = self.doc_dir(document_id) / "assets" / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, data)
        return target.relative_to(self.data_dir).as_posix()

    def write_record(self, record: DocumentRecord) -> None:
        path = self.doc_dir(record.document_id) / "document.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, record.model_dump_json(indent=2).encode())

    def write_elements(self, document_id: str, elements: list[DocumentElement]) -> None:
        lines = "".join(e.model_dump_json() + "\n" for e in elements)
        atomic_write_bytes(self.doc_dir(document_id) / "elements.jsonl", lines.encode())

    def clear_derived(self, document_id: str) -> None:
        """Remove extraction outputs (before re-processing) but keep the original file."""
        doc_dir = self.doc_dir(document_id)
        (doc_dir / "elements.jsonl").unlink(missing_ok=True)
        shutil.rmtree(doc_dir / "assets", ignore_errors=True)
        shutil.rmtree(doc_dir / "chunks", ignore_errors=True)  # stale once elements change

    # ------------------------------------------------------------------ chunks
    def write_chunks(self, document_id: str, config_id: str, chunks: list[Any]) -> None:
        if not _CONFIG_ID.fullmatch(config_id):
            raise ValueError(f"Invalid chunk config id: {config_id!r}")
        lines = "".join(c.model_dump_json() + "\n" for c in chunks)
        target = self.doc_dir(document_id) / "chunks" / f"{config_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, lines.encode())

    def chunk_config_ids(self, document_id: str) -> list[str]:
        return sorted(p.stem for p in (self.doc_dir(document_id) / "chunks").glob("*.jsonl"))

    def read_chunks(self, document_id: str, config_id: str) -> list[Any]:
        from omnirag.chunking.models import Chunk  # local import: chunking depends on us

        if not _CONFIG_ID.fullmatch(config_id):
            raise ValueError(f"Invalid chunk config id: {config_id!r}")
        path = self.doc_dir(document_id) / "chunks" / f"{config_id}.jsonl"
        if not path.exists():
            return []
        with path.open("rb") as fh:
            return [Chunk.model_validate_json(line) for line in fh if line.strip()]

    def delete(self, document_id: str) -> bool:
        doc_dir = self.doc_dir(document_id)
        if not doc_dir.exists():
            return False
        shutil.rmtree(doc_dir)
        return True

    # ------------------------------------------------------------------ reads
    def read_record(self, document_id: str) -> DocumentRecord | None:
        path = self.doc_dir(document_id) / "document.json"
        if not path.exists():
            return None
        return DocumentRecord.model_validate_json(path.read_bytes())

    def read_elements(self, document_id: str) -> list[DocumentElement]:
        path = self.doc_dir(document_id) / "elements.jsonl"
        if not path.exists():
            return []
        with path.open("rb") as fh:
            return [DocumentElement.model_validate_json(line) for line in fh if line.strip()]

    def list_records(self) -> list[DocumentRecord]:
        records = [
            DocumentRecord.model_validate_json(p.read_bytes())
            for p in self.documents_dir.glob("*/document.json")
        ]
        return sorted(records, key=lambda r: r.created_at)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
