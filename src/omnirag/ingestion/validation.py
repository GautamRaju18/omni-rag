"""File validation: decide *what* a file is before any parser touches it.

Extensions are a claim, not a fact. A renamed ``.exe`` uploaded as ``report.pdf`` must be
rejected before PyMuPDF tries to parse it, so we check magic bytes (file signatures) and, for
Office formats, the internal ZIP layout. This uses only the standard library (no libmagic).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from omnirag.core.errors import FileValidationError, UnsupportedFileTypeError
from omnirag.ingestion.models import FileKind

EXTENSION_KINDS: dict[str, FileKind] = {
    ".pdf": FileKind.PDF,
    ".docx": FileKind.DOCX,
    ".xlsx": FileKind.XLSX,
    ".pptx": FileKind.PPTX,
    ".csv": FileKind.CSV,
    ".txt": FileKind.TXT,
    ".md": FileKind.MARKDOWN,
    ".markdown": FileKind.MARKDOWN,
    ".json": FileKind.JSON,
    ".html": FileKind.HTML,
    ".htm": FileKind.HTML,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
    ".png": FileKind.IMAGE,
}

TEXT_KINDS = {FileKind.CSV, FileKind.TXT, FileKind.MARKDOWN, FileKind.JSON, FileKind.HTML}

# Office Open XML files are ZIP archives; this member proves which application wrote it.
_OOXML_MARKERS = {
    FileKind.DOCX: "word/document.xml",
    FileKind.XLSX: "xl/workbook.xml",
    FileKind.PPTX: "ppt/presentation.xml",
}

# Zip-bomb guard: a 1 MB .docx can declare gigabytes of XML. python-docx, python-pptx and
# openpyxl decompress and parse members fully, so an absurd declared size would exhaust the
# 8 GB WSL VM. Legitimate Office files rarely exceed ~10x compression on their XML.
MAX_OOXML_UNCOMPRESSED_BYTES = 512 * 1024 * 1024

_SNIFF_BYTES = 8192


def kind_for_path(path: Path) -> FileKind:
    """Map the extension to a ``FileKind`` or raise ``UnsupportedFileTypeError``."""
    kind = EXTENSION_KINDS.get(path.suffix.lower())
    if kind is None:
        supported = ", ".join(sorted(EXTENSION_KINDS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{path.suffix or '<none>'}'",
            details={"file": path.name, "supported": supported},
        )
    return kind


def validate_file(path: Path, *, max_bytes: int) -> FileKind:
    """Return the verified ``FileKind`` of ``path`` or raise a validation error."""
    kind = kind_for_path(path)
    if not path.is_file():
        raise FileValidationError(f"Not a file: {path.name}")
    size = path.stat().st_size
    if size == 0:
        raise FileValidationError(f"File is empty: {path.name}")
    if size > max_bytes:
        raise FileValidationError(
            f"File too large: {path.name}",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )

    with path.open("rb") as fh:
        head = fh.read(_SNIFF_BYTES)

    if kind is FileKind.PDF and not head.startswith(b"%PDF-"):
        raise FileValidationError(f"{path.name} has a .pdf extension but no PDF signature")
    if kind is FileKind.IMAGE and not _is_image(head):
        raise FileValidationError(f"{path.name} is not a valid PNG or JPEG image")
    if kind in _OOXML_MARKERS:
        _check_ooxml(path, kind)
    if kind in TEXT_KINDS and b"\x00" in head:
        raise FileValidationError(f"{path.name} looks binary (NUL bytes) but claims to be text")
    return kind


def _is_image(head: bytes) -> bool:
    return head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff")


def _check_ooxml(path: Path, kind: FileKind) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise FileValidationError(f"{path.name} is not a valid Office file (bad ZIP)") from exc
    if _OOXML_MARKERS[kind] not in {info.filename for info in infos}:
        raise FileValidationError(
            f"{path.name} is a ZIP archive but not a {kind.value.upper()} document"
        )
    # Sizes come from the ZIP directory, so this check decompresses nothing.
    uncompressed = sum(info.file_size for info in infos)
    if uncompressed > MAX_OOXML_UNCOMPRESSED_BYTES:
        raise FileValidationError(
            f"{path.name} expands to {uncompressed // (1024 * 1024)} MiB when unpacked",
            details={"uncompressed_bytes": uncompressed, "max": MAX_OOXML_UNCOMPRESSED_BYTES},
        )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream the file in 1 MiB chunks so large files never load fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def document_id_for(sha256: str) -> str:
    """Content-addressed ID: identical bytes always get the same ID (dedup for free)."""
    return f"doc_{sha256[:16]}"
