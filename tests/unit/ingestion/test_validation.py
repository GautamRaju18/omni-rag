from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from omnirag.core.errors import FileValidationError, UnsupportedFileTypeError
from omnirag.ingestion.models import FileKind
from omnirag.ingestion.validation import document_id_for, sha256_file, validate_file

MAX = 1024 * 1024


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _zip(tmp_path: Path, name: str, member: str) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, "<xml/>")
    return path


@pytest.mark.parametrize(
    ("name", "data", "kind"),
    [
        ("a.pdf", b"%PDF-1.7\n...", FileKind.PDF),
        ("a.PNG", b"\x89PNG\r\n\x1a\n....", FileKind.IMAGE),
        ("a.jpeg", b"\xff\xd8\xff\xe0....", FileKind.IMAGE),
        ("a.md", b"# hi", FileKind.MARKDOWN),
        ("a.htm", b"<p>x</p>", FileKind.HTML),
        ("a.csv", b"a,b\n1,2", FileKind.CSV),
        ("a.json", b"{}", FileKind.JSON),
        ("a.txt", b"hello", FileKind.TXT),
    ],
)
def test_valid_files(tmp_path: Path, name: str, data: bytes, kind: FileKind) -> None:
    assert validate_file(_write(tmp_path, name, data), max_bytes=MAX) is kind


@pytest.mark.parametrize(
    ("name", "member", "kind"),
    [
        ("r.docx", "word/document.xml", FileKind.DOCX),
        ("r.xlsx", "xl/workbook.xml", FileKind.XLSX),
        ("r.pptx", "ppt/presentation.xml", FileKind.PPTX),
    ],
)
def test_office_files_checked_by_zip_layout(
    tmp_path: Path, name: str, member: str, kind: FileKind
) -> None:
    assert validate_file(_zip(tmp_path, name, member), max_bytes=MAX) is kind


def test_xlsx_renamed_to_docx_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileValidationError, match="not a DOCX"):
        validate_file(_zip(tmp_path, "r.docx", "xl/workbook.xml"), max_bytes=MAX)


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("fake.pdf", b"MZ\x90\x00 windows executable", "no PDF signature"),
        ("fake.png", b"GIF89a....", "not a valid PNG or JPEG"),
        ("fake.docx", b"plain text", "bad ZIP"),
        ("bin.txt", b"abc\x00\x00def", "looks binary"),
        ("empty.md", b"", "empty"),
    ],
)
def test_invalid_files(tmp_path: Path, name: str, data: bytes, message: str) -> None:
    with pytest.raises(FileValidationError, match=message):
        validate_file(_write(tmp_path, name, data), max_bytes=MAX)


def test_oversized_file(tmp_path: Path) -> None:
    with pytest.raises(FileValidationError, match="too large"):
        validate_file(_write(tmp_path, "big.txt", b"x" * 101), max_bytes=100)


@pytest.mark.parametrize("name", ["virus.exe", "archive.zip", "noext"])
def test_unsupported_extension(tmp_path: Path, name: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_file(_write(tmp_path, name, b"data"), max_bytes=MAX)


def test_hash_and_id_are_content_addressed(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.txt", b"same bytes")
    b = _write(tmp_path, "b.txt", b"same bytes")
    c = _write(tmp_path, "c.txt", b"other bytes")
    assert sha256_file(a) == sha256_file(b) != sha256_file(c)
    assert document_id_for(sha256_file(a)) == f"doc_{sha256_file(a)[:16]}"
