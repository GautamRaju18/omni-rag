from __future__ import annotations

import pytest

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import ElementType, ExtractionMethod
from omnirag.ingestion.normalize import normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ﬁle ﬂow", "file flow"),  # PDF ligatures
        ("100 Wh/kg", "100 Wh/kg"),  # non-breaking space
        ("zero​width﻿", "zerowidth"),  # invisible characters
        ("soft­hyphen", "softhyphen"),
        ("bell\x07 char", "bell char"),  # control characters
        ("  many    spaces\t\there  ", "many spaces here"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("p1\n\n\n\n\np2", "p1\n\np2"),
        ("é", "é"),  # NFC composes combining accents
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_nfc_does_not_rewrite_technical_symbols() -> None:
    # NFKC would turn these into "x2", "1⁄2" and "Na2": NFC must keep them.
    assert normalize_text("x² ½ Na₂") == "x² ½ Na₂"


def test_preserve_layout_keeps_indentation() -> None:
    code = "\n    def f():\n        return 1   \n"
    assert normalize_text(code, preserve_layout=True) == "    def f():\n        return 1"


def test_context_ids_are_deterministic_and_skip_empty() -> None:
    ctx = ExtractionContext("doc_abc", "x.txt")
    method = ExtractionMethod.TEXT_DECODE
    first = ctx.add(ElementType.PARAGRAPH, "hello", extraction_method=method)
    skipped = ctx.add(ElementType.PARAGRAPH, " \n​ ", extraction_method=method)
    second = ctx.add(ElementType.PARAGRAPH, "world", extraction_method=method)
    assert (first, skipped, second) == ("doc_abc-e00000", None, "doc_abc-e00001")
    assert [e.order for e in ctx.elements] == [0, 1]


@pytest.mark.parametrize("separator", ["\v", "\f", "\u2028"])
def test_line_separators_become_newlines_not_deleted(separator: str) -> None:
    # PowerPoint soft breaks arrive as "\v"; deleting them would glue words together.
    assert normalize_text(f"line{separator}break") == "line\nbreak"
