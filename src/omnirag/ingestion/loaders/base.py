"""Loader contract and helpers shared by all loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import FileKind


class Loader(Protocol):
    """A loader turns one file into elements via ``ctx.add`` and returns document metadata.

    Loaders are synchronous: parsing is CPU-bound, so the async API layer runs them in a worker
    thread instead of blocking the event loop.
    """

    name: str
    kinds: tuple[FileKind, ...]

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str
    confidence: float  # 1.0 when the encoding is certain, lower when we fell back


# Order matters: UTF-8 is strict, so a successful decode is almost certainly right.
# cp1252 (Windows "ANSI") covers most legacy Western files; latin-1 never fails (last resort).
_ENCODINGS: tuple[tuple[str, float], ...] = (("utf-8-sig", 1.0), ("cp1252", 0.8), ("latin-1", 0.6))


def read_text(path: Path) -> DecodedText:
    """Decode a text file, recording which encoding worked and how sure we are."""
    raw = path.read_bytes()
    for encoding, confidence in _ENCODINGS:
        try:
            return DecodedText(raw.decode(encoding), encoding, confidence)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decoding cannot fail")  # pragma: no cover
