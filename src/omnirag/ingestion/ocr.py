"""OCR with the Tesseract command-line tool.

We call the ``tesseract`` binary directly instead of a wrapper library. The integration is
one subprocess call, and its TSV output gives what a RAG system needs: **per-word confidence**
and **pixel boxes**, grouped by Tesseract into blocks, paragraphs and lines.

    tesseract stdin stdout -l eng --psm 3 [--dpi 300] tsv

TSV columns: level, page_num, block_num, par_num, line_num, word_num, left, top, width, height,
conf, text. Rows with ``conf == -1`` are layout rows (block/paragraph/line), not words.

Security: the image arrives on stdin, arguments are a list (no shell), and the language string
is validated by ``Settings`` (``eng`` / ``eng+deu``). Nothing from the document reaches the
command line.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess  # noqa: S404 (fixed argument list, no shell)
import time
from dataclasses import dataclass, field
from functools import cached_property


class OcrError(RuntimeError):
    """Tesseract is missing, timed out or failed."""


@dataclass(frozen=True)
class OcrParagraph:
    text: str
    confidence: float  # mean word confidence, 0..1
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels
    words: int
    low_confidence_words: int  # words below 60%: likely misreads worth flagging


@dataclass(frozen=True)
class OcrResult:
    paragraphs: list[OcrParagraph] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def mean_confidence(self) -> float:
        words = sum(p.words for p in self.paragraphs)
        if not words:
            return 0.0
        return sum(p.confidence * p.words for p in self.paragraphs) / words


_LOW_WORD_CONF = 60.0


def parse_tsv(tsv: str) -> list[OcrParagraph]:
    """Group Tesseract TSV word rows into paragraphs, keeping OCR line breaks.

    Lines stay separate: in notebooks, forms and tables each line is often its own fact, and
    Tesseract's paragraph grouping is only a guess. Downstream chunking can rejoin prose.
    """
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text or float(row["conf"]) < 0:
            continue
        groups.setdefault((int(row["block_num"]), int(row["par_num"])), []).append(row)

    paragraphs = []
    for words in groups.values():
        lines: dict[int, list[str]] = {}
        for w in words:
            lines.setdefault(int(w["line_num"]), []).append(w["text"].strip())
        confs = [float(w["conf"]) for w in words]
        x0 = min(int(w["left"]) for w in words)
        y0 = min(int(w["top"]) for w in words)
        x1 = max(int(w["left"]) + int(w["width"]) for w in words)
        y1 = max(int(w["top"]) + int(w["height"]) for w in words)
        paragraphs.append(
            OcrParagraph(
                text="\n".join(" ".join(line) for line in lines.values()),
                confidence=round(sum(confs) / len(confs) / 100, 4),
                bbox=(x0, y0, x1, y1),
                words=len(words),
                low_confidence_words=sum(c < _LOW_WORD_CONF for c in confs),
            )
        )
    return paragraphs


class TesseractOcr:
    """Stateless (safe to share across threads); every call is its own subprocess."""

    def __init__(
        self,
        cmd: str = "tesseract",
        languages: str = "eng",
        timeout_s: float = 120.0,
        page_segmentation_mode: int = 3,  # 3 = fully automatic layout analysis
    ) -> None:
        self.cmd = cmd
        self.languages = languages
        self.timeout_s = timeout_s
        self.psm = page_segmentation_mode

    @cached_property
    def executable(self) -> str | None:
        return shutil.which(self.cmd)

    def available(self) -> bool:
        return self.executable is not None

    @cached_property
    def version(self) -> str:
        out = self._run(["--version"], b"", timeout_s=10).decode(errors="replace")
        return out.splitlines()[0].strip() if out else "tesseract (unknown version)"

    def recognize(self, image: bytes, *, dpi: int | None = None) -> OcrResult:
        """OCR one encoded image (PNG/JPEG bytes). ``dpi`` helps Tesseract size its model."""
        args = ["stdin", "stdout", "-l", self.languages, "--psm", str(self.psm)]
        if dpi:
            args += ["--dpi", str(dpi)]
        started = time.perf_counter()
        tsv = self._run([*args, "tsv"], image, timeout_s=self.timeout_s)
        return OcrResult(
            paragraphs=parse_tsv(tsv.decode("utf-8", errors="replace")),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _run(self, args: list[str], stdin: bytes, *, timeout_s: float) -> bytes:
        if self.executable is None:
            raise OcrError(f"Tesseract not found ({self.cmd!r}); install tesseract-ocr")
        try:
            proc = subprocess.run(  # noqa: S603 (absolute path, list args, no shell)
                [self.executable, *args],
                input=stdin,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrError(f"Tesseract timed out after {timeout_s:.0f} s") from exc
        if proc.returncode != 0:
            raise OcrError(f"Tesseract failed: {proc.stderr.decode(errors='replace')[-300:]}")
        return proc.stdout
