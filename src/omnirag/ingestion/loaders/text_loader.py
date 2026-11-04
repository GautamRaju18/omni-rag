"""Plain text (.txt): paragraphs separated by blank lines.

Plain text has no reliable structure markers, so we deliberately do not guess headings here.
Guessing wrong would put false ``section_path`` values into citations. Line numbers are kept
so answers can still point to an exact location.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import read_text
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind

_PARAGRAPH_BREAK = re.compile(r"\n(?:[ \t]*\n)+")  # one or more blank lines


class TextLoader:
    name = "text"
    kinds = (FileKind.TXT,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        decoded = read_text(path)
        if decoded.confidence < 1.0:
            ctx.warn(f"Not valid UTF-8; decoded as {decoded.encoding}")
        text = decoded.text.replace("\r\n", "\n").replace("\r", "\n")

        line = 1
        cursor = 0
        for match in [*_PARAGRAPH_BREAK.finditer(text), None]:
            end = match.start() if match else len(text)
            block = text[cursor:end]
            # Leading newlines (only possible at the very start of the file) shift the start.
            start = line + len(block) - len(block.lstrip("\n"))
            ctx.add(
                ElementType.PARAGRAPH,
                block,
                extraction_method=ExtractionMethod.TEXT_DECODE,
                confidence=decoded.confidence,
                metadata={"line_start": start, "line_end": start + block.strip("\n").count("\n")},
            )
            if match:
                line += block.count("\n") + match.group().count("\n")
                cursor = match.end()
        return {"encoding": decoded.encoding, "line_count": text.count("\n") + 1}
