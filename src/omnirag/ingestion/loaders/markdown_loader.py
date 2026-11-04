"""Markdown (.md): structure-preserving extraction with markdown-it-py.

Rather than splitting on ``#`` with regexes (which breaks on ``#`` inside code blocks, setext
headings, etc.), we use a CommonMark-compliant parser and walk its top-level block tokens.
Each block's ``map`` gives its source line range, kept as provenance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from markdown_it.token import Token

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import read_text
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_METHOD = ExtractionMethod.MARKDOWN_IT


def _close_index(tokens: list[Token], start: int) -> int:
    """Index of the token closing the block opened at ``start``."""
    level = tokens[start].level
    for j in range(start + 1, len(tokens)):
        if tokens[j].level == level and tokens[j].nesting == -1:
            return j
    return len(tokens) - 1


def _lines(token: Token) -> dict[str, int]:
    return {"line_start": token.map[0] + 1, "line_end": token.map[1]} if token.map else {}


class MarkdownLoader:
    name = "markdown"
    kinds = (FileKind.MARKDOWN,)

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark").enable("table")

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        decoded = read_text(path)
        text = decoded.text.replace("\r\n", "\n")
        front_matter = None
        if match := _FRONT_MATTER.match(text):
            front_matter = match.group()[4:-5]
            # Blank the front matter instead of removing it so line numbers stay correct.
            text = "\n" * match.group().count("\n") + text[match.end() :]

        source_lines = text.split("\n")
        tokens = self._md.parse(text)
        headings: list[tuple[int, str]] = []
        title: str | None = None
        conf = decoded.confidence

        def section() -> list[str]:
            return [t for _, t in headings]

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            end = _close_index(tokens, i) if tok.nesting == 1 else i
            meta = _lines(tok)

            if tok.type == "heading_open":
                level = int(tok.tag[1])
                heading = tokens[i + 1].content
                headings = [h for h in headings if h[0] < level]
                ctx.add(
                    ElementType.HEADING,
                    heading,
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata={**meta, "level": level},
                )
                headings.append((level, heading))
                if level == 1 and title is None:
                    title = heading

            elif tok.type == "paragraph_open":
                self._paragraph(ctx, tokens[i + 1], meta, section(), conf)

            elif tok.type in ("fence", "code_block"):
                ctx.add(
                    ElementType.CODE,
                    tok.content,
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata={**meta, "language": tok.info.strip() or None},
                )

            elif tok.type in ("bullet_list_open", "ordered_list_open"):
                ctx.add(
                    ElementType.LIST,
                    "\n".join(source_lines[tok.map[0] : tok.map[1]]) if tok.map else "",
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata={**meta, "ordered": tok.type == "ordered_list_open"},
                )

            elif tok.type == "blockquote_open":
                quote = "\n".join(source_lines[tok.map[0] : tok.map[1]]) if tok.map else ""
                ctx.add(
                    ElementType.PARAGRAPH,
                    re.sub(r"^\s*>\s?", "", quote, flags=re.MULTILINE).replace("\n", " "),
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata={**meta, "blockquote": True},
                )

            elif tok.type == "table_open":
                header, rows = self._table_cells(tokens[i : end + 1])
                add_table(
                    ctx,
                    header,
                    rows,
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata=meta,
                )

            elif tok.type == "html_block":
                ctx.add(
                    ElementType.PARAGRAPH,
                    BeautifulSoup(tok.content, "html.parser").get_text(" "),
                    extraction_method=_METHOD,
                    confidence=conf,
                    section_path=section(),
                    metadata={**meta, "raw_html": True},
                )

            i = end + 1

        return {
            "title": title,
            "encoding": decoded.encoding,
            "front_matter": front_matter,
            "line_count": len(source_lines),
        }

    @staticmethod
    def _paragraph(
        ctx: ExtractionContext,
        inline: Token,
        meta: dict[str, int],
        section_path: list[str],
        conf: float,
    ) -> None:
        children = inline.children or []
        images = [c for c in children if c.type == "image"]
        has_text = any(c.type == "text" and c.content.strip() for c in children)
        if has_text or not images:
            ctx.add(
                ElementType.PARAGRAPH,
                # CommonMark: a single newline inside a paragraph is a soft break (= space).
                inline.content.replace("\n", " "),
                extraction_method=_METHOD,
                confidence=conf,
                section_path=section_path,
                metadata=meta,
            )
        for image in images:
            # Referenced images are not fetched (may be remote); alt text + src are kept.
            ctx.add(
                ElementType.IMAGE,
                image.content,
                extraction_method=_METHOD,
                confidence=conf,
                section_path=section_path,
                metadata={**meta, "src": image.attrs.get("src"), "fetched": False},
            )

    @staticmethod
    def _table_cells(tokens: list[Token]) -> tuple[list[str], list[list[str]]]:
        header: list[str] = []
        rows: list[list[str]] = []
        current: list[str] = []
        in_head = False
        for tok in tokens:
            if tok.type == "thead_open":
                in_head = True
            elif tok.type == "thead_close":
                in_head = False
            elif tok.type == "tr_open":
                current = []
            elif tok.type == "inline":
                current.append(tok.content)
            elif tok.type == "tr_close":
                if in_head:
                    header = current
                else:
                    rows.append(current)
        return header, rows
