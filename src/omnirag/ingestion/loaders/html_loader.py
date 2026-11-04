"""HTML (.html/.htm): main-content extraction with BeautifulSoup.

Web pages are mostly boilerplate (navigation, cookie banners, footers). Indexing it pollutes
retrieval: every page "mentions" the same menu items. We:

1. drop non-content tags (script, style, nav, footer, aside, forms, ...),
2. pick the main content root: ``<main>`` → ``<article>`` → ``<body>``,
3. walk blocks in document order, keeping heading hierarchy as ``section_path``.

Images are recorded (alt text + src) but never fetched: downloading arbitrary URLs from an
uploaded file would be an SSRF risk, and the platform is offline-first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import read_text
from omnirag.ingestion.models import ElementType, ExtractionMethod, FileKind
from omnirag.ingestion.tables import add_table

_METHOD = ExtractionMethod.BEAUTIFULSOUP
_REMOVE = (
    "script style noscript template svg iframe form nav footer aside button select input dialog"
).split()
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_INLINE = set(
    "a abbr b bdi bdo cite code data dfn em i kbd mark q s samp small span strong sub sup "
    "time u var label del ins".split()
)
_TEXT_BLOCKS = {"p", "blockquote", "address", "dd", "dt", "summary"}


def _meta_content(soup: BeautifulSoup, **attrs: str) -> str | None:
    tag = soup.find("meta", attrs=attrs)
    return tag.get("content") if isinstance(tag, Tag) else None


class _Walker:
    def __init__(self, ctx: ExtractionContext, confidence: float) -> None:
        self.ctx = ctx
        self.conf = confidence
        self.headings: list[tuple[int, str]] = []
        self.first_h1: str | None = None

    @property
    def section(self) -> list[str]:
        return [t for _, t in self.headings]

    def add(self, etype: ElementType, text: str, **kw: Any) -> str | None:
        if etype not in (ElementType.LIST, ElementType.CODE):
            text = " ".join(text.split())  # HTML whitespace (incl. newlines) is insignificant
        return self.ctx.add(
            etype,
            text,
            extraction_method=_METHOD,
            confidence=self.conf,
            section_path=self.section,
            **kw,
        )

    def walk(self, node: Tag) -> None:
        buffer: list[str] = []  # loose inline text directly inside containers

        def flush() -> None:
            if buffer:
                self.add(ElementType.PARAGRAPH, " ".join(buffer))
                buffer.clear()

        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                if child.strip():
                    buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in _INLINE:
                buffer.append(child.get_text(" "))
                continue
            if name == "br":
                continue
            flush()
            if name in _HEADINGS:
                self.heading(child)
            elif name in _TEXT_BLOCKS:
                self.add(ElementType.PARAGRAPH, child.get_text(" "), metadata={"tag": name})
            elif name in ("ul", "ol"):
                self.add(ElementType.LIST, _list_text(child), metadata={"ordered": name == "ol"})
            elif name == "pre":
                self.add(ElementType.CODE, child.get_text())
            elif name == "table":
                self.table(child)
            elif name == "img":
                self.image(child)
            elif name == "figure":
                self.figure(child)
            elif name == "hr":
                continue
            else:  # containers: div, section, article, header, main, ...
                self.walk(child)
        flush()

    def heading(self, tag: Tag) -> None:
        level = int(tag.name[1])
        text = " ".join(tag.get_text(" ").split())
        if not text:
            return
        self.headings = [h for h in self.headings if h[0] < level]
        self.add(ElementType.HEADING, text, metadata={"level": level})
        self.headings.append((level, text))
        if level == 1 and self.first_h1 is None:
            self.first_h1 = text

    def table(self, tag: Tag) -> None:
        rows = [
            [cell.get_text(" ") for cell in tr.find_all(["th", "td"])] for tr in tag.find_all("tr")
        ]
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            return
        first_tr = tag.find("tr")
        has_th = isinstance(first_tr, Tag) and first_tr.find("th") is not None
        caption = tag.find("caption")
        add_table(
            self.ctx,
            rows[0],
            rows[1:],
            extraction_method=_METHOD,
            # Without <th>, "first row is the header" is a guess: lower confidence.
            confidence=self.conf if has_th else self.conf * 0.8,
            section_path=self.section,
            metadata={
                "header_inferred": not has_th,
                "caption": caption.get_text(" ", strip=True) if caption else None,
            },
        )

    def image(self, tag: Tag, parent_id: str | None = None) -> str | None:
        return self.add(
            ElementType.IMAGE,
            tag.get("alt") or "",
            metadata={"src": tag.get("src"), "fetched": False},
            parent_id=parent_id,
        )

    def figure(self, tag: Tag) -> None:
        figure_id = None
        img = tag.find("img")
        if isinstance(img, Tag):
            figure_id = self.image(img)
        caption = tag.find("figcaption")
        if isinstance(caption, Tag):
            self.add(ElementType.CAPTION, caption.get_text(" "), parent_id=figure_id)


def _list_text(tag: Tag, depth: int = 0) -> str:
    """Render nested lists as indented ``-`` / ``1.`` lines."""
    lines = []
    for index, li in enumerate(tag.find_all("li", recursive=False), start=1):
        nested = li.find_all(["ul", "ol"], recursive=False)
        for n in nested:
            n.extract()
        marker = f"{index}." if tag.name == "ol" else "-"
        lines.append(f"{'  ' * depth}{marker} {' '.join(li.get_text(' ').split())}")
        lines.extend(_list_text(n, depth + 1) for n in nested)
    return "\n".join(line for line in lines if line.strip())


class HtmlLoader:
    name = "html"
    kinds = (FileKind.HTML,)

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        decoded = read_text(path)
        soup = BeautifulSoup(decoded.text, "html.parser")

        canonical = soup.find("link", rel="canonical")
        metadata = {
            "title": soup.title.get_text(strip=True) if soup.title else None,
            "description": _meta_content(soup, name="description"),
            "author": _meta_content(soup, name="author"),
            "language": soup.html.get("lang") if soup.html else None,
            "canonical_url": canonical.get("href") if isinstance(canonical, Tag) else None,
            "encoding": decoded.encoding,
        }

        for tag in soup.find_all(_REMOVE):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        metadata["content_root"] = root.name if isinstance(root, Tag) else "document"

        walker = _Walker(ctx, decoded.confidence)
        walker.walk(root)  # BeautifulSoup itself is a Tag, so every branch is walkable
        metadata["title"] = metadata["title"] or walker.first_h1
        return metadata
