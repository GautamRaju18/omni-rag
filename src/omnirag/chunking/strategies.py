"""The chunking strategies.

All of them turn one document's elements into ``Chunk``s, and all are token-aware (bge-m3
tokens). They differ in what they respect:

| Strategy     | Unit                  | Respects                            | Tables / figures   |
|--------------|-----------------------|-------------------------------------|--------------------|
| fixed        | token window          | nothing (baseline)                  | cut anywhere       |
| recursive    | paragraph→word pieces | paragraph / sentence boundaries     | cut at line breaks |
| structure    | whole elements        | sections, element boundaries        | treated as text    |
| table        | whole elements        | + tables are their own chunks       | header repeated    |
| multimodal   | whole elements        | + image/chart groups                | caption+OCR+desc.  |
| hierarchical | 3 nested levels       | top level = whole top-level section | as multimodal      |
| parent_child | small children        | children indexed, parents returned  | as multimodal      |

Structure-family strategies never split an element unless it alone exceeds the size. Then it
is split recursively, and the pieces are marked ``partial``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omnirag.chunking.models import Chunk, ChunkingConfig, ChunkingStrategy
from omnirag.chunking.semantic import Embedder, distances, sentence_spans, window_texts
from omnirag.chunking.text import (
    Assembled,
    Segment,
    split_spans,
    strip_span,
    token_windows,
)
from omnirag.chunking.tokenizer import TokenCounter
from omnirag.ingestion.models import DocumentElement, DocumentRecord, ElementType

S = ChunkingStrategy
_SKIP_IN_FLOW = {ElementType.TABLE_ROW}  # rows are already inside their TABLE's Markdown
_FIGURES = {ElementType.IMAGE, ElementType.FIGURE}
_FAMILY = {"table_row": "table", "image": "figure", "caption": "figure", "description": "figure"}


def _segment(e: DocumentElement) -> Segment:
    return Segment(0, 0, e.element_id, e.element_type.value, e.page_number, e.section_path,
                   e.origin.value, e.confidence)  # fmt: skip


@dataclass
class Block:
    """A unit the structure-family strategies place whole: one element, a table, or a figure."""

    kind: str  # "text" | "heading" | "table" | "figure"
    parts: list[tuple[str, DocumentElement | None]]  # (text, element) pieces in order
    section_path: list[str]
    standalone: bool = False  # tables/figures in table/multimodal mode: never mixed with prose
    table_rows: list[tuple[str, DocumentElement | None]] = field(default_factory=list)
    table_header: str = ""  # "| a | b |\n| --- | --- |" (+ optional title line)


class _Builder:
    """Shared machinery: element lookup, block building, packing, chunk construction."""

    def __init__(
        self,
        record: DocumentRecord,
        elements: list[DocumentElement],
        config: ChunkingConfig,
        counter: TokenCounter,
    ) -> None:
        self.record, self.elements, self.config, self.counter = record, elements, config, counter
        self.children: dict[str, list[DocumentElement]] = defaultdict(list)
        for e in elements:
            if e.parent_element_id:
                self.children[e.parent_element_id].append(e)
        self.chunks: list[Chunk] = []
        # chunk_id -> (assembled text, span): children split their parent's exact span, so
        # every level keeps element-level provenance without searching for strings.
        self.sources: dict[str, tuple[Assembled, tuple[int, int]]] = {}
        title = record.metadata.get("title") or record.source_file
        self.title = str(title)

    # ------------------------------------------------------------------ blocks
    def descendants(self, element: DocumentElement) -> list[DocumentElement]:
        out, stack = [], list(reversed(self.children.get(element.element_id, [])))
        while stack:
            e = stack.pop()
            out.append(e)
            stack.extend(reversed(self.children.get(e.element_id, [])))
        return out

    def blocks(self, mode: S) -> list[Block]:
        tables = mode in (S.TABLE, S.MULTIMODAL)
        figures = mode is S.MULTIMODAL
        consumed: set[str] = set()
        blocks: list[Block] = []
        for e in self.elements:
            if e.element_id in consumed or e.element_type in _SKIP_IN_FLOW:
                continue
            if figures and e.element_type in _FIGURES:
                group = self.descendants(e)
                consumed.update(g.element_id for g in group)
                if (block := self._figure_block(e, group)) is not None:
                    blocks.append(block)
                continue
            if e.element_type is ElementType.TABLE:
                rows = [c for c in self.children.get(e.element_id, [])
                        if c.element_type is ElementType.TABLE_ROW]  # fmt: skip
                blocks.append(self._table_block(e, rows, standalone=tables))
                continue
            if not e.content:
                continue
            kind = (
                "heading" if e.element_type in (ElementType.HEADING, ElementType.TITLE) else "text"
            )
            path = e.section_path + [e.content] if kind == "heading" else e.section_path
            blocks.append(Block(kind, [(e.content, e)], path))
        return blocks

    def _table_block(self, table: DocumentElement, rows: list[DocumentElement], *,
                     standalone: bool) -> Block:  # fmt: skip
        lines = table.content.split("\n")
        caption = table.metadata.get("caption")
        title = f"Table: {caption}" if caption else ""
        header = "\n".join(([title] if title else []) + lines[:2])
        by_index = {r.metadata.get("row_index"): r for r in rows}
        row_parts = [(line, by_index.get(i)) for i, line in enumerate(lines[2:])]
        block = Block("table", [(header, table), *row_parts], table.section_path, standalone)
        block.table_header, block.table_rows = header, row_parts
        return block

    def _figure_block(self, figure: DocumentElement, group: list[DocumentElement]) -> Block | None:
        parts: list[tuple[str, DocumentElement | None]] = []
        captions = [g for g in group if g.element_type is ElementType.CAPTION]
        for c in captions:
            parts.append((f"Figure: {c.content}", c))
        if figure.element_type is ElementType.FIGURE and figure.content:
            parts.append((figure.content, figure))  # e.g. "Chart: Capacity retention ..."
        elif figure.content:
            parts.append((f"Alt text: {figure.content}", figure))
        for g in group:
            if g.element_type is ElementType.TABLE:  # chart data
                parts.append((g.content, g))
                parts.extend((r.content, r) for r in self.children.get(g.element_id, []))
            elif g.origin.value == "ocr":
                parts.append(
                    (f"Text in image (OCR, confidence {g.confidence:.2f}): {g.content}", g)
                )
            elif g.element_type is ElementType.DESCRIPTION:
                model = g.metadata.get("model", "a vision model")
                parts.append((f"Image description (generated by {model}): {g.content}", g))
        if not parts:
            return None  # nothing searchable: no caption, alt text, OCR or description
        # Keep the IMAGE itself in the chunk's provenance even when it has no text.
        if figure.element_type is ElementType.IMAGE and not figure.content:
            parts.insert(0, ("", figure))
        return Block("figure", parts, figure.section_path, standalone=True)

    def render(self, blocks: list[Block]) -> Assembled:
        out = Assembled("")
        for b in blocks:
            sep = "\n" if b.kind in ("table", "figure") else "\n\n"  # rows/labels on own lines
            for i, (text, element) in enumerate(b.parts):
                seg = _segment(element) if element is not None else None
                out.add(text, seg, sep="\n\n" if i == 0 else sep)
        return out

    # ----------------------------------------------------------------- packing
    def pack(self, blocks: list[Block], size: int, depth: int | None = None) -> list[list[Block]]:
        """Group blocks into chunks of at most ``size`` tokens: never across a section, never
        mixing standalone blocks with prose. Oversized blocks become their own group (split
        later).

        ``depth=1`` groups by *top-level* section instead of exact section, so subsections
        (and their tables/figures) join their parent section: the hierarchical top level."""
        grouped = depth is not None
        groups: list[list[Block]] = []
        current: list[Block] = []
        tokens, key = 0, None
        pending: list[Block] = []  # headings wait for the content they introduce

        def flush() -> None:
            nonlocal current, tokens
            if current:
                groups.append(pending + current)
                pending.clear()
            current, tokens = [], 0

        for b in blocks:
            k = tuple(b.section_path[:depth]) if grouped else tuple(b.section_path)
            if b.kind == "heading" and not (grouped and k == key and current):
                flush()
                pending.append(b)
                key = k
                continue
            n = self.counter.count(self.render([b]).text)
            if (b.standalone and not grouped) or n > size:
                flush()
                groups.append(pending + [b])
                pending.clear()
                continue
            if k != key or tokens + n > size:
                flush()
                key = k
            current.append(b)
            tokens += n
        flush()
        if pending:  # trailing headings with no content
            groups.append(pending)
        return groups

    # ------------------------------------------------------------------ chunks
    def emit_group(self, group: list[Block], size: int, **kw: Any) -> list[Chunk]:
        """One chunk per group, or several if it's oversized (tables split by rows)."""
        # A standalone figure/table group *is* that type, whatever has the most characters
        # (a chart's data rows outweigh its title, but the chunk is still a figure).
        standalone = [b.kind for b in group if b.standalone]
        if standalone and "element_type" not in kw:
            kw["element_type"] = standalone[0]
        assembled = self.render(group)
        if self.counter.count(assembled.text) <= size:
            return [self.emit(assembled, (0, len(assembled.text)), **kw)]
        table = next((b for b in group if b.kind == "table"), None)
        if table is not None and len(group) <= 2:  # [heading?] + table: split by rows
            return self._split_table(group, table, size, **kw)
        spans = split_spans(assembled.text, (0, len(assembled.text)), self.counter, size,
                            self.config.chunk_overlap,
                            keep_with_next=assembled.heading_ends())  # fmt: skip
        return [self.emit(assembled, span, **kw) for span in spans]

    def _split_table(self, group: list[Block], table: Block, size: int, **kw: Any) -> list[Chunk]:
        prefix = [b for b in group if b is not table]
        header_part = table.parts[0]
        out: list[Chunk] = []
        batch: list[tuple[str, DocumentElement | None]] = []

        def emit_batch() -> None:
            if not batch:
                return
            part = Block("table", [header_part, *batch], table.section_path, True)
            assembled = self.render(prefix + [part])
            out.append(self.emit(assembled, (0, len(assembled.text)), **kw))
            prefix.clear()  # the heading goes with the first part only

        for row in table.table_rows:
            trial = Block("table", [header_part, *batch, row], table.section_path, True)
            if batch and self.counter.count(self.render(prefix + [trial]).text) > size:
                emit_batch()
                batch = []
            batch.append(row)
        emit_batch()
        return out

    def emit(
        self,
        assembled: Assembled,
        span: tuple[int, int],
        *,
        parent: Chunk | None = None,
        level: int = 0,
        retrievable: bool = True,
        element_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        a, b = strip_span(assembled.text, span)
        content = assembled.text[a:b]
        hits = assembled.overlapping((a, b))
        segs = [s for s, _ in hits]
        type_chars: Counter[str] = Counter()
        for s in segs:
            type_chars[s.element_type] += max(s.end - s.start, 1)
        types = {s.element_type for s in segs}
        # Majority by characters, with figure parts and table rows reported as their whole.
        majority = type_chars.most_common(1)[0][0] if type_chars else "paragraph"
        dominant = element_type or _FAMILY.get(majority, majority)
        pages = sorted({s.page_number for s in segs if s.page_number is not None})
        sections = [s.section_path for s in segs if s.element_type not in ("heading", "title")]
        index = len(self.chunks)
        config_id = self.config.config_id
        chunk = Chunk(
            chunk_id=f"{self.record.document_id}-{config_id}-{index:05d}",
            document_id=self.record.document_id,
            source_file=self.record.source_file,
            chunking_strategy=self.config.strategy,
            config_id=config_id,
            chunk_index=index,
            content=content,
            token_count=self.counter.count(content),
            page_number=pages[0] if pages else None,
            page_numbers=pages,
            section_path=(sections[0] if sections else (segs[0].section_path if segs else [])),
            element_type=dominant,
            element_ids=list(dict.fromkeys(s.element_id for s in segs)),
            partial_element_ids=list(dict.fromkeys(s.element_id for s, p in hits if p)),
            parent_chunk_id=parent.chunk_id if parent else None,
            level=level,
            retrievable=retrievable,
            origins=sorted({s.origin for s in segs}),
            min_confidence=min((s.confidence for s in segs), default=1.0),
            metadata={"element_types": sorted(types), "char_span": [a, b], **(metadata or {})},
        )
        if self.config.contextual_headers:
            section = " > ".join(chunk.section_path)
            chunk.context = f"Document: {self.title}" + (f"\nSection: {section}" if section else "")
            chunk.metadata["context_token_count"] = self.counter.count(chunk.context)
        self.chunks.append(chunk)
        self.sources[chunk.chunk_id] = (assembled, (a, b))
        return chunk

    def split_children(
        self,
        parent: Chunk,
        size: int,
        overlap: int,
        *,
        level: int,
        retrievable: bool = True,
        skip_if_fits: bool = False,
    ) -> list[Chunk]:
        """Children of at most ``size`` tokens from the parent's own span (exact provenance).

        ``skip_if_fits``: a parent already within ``size`` gets no children, because a child
        identical to its parent would only be a duplicate hit in the index."""
        if skip_if_fits and parent.token_count <= size:
            return []
        assembled, span = self.sources[parent.chunk_id]
        return [
            self.emit(assembled, child, parent=parent, level=level, retrievable=retrievable)
            for child in split_spans(
                assembled.text,
                span,
                self.counter,
                size,
                overlap,
                keep_with_next=assembled.heading_ends(),
            )  # fmt: skip
        ]


# ---------------------------------------------------------------- strategies
def _linear(builder: _Builder) -> Assembled:
    """Plain document text for the baselines: every element in order, tables as Markdown."""
    out = Assembled("")
    for e in builder.elements:
        if e.element_type not in _SKIP_IN_FLOW and e.content:
            out.add(e.content, _segment(e))
    return out


def chunk_document(
    record: DocumentRecord,
    elements: list[DocumentElement],
    config: ChunkingConfig,
    counter: TokenCounter,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    b = _Builder(record, elements, config, counter)
    c = config
    if c.strategy is S.SEMANTIC:
        if embedder is None:
            raise ValueError("Semantic chunking needs an embedder (bge-m3 via Ollama)")
        _semantic(b, embedder)
        return b.chunks
    if c.strategy is S.FIXED:
        text = _linear(b)
        for span in token_windows(text.text, counter, c.chunk_size, c.chunk_overlap):
            b.emit(text, span)
    elif c.strategy is S.RECURSIVE:
        text = _linear(b)
        for span in split_spans(text.text, (0, len(text.text)), counter, c.chunk_size,
                                c.chunk_overlap):  # fmt: skip
            b.emit(text, span)
    elif c.strategy in (S.STRUCTURE, S.TABLE, S.MULTIMODAL):
        for group in b.pack(b.blocks(c.strategy), c.chunk_size):
            b.emit_group(group, c.chunk_size)
    elif c.strategy is S.HIERARCHICAL:
        # Level 0: section-respecting groups up to parent_size. Levels 1 and 2: recursive
        # splits of the parent's own span (paragraph breaks first, so elements mostly stay
        # whole). Every level is retrievable; retrieval can match coarse or fine.
        leaf_overlap = min(c.chunk_overlap, c.child_size // 4)
        for group in b.pack(b.blocks(S.MULTIMODAL), c.parent_size, depth=1):
            for top in b.emit_group(group, c.parent_size, level=0):
                # A top chunk that already fits chunk_size has no middle level, but still
                # gets small leaves if it is longer than child_size.
                mids = b.split_children(top, c.chunk_size, 0, level=1, skip_if_fits=True)
                for mid in mids or [top]:
                    b.split_children(mid, c.child_size, leaf_overlap, level=mid.level + 1,
                                     skip_if_fits=True)  # fmt: skip
    elif c.strategy is S.PARENT_CHILD:
        # Parents (multimodal chunks) are what the LLM reads; small children are what the
        # retriever matches. Parents are stored but not indexed.
        for group in b.pack(b.blocks(S.MULTIMODAL), c.chunk_size):
            for parent in b.emit_group(group, c.chunk_size, retrievable=False):
                b.split_children(parent, c.child_size, min(c.chunk_overlap, c.child_size // 4),
                                 level=1)  # fmt: skip
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"Unknown strategy {c.strategy}")
    return b.chunks


def _semantic(b: _Builder, embedder: Embedder) -> None:
    c = b.config
    plan: list[tuple[str, Any]] = []
    windows: list[str] = []
    for group in b.pack(b.blocks(S.MULTIMODAL), size=1 << 30):  # one group per section
        if any(block.standalone for block in group):
            plan.append(("group", group))  # tables/figures: as multimodal
            continue
        assembled = b.render(group)
        sentences = sentence_spans(assembled.text, 0, len(assembled.text))
        plan.append(("prose", (assembled, sentences, len(windows))))
        windows += window_texts(assembled.text, sentences)
    vectors = embedder.embed(windows) if windows else None
    gaps: dict[int, np.ndarray] = {}
    for kind, payload in plan:
        if kind == "prose":
            _, sentences, offset = payload
            gaps[offset] = distances(vectors[offset : offset + len(sentences)])
    pooled = np.concatenate(list(gaps.values())) if gaps else np.zeros(0)
    threshold = float(np.percentile(pooled, c.breakpoint_percentile)) if pooled.size else 2.0
    meta = {"breakpoint_threshold": round(threshold, 4)}

    def emit_span(assembled: Assembled, span: tuple[int, int]) -> None:
        if b.counter.count(assembled.text[span[0] : span[1]]) <= c.chunk_size:
            b.emit(assembled, span, metadata=meta)
            return
        for piece in split_spans(assembled.text, span, b.counter, c.chunk_size, 0,
                                 keep_with_next=assembled.heading_ends()):  # fmt: skip
            b.emit(assembled, piece, metadata=meta)

    for kind, payload in plan:
        if kind == "group":
            b.emit_group(payload, c.chunk_size, metadata=meta)
            continue
        assembled, sentences, offset = payload
        gap, headings = gaps[offset], assembled.heading_ends()
        first, tokens = 0, 0
        for i, (start, end) in enumerate(sentences):
            n = b.counter.count(assembled.text[start:end])
            if i > first and tokens + n > c.chunk_size:  # size limit: break before i
                emit_span(assembled, (sentences[first][0], sentences[i - 1][1]))
                first, tokens = i, 0
            tokens += n
            if i < len(gap) and gap[i] > threshold and end not in headings:
                emit_span(assembled, (sentences[first][0], end))  # topic shift after i
                first, tokens = i + 1, 0
        if first < len(sentences):
            emit_span(assembled, (sentences[first][0], sentences[-1][1]))
