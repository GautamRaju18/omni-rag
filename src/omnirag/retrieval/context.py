"""Context assembly (Phase 9c): turn ranked chunks into the context a generator will read.

Retrieval returns ranked chunks; generation (Phase 10) needs a bounded, citable context. Three
choices, each measured (``evaluation/context_benchmark.py``, ADR-0018):

* **Budget, not k.** Chunks range from ~10 to ~250 tokens (bge-m3), so "top 5" can be 150 or
  900 tokens. Blocks are added in rank order while they fit ``budget_tokens``; a block that
  doesn't fit is skipped and smaller later ones may still fit.
* **Expansion** (the parent-child idea at query time): ``neighbors`` adds the chunks before
  and after each hit in reading order (``NEXT_CHUNK``); ``section`` adds every chunk of the
  hit's section (``Section-[:CONTAINS]``). Touching or overlapping chunks of one document
  merge into one block, in reading order, so no text appears twice.
* **Compression** (``retrieval/compression.py``): extractive only. Blocks are cut into
  sentences and table rows, units are scored against the question, and the best units are kept
  verbatim (a table keeps its header with any kept row). The model never rewrites evidence:
  a citation must point at the document's own words.

Every block keeps its provenance (document, section, pages, chunk IDs, element IDs, origins)
for citations. Token counts include the source line a prompt will show for each block.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from omnirag.retrieval.models import RetrievedChunk

Expand = Literal["none", "neighbors", "section"]
Compress = Literal["none", "lexical", "embedding", "llm"]


class ContextOptions(BaseModel):
    budget_tokens: int = Field(default=1024, ge=64, le=8192)
    expand: Expand = "none"
    compress: Compress = "none"
    max_candidates: int = Field(default=20, ge=1, le=100)  # ranked hits considered


class ContextBlock(BaseModel):
    """One contiguous piece of one document, as the generator will see it."""

    document_id: str
    source_file: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    chunk_ids: list[str]  # reading order
    hit_chunk_ids: list[str]  # the retrieved ones (the rest came from expansion)
    element_ids: list[str] = Field(default_factory=list)
    origins: list[str] = Field(default_factory=list)
    rank: int  # best retrieval rank among its hits
    text: str
    tokens: int  # header line + text
    units_kept: int | None = None  # compression: sentences / rows kept of units_total
    units_total: int | None = None

    def header(self) -> str:
        where = " > ".join(self.section_path)
        return f"{self.source_file}" + (f" > {where}" if where else "")


class AssembledContext(BaseModel):
    blocks: list[ContextBlock]
    tokens: int
    budget_tokens: int
    options: ContextOptions
    candidates: int
    skipped: list[str] = Field(default_factory=list)  # chunk IDs that didn't fit
    ms: float = 0.0
    compression: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Numbered sources, the format Phase 10's prompt will use."""
        return "\n\n".join(f"[{i}] {b.header()}\n{b.text}" for i, b in enumerate(self.blocks, 1))


# What expansion needs from the graph for each chunk (same projection everywhere).
_FIELDS = (".chunk_id, .document_id, .source_file, .content, .chunk_index, .section_path, "
           ".page_numbers, .element_ids, .origins, .config_id")  # fmt: skip


def expansion_rows(store: Any, chunk_ids: list[str], mode: Expand) -> dict[str, list[dict]]:
    """For each chunk ID: the chunks its expansion covers (itself included), from the graph."""
    if mode == "none" or not chunk_ids:
        return {}
    if mode == "neighbors":
        cypher = (
            "UNWIND $ids AS id MATCH (c:Chunk {chunk_id: id}) "
            "OPTIONAL MATCH (p:Chunk)-[:NEXT_CHUNK]->(c) "
            "OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(n:Chunk) "
            f"RETURN id, [x IN [p, c, n] WHERE x IS NOT NULL | x {{{_FIELDS}}}] AS rows"
        )
    else:  # section: every chunk of the same set in the hit's (deepest) section
        cypher = (
            "UNWIND $ids AS id MATCH (c:Chunk {chunk_id: id}) "
            "OPTIONAL MATCH (s:Section)-[:CONTAINS]->(c) "
            "OPTIONAL MATCH (s)-[:CONTAINS]->(m:Chunk {config_id: c.config_id}) "
            f"WITH id, c, collect(m {{{_FIELDS}}}) AS members "
            f"RETURN id, CASE WHEN size(members) = 0 THEN [c {{{_FIELDS}}}] "
            "ELSE members END AS rows"
        )
    return {r["id"]: r["rows"] for r in store.query(cypher, ids=chunk_ids)}


def _row_of(hit: RetrievedChunk) -> dict[str, Any]:
    return {"chunk_id": hit.chunk_id, "document_id": hit.document_id,
            "source_file": hit.source_file, "content": hit.content, "chunk_index": None,
            "section_path": hit.section_path, "page_numbers": hit.page_numbers,
            "element_ids": hit.element_ids, "origins": hit.origins}  # fmt: skip


def _block(
    rows: list[dict], hits: set[str], rank: int, count: Callable[[str], int]
) -> ContextBlock:
    rows = sorted(rows, key=lambda r: (r.get("chunk_index") is None, r.get("chunk_index") or 0))
    first = rows[0]
    sections = [r.get("section_path") or [] for r in rows]
    # The shared section prefix names the block; a merged block may span subsections.
    prefix: list[str] = []
    for parts in zip(*sections, strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    text = "\n\n".join(r["content"] for r in rows)
    block = ContextBlock(
        document_id=first["document_id"], source_file=first["source_file"],
        section_path=prefix if len(rows) > 1 else list(sections[0]),
        page_numbers=sorted({p for r in rows for p in (r.get("page_numbers") or [])}),
        chunk_ids=[r["chunk_id"] for r in rows],
        hit_chunk_ids=[r["chunk_id"] for r in rows if r["chunk_id"] in hits],
        element_ids=list(dict.fromkeys(e for r in rows for e in (r.get("element_ids") or []))),
        origins=sorted({o for r in rows for o in (r.get("origins") or [])}),
        rank=rank, text=text, tokens=0,
    )  # fmt: skip
    block.tokens = count(f"{block.header()}\n{text}")
    return block


def build_blocks(hits: list[RetrievedChunk], expansions: dict[str, list[dict]],
                 count: Callable[[str], int]) -> list[ContextBlock]:  # fmt: skip
    """Blocks in rank order; expansions that touch or overlap one document's earlier block
    merge into it (the merged block keeps the better rank)."""
    groups: list[dict[str, Any]] = []  # {"doc", "rows": {id: row}, "hits": set, "rank"}
    for hit in hits:
        rows = expansions.get(hit.chunk_id) or [_row_of(hit)]
        ids = {r["chunk_id"] for r in rows}
        indexes = {r.get("chunk_index") for r in rows} - {None}

        def joins(g: dict[str, Any], ids=ids, indexes=indexes, doc=hit.document_id) -> bool:
            g_idx = {r.get("chunk_index") for r in g["rows"].values()} - {None}
            touching = any(abs(a - b) <= 1 for a in indexes for b in g_idx)
            return g["doc"] == doc and (bool(ids & set(g["rows"])) or touching)

        matches = [g for g in groups if joins(g)]
        if matches:  # one expansion can bridge two earlier blocks: merge them all
            target = matches[0]
            for g in matches[1:]:
                target["rows"].update(g["rows"])
                target["hits"] |= g["hits"]
                groups.remove(g)
        else:
            target = {"doc": hit.document_id, "rows": {}, "hits": set(), "rank": hit.rank}
            groups.append(target)
        for r in rows:
            target["rows"].setdefault(r["chunk_id"], r)
        target["hits"].add(hit.chunk_id)
    return [_block(list(g["rows"].values()), g["hits"], g["rank"], count) for g in groups]


def fill(blocks: list[ContextBlock], budget: int) -> tuple[list[ContextBlock], list[str]]:
    """Whole blocks in rank order while they fit; a block too big is skipped, not cut."""
    kept, skipped, used = [], [], 0
    for b in sorted(blocks, key=lambda b: b.rank):
        if used + b.tokens <= budget:
            kept.append(b)
            used += b.tokens
        else:
            skipped += b.hit_chunk_ids
    return kept, skipped


class ContextAssembler:
    """Builds the context from ranked hits. ``store`` (graph) is needed for expansion,
    ``compressor`` for compression; both are optional, and asking for a missing one raises."""

    def __init__(self, count: Callable[[str], int], store: Any = None,
                 compressor: Any = None) -> None:  # fmt: skip
        self.count = count
        self.store = store
        self.compressor = compressor

    def assemble(self, question: str, hits: list[RetrievedChunk],
                 options: ContextOptions | None = None) -> AssembledContext:  # fmt: skip
        o = options or ContextOptions()
        started = time.perf_counter()
        candidates = hits[: o.max_candidates]
        if o.expand != "none" and self.store is None:
            raise ValueError("context expansion needs the graph store")
        expansions = expansion_rows(self.store, [h.chunk_id for h in candidates], o.expand)
        blocks = build_blocks(candidates, expansions, self.count)
        info: dict[str, Any] = {}
        if o.compress == "none":
            kept, skipped = fill(blocks, o.budget_tokens)
        else:
            if self.compressor is None:
                raise ValueError("context compression needs a compressor")
            kept, skipped, info = self.compressor.compress(question, blocks, o.budget_tokens,
                                                           o.compress, self.count)  # fmt: skip
        return AssembledContext(
            blocks=kept, tokens=sum(b.tokens for b in kept), budget_tokens=o.budget_tokens,
            options=o, candidates=len(candidates), skipped=skipped, compression=info,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )  # fmt: skip
