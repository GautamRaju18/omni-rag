"""Extractive context compression (Phase 9c): keep the sentences and table rows that matter.

Abstractive compression (asking the model to summarise the passages) is rejected on purpose:
a summary can't be cited, and a 4B model rewrites numbers and names. Here every kept unit is
the document's own text, so a citation still points at real words.

1. **Units:** each block is cut into table rows (a table's header and separator lines travel
   with any kept row) and sentences (no break after "Dr.", "e.g." and other abbreviations:
   "Reviewed by Dr. Priya Raman." must stay one unit). List items and short lines are units
   too; a hard-wrapped sentence ("...after 800\ncycles on AUR-C104.") is joined back into one.
2. **Scores** (``method``):
   * ``lexical``: question words + 2 × question identifiers found in the unit (free);
   * ``embedding``: bge-m3 cosine between question and unit (one batched Ollama call);
   * ``llm``: the model returns the *numbers* of the units it needs; code keeps those units
     verbatim. On ``LLMError`` it falls back to ``lexical`` (recorded).
3. **Selection** under the token budget: units by score (then block rank, then position)
   while they fit; a block's source line is paid once. Kept units are shown in document
   order; adjacent units keep their original spacing, gaps become " … ".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from omnirag.knowledge_base.entities import norm
from omnirag.llm.chat import LLMError
from omnirag.retrieval.context import ContextBlock
from omnirag.retrieval.sparse import identifiers

_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "e.g", "i.e", "vs", "no", "fig", "approx", "etc",
           "inc", "ltd", "st", "cf", "al"}  # fmt: skip
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_TABLE_SEP = re.compile(r"^\|\s*:?-{3,}")
# A line starts a new unit (never joins the previous one) when it is a list item, heading or
# table row, or starts with a capitalised word ("The team...": after a heading). A line wraps
# onto the next only if it is long and doesn't end a sentence/label: short lines are
# "key: value" records, OCR lines, titles. Continuations start lowercase, "(", a digit, "E-7".
_ITEM = re.compile(r"^\s*(?:[-*•#|]|\d+[.)]\s)")
_NEW_SENTENCE = re.compile(r"^\s*[A-Z][a-z]")
_TERMINAL = (".", "!", "?", ":", ";")
_WRAP_MIN = 50
_WORDS = re.compile(r"[\w][\w.\-/]*")
_STOP = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the "
    "this to was were what when where which who whom whose why with after before".split()
)
GAP = " … "

LLM_SYSTEM = (
    "You select, from numbered sentences and table rows of technical documents, the ones "
    "needed to answer a question. Return the numbers of every unit that states a fact the "
    "answer needs, including units that identify what the question only describes. If "
    "unsure, include the unit. Do not answer the question."
)
LLM_SCHEMA = {"type": "object", "properties": {"keep": {"type": "array",
              "items": {"type": "integer"}}}, "required": ["keep"]}  # fmt: skip


@dataclass
class Unit:
    block: int  # index into the blocks list
    start: int  # character span in the block's text
    end: int
    table: int | None = None  # table group within the block (rows share their header)
    header: bool = False  # a table header / separator line: kept only with a row

    def text(self, blocks: list[ContextBlock]) -> str:
        return blocks[self.block].text[self.start : self.end]


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, s = [], start
    for m in _BOUNDARY.finditer(text, start, end):
        tail = text[s : m.start()].split()
        word = tail[-1].rstrip(".").lower() if tail else ""
        if word in _ABBREV or (len(word) == 1 and word.isalpha()):  # "Dr." / initials
            continue
        if len(tail) == 1 and word.isdigit():  # "1. High-temperature failure": a list number
            continue
        spans.append((s, m.start()))
        s = m.end()
    if s < end:
        spans.append((s, end))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _wraps(prev: str, line: str) -> bool:
    """``line`` continues ``prev`` (hard-wrapped prose: "...after 800\\ncycles on AUR-C104")."""
    return (len(prev.strip()) >= _WRAP_MIN and not prev.rstrip().endswith(_TERMINAL)
            and not _ITEM.match(line) and not _NEW_SENTENCE.match(line))  # fmt: skip


def split_units(text: str, block: int = 0) -> list[Unit]:
    units: list[Unit] = []
    table, prev_table_line, rows_in_table = -1, False, 0
    lines = list(re.finditer(r"[^\n]+", text))
    i = 0
    while i < len(lines):
        m = lines[i]
        line = m.group(0)
        if line.lstrip().startswith("|"):
            if not prev_table_line:
                table += 1
                rows_in_table = 0
            header = rows_in_table == 0 or (rows_in_table == 1 and bool(_TABLE_SEP.match(line)))
            units.append(Unit(block, m.start(), m.end(), table, header))
            rows_in_table += 1
            prev_table_line = True
            i += 1
            continue
        prev_table_line = False
        j = i  # a paragraph: this line plus the lines it wraps onto (directly adjacent only)
        while (j + 1 < len(lines) and lines[j + 1].start() == lines[j].end() + 1
               and _wraps(lines[j].group(0), lines[j + 1].group(0))):  # fmt: skip
            j += 1
        units += [Unit(block, a, b) for a, b in _sentences(text, m.start(), lines[j].end())]
        i = j + 1
    return units


def lexical_scores(question: str, texts: list[str]) -> list[float]:
    terms = {w for w in _WORDS.findall(norm(question)) if w not in _STOP and len(w) > 1}
    ids = {norm(i) for i in identifiers(question)}
    out = []
    for t in texts:
        words = set(_WORDS.findall(norm(t)))
        out.append(len(terms & words) + 2.0 * sum(1 for i in ids if i in norm(t)))
    return out


class Compressor:
    def __init__(self, embedder: Any = None, llm: Any = None, cache_size: int = 5000) -> None:
        self.embedder = embedder
        self.llm = llm
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        # The model's selection depends on the question and the units, not on the budget:
        # one call serves every budget (and every repeat of the same question).
        self._llm_cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    # ---------------------------------------------------------------- scoring
    def _embedding_scores(self, question: str, texts: list[str]) -> list[float]:
        if self.embedder is None:
            raise ValueError("embedding compression needs an embedder")
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = self.embedder.embed_documents(missing)
            if len(self._cache) + len(missing) > self._cache_size:
                self._cache.clear()
            self._cache.update(zip(missing, vectors, strict=True))
        q = self.embedder.embed_query(question)
        return [float(np.dot(q, self._cache[t])) for t in texts]

    def _llm_scores(self, question: str, units: list[Unit],
                    blocks: list[ContextBlock]) -> list[float]:  # fmt: skip
        if self.llm is None:
            raise ValueError("LLM compression needs a chat model")
        key = (question, tuple(u.text(blocks) for u in units))
        if key in self._llm_cache:
            return self._llm_cache[key]
        lines, current = [], -1
        for n, u in enumerate(units, start=1):
            if u.block != current:
                current = u.block
                lines.append(f"({blocks[u.block].source_file})")
            lines.append(f"[{n}] {' '.join(u.text(blocks).split())}")
        r = self.llm.chat(LLM_SYSTEM, f"Question: {question}\n\n" + "\n".join(lines),
                          schema=LLM_SCHEMA, num_predict=400)  # fmt: skip
        keep = {n for n in r.data.get("keep", []) if isinstance(n, int)}
        scores = [1.0 if n in keep else 0.0 for n in range(1, len(units) + 1)]
        if len(self._llm_cache) > 500:
            self._llm_cache.clear()
        self._llm_cache[key] = scores
        return scores

    # ---------------------------------------------------------------- selection
    def compress(
        self,
        question: str,
        blocks: list[ContextBlock],
        budget: int,
        method: str,
        count: Callable[[str], int],
    ) -> tuple[list[ContextBlock], list[str], dict]:
        units = [u for i, b in enumerate(blocks) for u in split_units(b.text, i)]
        scorable = [u for u in units if not u.header]
        texts = [u.text(blocks) for u in scorable]
        info: dict[str, Any] = {"method": method, "units": len(scorable)}
        if method == "embedding":
            scores = self._embedding_scores(question, texts)
        elif method == "llm":
            try:
                scores = self._llm_scores(question, scorable, blocks)
            except LLMError as exc:  # no selection from the model: fall back, say so
                info["fallback"] = f"lexical ({exc})"[:200]
                scores = lexical_scores(question, texts)
        else:
            scores = lexical_scores(question, texts)
        floor = -1.0 if method == "embedding" else 0.0  # lexical / llm: only units that matched

        def priority(i: int) -> tuple:
            u = scorable[i]
            return (-scores[i], blocks[u.block].rank, u.block, u.start)

        order = sorted(range(len(scorable)), key=priority)
        headers = {(u.block, u.table): [h for h in units if h.header and h.block == u.block
                                        and h.table == u.table] for u in scorable
                   if u.table is not None}  # fmt: skip
        chosen: set[int] = set()  # ids of kept units (scorable and headers)
        used, opened = 0, set()
        for i in order:
            if scores[i] <= floor:
                break
            u = scorable[i]
            extra = [h for h in headers.get((u.block, u.table), []) if id(h) not in chosen]
            cost = count(texts[i]) + 1 + sum(count(h.text(blocks)) + 1 for h in extra)
            if u.block not in opened:
                cost += count(blocks[u.block].header()) + 1
            if used + cost > budget:
                continue
            used += cost
            opened.add(u.block)
            chosen.add(id(u))
            chosen.update(id(h) for h in extra)
        kept, skipped = [], []
        for i, b in enumerate(blocks):
            mine = [u for u in units if u.block == i and id(u) in chosen]
            if not mine:
                skipped += b.hit_chunk_ids
                continue
            kept.append(_render(b, mine, len([u for u in units if u.block == i and not u.header]),
                                count))  # fmt: skip
        while kept and sum(b.tokens for b in kept) > budget:  # tokenisation of joined text
            skipped += kept.pop().hit_chunk_ids
        info.update({"kept": sum(b.units_kept or 0 for b in kept),
                     "scored": sum(1 for s in scores if s > floor)})  # fmt: skip
        return kept, skipped, info


def _render(block: ContextBlock, mine: list[Unit], total: int,
            count: Callable[[str], int]) -> ContextBlock:  # fmt: skip
    mine.sort(key=lambda u: u.start)
    parts, prev_end = [], None
    for u in mine:
        if prev_end is not None:
            between = block.text[prev_end : u.start]
            parts.append(between if not between.strip() else GAP)  # adjacent: keep spacing
        parts.append(block.text[u.start : u.end])
        prev_end = u.end
    text = "".join(parts)
    out = block.model_copy(
        update={
            "text": text,
            "units_total": total,
            "units_kept": sum(1 for u in mine if not u.header),
        }
    )
    out.tokens = count(f"{out.header()}\n{text}")
    return out
