"""Lexical (BM25) retrieval through the Neo4j full-text index, with a safe Lucene query builder.

User text is never handed to Lucene as-is. Measured 2026-11-05:
* "What is the N/P ratio of the anode?" fails to parse (``/`` starts a regex);
* "AND OR NOT" fails (bare operators);
* "Why did cell AUR-C103 fail?" parses, but none of its top 3 contains AUR-C103: the
  trailing ``?`` is a wildcard and the identifier is split by the analyzer.

``lucene_query`` rebuilds the query from words, escapes every Lucene special character,
keeps the user's own "quoted phrases" as phrases (exact keyword matching) and handles
*technical identifiers* (tokens mixing letters and digits: AUR-C103, E-7, M3, NaPF6) in
one of three modes:

* ``terms``: identifiers are ordinary words (the analyzer may split them);
* ``boost``: each identifier is also an exact phrase with a boost, so chunks naming the
  identifier exactly rank above chunks that only share its parts ("aur");
* ``require``: every identifier must occur (``+"aur-c103"``). No match means no result,
  which is right for a question about a cell that doesn't exist.

Full-text SEARCH accepts no WHERE, so all metadata filters are post-filters (over-fetch).
"""

from __future__ import annotations

import re
import time
from typing import Literal

from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.retrieval.dense import index_name
from omnirag.retrieval.filters import fetch_size, predicates
from omnirag.retrieval.models import (
    CHUNK_FIELDS,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)

IdentifierMode = Literal["terms", "boost", "require"]

_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/<>=])')
_PHRASE = re.compile(r'"([^"]+)"')
_WORD = re.compile(r"[^\W_]+(?:[-./+'][^\W_]+)*")
# Letters and digits in one token: AUR-C103, E-7, M3, R2, NaPF6, Na2Fe (not "2026-03-19").
_IDENTIFIER = re.compile(r"(?=[^\s]*[^\W\d_])(?=[^\s]*\d)[^\W_]+(?:[-./][^\W_]+)*")


def escape(term: str) -> str:
    return _SPECIAL.sub(r"\\\1", term)


def identifiers(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text):
        if _IDENTIFIER.fullmatch(word) and word.lower() not in (o.lower() for o in out):
            out.append(word)
    return out


def lucene_query(text: str, mode: IdentifierMode = "boost", boost: float = 2.0) -> str:
    """A safe Lucene query for ``text``. Returns "" if there is nothing to search for."""
    phrases = [p.strip() for p in _PHRASE.findall(text) if p.strip()]
    rest = _PHRASE.sub(" ", text)
    words = [w.lower() for w in _WORD.findall(rest)]
    parts = [f'"{escape(p)}"' for p in phrases]
    parts += [escape(w) for w in dict.fromkeys(words)]
    ids = identifiers(rest) if mode != "terms" else []
    if mode == "boost":
        parts += [f'"{escape(i.lower())}"^{boost:g}' for i in ids]
    elif mode == "require":
        parts += [f'+"{escape(i.lower())}"' for i in ids]
    return " ".join(parts)


class LexicalRetriever:
    name = "bm25"

    def __init__(
        self,
        store: GraphStore,
        *,
        config_id: str,
        index: str = "chunk_fulltext",
        identifier_mode: IdentifierMode = "require",
        overfetch: int = 10,
        max_fetch: int = 200,
    ) -> None:
        self.store = store
        self.config_id = config_id
        self.index = index_name(index)
        self.identifier_mode = identifier_mode
        self.overfetch = overfetch
        self.max_fetch = max_fetch

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: RetrievalFilters | None = None,
        identifier_mode: IdentifierMode | None = None,
    ) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        started = time.perf_counter()
        lucene = lucene_query(query, identifier_mode or self.identifier_mode)
        _, after, params = predicates(filters, self.config_id)  # nothing runs in-index
        fetch = fetch_size(k, after, self.overfetch, self.max_fetch)
        hits: list[RetrievedChunk] = []
        if lucene:
            rows = self.store.query(
                f"CYPHER 25 MATCH (c:Chunk) SEARCH c IN (FULLTEXT INDEX {self.index} "
                f"FOR $lucene LIMIT $fetch) SCORE AS score "
                f"WITH c, score WHERE {' AND '.join(after)} "
                f"RETURN {CHUNK_FIELDS}, score ORDER BY score DESC, chunk_id LIMIT $k",
                lucene=lucene, fetch=fetch, k=k, **params,
            )  # fmt: skip
            hits = [RetrievedChunk(rank=i + 1, retriever="bm25", **r) for i, r in enumerate(rows)]
        return RetrievalResult(
            query=query, retriever="bm25", k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={"lucene": lucene, "identifiers": identifiers(query),
                     "identifier_mode": identifier_mode or self.identifier_mode,
                     "fetch": fetch, "post_filters": after,
                     "underfilled": len(hits) < k and fetch > k},
        )  # fmt: skip
