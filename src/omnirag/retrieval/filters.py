"""Turn ``RetrievalFilters`` into Cypher predicates: inside the index where possible, else after.

Measured on Neo4j 2026.10.1 (2026-11-05):
* Vector ``SEARCH ... WHERE`` accepts only AND-ed equality / ``IN`` predicates on properties
  declared when the index was created (``WITH [...]``). OR, ``<>`` and undeclared
  properties are errors. Those filters run *inside* the search, so the top-k already
  satisfies them.
* A ``MATCH (c {document_id: ...})`` before ``SEARCH`` does **not** pre-filter: it filters
  the global top-k afterwards (it returned 0 rows in the probe).
* Full-text ``SEARCH`` accepts no WHERE at all.

Everything that can't go into the index is a *post-filter*: the retriever over-fetches
(``k × overfetch``, capped) and filters afterwards, and reports when fewer than k survived.
"""

from __future__ import annotations

from typing import Any

from omnirag.retrieval.models import RetrievalFilters

# Filter properties declared on the vector index (schema.VECTOR_FILTER_PROPERTIES).
VECTOR_INDEX_FILTERS = frozenset({"config_id", "is_latest", "document_id", "element_type"})


def predicates(
    filters: RetrievalFilters, config_id: str, in_index: frozenset[str] = frozenset()
) -> tuple[list[str], list[str], dict[str, Any]]:
    """(in-index predicates, post predicates, params). Values only ever travel as params."""
    inside: list[str] = []
    after: list[str] = []
    params: dict[str, Any] = {"config_id": filters.config_id or config_id}

    def add(prop: str, clause: str) -> None:
        (inside if prop in in_index else after).append(clause)

    add("config_id", "c.config_id = $config_id")
    if filters.latest_only:
        add("is_latest", "c.is_latest = true")
    if filters.document_ids:
        add("document_id", "c.document_id IN $document_ids")
        params["document_ids"] = filters.document_ids
    if filters.element_types:
        add("element_type", "c.element_type IN $element_types")
        params["element_types"] = filters.element_types
    if filters.source_files:
        after.append("c.source_file IN $source_files")
        params["source_files"] = filters.source_files
    if filters.exclude_generated:
        after.append("NOT 'generated' IN c.origins")
    if filters.min_confidence is not None:
        after.append("c.min_confidence >= $min_confidence")
        params["min_confidence"] = filters.min_confidence
    return inside, after, params


def fetch_size(k: int, post: list[str], overfetch: int, max_fetch: int) -> int:
    return min(max_fetch, max(k, k * overfetch)) if post else k
