"""Graph retrieval: link the query to entities, traverse the graph, rank chunks with provenance.

1. **Entity linking (dictionary).** The query is matched against every entity's canonical
   key and aliases (``entities.norm`` at word boundaries, the same rules as extraction).
   No model call, deterministic, exact for identifiers. Measured p50 (82 entities): 0.7 ms
   matching + 11 ms loading the entity list from Neo4j per query (to be cached when the
   graph grows: Phase 13). The longest
   match wins at full weight; names nested inside it ("E-7" in "E-7 qualification", "Aurora"
   in "Aurora S1") are kept at half weight. ``SAME_AS`` links add the aliased entity too.
   Questions that describe an entity instead of naming it ("the cell that failed at 45 °C")
   link nothing: graph retrieval then returns no chunks, and fusion relies on the others.
2. **Seed weight** = match weight × IDF of the entity over latest documents,
   ``ln(1 + N / df)``. An entity in one document counts more than one named in eight.
3. **Traversal** (all filters applied to the chunks):
   * chunks that MENTION a seed entity: Σ seed weights;
   * evidence chunks of RELATED_TO edges up to ``max_hops`` from a seed: weight ×
     (1.0 stated / 0.5 inferred) × 0.5^(hop-1), doubled when both ends are seeds (the
     edge connects two things the question names). A chunk takes its *best* edge only
     (max, not sum): measured 2026-11-05, summing let chunks full of hop-2 edges around a
     hub entity (E-7) outrank direct mentions. Only latest-version edges when
     ``latest_only``.
   Ties (most questions name one entity, so every chunk mentioning it scores the same) go
   to the chunk with more seed weight, then to the chunk mentioning fewer other entities
   (a chunk *about* the seed), never to an arbitrary ID.
4. **Provenance.** Each hit lists the paths that found it, e.g. ``MENTIONS AUR-C103`` or
   ``AUR-C103 -[USES, stated]-> E-5 (hop 1)``. ``details`` lists the linked entities and the
   *related documents*: documents ranked by how many seed entities they mention.

**Measured (2026-11-05, ADR-0014): adding this retriever to RRF lowers ranking quality** on
both eval sets (seed MRR 0.928 → ≤0.825; relational 0.838 → ≤0.781). Two refinements, a
relation-intent cue boost and evidence-first scoring, didn't help on both sets and were
removed (results kept in docs/benchmarks/graph-variants-*.json). The default mode is
``dense+bm25``; graph retrieval is for provenance, related documents and entity questions.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from omnirag.knowledge_base.entities import contains, entity_key, norm
from omnirag.knowledge_base.graph_store import GraphStore
from omnirag.retrieval.filters import predicates
from omnirag.retrieval.models import (
    CHUNK_FIELDS,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)

SUPPORT_WEIGHT = {"stated": 1.0, "inferred": 0.5}


def link(query: str, entities: list[dict[str, Any]], n_docs: int) -> list[dict[str, Any]]:
    """Entities named in ``query``: [{entity_id, name, type, surface, weight, nested}]."""
    text = norm(query)
    found: dict[str, dict[str, Any]] = {}
    for e in entities:
        forms = {e["key"]} | {norm(a) for a in e["aliases"] or []}
        forms |= {entity_key(a) for a in e["aliases"] or []}
        hits = [f for f in forms if len(f) >= 2 and contains(text, f)]
        if hits:
            surface = max(hits, key=len)
            found[e["entity_id"]] = {"entity_id": e["entity_id"], "name": e["name"],
                                     "type": e["type"], "surface": surface,
                                     "document_count": e["document_count"] or 1}  # fmt: skip
    # Nested matches ("e-7" inside "e-7 qualification") keep half weight.
    surfaces = [f["surface"] for f in found.values()]
    for f in found.values():
        f["nested"] = any(o != f["surface"] and contains(o, f["surface"]) for o in surfaces)
        idf = math.log(1 + max(n_docs, 1) / f["document_count"])
        f["weight"] = round(idf * (0.5 if f["nested"] else 1.0), 4)
    return sorted(found.values(), key=lambda f: (-f["weight"], f["name"]))


class GraphRetriever:
    name = "graph"

    def __init__(
        self,
        store: GraphStore,
        *,
        config_id: str,
        max_hops: int = 2,
    ) -> None:
        if not 0 <= max_hops <= 3:
            raise ValueError("max_hops must be between 0 and 3")
        self.store = store
        self.config_id = config_id
        self.max_hops = max_hops

    def _entities(self) -> tuple[list[dict[str, Any]], int]:
        rows = self.store.query(
            "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.key AS key, e.name AS name, "
            "e.type AS type, e.aliases AS aliases, e.document_count AS document_count"
        )
        n_docs = self.store.query("MATCH (d:Document {is_latest: true}) RETURN count(d) AS n")
        return rows, n_docs[0]["n"]

    def retrieve(
        self, query: str, k: int = 5, filters: RetrievalFilters | None = None
    ) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        started = time.perf_counter()
        entities, n_docs = self._entities()
        linked = link(query, entities, n_docs)
        # SAME_AS: "Marco" (whiteboard) <-> "Marco Lindqvist", at the same weight.
        weights = {f["entity_id"]: f["weight"] for f in linked}
        if weights:
            for r in self.store.query(
                "MATCH (a:Entity)-[:SAME_AS]-(b:Entity) WHERE a.entity_id IN $ids "
                "RETURN a.entity_id AS a, b.entity_id AS b, b.name AS name", ids=list(weights)
            ):  # fmt: skip
                weights.setdefault(r["b"], weights[r["a"]])
        link_ms = (time.perf_counter() - started) * 1000

        _, after, params = predicates(filters, self.config_id)
        chunk_where = " AND ".join(after)
        score: dict[str, float] = defaultdict(float)
        paths: dict[str, list[str]] = defaultdict(list)
        seeds_in: dict[str, float] = {}  # tie-breaks: more seed weight, then fewer
        focus: dict[str, int] = {}  # other entities (a chunk *about* the seed comes first)
        if weights:
            for r in self.store.query(
                f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.entity_id IN $ids "
                f"AND {chunk_where} RETURN c.chunk_id AS chunk_id, "
                "collect(DISTINCT e.entity_id) AS ids, collect(DISTINCT e.name) AS names, "
                "COUNT { (c)-[:MENTIONS]->() } AS total",
                ids=list(weights), **params,
            ):  # fmt: skip
                seeds_in[r["chunk_id"]] = sum(weights[i] for i in r["ids"])
                focus[r["chunk_id"]] = r["total"]
                score[r["chunk_id"]] += seeds_in[r["chunk_id"]]
                paths[r["chunk_id"]] += [f"MENTIONS {n}" for n in sorted(r["names"])]
            if self.max_hops:
                self._traverse(weights, filters, chunk_where, params, score, paths)

        ranked = sorted(score, key=lambda c: (-score[c], -seeds_in.get(c, 0.0),
                                              focus.get(c, 10**6), c))[:k]  # fmt: skip
        hits: list[RetrievedChunk] = []
        if ranked:
            rows = {r["chunk_id"]: r for r in self.store.query(
                f"MATCH (c:Chunk) WHERE c.chunk_id IN $ids RETURN {CHUNK_FIELDS}", ids=ranked
            )}  # fmt: skip
            hits = [RetrievedChunk(rank=i + 1, retriever="graph", score=round(score[c], 4),
                                   graph_paths=paths[c][:12], **rows[c])
                    for i, c in enumerate(ranked) if c in rows]  # fmt: skip
        related = []
        if weights:
            related = self.store.query(
                "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
                f"WHERE e.entity_id IN $ids AND {chunk_where} "
                "RETURN d.document_id AS document_id, d.title AS title, "
                "count(DISTINCT e) AS shared_entities ORDER BY shared_entities DESC, title "
                "LIMIT 10", ids=list(weights), **params,
            )  # fmt: skip
        return RetrievalResult(
            query=query, retriever="graph", k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={"linked": linked, "seed_count": len(weights), "candidates": len(score),
                     "link_ms": round(link_ms, 2), "max_hops": self.max_hops,
                     "related_documents": related},
        )  # fmt: skip

    def _traverse(self, weights, filters, chunk_where, params, score, paths) -> None:
        """Evidence chunks of RELATED_TO edges within ``max_hops`` of a seed entity."""
        latest = "AND ALL(x IN relationships(p) WHERE x.is_latest)" if filters.latest_only else ""
        rows = self.store.query(
            f"MATCH p = (s:Entity)-[:RELATED_TO*1..{self.max_hops}]-(:Entity) "
            f"WHERE s.entity_id IN $ids {latest} "
            "UNWIND range(0, size(relationships(p)) - 1) AS i "
            "WITH s, relationships(p)[i] AS r, i + 1 AS hop "
            "MATCH (c:Chunk {chunk_id: r.chunk_id}) "
            f"WHERE {chunk_where} "
            "RETURN DISTINCT s.entity_id AS seed, hop, r.chunk_id AS chunk_id, r.type AS type, "
            "r.support AS support, startNode(r).entity_id AS a, startNode(r).name AS a_name, "
            "endNode(r).entity_id AS b, endNode(r).name AS b_name",
            ids=list(weights), **params,
        )  # fmt: skip
        # A chunk's relation evidence counts once, as its best edge: summing would let a chunk
        # full of a hub entity's hop-2 edges (E-7 has dozens) outrank a direct mention.
        best: dict[str, float] = {}
        edges: dict[tuple[str, str, str, str], tuple[int, str]] = {}  # shortest hop per edge
        for r in rows:
            w = weights[r["seed"]] * SUPPORT_WEIGHT.get(r["support"], 0.5) * 0.5 ** (r["hop"] - 1)
            if r["a"] in weights and r["b"] in weights:
                w *= 2
            best[r["chunk_id"]] = max(best.get(r["chunk_id"], 0.0), w)
            edge = (r["chunk_id"], r["a"], r["type"], r["b"])
            label = f"{r['a_name']} -[{r['type']}, {r['support']}]-> {r['b_name']}"
            if edge not in edges or r["hop"] < edges[edge][0]:
                edges[edge] = (r["hop"], label)
        for (chunk_id, *_), (hop, label) in sorted(edges.items(), key=lambda kv: kv[1]):
            paths[chunk_id].append(f"{label} (hop {hop})")
        for chunk_id, w in best.items():
            score[chunk_id] += w
