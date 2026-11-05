"""Graph schema: uniqueness constraints, lookup indexes, the vector and full-text indexes.

Measured on Neo4j 2026.10.1 Community (2026-11-05):
* New vector indexes default to ``vector.quantization.type: BINARY`` (1 bit per dimension
  for candidate search). We set the type explicitly; the default here is ``NONE`` (exact
  float32 vectors): at this corpus size memory is no concern, and lossy candidates would
  blur strategy and model comparisons. Valid values: NONE, SCALAR, BINARY.
* A vector index can carry extra filter properties (Cypher 25 ``WITH [...]``): equality /
  ``IN`` filters on them run inside the search (Phase 6). Declared: chunk set, latest
  version, document and element type (``VECTOR_FILTER_PROPERTIES``).
* Only one full-text index can cover a label + property set: ``CREATE FULLTEXT INDEX ...
  IF NOT EXISTS`` with another name is silently a no-op. So an analyzer change must drop and
  recreate the index, and a changed setting is reported, never silently ignored.
* The full-text "standard" analyzers split identifiers (``AUR-C103`` → aur, c103); the
  ``classic`` analyzer keeps tokens with digits whole. Phase 6 measures which one to use.

Index options can't be query parameters, so they are built from validated values only
(an int dimension and enum members); labels and property names are constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omnirag.knowledge_base.graph_store import GraphStore

Quantization = Literal["NONE", "SCALAR", "BINARY"]
# Must match retrieval.filters.VECTOR_INDEX_FILTERS (asserted by a test).
VECTOR_FILTER_PROPERTIES = ("config_id", "is_latest", "document_id", "element_type")

# (label, property) pairs that identify a node: one uniqueness constraint each.
UNIQUE_KEYS = [
    ("Source", "source_key"),
    ("Document", "document_id"),
    ("Page", "page_id"),
    ("Section", "section_id"),
    ("Chunk", "chunk_id"),
    ("Table", "table_id"),
    ("Figure", "figure_id"),
    ("Entity", "entity_id"),
]
# Lookup indexes for per-document replacement/deletion and common filters.
LOOKUPS = [
    ("Chunk", "document_id"),
    ("Chunk", "config_id"),
    ("Page", "document_id"),
    ("Section", "document_id"),
    ("Table", "document_id"),
    ("Figure", "document_id"),
    ("Document", "is_latest"),
    ("Entity", "key"),
    ("Entity", "type"),
]
# Relationship property index: per-document deletion of extracted relations.
REL_LOOKUPS = [("RELATED_TO", "document_id")]


class SchemaMismatchError(RuntimeError):
    """An existing index has a different configuration (e.g. a new embedding dimension)."""


@dataclass
class SchemaReport:
    created: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)
    recreated: list[str] = field(default_factory=list)


def ensure_schema(
    store: GraphStore,
    *,
    vector_dim: int,
    vector_index: str = "chunk_embedding",
    quantization: Quantization = "NONE",
    fulltext_index: str = "chunk_fulltext",
    analyzer: str = "classic",
    recreate_vector_index: bool = False,
    recreate_fulltext_index: bool = False,
) -> SchemaReport:
    """Idempotently create the schema. Raises ``SchemaMismatchError`` instead of silently
    keeping an index whose configuration (dimension, quantization, filter properties,
    analyzer) differs from what's requested."""
    if not isinstance(vector_dim, int) or vector_dim <= 0:
        raise ValueError(f"Invalid vector dimension {vector_dim!r}")
    if quantization not in ("NONE", "SCALAR", "BINARY"):
        raise ValueError(f"Invalid quantization {quantization!r}")
    report = SchemaReport()
    existing = {r["name"]: r for r in store.schema(
        "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options")}  # fmt: skip
    constraints = {r["name"] for r in store.schema("SHOW CONSTRAINTS YIELD name")}

    for label, prop in UNIQUE_KEYS:
        name = f"{label.lower()}_{prop}_unique"
        _note(report, name, name in constraints)
        store.schema(f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) "
                     f"REQUIRE n.{prop} IS UNIQUE")  # fmt: skip
    for label, prop in LOOKUPS:
        name = f"{label.lower()}_{prop}"
        _note(report, name, name in existing)
        store.schema(f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
    for rel, prop in REL_LOOKUPS:
        name = f"{rel.lower()}_{prop}"
        _note(report, name, name in existing)
        store.schema(f"CREATE INDEX {name} IF NOT EXISTS FOR ()-[r:{rel}]-() ON (r.{prop})")

    current = existing.get(vector_index)
    if current is not None:
        cfg = current["options"]["indexConfig"]
        have = (cfg.get("vector.dimensions"), cfg.get("vector.quantization.type"),
                tuple(current["properties"][1:]))  # fmt: skip
        want = (vector_dim, quantization, VECTOR_FILTER_PROPERTIES)
        if have != want and not recreate_vector_index:
            raise SchemaMismatchError(
                f"Vector index {vector_index} has dimension / quantization / filter properties "
                f"{have}, requested {want}. Run `init --recreate-vector-index` (vectors stay on "
                "the nodes and are re-indexed; re-embed first if the dimension changed)."
            )
        if have != want:
            store.schema(f"DROP INDEX {vector_index}")
            report.recreated.append(vector_index)
            current = None
    if current is None:
        with_props = ", ".join(f"c.{p}" for p in VECTOR_FILTER_PROPERTIES)
        store.schema(
            f"CYPHER 25 CREATE VECTOR INDEX {vector_index} IF NOT EXISTS FOR (c:Chunk) "
            f"ON (c.embedding) WITH [{with_props}] OPTIONS {{indexConfig: {{"
            f"`vector.dimensions`: {vector_dim}, `vector.similarity_function`: 'cosine', "
            f"`vector.quantization.type`: '{quantization}'}}}}"
        )
        if vector_index not in report.recreated:
            report.created.append(vector_index)
    else:
        report.existing.append(vector_index)

    if not analyzer.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid analyzer name {analyzer!r}")
    current = existing.get(fulltext_index)
    if current is not None:
        have_analyzer = current["options"]["indexConfig"].get("fulltext.analyzer")
        if have_analyzer != analyzer and not recreate_fulltext_index:
            raise SchemaMismatchError(
                f"Full-text index {fulltext_index} uses analyzer {have_analyzer!r}, requested "
                f"{analyzer!r}. Run `init --recreate-fulltext-index` (rebuilt from the nodes)."
            )
        if have_analyzer != analyzer:
            store.schema(f"DROP INDEX {fulltext_index}")
            report.recreated.append(fulltext_index)
            current = None
    if current is None:
        store.schema(
            f"CREATE FULLTEXT INDEX {fulltext_index} IF NOT EXISTS FOR (c:Chunk) "
            "ON EACH [c.content, c.context] OPTIONS {indexConfig: {"
            f"`fulltext.analyzer`: '{analyzer}'}}}}"
        )
        if fulltext_index not in report.recreated:
            report.created.append(fulltext_index)
    else:
        report.existing.append(fulltext_index)
    store.schema("CALL db.awaitIndexes(120)")
    return report


def _note(report: SchemaReport, name: str, exists: bool) -> None:
    (report.existing if exists else report.created).append(name)
