"""Graph schema: uniqueness constraints, lookup indexes, the vector and full-text indexes.

Measured on Neo4j 2026.10.1 Community (2026-11-05):
* New vector indexes default to ``vector.quantization.type: BINARY`` (1 bit per dimension
  for candidate search). We set the type explicitly; the default here is ``NONE`` (exact
  float32 vectors): at this corpus size memory is no concern, and lossy candidates would
  blur strategy and model comparisons. Valid values: NONE, SCALAR, BINARY.
* A vector index can carry extra filter properties (Cypher 25 ``WITH [...]``). We include
  ``config_id`` and ``is_latest`` so Phase 6 can filter inside the index.
* The full-text "standard" analyzers split identifiers (``AUR-C103`` → aur, c103), so
  ``AUR-C103`` also matches ``AUR-C105``. Phase 6 evaluates analyzers and query forms.

Index options can't be query parameters, so they are built from validated values only
(an int dimension and enum members); labels and property names are constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from omnirag.knowledge_base.graph_store import GraphStore

Quantization = Literal["NONE", "SCALAR", "BINARY"]

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
    analyzer: str = "standard-no-stop-words",
    recreate_vector_index: bool = False,
) -> SchemaReport:
    """Idempotently create the schema. Raises ``SchemaMismatchError`` instead of silently
    keeping an index whose dimension or quantization differs from what's requested."""
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
        same = (cfg.get("vector.dimensions") == vector_dim
                and cfg.get("vector.quantization.type") == quantization)  # fmt: skip
        if not same and not recreate_vector_index:
            raise SchemaMismatchError(
                f"Vector index {vector_index} has dimension {cfg.get('vector.dimensions')} / "
                f"quantization {cfg.get('vector.quantization.type')}, requested {vector_dim} / "
                f"{quantization}. Re-embed and run `init --recreate-vector-index`."
            )
        if not same:
            store.schema(f"DROP INDEX {vector_index}")
            report.recreated.append(vector_index)
            current = None
    if current is None:
        store.schema(
            f"CYPHER 25 CREATE VECTOR INDEX {vector_index} IF NOT EXISTS FOR (c:Chunk) "
            "ON (c.embedding) WITH [c.config_id, c.is_latest] OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {vector_dim}, `vector.similarity_function`: 'cosine', "
            f"`vector.quantization.type`: '{quantization}'}}}}"
        )
        if vector_index not in report.recreated:
            report.created.append(vector_index)
    else:
        report.existing.append(vector_index)

    _note(report, fulltext_index, fulltext_index in existing)
    if not analyzer.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid analyzer name {analyzer!r}")
    store.schema(
        f"CREATE FULLTEXT INDEX {fulltext_index} IF NOT EXISTS FOR (c:Chunk) "
        "ON EACH [c.content, c.context] OPTIONS {indexConfig: {"
        f"`fulltext.analyzer`: '{analyzer}'}}}}"
    )
    store.schema("CALL db.awaitIndexes(60)")
    return report


def _note(report: SchemaReport, name: str, exists: bool) -> None:
    (report.existing if exists else report.created).append(name)
