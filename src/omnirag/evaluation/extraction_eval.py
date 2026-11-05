"""Measure entity/relation extraction against the gold set (``data/eval/entities_seed.json``).

Works on the stored extraction files only: no model calls, no Neo4j. It applies the same
resolution rules as the graph (canonical keys, plus person aliases across documents).

* **Entity recall**: a gold entity counts as found if any extracted entity key equals the
  key of its name or one of its aliases (or is a SAME_AS alias of one).
* **Fragmentation**: a gold entity found under several keys that no SAME_AS link joins
  (e.g. "aurora" and "project aurora") is a resolution miss.
* **Type agreement**: the entity's majority type equals the gold type.
* **Relation recall**: *pair* = any extracted relation links the two gold entities (any type,
  any direction); *typed* = with an acceptable type and the gold direction.
* **Unlisted** extracted entities match no gold entry. They are listed for review, not counted
  as errors: the gold set may be incomplete, and there's no automatic precision number.

    uv run python -m omnirag.evaluation.extraction_eval [--out docs/benchmarks/x.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from omnirag.core.config import get_settings
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base.entities import (
    DocumentExtraction,
    EntityType,
    entity_key,
    person_alias_map,
)

GOLD = Path(__file__).parents[3] / "data" / "eval" / "entities_seed.json"


class GoldEntity(BaseModel):
    name: str
    type: EntityType
    aliases: list[str] = []

    def keys(self) -> set[str]:
        names = [self.name, *self.aliases]
        return {entity_key(n, self.type) for n in names} | {entity_key(n) for n in names}


class GoldRelation(BaseModel):
    source: str
    target: str
    types: list[str]
    file: str
    evidence: str


class GoldSet(BaseModel):
    version: int
    description: str
    entities: list[GoldEntity]
    relations: list[GoldRelation]

    @classmethod
    def load(cls, path: Path = GOLD) -> GoldSet:
        return cls.model_validate_json(path.read_bytes())


def evaluate(gold: GoldSet, extractions: list[DocumentExtraction]) -> dict[str, Any]:
    types: dict[str, Counter] = defaultdict(Counter)
    for x in extractions:
        for c in x.chunks:
            for m in c.mentions:
                types[m.key][m.type.value] += 1
    majority = {
        k: sorted(t.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, t in types.items()
    }
    persons = {k for k, t in majority.items() if t == "PERSON"}
    full = {k for k in persons if len(k.split()) >= 2 and "." not in k.split()[0]}
    same_as = {s: f for s, (f, _) in person_alias_map(full, persons - full).items()}

    def keys_of(g: GoldEntity) -> set[str]:
        wanted = g.keys()
        return {k for k in types if k in wanted or same_as.get(k) in wanted}

    entity_rows, matched_keys = [], set()
    for g in gold.entities:
        found = keys_of(g)
        matched_keys |= found
        entity_rows.append({
            "name": g.name, "type": g.type.value, "found": bool(found), "keys": sorted(found),
            # several keys that no SAME_AS joins = one real-world entity split into nodes
            "fragmented": len({same_as.get(k, k) for k in found}) > 1,
            "type_ok": bool(found) and Counter(majority[k] for k in found).most_common(1)[0][0]
            == g.type.value,
        })  # fmt: skip
    by_name = {g.name: g for g in gold.entities}

    relations = [(r.source, r.type.value, r.target, r.support)
                 for x in extractions for c in x.chunks for r in c.relations]  # fmt: skip
    relation_rows = []
    for gr in gold.relations:
        src, tgt = keys_of(by_name[gr.source]), keys_of(by_name[gr.target])
        pair = [r for r in relations if {r[0], r[2]} & src and {r[0], r[2]} & tgt]
        typed = [r for r in pair if r[0] in src and r[2] in tgt and r[1] in gr.types]
        relation_rows.append({
            "source": gr.source, "target": gr.target, "types": gr.types,
            "pair": bool(pair), "typed": bool(typed),
            "stated": any(r[3] == "stated" for r in typed),
            "extracted": sorted({f"{r[0]} {r[1]} {r[2]} ({r[3]})" for r in pair}),
        })  # fmt: skip

    unlisted = sorted(
        ({"key": k, "type": majority[k], "mentions": sum(types[k].values())}
         for k in types if k not in matched_keys),
        key=lambda r: (-r["mentions"], r["key"]),
    )  # fmt: skip
    totals = Counter()
    for x in extractions:
        totals.update(x.counts())
    per_type = defaultdict(lambda: [0, 0])
    for row in entity_rows:
        per_type[row["type"]][0] += row["found"]
        per_type[row["type"]][1] += 1
    n_e, n_r = len(entity_rows), len(relation_rows)
    summary = {
        "gold_entities": n_e,
        "entity_recall": round(sum(r["found"] for r in entity_rows) / n_e, 3),
        "entity_recall_by_type": {t: f"{a}/{b}" for t, (a, b) in sorted(per_type.items())},
        "type_agreement": round(
            sum(r["type_ok"] for r in entity_rows) / max(1, sum(r["found"] for r in entity_rows)), 3
        ),  # fmt: skip
        "fragmented": [r["name"] for r in entity_rows if r["fragmented"]],
        "gold_relations": n_r,
        "relation_pair_recall": round(sum(r["pair"] for r in relation_rows) / n_r, 3),
        "relation_typed_recall": round(sum(r["typed"] for r in relation_rows) / n_r, 3),
        "relation_typed_stated": sum(r["stated"] for r in relation_rows),
        "extracted_entities": len(types),
        "unlisted_entities": len(unlisted),
        "same_as_links": len(same_as),
        "totals": {k: totals[k] for k in sorted(totals)},
    }
    return {"summary": summary, "entities": entity_rows, "relations": relation_rows,
            "unlisted": unlisted}  # fmt: skip


def load_extractions(store: DocumentStore, config_id: str) -> list[DocumentExtraction]:
    out = []
    for r in store.list_records():
        if r.status is IngestionStatus.COMPLETED:
            x = store.read_extraction(r.document_id, config_id)
            if x is not None:
                out.append(x)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.extraction_eval")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    extractions = load_extractions(DocumentStore(settings.data_dir), settings.extraction_config_id)
    result = evaluate(GoldSet.load(), extractions)
    if extractions:
        x = extractions[0]
        result["setup"] = {"model": x.model, "model_digest": x.model_digest,
                           "prompt_version": x.prompt_version, "config_id": x.config_id,
                           "documents": len(extractions)}  # fmt: skip
    print(json.dumps(result["summary"], indent=2))
    for row in result["entities"]:
        if not row["found"] or not row["type_ok"] or row["fragmented"]:
            print(f"entity  {row['name']:<20} found={row['found']} type_ok={row['type_ok']} "
                  f"keys={row['keys']}")  # fmt: skip
    for row in result["relations"]:
        mark = "OK " if row["typed"] else ("~  " if row["pair"] else "MISS")
        print(f"rel {mark} {row['source']} -[{'/'.join(row['types'])}]-> {row['target']}  "
              f"{row['extracted']}")  # fmt: skip
    print("unlisted:", ", ".join(f"{u['key']} ({u['type']})" for u in result["unlisted"]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
