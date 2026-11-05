"""Validate LLM-extracted entities and relations against the source text, then resolve them.

The model proposes; this module decides. Nothing reaches the graph unless it is grounded in
the chunk it came from:

* **Entities** must appear verbatim (after Unicode/whitespace/dash normalisation, at word
  boundaries) in the chunk's own text, not in its contextual header. Names that look like
  measurements ("5%", "100 cycles") are rejected, and so is anything that appears only in
  model-written text (an image description), because that text is not a source.
* **Relations** need two accepted endpoints in the same chunk and a type signature that fits
  (``TESTED_BY`` goes from a sample to a person). A relation that only fits reversed is
  flipped and flagged ``direction_fixed``.
* **Support.** A relation is ``stated`` when its evidence quote is found in the source text
  (verbatim, or as a token subsequence of one line, since models drop table columns) and that
  passage names both endpoints. Otherwise it is kept as ``inferred`` and labelled as such.
* **Resolution.** Every entity gets a canonical key (lowercase, honorifics and "Project"
  prefixes removed), so "Dr. Priya Raman" and "Priya Raman" are one node. Within a document a
  first name or "A. Tanaka" maps to the unique full name it abbreviates. Every decision is
  recorded on the mention (``resolution``).

Everything here is pure Python with no I/O, so the rules are unit-tested on exact cases.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from omnirag.chunking.models import Chunk
from omnirag.ingestion.models import ContentOrigin, DocumentElement


class EntityType(StrEnum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PROJECT = "PROJECT"
    PRODUCT = "PRODUCT"
    MATERIAL = "MATERIAL"  # chemicals, electrolytes, components
    SAMPLE = "SAMPLE"  # individual test cells / specimens
    MILESTONE = "MILESTONE"
    DOCUMENT = "DOCUMENT"


class RelationType(StrEnum):
    PART_OF = "PART_OF"
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"
    MEMBER_OF = "MEMBER_OF"
    AUTHORED = "AUTHORED"
    USES = "USES"
    TESTED_BY = "TESTED_BY"
    REPLACES = "REPLACES"
    DEPENDS_ON = "DEPENDS_ON"
    SUPPLIES = "SUPPLIES"
    RELATED_TO = "RELATED_TO"  # a stated link that fits no other type


E = EntityType
_THINGS = {E.PROJECT, E.PRODUCT, E.MATERIAL, E.SAMPLE, E.MILESTONE, E.DOCUMENT}
_AGENTS = {E.PERSON, E.ORGANIZATION}
# Allowed (source types, target types) per relation; RELATED_TO accepts anything.
SIGNATURES: dict[RelationType, tuple[set[EntityType], set[EntityType]]] = {
    RelationType.PART_OF: (set(E) - {E.PERSON}, {E.PROJECT, E.PRODUCT, E.ORGANIZATION, E.DOCUMENT}),
    RelationType.RESPONSIBLE_FOR: (_AGENTS, _THINGS),
    RelationType.MEMBER_OF: ({E.PERSON, E.ORGANIZATION}, {E.ORGANIZATION, E.PROJECT}),
    RelationType.AUTHORED: (_AGENTS, {E.DOCUMENT}),
    RelationType.USES: ({E.SAMPLE, E.PRODUCT, E.MATERIAL, E.PROJECT}, {E.MATERIAL, E.PRODUCT}),
    RelationType.TESTED_BY: ({E.SAMPLE, E.PRODUCT, E.MATERIAL}, _AGENTS),
    RelationType.REPLACES: ({E.MATERIAL, E.PRODUCT, E.SAMPLE}, {E.MATERIAL, E.PRODUCT, E.SAMPLE}),
    RelationType.DEPENDS_ON: ({E.MILESTONE, E.PROJECT, E.PRODUCT}, set(E) - {E.PERSON}),
    RelationType.SUPPLIES: ({E.ORGANIZATION}, {E.MATERIAL, E.PRODUCT}),
    RelationType.RELATED_TO: (set(E), set(E)),
}

Support = Literal["stated", "inferred"]
_ORIGIN_RANK = {ContentOrigin.ORIGINAL: 0, ContentOrigin.OCR: 1, ContentOrigin.GENERATED: 2}


# ---------------------------------------------------------------- raw model output
class RawEntity(BaseModel):
    name: str
    type: EntityType


class RawRelation(BaseModel):
    source: str
    type: RelationType
    target: str
    evidence: str = ""


class RawOutput(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)


# ---------------------------------------------------------------- validated results
class Mention(BaseModel):
    key: str  # canonical name; Entity identity
    surface: str  # as written in the text (the model's copy, verified)
    type: EntityType
    origin: ContentOrigin  # best origin among the elements containing it
    confidence: float  # element confidence (OCR mean word confidence, else 1.0)
    resolution: str | None = None  # e.g. "alias:first_name" when remapped to a full name


class Relation(BaseModel):
    source: str  # entity keys
    target: str
    type: RelationType
    support: Support
    evidence: str
    evidence_verified: bool
    evidence_origin: ContentOrigin | None = None
    direction_fixed: bool = False


class Rejection(BaseModel):
    kind: Literal["output", "entity", "relation"]
    item: dict[str, Any]
    reason: str


class ChunkExtraction(BaseModel):
    chunk_id: str
    input_sha256: str  # of the prompt input: reuse the raw output while it is unchanged
    raw: dict[str, Any] | None = None  # the model's JSON as returned (audit trail)
    error: str | None = None
    mentions: list[Mention] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    document_id: str
    config_id: str
    model: str
    model_digest: str
    prompt_version: str
    extracted_at: str
    chunks: list[ChunkExtraction] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        reasons = Counter(r.reason for c in self.chunks for r in c.rejected)
        support = Counter(r.support for c in self.chunks for r in c.relations)
        return {
            "chunks": len(self.chunks),
            "errors": sum(1 for c in self.chunks if c.error),
            "mentions": sum(len(c.mentions) for c in self.chunks),
            "entities": len({m.key for c in self.chunks for m in c.mentions}),
            "relations": sum(len(c.relations) for c in self.chunks),
            "stated": support["stated"],
            "inferred": support["inferred"],
            "rejected": sum(reasons.values()),
            **{f"rejected_{k}": v for k, v in sorted(reasons.items())},
        }


# ---------------------------------------------------------------- normalisation
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_HONORIFIC = re.compile(r"^(dr|prof|mr|mrs|ms|mx)\.?\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-+/][a-z0-9]+)*")
_MEASUREMENT = re.compile(r"^[\d.,\s%~<>±+-]")  # "5%", "100 cycles", "45 C", "-20 °C"
_MONEY = re.compile(r"^(eur|usd|gbp|chf|jpy|[$€£¥])\s?[\d.,]")  # "EUR 85,000"
_FIELD_VALUE = re.compile(r"^[a-z]+(_[a-z]+)+$")  # "in_progress": a status, not a name
_LABEL_VALUE = re.compile(r"^[a-z][a-z _]*:\s+[\d.,%+-]")  # "Category: 400", not "EC:DEC (1:1)"
_STATUS = {"passed", "failed", "done", "true", "false", "pending", "active", "completed",
           "in progress", "on track", "at risk", "open", "closed"}  # fmt: skip
# A class word before an identifier: "cell AUR-C103", "milestone M2", "electrolyte E-5".
_CLASS_PREFIX = re.compile(r"^(cells?|samples?|electrolytes?|milestones?|materials?)\s+(?=\S*\d)")


def norm(text: str) -> str:
    """NFKC (₆ → 6, ﬁ → fi), unified dashes, no markdown emphasis, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = text.replace("*", "").replace("`", "")
    return " ".join(text.casefold().split())


def entity_key(name: str, entity_type: EntityType | str | None = None) -> str:
    key = norm(name).strip(" .,;:!?\"'")
    # Brackets only go when unbalanced: keep "ec:dec (1:1)" and "na2fe[fe(cn)6]" whole.
    for opening, closing in ("()", "[]"):
        while key.endswith(closing) and key.count(closing) > key.count(opening):
            key = key[:-1].rstrip()
        while key.startswith(opening) and key.count(opening) > key.count(closing):
            key = key[1:].lstrip()
    key = _HONORIFIC.sub("", key)
    key = _CLASS_PREFIX.sub("", key)
    key = re.sub(r"\s*\+\s*", " + ", key)
    if entity_type == EntityType.PROJECT and key.startswith("project "):
        key = key[len("project ") :]
    return key


def entity_id(key: str) -> str:
    return "ent_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def contains(haystack: str, needle: str) -> bool:
    """``needle`` in ``haystack`` (both normalised) at word boundaries: "EC" not in "DEC"."""
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def find_evidence(evidence: str, text: str) -> str | None:
    """The normalised source passage that backs ``evidence``, or None.

    1. verbatim (normalised) substring of the text;
    2. else the evidence's tokens in order within one line of the text (a table row whose
       columns the model dropped, or a quote with changed punctuation).
    """
    ev = norm(evidence).strip(" .")
    if len(ev) < 3:
        return None
    flat = norm(text)
    if ev in flat:
        return ev
    tokens = _TOKEN.findall(ev)
    if not tokens:
        return None
    for line in text.splitlines():
        line_tokens = iter(_TOKEN.findall(norm(line)))
        if all(t in line_tokens for t in tokens):  # subsequence check (iterator consumed)
            return norm(line)
    return None


# ---------------------------------------------------------------- validation
def _origin_of(needle: str, elements: list[DocumentElement], chunk: Chunk):
    """(best origin, confidence) of the elements whose text contains ``needle``."""
    hits = [e for e in elements if contains(norm(e.content), needle)]
    if not hits:  # spans elements or chunker labels: be conservative
        worst = max((ContentOrigin(o) for o in chunk.origins), key=_ORIGIN_RANK.get,
                    default=ContentOrigin.ORIGINAL)  # fmt: skip
        return worst, chunk.min_confidence
    best = min((e.origin for e in hits), key=_ORIGIN_RANK.get)
    return best, min(e.confidence for e in hits if e.origin is best)


def validate_chunk(
    chunk: Chunk, elements: list[DocumentElement], raw: dict[str, Any] | None
) -> tuple[list[Mention], list[Relation], list[Rejection]]:
    """Accept or reject every proposed entity and relation of one chunk."""
    rejected: list[Rejection] = []
    try:
        output = RawOutput.model_validate(raw or {})
    except ValidationError as exc:
        reason = f"schema: {exc.errors()[0]['msg']}"
        return [], [], [Rejection(kind="output", item={"raw": raw}, reason=reason)]

    text = norm(chunk.content)
    by_surface: dict[str, Mention] = {}
    for ent in output.entities:
        surface = norm(ent.name).strip(" .,;:")
        item = ent.model_dump(mode="json")
        reason = None
        if len(surface) < 2 or len(surface) > 80:
            reason = "bad_length"
        elif _MEASUREMENT.match(surface) or _MONEY.match(surface) or "%" in surface:
            reason = "measurement"
        elif _FIELD_VALUE.match(surface) or _LABEL_VALUE.search(surface) or surface in _STATUS:
            reason = "field_value"
        elif not contains(text, surface):
            reason = "not_in_text"
        if reason is None:
            origin, confidence = _origin_of(surface, elements, chunk)
            if origin is ContentOrigin.GENERATED:
                reason = "generated_only"  # only in a model-written description
        if reason:
            rejected.append(Rejection(kind="entity", item=item, reason=reason))
            continue
        if surface in by_surface:
            continue  # same name listed twice
        by_surface[surface] = Mention(
            key=entity_key(ent.name, ent.type), surface=ent.name.strip(), type=ent.type,
            origin=origin, confidence=round(confidence, 3),
        )  # fmt: skip
    _link_abbreviations(text, list(by_surface.values()))
    mentions = _dedupe_keys(list(by_surface.values()))
    by_key = {m.key: m for m in mentions}

    def lookup(name: str) -> Mention | None:
        surface = norm(name).strip(" .,;:")
        if surface in by_surface:
            return by_key[by_surface[surface].key]
        return next((m for m in mentions if m.key == entity_key(name, m.type)), None)

    relations: dict[tuple[str, str, str], Relation] = {}
    for rel in output.relations:
        item = rel.model_dump(mode="json")
        src, tgt = lookup(rel.source), lookup(rel.target)
        if src is None or tgt is None:
            rejected.append(Rejection(kind="relation", item=item, reason="unknown_endpoint"))
            continue
        if src.key == tgt.key:
            rejected.append(Rejection(kind="relation", item=item, reason="self_loop"))
            continue
        sources, targets = SIGNATURES[rel.type]
        fixed = False
        if not (src.type in sources and tgt.type in targets):
            if tgt.type in sources and src.type in targets:
                src, tgt, fixed = tgt, src, True
            else:
                rejected.append(Rejection(kind="relation", item=item, reason="type_signature"))
                continue
        passage = find_evidence(rel.evidence, chunk.content)
        names_both = passage is not None and all(_named_in(m, passage) for m in (src, tgt))
        origin = _origin_of(passage, elements, chunk)[0] if passage else None
        stated = names_both and origin is not ContentOrigin.GENERATED
        relations.setdefault((src.key, rel.type.value, tgt.key), Relation(
            source=src.key, target=tgt.key, type=rel.type,
            support="stated" if stated else "inferred", evidence=rel.evidence.strip(),
            evidence_verified=passage is not None, evidence_origin=origin,
            direction_fixed=fixed,
        ))  # fmt: skip
    return mentions, list(relations.values()), rejected


def _named_in(mention: Mention, passage: str) -> bool:
    """The passage names the entity: its surface, or for a person any name part ("Aiko")."""
    if contains(passage, norm(mention.surface)) or contains(passage, mention.key):
        return True
    if mention.type is EntityType.PERSON:
        return any(len(t) >= 3 and contains(passage, t) for t in mention.key.split())
    return False


def _dedupe_keys(mentions: list[Mention]) -> list[Mention]:
    """Two surfaces with one key ("Dr. Priya Raman", "Priya Raman"): keep the first."""
    seen: dict[str, Mention] = {}
    for m in mentions:
        seen.setdefault(m.key, m)
    return list(seen.values())


# ---------------------------------------------------------------- resolution
def person_alias_map(full_names: set[str], short_names: set[str]) -> dict[str, tuple[str, str]]:
    """Map abbreviated person keys to the unique full key they abbreviate.

    ``"aiko"`` → ``"aiko tanaka"`` (first name), ``"tanaka"`` → (last name),
    ``"a. tanaka"`` → (initial + last name). Ambiguous names (two Marcos) stay unresolved.
    Returns ``{short: (full, rule)}``.
    """
    out: dict[str, tuple[str, str]] = {}
    for short in short_names:
        parts = short.replace(".", ". ").split()
        candidates: dict[str, str] = {}
        for full in full_names:
            fp = full.split()
            if len(fp) < 2 or full == short:
                continue
            if len(parts) == 1 and parts[0] == fp[0]:
                candidates[full] = "alias:first_name"
            elif len(parts) == 1 and parts[0] == fp[-1]:
                candidates[full] = "alias:last_name"
            elif (len(parts) == 2 and len(parts[0].rstrip(".")) == 1
                  and fp[0].startswith(parts[0].rstrip(".")) and parts[1] == fp[-1]):  # fmt: skip
                candidates[full] = "alias:initial"
        if len(candidates) == 1:
            out[short] = next(iter(candidates.items()))
    return out


def is_abbreviation(short: str, long_: str) -> bool:
    """Schwartz–Hearst style test: ``short`` is one token whose letters/digits appear in
    order in ``long_``, starting with its first letter ("ec" / "ethylene carbonate").
    Rules out parentheticals that aren't definitions: "cell aur-c103 (electrolyte e-5)"."""
    chars = re.sub(r"[^a-z0-9]", "", short)
    if " " in short or not 2 <= len(chars) <= 10 or not long_ or chars[0] != long_[0]:
        return False
    rest = iter(long_)
    return all(ch in rest for ch in chars)


def _link_abbreviations(text: str, mentions: list[Mention]) -> None:
    """ "ethylene carbonate (EC)" or "EC (ethylene carbonate)" in the text: the long form
    takes the abbreviation's key (the shorter, more frequent form across documents)."""
    for a in mentions:
        for b in mentions:
            if a is b or a.key == b.key or len(b.key) >= len(a.key):
                continue
            long_, short = norm(a.surface), norm(b.surface)
            defined = f"{long_} ({short})" in text or f"{short} ({long_})" in text
            if defined and is_abbreviation(short, long_):
                a.key, a.resolution = b.key, "alias:abbreviation"


def resolve_document(chunks: list[ChunkExtraction]) -> None:
    """Within one document, remap aliases to one key (in place): abbreviations defined in any
    chunk apply to the whole document, and short person names map to unique full names."""
    aliases: dict[str, tuple[str, str]] = {
        entity_key(m.surface, m.type): (m.key, m.resolution)
        for c in chunks for m in c.mentions if m.resolution == "alias:abbreviation"
    }  # fmt: skip
    persons = {m.key for c in chunks for m in c.mentions if m.type is EntityType.PERSON}
    full = {k for k in persons if len(k.split()) >= 2 and "." not in k.split()[0]}
    aliases.update(person_alias_map(full, persons - full))
    if not aliases:
        return
    for c in chunks:
        for m in c.mentions:
            if m.key in aliases:
                m.key, m.resolution = aliases[m.key]
        c.mentions = _dedupe_keys(c.mentions)
        merged: dict[tuple[str, str, str], Relation] = {}
        for r in c.relations:
            r.source = aliases.get(r.source, (r.source,))[0]
            r.target = aliases.get(r.target, (r.target,))[0]
            if r.source != r.target:
                merged.setdefault((r.source, r.type.value, r.target), r)
        c.relations = list(merged.values())


def entity_summary(mentions: list[dict[str, Any]]) -> dict[str, Any]:
    """Name, type and aliases of one entity from all its mentions (order-independent)."""
    surfaces = Counter(m["surface"] for m in mentions)
    types = Counter(m["type"] for m in mentions)
    name = sorted(surfaces.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]
    ranked = sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "name": name,
        "type": ranked[0][0],
        "types": [t for t, _ in ranked],
        "type_counts": [n for _, n in ranked],
        "aliases": sorted(surfaces),
        "mention_count": len(mentions),
        "document_count": len({m["document_id"] for m in mentions}),
    }


def group_mentions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["entity_id"]].append(r)
    return grouped
