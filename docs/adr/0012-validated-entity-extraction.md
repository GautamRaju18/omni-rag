# ADR-0012: LLM entity/relation extraction, validated against the source and stored with provenance

**Status:** accepted (2026-11-05, Phase 5b)

## Context
The brief's Phase 5 needs Entity nodes and MENTIONS / RELATED_TO / CITES relationships. It also says:

> Do not blindly trust LLM-generated entities or relationships. Validate extracted graph data, preserve provenance, and distinguish inferred relationships from source-supported facts.

The only local model that fits next to bge-m3 on a 6 GB GPU is `qwen3:4b-instruct`.

What the probe on 4 chunks found (2026-11-05, Ollama 0.40.14):
- **Speed:** 12–22 s per chunk.
- **Fabricated entities:** measurements ("5%", "100 cycles") and status values (`in_progress`) came back as entities.
- **Bad relations:** reversed directions ("E-5 USES AUR-C103"), impossible types ("AUR-C105 TESTED_BY E-7"), endpoints that aren't entities ("45 C"), and table rows quoted with columns dropped.
- **JSON shape:** Ollama's `format` (JSON schema) guaranteed parseable JSON with valid enums every time.

## Decision
**Extraction.**
- **One call per chunk:** each retrievable chunk of the graph's chunk set gets one `/api/chat` call with `format` = JSON schema, `temperature 0` and `seed 42`.
- **Closed vocabulary:** 8 entity types and 10 relation types (RELATED_TO is the fallback).
- **Context:** the prompt includes the chunk's contextual header for disambiguation. Grounding only counts the chunk text.
- **Versioned prompt:** `PROMPT_VERSION` (`kg-v2`) is part of the cache key.

**Validation (deterministic, `knowledge_base/entities.py`).**
- **Grounding and normalisation:** an entity must occur in the chunk text at word boundaries, after NFKC, dash and whitespace normalisation and with markdown emphasis removed.
- **Entities are rejected for:**
  - looking like a measurement or date (starting with a digit or a sign);
  - being a snake_case field value;
  - a bad length;
  - not being in the text;
  - appearing **only in model-written text** (a vision description): `generated_only`.
- **Relations are rejected for:**
  - an endpoint that isn't an accepted entity in the same chunk;
  - a self-loop;
  - a type signature that fits neither way.
- **Reversed relations:** if a relation fits only the other way round, it is flipped and flagged `direction_fixed`.
- **Support labels:**
  - `stated`: the evidence quote is found in source text (verbatim, or as a token subsequence of one line) and that passage names both endpoints.
  - `inferred`: anything else, including evidence that exists only in generated text.
- **Rejections are kept:** every rejection is stored with a reason. Nothing is silently dropped.

**Resolution.**
- **Canonical key:** an entity's key is its normalised name, with honorifics and a "Project " prefix (for PROJECT) removed. `entity_id = ent_ + sha256(key)[:16]`, so one name in many documents is one node.
- **Type:** the majority type over all mentions; all type votes are stored.
- **Person aliases within a document** are resolved to the unique full name they abbreviate: first name, last name, or initial + last name. The rule is stored on the mention (`resolution`).
- **Across documents**, no merging: a recomputed `(:Entity)-[:SAME_AS {rule}]->(:Entity)` link joins a single-name person to a unique full name. Ambiguous names stay apart.

**Storage and graph.**
- **Files:** `extractions/<config>.json` per document holds the raw replies plus the validated result, model digest and prompt version.
- **Reuse:** a chunk whose input is unchanged reuses its stored reply. `revalidate` re-applies changed rules without any model call.
- **Graph shape:**
  - `(:Chunk)-[:MENTIONS {surface, type, origin, confidence, resolution, model, prompt_version}]->(:Entity)`.
  - `(:Entity)-[:RELATED_TO {type, support, evidence, evidence_verified, evidence_origin, direction_fixed, chunk_id, document_id, is_latest, model, prompt_version}]->(:Entity)`: one edge per fact per chunk, so provenance is never merged away.
- **Same transaction:** entities are written with the document's structure. They are refreshed (name, type, aliases, counts) from their mentions, and orphans are deleted.
- **Versions:** RELATED_TO edges get `is_latest` from their document version.
- **CITES:** `(:Chunk)-[:CITES]->(:Document)` when a chunk contains another latest document's title (at least 12 characters). It's recomputed corpus-wide.
- **Not at ingestion by default:** `extraction_enabled=false`, because of the cost (≈15 s per chunk), the VRAM contention with the vision model during uploads, and because the step can always be run later (`python -m omnirag.knowledge_base extract`).

## Measured (2026-11-05, 58 chunks, gold set of 37 entities / 14 facts)
| | Value |
|---|---|
| Cost | 16.2 s per chunk (939 s total), 21,303 output tokens; rerun with stored replies: 0 calls, 6.1 s |
| Graph | 82 entities, 217 mentions, 97 relations (71 stated / 26 inferred), 1 SAME_AS, 0 CITES |
| Rejected | 203 (unknown_endpoint 118, measurement 38, not_in_text 24, type_signature 17, field_value 6) |
| Entity recall / type agreement | 36/37 / 36/36; 2 fragmented (NaDFOB, EC: long forms in other chunks) |
| Relation recall | pair 11/14, typed 10/14 (all stated) |
| Known contradiction | "E-5 REPLACES E-7" and "E-7 REPLACES E-5", both stated |

Details and the per-fact table: [learning/phase-05b](../learning/phase-05b-entity-extraction.md) §6.

## Consequences
- The graph only holds entities that are literally in the source. Paraphrased or implied entities are lost; that is the price of trust.
- `stated` means "a source passage names both endpoints", not "the relation type and direction are right". The model can still assign a wrong type or direction between two legitimately co-mentioned entities when both directions fit the signature (e.g. E-5/E-7 REPLACES). Phase 7 should weigh `stated` over `inferred` and show the evidence quote.
- Keys are type-independent, except that a "Project " prefix is only stripped for PROJECT. A project typed as ORGANIZATION in one chunk becomes a second node (measured as fragmentation).
- Quality is measured against a hand-built gold set (`data/eval/entities_seed.json`: 37 entities, 14 relations, every name and quote checked against the samples by a test). There is no automatic precision figure: extracted entities outside the gold set are listed for review.
