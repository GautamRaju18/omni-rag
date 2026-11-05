# Phase 5b: Entities and relations, extracted by an LLM and checked against the source

> Decision record: [ADR-0012](../adr/0012-validated-entity-extraction.md). Builds on the document graph from [Phase 5a](phase-05a-graph-structure.md). Raw results: [docs/benchmarks/extraction-2026-11-05.json](../benchmarks/extraction-2026-11-05.json).

## 1. Problem being solved
Phase 5a's graph knows *where* text is: documents, sections, pages, chunks. It doesn't know *what the text is about*. A question such as "which cells use E-7 and who tests them?" needs:
- **entities:** "AUR-C105", "E-7", "Aiko Tanaka", whichever document they appear in;
- **relations:** "AUR-C105 USES E-7", "AUR-C105 TESTED_BY Aiko Tanaka".

A local 4B model can propose those, but it also proposes things that aren't there. The brief is explicit: don't trust it blindly, keep provenance, and separate what the source states from what was inferred.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Entity / mention** | An *Entity* is one real-world thing (one node). A *mention* is one place a chunk names it (a `MENTIONS` edge with the surface form, origin and confidence). |
| **Structured output** | Ollama's `format` takes a JSON schema and constrains decoding to it. The reply always parses, and types are always from the enum. It guarantees *shape*, never *truth*. |
| **Grounding** | An entity is kept only if its name occurs in the chunk's own text. This is the main defence against invented entities. |
| **Type signature** | Which entity types a relation may connect. `TESTED_BY` goes from a SAMPLE/PRODUCT/MATERIAL to a PERSON/ORGANIZATION. It catches impossible and reversed relations. |
| **Stated vs. inferred** | `stated`: the evidence quote is really in the source and names both endpoints. `inferred`: everything else (missing, invented or generated-only evidence). Both are kept, labelled. |
| **Entity resolution** | Deciding that two names are one thing. Here it's conservative: canonical keys ("Dr. Priya Raman" = "Priya Raman"), person aliases within a document, and `SAME_AS` links across documents instead of merging. |
| **Provenance** | For every fact: which chunk, document, model digest and prompt version produced it, from which origin (original, OCR or generated), with what evidence. |

## 3. Architecture
```
knowledge_base/
  extractor.py   OllamaExtractor: prompt (kg-v2) + JSON schema, one call per chunk, reuse of
                 unchanged replies; ExtractionStep (pipeline); revalidate (no model calls)
  entities.py    pure rules: norm / entity_key / contains / find_evidence, validate_chunk,
                 person_alias_map, resolve_document, entity_summary; result models
  graph_builder  + MENTIONS / RELATED_TO / Entity refresh in the document transaction,
                 + corpus pass: SAME_AS (person aliases) and CITES (title references)
  traversal.py   + entities(), entity_detail()
evaluation/extraction_eval.py   gold set data/eval/entities_seed.json (37 entities, 14 facts)

<data_dir>/documents/<doc>/extractions/multimodal-256-ctx.json
    raw model replies + validated mentions/relations + every rejection with its reason
```

```
(:Chunk)-[:MENTIONS {surface, type, origin, confidence, resolution, model, prompt_version}]->(:Entity)
(:Entity)-[:RELATED_TO {type, support, evidence, evidence_verified, evidence_origin,
                        direction_fixed, chunk_id, document_id, is_latest, ...}]->(:Entity)
(:Entity)-[:SAME_AS {rule}]->(:Entity)        "Marco" -> "Marco Lindqvist"
(:Chunk)-[:CITES {match: 'title'}]->(:Document)
```

## 4. What happens to one chunk
1. **Prompt:** the system prompt (types, directions, "copy names exactly", "evidence = exact quote"), then the chunk's header (document > section), then `TEXT:` and the chunk content.
2. **Model call:** Ollama returns JSON that matches the schema (`temperature 0`, `seed 42`). The raw reply is stored.
3. **Entities:** normalise (NFKC turns NaPF₆ into napf6; dashes are unified; `**bold**` is removed), then check in order:
   - length 2–80;
   - not a measurement or date;
   - not a `snake_case` value;
   - **found in the chunk text at word boundaries** (so "EC" is not found inside "DEC");
   - not found *only* in a model-written image description.
4. **Relations:**
   - both endpoints must be accepted entities of this chunk, and not the same one;
   - the type signature must fit. If it only fits reversed, the relation is flipped and flagged `direction_fixed`;
   - the evidence is located (verbatim, or as the quote's tokens in order within one line, since models drop table columns);
   - `stated` if found in original/OCR text and naming both endpoints, else `inferred`.
5. **Document level:** "Aiko" and "A. Tanaka" map to "Aiko Tanaka" when that is the only matching full name in the document.
6. **Graph:**
   - mentions and relations are written in the document's transaction;
   - each touched entity's name, type, aliases and counts are recomputed from all its mentions;
   - orphans are deleted;
   - SAME_AS and CITES are recomputed corpus-wide.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.knowledge_base init                  # adds Entity constraint + indexes
uv run python -m omnirag.knowledge_base extract --all         # ~20 min for the 58 sample chunks
uv run python -m omnirag.knowledge_base entities --type PERSON
uv run python -m omnirag.knowledge_base entity "AUR-C103"
uv run python -m omnirag.evaluation.extraction_eval           # against the gold set
uv run python -m omnirag.knowledge_base revalidate --all      # after changing a rule: no LLM
uv run python -m omnirag.knowledge_base extract --all         # again: replies reused, ~seconds
```
In the Neo4j Browser:
```cypher
MATCH (s:Entity)-[r:RELATED_TO {support: 'stated', is_latest: true}]->(t:Entity)
RETURN s.name, r.type, t.name, r.evidence LIMIT 50
```

## 6. Expected outputs (measured 2026-11-05)
Setup: Ollama 0.40.14, `qwen3:4b-instruct` (digest in the benchmark JSON), prompt `kg-v2`, chunk set `multimodal-256-ctx` (58 chunks, 12 documents), RTX 3060 6 GB.

**Cost.** `extract --all`:
- 58 model calls, 939 s of model time (16.2 s per chunk), 957 s wall, 21,303 output tokens, `errors=0`.
- GPU memory during the run: 4,563 / 6,144 MiB; qwen3:4b-instruct used 3.31 GB of VRAM.
- A second `extract --all` made **0 model calls** (58 reused) and took 6.1 s.

**What validation did.** Final rules (after the fixes in §7), per document:

| Document | entities | relations | stated | inferred | rejected |
|---|---|---|---|---|---|
| aurora_config.json | 16 | 6 | 0 | 6 | 11 |
| aurora_research_notes.md | 18 | 14 | 12 | 2 | 13 |
| cell_test_results.csv | 13 | 16 | 16 | 0 | 24 |
| helios_aurora_s1.html | 13 | 4 | 3 | 1 | 10 |
| meeting_notes_2026-03-12.txt | 15 | 12 | 9 | 3 | 2 |
| aurora_budget_2026.xlsx | 19 | 9 | 9 | 0 | 17 |
| aurora_q1_review.pptx | 17 | 5 | 3 | 2 | 46 |
| aurora_status_report_2026-03.docx | 24 | 10 | 7 | 3 | 32 |
| aurora_lab_notebook_scan.pdf | 4 | 4 | 0 | 4 | 5 |
| aurora_safety_review.pdf | 24 | 14 | 10 | 4 | 23 |
| whiteboard_2026-03-20.jpg | 3 | 1 | 0 | 1 | 7 |
| e7_technical_note.pdf | 13 | 2 | 2 | 0 | 13 |

- **Totals:** 217 mentions of **82 entities**; **97 relations** (71 stated, 26 inferred); **203 rejections**.
- **Rejections by reason:** unknown_endpoint 118, measurement 38, not_in_text 24, type_signature 17, field_value 6. 12 relations were stored with `direction_fixed`.
- **Mention origins:** 7 OCR, the rest original. No entity survived only in generated text.
- **Why some documents have no stated relations:**
  - aurora_config.json: each JSON record names one thing per line (`lead: Dr. Priya Raman`), so no single line names both endpoints.
  - The scanned notebook: its OCR lines name a cell or a person, never both.
- **Resolution:**
  - 4 mentions in the meeting notes were resolved to full names ("Priya", "Aiko", "Marco", "Samuel", all `alias:first_name`).
  - 1 `SAME_AS` link: "Marco" (whiteboard, OCR) → "Marco Lindqvist".
  - 0 `CITES`: no document names another by title.
- **Most-mentioned entities:**
  - E-5: 16 mentions in 7 documents.
  - E-7: 16 mentions in 8 documents.
  - Dr. Priya Raman: 9 mentions in 5 documents.
  - AUR-C105: 9 mentions in 7 documents.

**Against the gold set** (`extraction_eval`, [results](../benchmarks/extraction-2026-11-05.json)):

| Metric | Value |
|---|---|
| Entity recall | **36 / 37** (0.973). Missed: M1 (only in tables, where the model extracted "Baseline chemistry frozen" instead) |
| Recall by type | MATERIAL 12/12, ORGANIZATION 5/5, PERSON 6/6, SAMPLE 8/8, PRODUCT 2/2, PROJECT 1/1, MILESTONE 2/3 |
| Type agreement (majority type = gold type) | 36 / 36 |
| Fragmented (one thing, several unlinked nodes) | 2: NaDFOB / "sodium difluoro(oxalato)borate", EC / "ethylene carbonate" |
| Relation pair recall (any edge between the two) | **11 / 14** (0.786) |
| Relation typed recall (right type and direction) | **10 / 14** (0.714), all 10 `stated` |
| Extracted entities outside the gold set | 43 of 82 (listed in the JSON for review) |

Per fact:

| Gold fact | Result |
|---|---|
| Marco Lindqvist RESPONSIBLE_FOR E-7 | ✅ stated |
| AUR-C103 USES E-5 | ✅ stated |
| AUR-C105 USES E-7 | ✅ stated |
| AUR-C106 USES E-7 | ✅ stated |
| E-7 REPLACES E-5 | ✅ stated, **but the reversed edge "E-5 REPLACES E-7" is also stated** (§7) |
| AUR-C103 TESTED_BY Aiko Tanaka | ✅ stated |
| AUR-C107 TESTED_BY Jonas Weber | ✅ stated (after the kg-v2 prompt fix) |
| M2 DEPENDS_ON E-7 | ✅ stated |
| Kokos Carbon Ltd SUPPLIES hard carbon | ✅ stated |
| E-7 USES NaDFOB | ✅ stated (via "sodium difluoro(oxalato)borate", the name used in that document) |
| Project Aurora PART_OF Helios Energy Lab | ~ pair found as "Helios Energy Lab RESPONSIBLE_FOR Aurora" (stated) + RELATED_TO (inferred) |
| Nordic Salts AB SUPPLIES NaPF6 | ❌ the model named the material "NaPF6 salt", a different key |
| Priya Raman RESPONSIBLE_FOR Project Aurora | ❌ no relation between them extracted from the JSON record |
| Aurora S1 USES Prussian white | ❌ no relation extracted |

Limits of these numbers: 37 entities and 14 facts, one model and one corpus. One fact is 0.071 of relation recall. The gold set was written by the same person who wrote the rules, so treat these as a regression baseline, not a quality claim.

## 7. Surprises found during verification
- **The probe found the model's failure modes before any code was written** (4 chunks):
  - measurements as entities ("5%", "100 cycles" as MILESTONE);
  - reversed relations ("E-5 USES AUR-C103");
  - impossible types ("AUR-C105 TESTED_BY E-7");
  - phantom endpoints ("45 C");
  - table rows quoted with a column dropped. That's why `find_evidence` accepts a token subsequence of one line.
- **Prompt v1 → v2.** The model used AUR-C107 and AUR-C108 in relations without listing them as entities, so true table facts were rejected as `unknown_endpoint`. It also listed `in_progress` as a MILESTONE. v2 adds "every name used in a relation must also be listed in entities" and "no status values". AUR-C107 TESTED_BY Jonas Weber is now extracted.
- **First full run → rule fixes, applied with `revalidate` (no model calls).**
  - The first validated run had 95 entities, 56 of them outside the gold set, and 108 relations (79 stated). Reading the unlisted ones showed general gaps:
    - chart labels ("Capacity retention (%)");
    - table fields ("Category: 400");
    - money ("EUR 85,000");
    - status words ("passed", "failed");
    - class-word duplicates ("cell AUR-C103", "milestone M2", "electrolyte E-5");
    - **a key-normalisation bug** that stripped closing brackets ("EC:DEC (1:1)" became "ec:dec (1:1").
  - After the fixes: 82 entities (43 unlisted) and 97 relations (71 stated). The lost relations were junk such as "AUR-C105 USES Capacity retention (%)" ×8 and "Category: 400 RELATED_TO AUR-C105", or duplicates collapsed by the shared key.
  - Gold recall didn't change. These rules come from reading this corpus's output: they're general, but measured only here.
- **A new rule caused a serious bug, and an acceptance check caught it.** The first abbreviation rule ("long (short)" means short is an alias) read "Cell AUR-C103 (electrolyte E-5)" as a definition. It merged AUR-C103 into E-5 across the whole safety review, and `entity "AUR-C103"` returned E-5. The rule now also requires a real abbreviation (Schwartz–Hearst: one token whose characters appear in order in the long form, starting with its first letter). A regression test covers it. With that guard the rule makes 0 links on this corpus, so EC / "ethylene carbonate" stays fragmented.
- **`stated` doesn't mean "correct".** The graph has both "E-7 REPLACES E-5" and "E-5 REPLACES E-7" as stated. Both are materials, so the signature fits both ways, and the evidence names both. Validation can't decide direction from co-occurrence. Contradictory pairs are easy to find (`MATCH (a)-[r1:RELATED_TO]->(b)-[r2:RELATED_TO]->(a) WHERE r1.type = r2.type`); Phase 7 should not trust either without the evidence.
- **The JSON config and the scanned notebook produce only inferred relations.** In both, each line names one thing, so no single passage names both endpoints. The rule is working as designed; the cost is that facts spread over adjacent lines stay `inferred`.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `ExtractionError: Model 'qwen3:4b-instruct' not found` | Pull it: `ollama.exe pull qwen3:4b-instruct`. Also check that port 11434 is the Windows Ollama 0.40.x (`curl localhost:11434/api/version`), not the other project's WSL server. |
| `extract` is very slow, or other models unload | With `OLLAMA_MAX_LOADED_MODELS=2`, the generator, bge-m3 and the vision model compete. Run extraction as a batch, not during uploads with vision on. |
| `errors=N` in the extract output | The reply was unusable twice (e.g. cut off at `num_predict`). The chunk is recorded with `model_error`; re-run `extract` for that document. |
| `entity "X"` says no entity | Names are looked up by canonical key or exact alias: try the full name, or `entities` to list them. |
| A relation you expected is missing | Check `rejected` in the document's `extractions/*.json`. The reason (`unknown_endpoint`, `type_signature`, …) says which rule dropped it. |
| Changing the prompt doesn't change results | `PROMPT_VERSION` is part of the cache key. Bump it, or run `extract --force`. |

## 9. Exercise (≈45 min)
1. **Read the rejections.** Open `extractions/multimodal-256-ctx.json` for `cell_test_results.csv`. Count rejections by reason. For two `unknown_endpoint` rejections, decide whether the model or the rule was wrong.
2. **Change a rule without the model.** In `entities.py`, make `_named_in` require the *full* surface of a person (drop the name-part rule). Run `revalidate --all` and `extraction_eval`. How many gold facts lose `stated`? Revert.
3. **Cypher.** Find every SAMPLE with its USES material and TESTED_BY person, from `stated` edges only. Which cells have no tester, and is that the model's fault or the corpus's?
4. **Think.** Why are cross-document person aliases a `SAME_AS` link instead of a merge? Construct a two-document example where merging "Marco" into "Marco Lindqvist" would be wrong.

## 10. Acceptance criteria — Phase 5b
- [ ] `extract --all` processes 12 documents with `errors=0`, and a second run makes no model calls (`calls=0 cached=58`).
- [ ] `extraction_eval` reports 36/37 entities, 11/14 pairs and 10/14 typed facts (§6). If your numbers differ, compare the model digest first.
- [ ] `entity "AUR-C103"` shows its relations with `support` and the verbatim evidence.
- [ ] No Entity node contains a measurement or a status value. `MATCH (e:Entity) WHERE e.key =~ '^[0-9].*' RETURN e` returns nothing.
- [ ] `uv run pytest` passes, and `uv run pytest -m integration` passes with `neo4j-test` running.
