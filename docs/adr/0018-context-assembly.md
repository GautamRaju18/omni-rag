# ADR-0018: A token budget, not a chunk count; expansion and extractive compression stay opt-in

**Status:** accepted (2026-11-06, Phase 9c)

## Context
Phases 6–9 return the top *k* = 5 chunks. Phase 10 puts *text* into a prompt, so the real constraint is tokens, and five chunks are anything from ~150 to ~900 tokens on this corpus (seed p50 508, max 891). Two retrieval-era facts point at the budget, not at k:
- **q22** (the known ground-truth gap, ADR-0015): its second quote is fully covered by rank 6, at 449 tokens.
- **r05**: covered by rank 7, at 509 tokens; **r02** only by rank 16 (1,682 tokens).

The brief (section 10) also asks for **parent-child retrieval**, **contextual retrieval** and **context compression**. Before building expansion, the corpus was checked (58 retrievable chunks, median 57 tokens, 4,938 tokens in total):
- no evidence quote spans a chunk boundary, so expansion can't *repair* a quote here;
- 46 of 58 chunks have a neighbour, 50 have a section, and a section holds at most 3 chunks (average 1.3).

So expansion was expected to add little on this corpus, and compression had to be measured, not assumed.

## Decision
A new last node, **assemble** (`retrieval/context.py`), builds the generator's context from the workflow's ranking:
```
… → check → assemble → END            (correct → retrieve loops back as before)
```
- **Budget, not k.** Blocks are taken in rank order while they fit `context_budget` (bge-m3 tokens, *including* each block's `file > section` header line). A block that doesn't fit is **skipped** (listed in `skipped`), never cut mid-text; smaller later blocks may still fit.
  - Candidates: the diversified top-k first (the order the check judged), then the other ranked hits, up to `depth` (20).
- **Expansion** (`expand`): `neighbors` adds the previous and next chunk (`NEXT_CHUNK`, same chunk set); `section` adds every chunk of the hit's deepest section with the same `config_id`. Overlapping or touching chunks of one document merge into one block in reading order, so no text appears twice. Each block keeps `hit_chunk_ids` (what retrieval found) separate from `chunk_ids` (what expansion added).
- **Compression** (`compress`, `retrieval/compression.py`) is **extractive only**: blocks are cut into units (sentences, table rows; a kept row brings its table's header and separator lines), units are scored, and the best are kept under the budget in document order, verbatim, with `" … "` marking gaps. Three scorers:
  - `lexical`: question words + 2 × question identifiers in the unit (free);
  - `embedding`: bge-m3 cosine, question vs unit (one batched call, cached);
  - `llm`: qwen3 returns the *numbers* of the units it needs; code keeps those units. Replies are cached per (question, unit texts); on `LLMError` it falls back to lexical and records `fallback`.

  Abstractive compression (asking the model to summarise) was rejected: a summary can't be cited, and a 4B model rewrites numbers and names.
- **API / CLI / dashboard:** `Strategy` gains `context_budget` (64–8192), `expand`, `compress`; `details.context` carries the blocks, tokens, skipped chunks, compression info and the rendered text (`[i] file > section\ntext`, the shape Phase 10's prompt will cite). The CLI has `--budget/--expand/--compress/--show-context`; the dashboard's inspector shows tokens used vs budget, each block's source and cost, and the exact text.
- **Bounded:** `recursion_limit = 4·(retries+1) + 2·steps + 6` (+1 for assemble).

## Measured (2026-11-06; dense+bm25, 20 candidates retrieved once per question, only the assembly varies; [seed](../benchmarks/context-2026-11-06.json), [relational](../benchmarks/context-relational-2026-11-06.json))
**Context recall** = share of a question's facts whose evidence quote is contained in the assembled text (the ADR-0009 containment rule). Tokens are bge-m3, headers included.

| Config | seed recall | seed tokens p50 / max | rel. recall | rel. tokens p50 / max | assembly p50 (seed / rel.) | LLM calls/q (seed / rel.) | vs none/none at the same budget (seed · rel.) |
|---|---|---|---|---|---|---|---|
| top5 (Phases 6-9) | 0.986 | 508 / 891 | 0.875 | 536 / 797 | 0 / 0 ms | 0.0 / 0.0 | – · – |
| all 20 candidates | 1.000 | 2079 / 2745 | 1.000 | 2114 / 2420 | 0 / 0 ms | 0.0 / 0.0 | – · – |
| none/none@256 | 0.932 | 248 / 260 | 0.562 | 248 / 254 | 3 / 13 ms | 0.0 / 0.0 | – · – |
| none/none@512 | 1.000 | 500 / 517 | 0.750 | 510 / 518 | 3 / 11 ms | 0.0 / 0.0 | – · – |
| **none/none@1024 (default)** | 1.000 | 1019 / 1043 | 0.938 | 1023 / 1034 | 3 / 10 ms | 0.0 / 0.0 | – · – |
| neighbors/none@1024 | 0.986 | 984 / 1027 | 0.688 | 989 / 1025 | 19 / 45 ms | 0.0 / 0.0 | −q22 · −r04 −r07 −r08 |
| section/none@1024 | 1.000 | 1012 / 1030 | 0.875 | 993 / 1027 | 63 / 76 ms | 0.0 / 0.0 | no change · +r02 −r08 |
| none/lexical@1024 | 0.811 | 658 / 1005 | 0.750 | 948 / 1003 | 6 / 39 ms | 0.0 / 0.0 | −q15 −q22 −q23 −q24 −q25 −q26 −q32 −q34 · −r04 −r06 |
| none/embedding@1024 | 0.932 | 1002 / 1011 | 0.875 | 1001 / 1004 | 13 / 22 ms | 0.0 / 0.0 | −q07 −q15 −q26 · +r02 −r08 |
| none/llm@256 | 0.959 | 185 / 251 | 0.812 | 240 / 243 | 3722 / 10964 ms | 1.0 / 1.0 | +q22 +q25 +q35 −q07 −q29 · +r04 +r07 +r08 |
| none/llm@512 | 0.959 | 185 / 499 | 0.875 | 484 / 492 | 12 / 53 ms | 0.0 / 0.0 | −q07 −q29 · +r02 +r08 −r03 |
| none/llm@1024 | 0.959 | 185 / 996 | 0.875 | 974 / 990 | 10 / 47 ms | 0.0 / 0.0 | −q07 −q29 · +r02 −r03 −r05 |
| section/llm@256 | 0.973 | 138 / 251 | 0.562 | 229 / 246 | 5150 / 7497 ms | 1.05 / 1.0 | +q22 +q25 +q35 −q20 · +r07 +r08 −r02 −r05 |

- The LLM selection is made once per question and cached (same question, same units), so its 512/1024 rows reuse the 256 row's call: **its real cost is the 256 row's latency** (p50 3.7 s seed, 11.0 s relational; the prompt holds all units of the 20 candidates, ~2,100 tokens).
- Wins and losses are per question against `none/none` at the same budget; reference rows compare against nothing.

What the numbers say:
- **A budget beats a count.** At 1,024 tokens every seed fact and 15 of 16 relational facts reach the context, against 0.986 / 0.875 for the top 5. Versus the top 5 it gains q22 (rank 6) and r05 (rank 7) and loses nothing on either set; the price is ~2× the tokens (p50 1,019 vs 508), well inside the generator's 8,192-token window.
  - 512 tokens costs the same as the top 5 and still gains q22, but loses r08 (its blocks don't all fit).
- **Expansion doesn't pay here.** At a fixed budget, neighbours crowd out other retrieved chunks (−q22 and −r04 −r07 −r08); a section adds r02 but loses r08. As predicted from the corpus (no quote crosses a chunk boundary, ≤ 3 chunks per section), there is nothing for expansion to repair.
- **Compression only helps when the budget is small, and only with the LLM.**
  - Lexical and embedding scoring drop whole facts at every budget. Example (q26): "Prepared by Lena Brandt (Safety Officer)" shares one word with the question, while a dozen units mention E-5.
  - The LLM selection keeps 0.959 / 0.812 of the facts in **256** tokens, where the uncompressed context keeps 0.932 / 0.562 (+r04 +r07 +r08, no relational loss), for 1 call (4–11 s). At 1,024 tokens it is worse than no compression (0.959 vs 1.000; −q07 −q29).
  - Some "losses" are the metric, not the context: q15's 300-cycle value survives in the slide's chart lines (`Category: 300 | AUR-C105: 95.9 | AUR-C106: 96.2`), but the evidence quote is the table row `| 300 | 95.9 | 96.2 |`. Under ADR-0009 quotes aren't loosened to make a method look better.
- **A bug found by measuring.** The first run split units line by line, so hard-wrapped prose ("A fallback is E-5 + FEC, which is at 83.1% after 800" / "cycles on AUR-C104.") became two units and every compression method cut evidence quotes in half; list numbers ("1.") became units of their own. After the fix (wrapped lines rejoined, list numbers kept, `test_wrapped_sentences_and_list_numbers_stay_whole`), relational LLM@256 went 0.500 → 0.812 and embedding@1024 0.812 → 0.875; uncompressed rows were unchanged.

**Calibration** (bge-m3 vs the generator's own count, Ollama `prompt_eval_count` on raw text, all 58 chunks): qwen3 counts 5,151 tokens where bge-m3 counts 4,938, a ratio of **1.043** overall (p50 1.055, min 0.702, max 1.571). A 1,024-token budget is ~1,070 qwen3 tokens; one table-heavy chunk can be 1.57× its bge-m3 count.

## Consequences
- **Defaults** (`Strategy` and `Settings`): budget 1,024 bge-m3 tokens, expand `none`, compress `none`. Plain `/retrieval/search` is unchanged.
- **The generator reads `details.context`**, not the top-k hits: Phase 10 builds its prompt from the rendered blocks and cites them by number.
- **LLM compression is the option for small budgets** (a smaller-context model, or many questions batched into one prompt): at 256 tokens it keeps the most evidence of anything measured, for one extra call per question.
- **Expansion stays available** for corpora with long sections and evidence that crosses chunk boundaries; re-measure there before turning it on.
- **Token counts are approximate for the generator** (+4% on average, up to +57% on one chunk): Phase 10 leaves headroom in `num_ctx` instead of filling it to the limit.
- **Open:** a dataset v2 with alternative quotes (q15-style chart lines) would make context recall fairer to compression; evidence-aware expansion (only when a quote is cut at a boundary) for other corpora.
