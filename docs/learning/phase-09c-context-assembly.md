# Phase 9c: Context assembly: a token budget, expansion and extractive compression

> Decision record: [ADR-0018](../adr/0018-context-assembly.md). Results: [seed](../benchmarks/context-2026-11-06.json), [relational](../benchmarks/context-relational-2026-11-06.json).

## 1. Problem being solved
Retrieval ends with a ranked list. Generation (Phase 10) needs **text in a prompt**, and a prompt is measured in tokens, not chunks. On this corpus "the top 5" is anything from ~150 to ~900 tokens, and some evidence sits just below the cut:
- q22's second fact is at rank 6 (the top 6 cost 449 tokens);
- r05's second fact is at rank 7 (509 tokens).

So the question of this phase is: **what is the best context for a fixed number of tokens?** Three levers: how much to take (budget), whether to add surrounding text (expansion), and whether to remove irrelevant text (compression).

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Token budget** | The maximum size of the context, in bge-m3 tokens, *including* the `file > section` line shown for each block. |
| **Block** | One contiguous piece of one document as the generator sees it: one chunk, or several merged ones. Carries document, section, pages, chunk IDs and element IDs for citations. |
| **Expansion** (parent-child at query time) | Add the chunks around a hit: its neighbours in reading order, or its whole section. |
| **Extractive compression** | Keep only some sentences and table rows, verbatim. The opposite, *abstractive* compression (a summary), can't be cited and lets a 4B model rewrite numbers. |
| **Unit** | What compression keeps or drops: a sentence (no break after "Dr." or "e.g."), a list item, a table row (with its header). |
| **Context recall** | Share of a question's facts whose evidence quote is contained in the assembled text (the ADR-0009 rule, applied to the context instead of the top k). |

## 3. Architecture
```
retrieval/context.py      ContextOptions · ContextBlock · AssembledContext · ContextAssembler
                          expansion_rows (parameterized Cypher) · build_blocks (merge) · fill (budget)
retrieval/compression.py  split_units · lexical_scores · Compressor (embedding cache, LLM selection cache)
workflows/retrieval_graph.py   + assemble node (last), Strategy.context_budget / expand / compress
evaluation/context_benchmark.py  retrieve once, vary assembly; --calibrate (qwen3 vs bge-m3 tokens)
frontend: Workflow inspector → "4 · Context for the generator" + the rendered context
```
```
… → check → assemble → END
assemble: ranked hits → [expand] → merge into blocks → [compress] → fill the budget → render "[i] file > section\ntext"
```

## 4. Walkthrough: "Why did cell AUR-C103 fail?" (defaults: 1,024 tokens, no expansion, no compression)
1. The workflow ranks 20 candidates (dense + BM25, RRF); the check says `sufficient`.
2. **assemble** takes the diversified top 5 first (the order the check judged), then the other 15 in rank order.
3. Each candidate becomes a block with its header (`aurora_safety_review.pdf > … > 1. Incident Summary`). Blocks are added while they fit: 8 blocks, **981 / 1,024 tokens**; the other 12 chunks didn't fit or weren't reached and are listed in `skipped`.
4. The rendered context starts with `[1] aurora_research_notes.md > Project Aurora: Research Notes > Results (as of 2026-03-10)`. Phase 10 asks the model to cite these numbers.

The same question with `--expand section --compress llm --budget 256`: the model picks the units it needs, code keeps them verbatim, and three blocks fit in 229 tokens:
```
[2] aurora_safety_review.pdf > Aurora E-5 High-Temperature Failure: Safety Review > 1. Incident Summary
Cell AUR-C103 (electrolyte E-5) failed at cycle 412 during 45 °C cycling. The pouch swelled by 14% and vented without flame.
```

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.retrieval advanced "Why did cell AUR-C103 fail?" --show-context
uv run python -m omnirag.retrieval advanced "Why did cell AUR-C103 fail?" --expand section --compress llm --budget 256 --show-context
uv run python -m omnirag.evaluation.context_benchmark --calibrate               # ~55 min (LLM rows)
uv run python -m omnirag.evaluation.context_benchmark --dataset data/eval/relational_seed.json
```
Dashboard: **Workflow inspector** → column "4 · Context for the generator" (budget, expand, compress) → the "Context for the generator" section shows tokens used vs budget, one row per block, and the exact text.

## 6. Expected outputs (measured 2026-11-06, bge-m3 tokens; full table in ADR-0018)
| | seed recall | seed tokens p50 | relational recall | relational tokens p50 |
|---|---|---|---|---|
| top 5 (Phases 6–9) | 0.986 | 508 | 0.875 | 536 |
| 512, none/none | 1.000 | 500 | 0.750 | 511 |
| **1,024, none/none (default)** | **1.000** | 1,019 | **0.938** | 1,023 |
| 1,024, section | 1.000 | 1,012 | 0.875 | 993 |
| 1,024, neighbours | 0.986 | 984 | 0.688 | 989 |
| 1,024, lexical compression | 0.811 | 658 | 0.750 | 948 |
| 1,024, embedding compression | 0.932 | 1,002 | 0.875 | 1,001 |
| 256, none/none | 0.932 | 248 | 0.562 | 248 |
| **256, LLM compression** | 0.959 | 185 | **0.812** | 240 |

- The default gains q22 and r05 over the top 5 and loses nothing.
- LLM compression is the best way to fit evidence into **256** tokens (+r04 +r07 +r08 relational, no loss), at one call per question (p50 3.7 s seed, 11.0 s relational).
- Calibration: the generator (qwen3) counts **1.043×** the bge-m3 tokens overall (p50 1.055, range 0.70–1.57).

## 7. Surprises found during verification
- **Expansion made things worse.** On a corpus where no evidence quote crosses a chunk boundary, the neighbours of a hit are mostly *other* facts' noise; at a fixed budget they push real hits out.
- **A wrapped line is not a sentence.** The first benchmark run showed every compression method losing evidence in `meeting_notes_2026-03-12.txt`. The text is hard-wrapped at ~90 characters, and the splitter treated each line as a unit: "Early E-7 cells" and "(AUR-C105 and AUR-C106) are above 89% retention…" were scored separately and one was dropped. Rule now: a long line that doesn't end a sentence continues on the next line unless that line starts a list item, a heading, a table row or a capitalised word.
- **"1." was a sentence** ("1. High-temperature failure of AUR-C103" split after the number). A one-word tail that is a number is no longer a boundary.
- **Some compression losses are the metric's.** q15's value survives as `Category: 300 | … AUR-C106: 96.2`, but the quote is the chart table's row. The quote stays (ADR-0009): a looser metric would make every method look better, including wrong ones.
- **The first CLI run is slow** (assemble 874 ms): a fresh process loads the bge-m3 tokenizer on first use; in the benchmark it's ~3 ms.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `ValueError: context expansion needs a graph store` | `ContextAssembler` built without `store` but `expand != none` (only in custom code; `build_workflow` passes it). |
| `ValueError: context compression needs a compressor` | Same for `compress != none`. |
| `context.compression.fallback = "lexical (...)"` | The LLM selection failed (timeout, unusable JSON twice): lexical scores were used instead. |
| Context has fewer tokens than the budget | Normal: the next block didn't fit and later ones were larger. `skipped` lists them. |
| 422 on `/retrieval/advanced` | `strategy.context_budget` outside 64–8192. |
| A compressed block shows ` … ` | Units between the kept ones were dropped. Text on both sides is verbatim. |

## 9. Exercise (≈45 min)
1. **Find the knee.** Run the benchmark with `--budgets 128,256,384,512,768,1024 --compress none` and plot recall vs tokens for both sets. Where does each set stop gaining?
2. **Break the splitter.** Write three lines of real-looking text that `split_units` cuts wrongly (hint: a wrapped line starting with a name). Add them to `test_wrapped_sentences_and_list_numbers_stay_whole` as a known limitation or fix the rule, then re-run the benchmark.
3. **Think.** Why is a *skipped* block better than a *truncated* one for citations? What would a truncated table row do to Phase 11's quote check?

## 10. Acceptance criteria — Phase 9c
- [ ] `advanced "Why did cell AUR-C103 fail?" --show-context` prints `context for generation: 8 blocks, ~981/1024 tokens` and `[1] aurora_research_notes.md > …`.
- [ ] The benchmark reproduces §6 within a question or two on the LLM rows (model nondeterminism across driver versions); the non-LLM rows exactly.
- [ ] The inspector's Default preset shows a "Context for the generator" section with tokens ≤ budget.
- [ ] `uv run pytest` and `uv run pytest -m integration` pass.
