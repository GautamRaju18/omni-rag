# Phase 9a: Query transforms, a context check and a bounded corrective loop (LangGraph)

> Decision record: [ADR-0016](../adr/0016-corrective-retrieval-workflow.md). Results: [workflow-2026-11-05.json](../benchmarks/workflow-2026-11-05.json) (seed set), [workflow-relational-2026-11-05.json](../benchmarks/workflow-relational-2026-11-05.json) (relational set). Phase 9b adds routing, self-query, context expansion, compression and agentic retrieval.

## 1. Problem being solved
Retrieval so far is "one query in, top-k out". Real questions can be:
- **badly phrased:** "that thing Marco owns";
- **multi-part:** "which electrolyte replaced it, and who owns it?";
- **about things that aren't there:** cell AUR-C109.

Advanced RAG reformulates queries, and before answering asks whether the retrieved context is actually enough. If it isn't, it tries again, a bounded number of times. Phase 8 showed that the reranker's relevance score can't make that decision.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Query rewriting** | An LLM turns the question into a cleaner search query. |
| **Multi-query** | The LLM writes several queries (different wording, or one per sub-question). All are searched and the lists fused with RRF. |
| **HyDE** | "Hypothetical Document Embeddings": the LLM writes a fake passage that *would* answer, and dense search uses its embedding. The passage contains invented facts; it is a search key, never evidence. |
| **Context quality check** | "Can these passages answer the question?" Here: identifier coverage (free), then an LLM grader (sufficient / partial / insufficient + what is missing). |
| **Corrective retrieval** | When the check fails, write one new query aimed at what is missing, retrieve, merge and re-check (the idea behind CRAG). |
| **LangGraph** | A library for stateful graphs of steps (nodes) with conditional edges and loops. A `recursion_limit` caps the number of steps. |
| **Bounded loop** | A retry counter in the state plus a recursion limit derived from it: the loop *cannot* run forever. |

## 3. Architecture
```
llm/chat.py                 OllamaChat: one /api/chat call, JSON-schema structured output,
                            temperature 0 + seed, retries on 5xx/429/transport, 1 retry on bad JSON
retrieval/transforms.py     QueryTransformer: rewrite, multi_query, hyde, corrective, check
                            (identifier guard: keep every question identifier, invent none)
workflows/retrieval_graph.py  RetrievalWorkflow (LangGraph StateGraph), Strategy, build_workflow
api: POST /api/v1/retrieval/advanced     cli: python -m omnirag.retrieval advanced
evaluation/workflow_benchmark.py        7 configurations + grader accuracy, both eval sets
```
```
START → transform → retrieve → [rerank] → check ──sufficient / no retries left──→ END
                        ↑                       └─partial/insufficient→ correct ─┘
state: question, queries[], hyde, lists{} (every ranked list so far), hits, attempts, checks[], trace[]
```

## 4. Walkthrough: three questions, `--transform multi_query`
1. **"What capacity retention did cell AUR-C109 reach?"**
   - transform: 3 extra queries.
   - retrieve: 8 lists fused.
   - check: **identifier coverage**. AUR-C109 is in no passage, so the verdict is `insufficient`, with no LLM call.
   - Rewording can't make a cell exist, so the loop **stops without retrying**.
2. **"Who is the CEO of Helios Energy Lab?"**
   - check (LLM): `partial`.
   - correct: a new query, "Who is the CEO of Helios Energy Lab as of June 2024?". The model added a date (a known flaw, §7).
   - retrieve, then check: `partial` again.
   - `max_retries` = 1 is reached, so the workflow ends with verdict `partial`.
3. **"Which electrolyte replaced the one used in the cell that failed at 45 °C, and who owns its qualification?"** (q07; graph retrieval couldn't link it in Phase 7):
   - Multi-query splits it into "What electrolyte replaced…?" and "Who owns the qualification…?".
   - The top 3 are the safety review (E-5 failed), the E-7 technical note (replaces E-5) and the meeting notes (Marco owns the E-7 qualification).
   - The check says `sufficient`.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync          # adds langgraph 1.2.19
uv run python -m omnirag.retrieval advanced "What capacity retention did cell AUR-C109 reach?"
uv run python -m omnirag.retrieval advanced \
  "Which electrolyte replaced the one used in the cell that failed at 45 °C, and who owns its qualification?" \
  --transform multi_query
curl -s -X POST localhost:8000/api/v1/retrieval/advanced -H 'content-type: application/json' \
  -d '{"query": "Who is the CEO of Helios Energy Lab?", "strategy": {"max_retries": 1}}'
uv run python -m omnirag.evaluation.workflow_benchmark                    # ~15 min
uv run python -m omnirag.evaluation.workflow_benchmark --dataset data/eval/relational_seed.json
```

## 6. Expected outputs (measured 2026-11-05/06, qwen3:4b-instruct, K=5)
| Config | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | rel. nDCG | seed p50 | LLM calls/q |
|---|---|---|---|---|---|---|---|---|
| base (= dense+bm25) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 19 ms | 0 |
| rewrite | 0.986 | 0.914 | 0.878 | 0.875 | 0.838 | 0.754 | 0.5 s | 1.0 |
| multi-query | 0.986 | 0.892 | 0.881 | 0.875 | 0.775 | 0.732 | 2.3 s | 1.0 |
| HyDE | 1.000 | 0.939 | 0.899 | 0.875 | 0.713 | 0.685 | 3.6 s | 1.0 |
| multi-query + HyDE | 1.000 | 0.932 | 0.907 | 0.875 | 0.775 | 0.732 | 6.5 s | 2.0 |
| corrective (check + ≤1 retry) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 1.6 s | 1.23 |

**Per question:**
- HyDE (and multi-query + HyDE) wins only q22 on the seed set. That's the ground-truth gap from Phase 8: the config record moves into the top 5.
- No other configuration wins or loses a question's recall on either set. The MRR differences are rank shifts.

**Context check** (sufficient vs not, against ground truth):
- **Seed: 37 / 40.**
  - All 3 unanswerable questions are not-sufficient: q40 via identifiers, q38 `insufficient`, q39 `partial`.
  - False alarms: q11 and q20 (fully covered, judged `partial`).
  - q22 is judged `sufficient`, which the ground truth (one quote per fact) counts as wrong.
- **Relational: 6 / 8.** r06 and r07 are false alarms.

**Corrective loop:**
- Retried on 5 / 40 seed and 4 / 8 relational questions, with no change in recall or MRR.
- p95 latency 3.0 s (seed).

## 7. Surprises found during verification
- **Prompt examples leak.** The first prompts said "identifiers such as AUR-C103, E-7, M2". The model put "AUR-C103 E-7 M2" into the rewrite of a question that names none of them, and HyDE wrote "The electrolyte … is AUR-C103. Its qualification is owned by E-7." With neutral examples ("XB-12, QZ-4"), HyDE copied *those*. Prompts now contain no example identifiers, and a guard rejects any rewrite that drops a question identifier or invents one.
- **The grader's JSON can be cut off.** "Unterminated string" at `num_predict=200`. The fix: a larger budget, a 15-word cap on "missing", and a failed check is recorded as `partial / llm_error` instead of crashing.
- **The model sometimes returns an empty reply.** A rewrite came back empty twice in a row and crashed the first benchmark run. Every transform and corrective step now degrades to the original question, and the error goes into the trace.
- **Transforms don't help well-formed questions.** Every eval question already names its entities and uses the corpus's wording; rewriting can only paraphrase away from it. q07 (§4) is where multi-query *looks* helpful, but base already had all of q07's facts in its top 5 (recall 1.0 in every configuration), so measurably it changed nothing.
- **HyDE invents with confidence:** "Cell AUR-C103 failed due to a thermal runaway event detected on 2023-10-15…". Only its *embedding* is used; the text must never reach a user or a prompt as evidence.
- **The corrective rewrite adds constraints** ("…as of June 2024"). The identifier guard catches identifiers, not dates.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| HTTP 503 "Model server unavailable" from `/retrieval/advanced` | Ollama is down or the model is missing (the grader and transforms need `qwen3:4b-instruct`). Plain `/retrieval/search` still works. |
| `verdict: partial` with `method: llm_error` in `checks` | The grader's reply was unusable; the workflow continued. Look at the trace, and retry. |
| `trace[0].errors` not empty | A transform failed and the original question was used. Retrieval quality equals `base`. |
| HTTP 422 for `max_retries: 5` | Retries are capped at 3 (`Strategy`) by design. |
| `GraphRecursionError` | Should be impossible: the limit is derived from `max_retries`. If you add nodes to the loop, update the formula in `RetrievalWorkflow.retrieve`. |
| A rewritten query contains things the user never asked | Identifiers are guarded; dates and constraints aren't (§7). Inspect `details.queries`. |

## 9. Exercise (≈60 min)
1. **Write vague questions.** Rephrase 5 seed questions the way a hurried user would ("that failed cell, what went wrong?", "who's on the electrolyte?"). Run base, rewrite and multi-query on them. Do the transforms help *here*? Keep your questions; Phase 12 needs vague queries.
2. **Break the grader.** Find a question where the grader says `sufficient` but the answer isn't in the passages. Was the identifier check any help? What would a stricter grader prompt say?
3. **Make the loop pay off.** For a question that ends `partial`, read `checks[*].missing` and the corrective query. Was the new query reasonable? Why didn't it find more?
4. **Think.** The brief says "avoid infinite agent loops". Name the two independent mechanisms that guarantee termination here. What happens if someone adds a node to the loop without changing `recursion_limit`?

## 10. Acceptance criteria — Phase 9a
- [ ] `advanced "What capacity retention did cell AUR-C109 reach?"` shows `verdict=insufficient attempts=0` with `via identifiers`, and no LLM call for the check.
- [ ] `advanced … q07 … --transform multi_query` shows the safety review, the E-7 note and the meeting notes in the top 3.
- [ ] `workflow_benchmark` reproduces §6 (base = dense+bm25 exactly; grader 37/40 on the seed set).
- [ ] You can explain why transforms are off by default and the grader is on.
- [ ] `uv run pytest` passes, and `uv run pytest -m integration` passes with Ollama and Neo4j running.
