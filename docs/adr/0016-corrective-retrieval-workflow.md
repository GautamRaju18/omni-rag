# ADR-0016: A bounded corrective retrieval workflow in LangGraph; the context check is the useful part

**Status:** accepted (2026-11-06, Phase 9a)

## Context
The brief (section 10) asks for advanced RAG built incrementally:
- query rewriting, multi-query, HyDE;
- parent-child, contextual, self-query and graph-enhanced retrieval;
- metadata filtering, context compression and query routing;
- multi-document, table- and image-aware QA;
- iterative and agentic retrieval.

It also asks for LangGraph orchestration with conditional branches, including a "Context Quality Check", and says: "If retrieval confidence is insufficient, allow controlled query rewriting and a limited number of retrieval retries. Avoid infinite agent loops."

Phase 8 showed that the reranker's score can't be the confidence signal: the nonexistent cell AUR-C109 scores 0.936.

Phase 9 is split:
- **9a (this ADR):** transforms, context check, bounded corrective loop.
- **9b:** routing, self-query, context expansion, compression, multi-document diversity, agentic retrieval.

## Decision
**LangGraph** `langgraph==1.2.19` (MIT; not the release published the same day). LangSmith comes in as a dependency but stays off: no tracing environment variables are set until Phase 12.

**Workflow** (`workflows/retrieval_graph.py`):
```
START → transform → retrieve → [rerank] → check → (sufficient | out of retries) → END
                        ↑                         ↘ correct ┘
```
- **transform:** `none | rewrite | multi_query | hyde | multi_query+hyde`. The original question always stays among the queries.
- **retrieve:** dense and BM25 for every query (dense for the HyDE passage), run in parallel. All lists from all attempts are fused with RRF, so a retry only adds evidence.
- **check:**
  1. **Identifier coverage**, free: any identifier in the question that appears in no passage makes the context `insufficient`, and stops the loop.
  2. **LLM grader** (structured output): `sufficient | partial | insufficient`, the supporting passages, and what is missing.
- **correct:** one new query aimed at what's missing.
- **Bounded loop:** `max_retries` is 0–3, validated by the API as well, and counted in the state. LangGraph's `recursion_limit` is derived from it (4 × (retries + 1) + 4), so an unplanned loop raises instead of spinning.
- **Degradation:** a failed transform or corrective rewrite falls back to the original question. It is recorded in the per-node trace and never fails the request.

**Guards on LLM-written queries:**
- A rewrite must keep every identifier in the question and add none; otherwise the original question is used.
- Prompts carry **no example identifiers.** The model copied corpus examples ("AUR-C103, E-7, M2") into unrelated queries, and then copied neutral ones ("XB-12, QZ-4") into HyDE passages.

**Interfaces:** `POST /api/v1/retrieval/advanced` (query, k, filters, optional strategy) and `python -m omnirag.retrieval advanced`. Each result carries the verdict, the queries, every check and a per-node trace.

**Defaults:** transform `none`, grader on, `max_retries` 1, rerank off.

## Measured (2026-11-05/06, qwen3:4b-instruct, K=5; [workflow-2026-11-05.json](../benchmarks/workflow-2026-11-05.json), [workflow-relational-2026-11-05.json](../benchmarks/workflow-relational-2026-11-05.json))
| Config | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | rel. nDCG | seed p50 | LLM calls/q |
|---|---|---|---|---|---|---|---|---|
| base (no transform; = dense+bm25) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 19 ms | 0 |
| rewrite | 0.986 | 0.914 | 0.878 | 0.875 | 0.838 | 0.754 | 0.5 s | 1.0 |
| multi-query (n=3) | 0.986 | 0.892 | 0.881 | 0.875 | 0.775 | 0.732 | 2.3 s | 1.0 |
| HyDE | 1.000 | 0.939 | 0.899 | 0.875 | 0.713 | 0.685 | 3.6 s | 1.0 |
| multi-query + HyDE | 1.000 | 0.932 | 0.907 | 0.875 | 0.775 | 0.732 | 6.5 s | 2.0 |
| corrective (check + ≤1 retry) | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 1.6 s | 1.23 |
| multi-query + corrective | 0.986 | 0.892 | 0.881 | 0.875 | 0.775 | 0.732 | 4.0 s | 2.2 |

What the numbers say:
- **No transform improves both sets.**
  - HyDE's seed recall gain is q22, the ground-truth gap described in ADR-0015. On the relational set HyDE loses 0.125 MRR.
  - Multi-query lowers MRR on both sets.
- **Corrective retries recovered no fact on either set.** They ran on 5 of 40 seed questions and 4 of 8 relational ones; recall and MRR are identical to base.

**Context check, against ground truth** (sufficient = the base top-5 covers every fact; unanswerable = insufficient):

| | seed (40) | relational (8) |
|---|---|---|
| agrees on "sufficient vs not" | **37 / 40 (0.925)** | 6 / 8 (0.75) |
| unanswerable judged not-sufficient | **3 / 3** (q38, q40 insufficient; q39 partial) | – |
| false alarms (covered, judged partial) | q11, q20 | r06, r07 |
| other | q22 judged sufficient (the ground-truth gap) | 2 partially covered judged insufficient |

Check latency: p50 1.6 s (seed, LLM grader); the identifier check takes about 0 ms.

## Consequences
- **The context check is the confidence signal Phase 10 will use.** "insufficient" or "partial" tells generation to say what the documents don't contain. Unlike the reranker's score, it caught all three unanswerable questions.
- **Its known failure mode is false alarms** (4 of the 42 fully covered questions across both sets). The cost is an unnecessary retry, or an over-cautious answer, not an invented one.
- **Transforms stay available but off by default.** Our eval questions are well-formed and name their entities. Rewriting is meant for vague or conversational queries, which these sets don't contain; Phase 12's larger set should include some.
- **The retry stays on (max 1)** for robustness to poorly phrased queries, at a measured cost of 0.23 extra LLM calls per question. On these sets it never helped.
- **HyDE writes plausible-sounding false facts** ("thermal runaway on 2023-10-15"). It is used only as a dense query, never as evidence or output; that rule must hold in every later phase.
- **The corrective rewrite can add constraints the question didn't contain** ("as of June 2024"): the identifier guard doesn't cover dates. This is acceptable for search queries and noted for 9b.
