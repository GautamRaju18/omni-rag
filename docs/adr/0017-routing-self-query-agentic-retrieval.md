# ADR-0017: Rule-based query routing and a grounded search agent; self-query stays opt-in

**Status:** accepted (2026-11-06, Phase 9b)

## Context
Phase 9a (ADR-0016) built the bounded LangGraph workflow. Its corrective step wrote new queries from the grader's one-line "missing" note, without seeing the passages, so it could only reword the question. On the relational set it never recovered a fact. Two relational questions fail at base because the second fact can only be found through a name that the *first* passage gives:
- **r02:** the salt in E-5 is NaPF₆; the supplier is a budget row.
- **r05:** the anode material is hard carbon; the second source is a risk-register row.

The brief (section 10) also asks for:
- query routing;
- metadata filtering and self-query retrieval;
- table- and image-aware QA;
- multi-document QA;
- agentic retrieval.

Context expansion and compression move to **Phase 9c**: they are judged by evidence kept per token sent to the model, which belongs with generation (Phase 10).

## Decision
**Workflow:**
```
START → route → transform → retrieve ⇄ plan → [rerank] → check → END | correct → retrieve
```
- **route** (`retrieval/router.py`): classifies the question as lookup, table, image or multi-hop.
  - Table and image intents add one *preference list*: the original question searched with an element-type filter (table + record, or figure). The list joins RRF, so it can move chunks up but never removes one.
  - Two routers share one interface: `RuleRouter` (regular expressions, free) and `LLMRouter` (one structured call).
- **self-query** (`retrieval/self_query.py`): the LLM proposes documents the question names, picking from an enum built from the real catalog. Code accepts a proposal only when the question contains the document's distinctive title words (or its format word: page, slides, spreadsheet). It runs in one of two modes:
  - `soft`: a preference list;
  - `hard`: a filter.
- **plan** (`retrieval/agent.py`): the agent reads the current top-k and chooses `search` (a new dense + BM25 query), `graph` (the Phase 7 retriever on one entity name) or `finish`. Guards:
  - every identifier in an agent query must occur in the question or in the passages shown (after `norm`, so NaPF₆ = NaPF6);
  - graph entities must be named in the question or the passages;
  - repeated queries are refused.

  Refusals end the agent's turn and stay in the trace.
- **BM25 requires only the user's identifiers.** LLM-written queries pass `required=identifiers(question)`, so an identifier the agent copied from a passage is a boost, not a filter. The first run showed why: the agent added "Aurora S1", and `require` mode then dropped every chunk without "S1".
- **Per-document cap** (`fusion.diversify`): at most N chunks per document in the top k. Chunks over the cap fill in when fewer documents were found.
- **No corrective retry after the agent ran.** The first run of the default combination lost r02 again:
  1. the agent found the supplier row;
  2. the grader said "insufficient";
  3. the 9a corrective step added "…NaPF₆ salt supplier unknown";
  4. its two new lists pushed the row out of the top 5.

  The agent is the evidence-aware version of that step, so the workflow now ends at the check once the agent has run. Corrective retries remain for questions the agent didn't handle.
- **Bounded:** `agent_steps` (0–3) and `max_retries` (0–3) are both counted in the state. `recursion_limit = 4·(retries+1) + 2·steps + 5`, and a unit test runs the worst case (agent always searches, grader never satisfied, every maximum).
- **Dashboard:** a Streamlit app (Phase 14's retrieval laboratory, pulled forward on request). It talks to the API only. `POST /retrieval/advanced` now returns every ranked list (top 10, with file names), and `GET /graph/stats` and `GET /graph/entities` were added.

## Measured (2026-11-06, qwen3:4b-instruct, K=5; [seed](../benchmarks/workflow-9b-2026-11-06.json), [relational](../benchmarks/workflow-9b-relational-2026-11-06.json), [routing](../benchmarks/routing-2026-11-06.json))

**Router accuracy** (question intents; "table" in the seed set means *the evidence is a table*, which the question often doesn't show):

| | seed (40) | relational (8) | held-out (24) | p50 |
|---|---|---|---|---|
| rules | **30 / 40** | **7 / 8** | **23 / 24** | 0.01 ms |
| LLM | 23 / 40 | 2 / 8 | 20 / 24 | 0.8–1.2 s |

- The held-out set was labelled before the rules were written. It's the same author, so it's a sanity check, not an unbiased test.
- The rules' seed misses are mostly "table" questions that read like lookups ("When is milestone M3 due?").
- The LLM over-predicts `table` and misses 6 of the 8 relational questions.

**Retrieval** (each option alone on top of base = dense + BM25, no grader, no retries):

| Config | seed R@5 | seed MRR | seed nDCG | rel. R@5 | rel. MRR | rel. nDCG | seed p50 / p95 | LLM calls/q (seed) | per-question vs base |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.986 | 0.928 | 0.889 | 0.875 | 0.838 | 0.754 | 39 / 75 ms | 0 | – |
| route (rules) | 0.986 | **0.941** | 0.883 | 0.875 | 0.838 | 0.754 | 36 / 59 ms | 0 | no change |
| route (LLM) | 0.932 | 0.858 | 0.806 | 0.812 | 0.838 | 0.735 | 0.8 / 1.1 s | 1.0 | −q24 −q35 −r03 |
| self-query soft | 0.986 | 0.928 | 0.878 | 0.812 | 0.838 | 0.716 | 0.7 / 6.6 s | 1.07 | −r07 |
| self-query hard | 0.986 | 0.928 | 0.878 | 0.750 | 0.838 | 0.667 | 0.8 / 7.9 s | 1.07 | −r04 −r07 |
| max 2 per document | 1.000 | 0.928 | 0.891 | 0.875 | 0.838 | 0.754 | 40 / 50 ms | 0 | +q22 (ground-truth gap) |
| agent auto (rules route) | 0.986 | 0.928 | 0.869 | **0.938** | **0.917** | **0.831** | 47 ms / 4.0 s | 0.2 | **+r02**, no loss |
| agent always | 0.959 | 0.901 | 0.867 | 0.938 | 0.917 | 0.831 | 2.6 / 5.8 s | 1.35 | +r02, −q29 |
| **default: route + agent auto + check + ≤1 retry** | 0.986 | 0.928 | 0.869 | **0.938** | **0.917** | **0.831** | 1.3 / 4.5 s | 1.38 | **+r02**, no loss |

What the numbers say:
- **The agent recovers r02**: it read "NaPF₆" in the first passage, searched "NaPF₆ salt supplier", and the budget row entered the top 5.
  - It doesn't recover r05: the Carbonix line sits in a risk-register row that no query reached.
  - Before the BM25 fix it also *lost* r06. Its query added "E-7", which BM25 then required, and its lists outranked an original fact. After the fix, r06 keeps both facts.
- **On every seed question the agent stays out of the way.** The router sends 5 of 40 to it, and recall and MRR are unchanged.
  - The nDCG cost (0.889 → 0.869) is lower-rank reordering from the extra lists.
  - Running the agent on every question (`always`) lost q29. After a graph lookup of "E-7" and a broader search, the swelling paragraph left the top 5.
- **Self-query picks are precise but don't help.** All 10 documents it accepted on the seed set held evidence, and grounding rejected 19 "might contain the answer" proposals. But:
  - on r04 and r07 it chose the budget spreadsheet while the other fact sits elsewhere (the status report for r04, the project config for r07), so a hard filter loses that evidence;
  - 3 of 48 calls (q26, q27, q28, all about the safety review) came back empty twice (degraded to no filter). Each failure costs two ~3.5 s attempts, which is where the 6.6–7.9 s p95 comes from; successful calls take p50 510 ms, max 808 ms.
- **The LLM router is worse than the rules** on every set, and costs a second per question.
- **The per-document cap** only "wins" q22, the known ground-truth gap (ADR-0015). It is kept as an option.

## Consequences
- **Defaults for `/retrieval/advanced`** (`Strategy` and `Settings`): `route=rules`, `agent=auto` (2 steps), grader on, ≤ 1 corrective retry (not after the agent), self-query off, cap off. Plain `/retrieval/search` is unchanged (`dense+bm25`).
  - Measured as a whole: relational recall 0.875 → 0.938 and MRR 0.838 → 0.917, +r02 with no loss.
  - Seed: no recall or MRR change; nDCG 0.889 → 0.869. The check flags all 3 unanswerable questions (q38 and q40 insufficient, q39 partial).
  - Cost: seed p50 1.3 s (the grader), relational p50 4.7 s (agent + grader).
- **Agent queries may only use names that a passage or the question contains.** This is the Phase 5b rule (the LLM proposes, code decides) applied to search: the agent follows evidence, it can't invent leads.
- **Identifiers the user typed are hard constraints; identifiers a model wrote are hints.** This holds for every LLM-written query: agent, corrective, multi-query.
- **Preferences are soft by default.** A hard self-query filter can only be as good as the document pick, and the pick is right about the named document but not about where the *other* fact lives.
- **Open:**
  - r05 (evidence in a table row that no query reached);
  - the rule router can't see that a question's answer lives in a table;
  - routing accuracy on real user questions is unknown until Phase 12's larger set.
