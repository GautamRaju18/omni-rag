# Phase 13: Failure analysis and observability

> Decision record: [ADR-0020](../adr/0020-evaluation-framework.md) (taxonomy, measured root causes). Code: `evaluation/failures.py`, `observability/analytics.py`, `generation/history.py` (traces), `api/routers/analytics.py`.

## 1. Problem being solved
A wrong answer can be lost at any of a dozen stages:
- a scan OCR'd badly;
- a fact split across chunks;
- a chunk missing from the graph;
- a retriever ranking it 30th;
- a context budget that ran out;
- a model that refused or invented.

Averages hide which. Debugging needs the **first stage that lost the evidence**, for every failed question, and a record of every live query detailed enough to replay what happened.

## 2. The taxonomy (`failures.py`, checked in pipeline order)
| Category | Test (per fact of the question) |
|---|---|
| parsing / OCR failure | The evidence quote is in no element of its file (OCR for images and scans). |
| chunking failure | The quote is in an element but in no single chunk. |
| graph construction failure | No covering chunk exists in Neo4j. |
| embedding failure | Covering chunks have no vector (dense modes). |
| retrieval failure | No covering chunk among the ranked candidates. |
| reranking failure | The first stage had it in its top 5; after reranking it's out of the context. |
| context construction failure | A candidate covers it; the assembled context doesn't. |
| generation failure | Every fact was in the context; the answer misses terms or refuses. |
| hallucination | An unsupported claim, or any answer to an unanswerable question. |
| citation failure | The answer is right, but a claim cited the wrong source (relocated). |

The root cause is the earliest stage; later stages aren't blamed again (they never had the evidence). Hallucination and citation failures are added on top.

## 3. The query lifecycle trace
Every `/chat` turn stores one JSON record in SQLite (`ChatStore`):
- the question;
- the strategy;
- rewritten queries;
- the route and agent steps;
- every ranked list (top 10, with scores and files);
- rerank scores per hit;
- the context blocks (chunk IDs, tokens);
- the answer with each claim's status and issues;
- the citations;
- latency per component: route, transform, retrieve, plan, rerank, check, correct, assemble, embedding, generation, total.

`GET /chat/turns/{id}` returns it; `GET /analytics` aggregates the stored turns (`turn_analytics`):
- answered, grounded and refused turns, and why;
- claim statuses and citation accuracy;
- live failure flags;
- verdicts and routes;
- latency p50/p95 per component;

It also includes the newest evaluation report's summary.

## 4. Measured root causes (2026-11-06; questions per first stage that lost them)
| | E1 Dense | E2 BM25 | E3 Dense+BM25 | E4 +Graph | E5 +Rerank (76 q) |
|---|---|---|---|---|---|
| retrieval | 1 | **15** | 1 | 3 | 0 |
| context construction | **10** | 1 | 3 | 3 | 0 |
| generation | 6 | 5 | 6 | 7 | 5 |
| hallucination | 2 | 1 | 5 | 2 | 1 |
| citation | 4 | 1 | 4 | 3 | 1 |

Reading it:
- **BM25 alone fails at retrieval.** 14 of its 15 retrieval failures are two-fact questions (multi-hop, cross-document, comparative): at least one of the two facts never reaches the candidates.
- **Dense alone fails at context construction.** The evidence is among its 20 candidates, but ranked too low to fit into 1,024 tokens.
- **With dense+BM25, most failures are the generator's.** The evidence was in the context: refusals (e08, e10), half answers (q17), wrong numbers (e48), a misattributed row (q33).
- **No parsing, OCR, chunking, graph or embedding failures on this corpus.**

## 5. How to run
```bash
curl -s localhost:8000/api/v1/analytics | jq '.chat | {turns, answered, grounded, refused_by, failure_flags, latency_ms}'
curl -s localhost:8000/api/v1/chat/turns/<turn_id> | jq '.retrieval.lists, .timings'
uv run python -m omnirag.evaluation.rag_eval --experiments E3 --limit 20 --out /tmp/e3.json && jq '.experiments.E3.failures' /tmp/e3.json
```
Dashboard: **Performance** (live traffic, latency per component, flags, recent turns, the evaluation's latency, memory, indexing). **Evaluation → Where questions fail** (root causes) and **Per question**.

## 6. Common errors
| Symptom | Cause / fix |
|---|---|
| `/analytics` shows 0 turns | Turns are stored per data directory: the containers and `uv run` share `~/omnirag-data` only if the bind mount points there. |
| A failure is blamed on generation, but the answer looks right | Its `must_include` term is missing (check the per-question table). If the term is wrong, record an erratum in the dataset and re-score with `--from-partial`. Never edit a single report. |
| Latency p95 much higher than p50 | Long answers (comparisons) and GPU throttling. Check `gpu` in the report. |

## 7. Exercise (≈45 min)
1. Find E1's ten context-construction failures in the report. Rerun one with `--budget 2048` in the CLI (`retrieval advanced`): does the fact enter the context?
2. Ask three questions on the Ask page, including one about AUR-C109, then open Performance. Which component dominates latency?
3. **Think.** Why must a root-cause classifier stop at the first failing stage? What would double counting do to the table above?

## 8. Acceptance criteria: Phase 13
- [ ] `test_failures_follow_the_pipeline_and_blame_the_first_stage` passes (every category, including OCR and reranking).
- [ ] After a few chat turns, `GET /analytics` shows them with per-component latency; `GET /chat/turns/{id}` returns the full trace.
- [ ] The Evaluation page's root-cause table matches the report's `failures.root_causes`.
