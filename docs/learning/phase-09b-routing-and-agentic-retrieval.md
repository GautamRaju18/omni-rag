# Phase 9b: Query routing, self-query, a grounded search agent, and a dashboard to watch it all

> Decision record: [ADR-0017](../adr/0017-routing-self-query-agentic-retrieval.md). Results: [seed](../benchmarks/workflow-9b-2026-11-06.json), [relational](../benchmarks/workflow-9b-relational-2026-11-06.json), [routing](../benchmarks/routing-2026-11-06.json). Context expansion and compression moved to Phase 9c.

## 1. Problem being solved
Phase 9a could reword a question but never *learn* from what it found.

Take "Who supplies the salt that the E-5 electrolyte is made with?" (r02). It can't be answered in one search:
1. Search: a passage says E-5 is "1.0 M NaPF₆ in EC:DEC".
2. Search: "NaPF₆ supplier" finds the budget row "NaPF6 salt | Nordic Salts AB".

The second query depends on the answer to the first. That chain is **multi-hop** retrieval, and choosing the next search from what was read is **agentic** retrieval.

Two smaller questions come along with it:
- **What kind of question is this?** Routing.
- **Does the question name a document?** Self-query.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Query routing** | Decide what a question needs before searching: `lookup`, `table`, `image` or `multi_hop`. |
| **Soft preference** | An extra ranked list restricted to preferred chunks (tables, figures, a named document) that joins RRF. It can lift chunks; it can't hide any. A *hard filter* can. |
| **Self-query** | Extract metadata filters from the question itself ("the safety review" → that document). Here, the LLM proposes and code decides. |
| **Bridge entity** | The name that links two facts (NaPF₆). It isn't in the question; it appears in the first passage. |
| **Agent / tool use** | The model picks an action (`search`, `graph`, `finish`) after reading the passages; code runs it. |
| **Grounding guard** | An agent query may only use identifiers that the question or a shown passage contains. The agent follows evidence; it can't invent leads. |
| **Required vs boosted terms** | In BM25 `require` mode, `+"aur-c103"` must match. Only identifiers the *user* typed are required; a model's identifiers are boosts. |
| **Diversity cap** | At most N chunks per document in the top k (for multi-document questions). |

## 3. Architecture
```
retrieval/router.py        RuleRouter (regexes) and LLMRouter (one call): intent → preferred element types
retrieval/self_query.py    catalog → enum schema → LLM proposal → descriptor-word grounding
retrieval/agent.py         AgentPlanner: reads the top-k, returns search / graph / finish; grounding guards
retrieval/fusion.py        + diversify(hits, k, max_per_document)
retrieval/sparse.py        + lucene_query(..., required=[...])
workflows/retrieval_graph.py  + route and plan nodes; recursion_limit(strategy)
evaluation/routing_eval.py  router accuracy on 3 labelled sets
frontend/                   Streamlit dashboard (app.py, pages.py, views.py, api_client.py)
api: GET /graph/stats, GET /graph/entities; /retrieval/advanced now returns every ranked list
```
```
START → route → transform → retrieve ─(agent on, steps left)→ plan ─(search | graph)→ retrieve …
                                  └──────────────→ [rerank] → check ─→ END  |  correct → retrieve
```

## 4. Walkthrough: r02 with the default strategy
1. **route**: the rule `described: the salt that` matches, so the intent is `multi_hop` and the agent is on. Cost: 0.01 ms.
2. **retrieve**: dense and BM25 for the question. The top passage is the research notes' "E-5: 1.0 M NaPF₆ in EC:DEC (1:1)".
3. **plan**: the agent reads the 5 passages and replies `search "NaPF₆ salt supplier"`, with the reason "E-5 contains NaPF₆; need to find who supplies this salt".
   - Guard: "NaPF₆" occurs in passage 1, so the query may run.
4. **retrieve**: two new lists. BM25 would require only the question's identifier, E-5, which this query doesn't contain, so nothing is required. "NaPF₆" isn't detected as an identifier at all, because the subscript ₆ isn't a decimal digit.
   - RRF over all 4 lists puts the budget supplier row at #3.
5. **plan**: `finish` ("Passage [3] states Nordic Salts AB supplies NaPF6 salt").
6. **check**: the grader judges the context.

In the dashboard's **Workflow inspector**, the **Default** preset shows all of this (the agent runs because the router says multi-hop):
- the path through the graph;
- a timeline;
- every node's detail;
- the RRF table, where the budget row's best list is `dense:NaPF₆ salt supplier`.

## 5. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync                 # adds streamlit 1.66.1 (frontend group)
uv run python -m omnirag.retrieval advanced "Who supplies the salt that the E-5 electrolyte is made with?" --route rules --agent auto
uv run python -m omnirag.retrieval advanced "What does the safety review recommend for pilot-line pouches?" --self-query soft
uv run python -m omnirag.evaluation.routing_eval
uv run python -m omnirag.evaluation.workflow_benchmark --configs 9b                    # ~10 min
uv run python -m omnirag.evaluation.workflow_benchmark --configs 9b --dataset data/eval/relational_seed.json
# dashboard
uv run uvicorn omnirag.api.main:app --no-access-log      # terminal 1
uv run streamlit run src/omnirag/frontend/app.py         # terminal 2 → http://localhost:8501
```

## 6. Expected outputs (measured 2026-11-06, qwen3:4b-instruct, K=5)
**Router accuracy:**

| | seed | relational | held-out |
|---|---|---|---|
| rules | 30/40 | 7/8 | 23/24 |
| LLM | 23/40 | 2/8 | 20/24 |

- The rules take 0.01 ms; the LLM router takes about 1 s.
- Rule misses:
  - seed: q09 (two-part, but both answers are in one chunk), q16 ("lab notebook" with no visual word), q17 ("noted in the" reads like a description), and seven "table" questions phrased as lookups;
  - relational: r05;
  - held-out: h11.

**Retrieval** (full table in ADR-0017):
- **Agent on multi-hop routes:** relational R@5 0.875 → **0.938** and MRR 0.838 → **0.917** (+r02, no loss). On the seed set, recall and MRR don't change; nDCG drops 0.889 → 0.869, and p95 latency is 4.0 s.
- **Rule router alone:** seed MRR 0.928 → 0.941, no recall change. The LLM router loses q24, q35 and r03.
- **Self-query:** all 10 accepted documents were right. A soft preference loses r07; a hard filter loses r04 and r07.
- **Max 2 per document:** only q22 changes (the known ground-truth gap).
- **Default (route + agent auto + check + ≤1 retry, no retry after the agent):**
  - relational R@5 **0.938**, MRR **0.917** (+r02, no loss), p50 4.7 s;
  - seed unchanged (R@5 0.986, MRR 0.928), p50 1.3 s;
  - all 3 unanswerable questions are flagged by the check.

## 7. Surprises found during verification
- **A copied identifier became a filter.**
  - The agent searched "hard carbon anode second source supplier Aurora S1". "S1" came from a passage, and BM25 `require` mode then dropped every chunk without "S1", including the one with the answer's wording.
  - Fix: only the user's identifiers are required. After the fix, r06 (which the agent had *lost*) was recovered.
- **The LLM router is worse than ten regexes.** It calls many plain lookups `table` ("What is the N/P ratio of the anode?") and misses described entities. One reason: its "table" judgement is about how the answer *looks*, while the eval label says where it *is*.
- **Self-query is precise but unhelpful.** It names the right document. But for "How much of the budget of the workstream owned by the electrolyte chemist has been spent?" (r07), the budget sheet is only half the answer: who the electrolyte chemist is comes from the project config. A preference list for one document pushes the other source down.
- **The corrective retry undid the agent.**
  - With the grader and one retry added, r02 was lost again: the 9a corrective query ("…NaPF₆ salt supplier unknown") added two lists that pushed the found supplier row out of the top 5.
  - RRF over *every* list means later, worse queries can dilute earlier, better ones.
  - Fix: no corrective retry once the agent has run.
- **The agent's graph tool is used for phrases.** It sent "next technical review meeting" as an "entity". That phrase is in the question, so the guard allows it, but graph linking finds nothing. Harmless, but it's a wasted step.
- **The dashboard diagram was blank** because a Graphviz node was called `graph`, a reserved word in the DOT language. There's now a test for that.
- **Streamlit caches imported modules.** After editing `pages.py`, restart the server.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `REFUSED identifier not in the question or passages: X` in `agent.steps` | The guard working: the model invented or misremembered a name. |
| `REFUSED repeats an earlier search` | The agent proposed the same query again; its turn ends. |
| `self_query.method = llm_error` | The model returned unusable (here: empty) JSON twice; no filter is applied (safe). Measured on 3 of 48 questions, all about the safety review; each failure costs ~7 s. |
| Dashboard: "API not reachable at http://localhost:8000" | Start uvicorn first, or set `OMNIRAG_FRONTEND_API_URL`. |
| Dashboard shows old behaviour after an edit | Restart `streamlit run` (module cache). |
| `GraphRecursionError` | Impossible by construction; if you add a node to a loop, update `recursion_limit()` and its test. |

## 9. Exercise (≈60 min)
1. **Break the router.** Write 5 questions the rules misclassify, *without* looking at `_CUES`. Run `routing_eval` on them (add a file next to the held-out one). Would one more regex fix them without breaking the held-out set?
2. **Chase r05.** In the inspector, run r05 with `agent=always` and 3 steps. Read the agent's queries and the RRF table. Which query would have found "Qualify Carbonix Materials as second supplier"? Why didn't the agent write it?
3. **Soft vs hard.** Run r07 with self-query soft, then hard. Use the "Where each chunk ranks" view in the lab to explain where the owner's chunk went.
4. **Think.** Why are a model's identifiers boosted while the user's are required? What would go wrong in each direction?

## 10. Acceptance criteria — Phase 9b
- [ ] `routing_eval` reproduces §6: rules 30/40, 7/8, 23/24. The LLM router may vary by a question or two.
- [ ] `advanced` on r02 with `--route rules --agent auto` shows an agent `search` containing NaPF and a hit with "Nordic Salts AB".
- [ ] The relational benchmark shows `agent_auto` +r02 with no losses; the seed set shows no recall or MRR change.
- [ ] The dashboard: Overview shows OK health and 12 documents. The inspector's **Default** preset on r02 shows `retrieve ×2`, `plan ×2` and the budget row in the final context.
- [ ] `uv run pytest` and `uv run pytest -m integration` pass.
