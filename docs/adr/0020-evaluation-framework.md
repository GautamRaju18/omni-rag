# ADR-0020: A 110-question evaluation; deterministic metrics first, a local judge second; failures by root cause

**Status:** accepted (2026-11-06, Phases 12-13)

## Context
The brief asks for:
- an evaluation set of at least 100 questions across eight kinds, each with a reference answer, relevant chunk IDs, expected sources and answer characteristics;
- retrieval metrics (Recall, Precision, MRR, nDCG, Hit Rate, context precision and recall);
- generation metrics (faithfulness, relevance, correctness, hallucination, citation accuracy, completeness);
- performance metrics;
- experiments E1–E6, with LangSmith and LLM-as-a-judge for the semantic metrics;
- a failure taxonomy and a dashboard.

Two constraints shape the design:
- **Locally available models only.** The judge is the same 4B model as the generator: a second model doesn't fit beside it on 6 GB.
- **LangSmith is an external service**, and sending documents there needs care.

## Decision
- **Dataset** `data/eval/omnirag_eval_v1.json`: 110 questions. It merges the 40 seed questions (Phase 3b) and the 8 relational ones (Phase 7), and adds 62 new ones written from the samples.
  - Types: factual 26, table 21, multi-hop 15, keyword 11, image 11, unanswerable 10, comparative 9, cross-document 7.
  - Every evidence quote is verbatim and resolves to element IDs: a unit test checks the non-layout files, an integration test checks everything including Docling and generated text.
  - `must_include` lists the terms a correct answer contains (alternatives allowed). A test asserts that the reference answer itself passes.
  - Relevant chunk IDs are derived per run from the evidence and recorded in the report (`ground_truth`). Chunk IDs change with the chunk set, element IDs don't (ADR-0008).
- **Experiments** (`evaluation/rag_eval.py`):
  - E1 dense, E2 BM25, E3 dense+BM25, E4 +graph, E5 +rerank, E6 +rerank+rewrite;
  - E7: the full `/chat` pipeline (router, agent, context check).

  All use the same context (1,024-token budget from 20 candidates) and the same generator, so only retrieval differs. E1–E5 have no context check: the model alone decides to refuse.
- **Metrics, deterministic first:**
  - retrieval at K=5 plus context recall and precision;
  - answer correctness (all `must_include` terms present) and completeness (share of terms);
  - faithfulness and citation accuracy (quote verification, ADR-0019);
  - hallucination (an unsupported claim, or any answer to an unanswerable question);
  - correct and false refusals;
  - latency per component, throughput, memory, GPU state.
- **LLM judge** (`judge.py`, `judge-v2`): correctness against the reference, relevance (1–5), supported. Temperature 0, JSON schema, short replies (v1's 300-character reasons cost 8 s per call).
  - **Refusals are not left to the judge.** Measured: in E1–E4 it called 25 of 27 refusals of answerable questions "correct". A refusal is scored "refused" on an answerable question and "correct" on an unanswerable one; the raw verdict is kept and counted (`refusal_overrides`).
  - The judge's agreement with the deterministic check is reported per experiment.
- **Failure root cause** (`failures.py`): each fact is followed through the pipeline (parsing/OCR → chunking → graph → embedding → retrieval → reranking → context → generation), and the first stage that lost it is blamed. Hallucination and citation failures are added on top. Live chat turns get the failure kinds visible without ground truth (`live_flags`).
- **Robust runs:** per-question records stream to `<out>.partial.jsonl` and are resumed from it. `--from-partial` rebuilds a report from the records, applying the dataset's recorded errata and the refusal rule to every experiment alike, with no model calls.
- **LangSmith** (`langsmith_sync.py`, `observability/tracing.py`) is opt-in and redacting:
  - the dataset carries questions, reference answers, terms and source file names;
  - experiments carry scores; answers are `[redacted]` unless `include_answers`;
  - tracing needs both `OMNIRAG_LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`, and hides inputs/outputs by default.

  The SDK calls were checked against the installed `langsmith` 0.14.17 and are tested with a fake client. They were **not run against the hosted service** (no API key on this machine).
- **API:** `POST /evaluate` (one run at a time, in the API process, 409 otherwise), `GET /evaluate[/{run_id}]`, `GET /analytics`.

## Measured (2026-11-06, qwen3:4b-instruct (generator and judge), bge-m3, multimodal-256-ctx chunks; [report](../benchmarks/rag-eval-2026-11-06.json), [per-question records](../benchmarks/rag-eval-2026-11-06.partial.jsonl), [same-76 comparison](../benchmarks/same76-rag-eval-2026-11-06.json))
The run was **stopped early at the user's request**:
- E1–E4 are complete (110 questions each);
- E5 covered the first 76 questions;
- E6 and E7 were not measured.

The laptop GPU was thermally throttling during the run (88 °C, core clock down to 307 MHz of 2,100 at 93% load, checked mid-run), so latencies are pessimistic.

**All 110 questions:**

| | E1 Dense | E2 BM25 | E3 Dense+BM25 | E4 +Graph |
|---|---|---|---|---|
| Recall@5 | 0.915 | 0.895 | **0.950** | 0.900 |
| MRR | 0.844 | 0.799 | **0.879** | 0.805 |
| nDCG@5 | 0.800 | 0.765 | **0.841** | 0.767 |
| Hit@5 | 0.950 | 0.960 | **0.990** | 0.960 |
| Precision@5 | 0.268 | **0.343** | 0.290 | 0.268 |
| Context recall (1,024-token context) | 0.930 | 0.900 | **0.980** | 0.960 |
| Context precision (token-weighted, top 5) | 0.294 | **0.377** | 0.320 | 0.301 |
| **Answer correctness** (all terms) | **0.900** | 0.840 | **0.900** | 0.890 |
| Completeness (share of terms) | 0.915 | 0.855 | 0.915 | 0.905 |
| Faithfulness (claims quote-verified) | 0.843 | 0.855 | 0.853 | 0.846 |
| Citation accuracy | 0.847 | **0.888** | 0.864 | 0.853 |
| Hallucination rate (any unsupported claim, or an unanswerable answered) | 0.045 | **0.009** | 0.064 | 0.036 |
| Unanswerable refused | 10/10 | 10/10 | 9/10 | 10/10 |
| False refusals (answerable) | 5 | 12 | 5 | 5 |
| Judge: correct / relevance / supported | 0.908 / 4.95 / 0.97 | 0.845 / 4.89 / 1.00 | 0.890 / 4.95 / 0.97 | 0.873 / 4.96 / 0.96 |
| Judge agreement with the term check | 0.963 | 0.955 | 0.954 | 0.955 |
| End to end p50 / p95 (s) | 6.0 / 15.5 | 4.9 / 12.5 | 5.0 / 13.7 | 5.0 / 12.9 |
| Retrieval p50 (ms) | 74 | 23 | 20 | 76 |
| Generation p50 / p95 (s) | 5.7 / 15.4 | 4.8 / 12.4 | 5.0 / 13.7 | 4.8 / 12.7 |
| Throughput (sequential, q/min) | 8.1 | 10.6 | 9.7 | 9.7 |
| Prompt → answer tokens (p50) | 1,364 → 92 | 1,260 → 89 | 1,354 → 90 | 1,350 → 89 |
| Failed questions (root causes) | 23 | 23 | 19 | 18 |

**The same 76 questions** (the first 76 in dataset order, which is all that E5 finished):

| | E1 | E2 | E3 | E4 | E5 +rerank |
|---|---|---|---|---|---|
| Recall@5 | 0.938 | 0.959 | 0.979 | 0.945 | **1.000** |
| MRR | 0.877 | 0.853 | 0.901 | 0.828 | **0.966** |
| nDCG@5 | 0.840 | 0.834 | 0.873 | 0.807 | **0.940** |
| Context recall | 0.938 | 0.966 | 0.993 | 0.966 | **1.000** |
| Answer correctness | 0.890 | 0.890 | 0.918 | 0.877 | **0.932** |
| Faithfulness | 0.870 | 0.869 | **0.874** | 0.865 | 0.820 |
| Hallucination rate | 0.026 | **0.000** | 0.013 | 0.013 | 0.039 |
| End to end p50 (s) | 5.7 | 5.0 | **4.9** | 4.9 | 22.4 (reranking 19.7 on CPU) |

**Per question** (not only means):
- **E5 vs E3** (same 76): +e10, +q33, +r02; −q04, −q34.
  - Reranking put the budget row with "Nordic Salts AB" into the context, which r02 needed.
- **E1 vs E3** (all 110): +q33, +e10, +e34, +e53, +e59; −q04, −q20, −q34, −q35.
- **E4 vs E3** (all 110): +e34, +e53, +e59; −q26, −q34, −e23.
- **E2 vs E3** (all 110): +q33, +e34, +e59; −q05, −r04, −r08, −e32, −e37, −e42, −e44, −e52.
  - BM25 alone loses two-fact questions: cross-document 0.571 and multi-hop 0.600 correct, against 0.857 and 0.800 for E3. 14 of its 15 retrieval failures are two-fact questions.

**Failure root causes** (questions per first stage that lost them):

| | E1 | E2 | E3 | E4 | E5 (76) |
|---|---|---|---|---|---|
| retrieval | 1 | **15** | 1 | 3 | 0 |
| context construction | **10** | 1 | 3 | 3 | 0 |
| generation | 6 | 5 | 6 | 7 | 5 |
| hallucination | 2 | 1 | 5 | 2 | 1 |
| citation | 4 | 1 | 4 | 3 | 1 |

No parsing, OCR, chunking, graph or embedding failures: every evidence quote is in an element, a chunk, the graph and the vector index.

What the numbers say:
- **Dense + BM25 (E3) is the best complete configuration.** It has the best retrieval and the best context recall (0.98), and ties for the best answer correctness (0.90). It stays the default (ADR-0014 and ADR-0017's retrieval base); E7 adds the router, agent and check on top.
- **Reranking (E5) is the best retriever** on the questions it covered (R@5 1.000, MRR 0.966, +r02), and the most correct, at 4.6× the latency on CPU. It stays the quality mode (ADR-0015).
- **Graph retrieval (E4) still doesn't pay**: lower MRR than E3, −1 correct answer. Same as ADR-0014, now with generation measured.
- **Most remaining failures are the 4B generator, not retrieval.** With the evidence in the context, the model:
  - refuses (e08 "What application is the Aurora S1 designed for?", e10);
  - drops part of a two-part answer (q17);
  - picks the wrong number (e48: the 1,500-cycle qualification instead of the 3,000-cycle target);
  - attributes a table row to the wrong cell (q33: "Aiko Tanaka tested AUR-C107"). That claim passes the quote check because the quote is real, from another row; the judge caught it.
- **What "hallucination rate" counts.** It counts every question where the model *wrote* an unsupported claim. The citation checker removed those claims before display, except for an unanswerable question answered with grounded-looking text (e59, E3: AUR-C107's 25 °C value given as 45 °C).
- **A correct claim can be removed.** e51 (two-hop) cited only one of the two blocks its claim needs, so the claim failed the identifier check.
- **The judge is unreliable on refusals** (25 of 27 called "correct"). With refusals decided by rule, it agrees with the term check on 95–96% of questions.

**Dataset errata (recorded in the file, applied to every experiment).** Reading the first answers showed four "which X?" questions (e27, e28, e29, e31) whose `must_include` listed the supporting numbers instead of the entity asked for. e27's correct "AUR-C106 retained more capacity" was scored wrong. Evidence quotes are unchanged; every experiment was re-scored from its stored answers. Effect: E1 and E3 correctness 0.89 → 0.90; E2 and E4 unchanged.

## Consequences
- Evaluate generation changes with `rag_eval` on at least E3 and E7. Report per-question wins and losses, and failure root causes. Delete `*.partial.jsonl` whenever the prompt, judge or dataset changes.
- **The deterministic checks are the primary signal.** The judge is a second opinion whose agreement is always reported.
- **Open:**
  - **finish E6 and E7, and the rest of E5:** `rag_eval --out docs/benchmarks/rag-eval-<date>.json` resumes from the records;
  - a dataset written by someone else (single author today);
  - a stronger judge when hardware allows;
  - run LangSmith upload and tracing against the hosted service.
