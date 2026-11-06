# Phase 12: RAG evaluation: 110 questions, deterministic metrics, a local judge, LangSmith

> Decision record and full tables: [ADR-0020](../adr/0020-evaluation-framework.md). Report: [rag-eval-2026-11-06.json](../benchmarks/rag-eval-2026-11-06.json) (records: `.partial.jsonl`).

## 1. Problem being solved
"It seems to work" is not a result. To compare retrieval configurations and to catch regressions, the system needs a fixed question set with verifiable ground truth, metrics for every stage, and per-question results, because one question is 0.9% of 110. The brief also asks for LLM-as-a-judge semantic metrics and LangSmith experiments, while keeping documents private.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Evidence quote** | Verbatim text from a sample file that states a fact; resolved to element IDs, so it is checked to exist. |
| **`must_include`** | Terms a correct answer contains (inner list = alternatives), normalised. Deterministic correctness. |
| **Recall / Precision / MRR / nDCG / Hit @5** | Ranking metrics on the top 5 retrieved chunks (relevance = the chunk contains a fact's quote). |
| **Context recall / precision** | Facts present in the assembled 1,024-token context / share of top-5 tokens in relevant chunks. |
| **Faithfulness / citation accuracy** | From Phase 11's quote checks: verified claims / claims; model-cited block holds the quote / claims. |
| **Hallucination** | A claim nothing supports (caught and removed), or any answer to an unanswerable question. |
| **LLM judge** | The local model grading correctness vs the reference, relevance (1–5), and support. A second opinion. |
| **Judge agreement** | How often the judge's "correct" equals the deterministic term check: the judge's own reliability, measured. |
| **Experiment** | One retrieval configuration over all questions, with the same context budget and generator. |

## 3. Architecture
```
data/eval/omnirag_eval_v1.json      110 questions, evidence quotes, must_include, errata
evaluation/rag_eval.py               Evaluator: per (experiment, question) → retrieve → context → answer → judge
                                     → record (metrics, timings, failures) → .partial.jsonl; aggregate; rebuild
evaluation/judge.py                  JUDGE_SYSTEM / JUDGE_SCHEMA (judge-v2)
evaluation/failures.py               root-cause classifier (Phase 13)
evaluation/langsmith_sync.py         redacted dataset + score upload (opt-in)
evaluation/runs.py                   POST /evaluate background runs
observability/tracing.py             LangSmith tracing switch (opt-in, redacting)
```

## 4. Walkthrough: one record
For E3 and q05 ("Why did cell AUR-C103 fail?"):
1. `dense+bm25` retrieves 20 candidates. Recall@5 etc. are computed on the top 5.
2. The context assembler fills 1,024 tokens. Context recall = facts whose quotes are in that text.
3. The generator answers. The citation checker marks claims verified/partial/unsupported.
4. Correctness = all of `["E-5", ["decompos", "gas"]]` in the answer.
5. The judge reads question, reference, answer and sources.
6. The failure classifier returns `[]` (or the first stage that lost a fact).
7. The record is appended to the `.partial.jsonl` file.

## 5. How to run
```bash
uv run python -m omnirag.evaluation.rag_eval --experiments E3,E7 --limit 20 --out /tmp/quick.json   # ~5 min
uv run python -m omnirag.evaluation.rag_eval --indexing --out docs/benchmarks/rag-eval-<date>.json   # all E1-E7 (~4-5 h on the laptop)
uv run python -m omnirag.evaluation.rag_eval --from-partial --out docs/benchmarks/rag-eval-<date>.json  # re-score, no model calls
LANGSMITH_API_KEY=... uv run python -m omnirag.evaluation.rag_eval --from-partial --out ... --langsmith
curl -X POST localhost:8000/api/v1/evaluate -H 'content-type: application/json' -d '{"experiments":["E3"],"limit":10}'
```
Dashboard: **Evaluation** (compare, chart any metric, root causes, per question, start a run).

## 6. Expected outputs (measured 2026-11-06; the run was stopped early: E1–E4 complete, E5 on 76 questions, E6/E7 not measured)
| | E1 Dense | E2 BM25 | E3 Dense+BM25 | E4 +Graph | E5 +Rerank (76 q) |
|---|---|---|---|---|---|
| Recall@5 | 0.915 | 0.895 | **0.950** | 0.900 | 1.000 |
| MRR | 0.844 | 0.799 | **0.879** | 0.805 | 0.966 |
| Context recall | 0.930 | 0.900 | **0.980** | 0.960 | 1.000 |
| Answer correctness | 0.900 | 0.840 | 0.900 | 0.890 | 0.932 |
| Faithfulness | 0.843 | 0.855 | 0.853 | 0.846 | 0.820 |
| Hallucination rate | 0.045 | 0.009 | 0.064 | 0.036 | 0.039 |
| Unanswerable refused | 10/10 | 10/10 | 9/10 | 10/10 | 3/3 |
| End to end p50 | 6.0 s | 4.9 s | 5.0 s | 5.0 s | 22.4 s |

On the same 76 questions, E3 scores 0.918 correct versus E5's 0.932 (+e10 +q33 +r02, −q04 −q34). The full table, with latency percentiles, judge and per-question lists, is in ADR-0020.

## 7. Surprises found during verification
- **The judge calls refusals correct.** In 25 of 27 refusals of answerable questions, the 4B judge said "correct" (why is not visible in its one-sentence reasons). Refusals are now scored by rule; the judge's raw verdict is kept and counted.
- **My own answer terms were wrong for four questions.** "Which E-7 cell retained more capacity?" expected the percentages; "AUR-C106" alone is the answer. Corrected as recorded errata and re-scored for every experiment, without regenerating anything.
- **The GPU throttles.** Halfway through, generation took 3× as long: the GPU was at 88 °C with its clock at 307 MHz. Every report now records GPU temperature and clock per experiment (via the Windows `nvidia-smi.exe`; the Linux one segfaults when started from Python in WSL).
- **Running the Docker stack during the evaluation crashed E5.** Loading torch's 440 MB library failed under memory pressure (8 GB VM: evaluation + Neo4j + API + dashboard containers). Records are resumable, so nothing was lost.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| Numbers from an old prompt in a new report | A stale `.partial.jsonl` was resumed. Delete it when the prompt, judge or dataset changes. |
| `Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so` | Out of memory when the reranker loads (E5/E6). Stop other containers; resume. |
| `--from-partial needs --out …` | Point `--out` at the report path whose `.partial.jsonl` exists. |
| `LangSmith upload skipped: LANGSMITH_API_KEY is not set` | Expected without a key: nothing leaves the machine. |
| 409 on `POST /evaluate` | One evaluation already runs in the API process. |

## 9. Exercise (≈60 min)
1. Run `rag_eval --experiments E3 --limit 20 --no-judge` twice. Which numbers change between runs, and why do latencies differ more than metrics?
2. Pick a failed question from E3's per-question table in the dashboard. Read its record in `.partial.jsonl`: which stage would you fix first?
3. **Think.** Why are the deterministic checks the primary metric and the judge a second opinion? What would you need before trusting the judge alone?

## 10. Acceptance criteria: Phase 12
- [ ] `uv run pytest tests/unit/test_rag_evaluation.py` passes (dataset shape, terms vs references, failures, aggregation, judge, LangSmith redaction, runs, API).
- [ ] `uv run pytest -m integration` verifies every quote of the 110 questions against the live store.
- [ ] `rag_eval --experiments E3 --limit 10` writes a report with retrieval, generation, judge, performance and failure sections.
- [ ] The Evaluation page shows the measured report, marks partial experiments, and lists the errata.
