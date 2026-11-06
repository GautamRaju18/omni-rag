"""Benchmark the Phase 9 workflow: query transforms, the context check, corrective retries.

    uv run python -m omnirag.evaluation.workflow_benchmark [--dataset ...] [--out x.json]

Configurations (all on dense + BM25 + RRF, K=5, no reranking unless named):
* ``base``: no transform, no check: must reproduce ``dense+bm25`` (sanity);
* ``rewrite`` / ``multi_query`` / ``hyde`` / ``multi_query+hyde``: transforms alone;
* ``corrective``: identifier check + LLM grader + at most 1 corrective retry;
* ``multi_query+corrective``: both.

Also reports the **grader's accuracy**. Ground truth "sufficient" means the base top-5 covers
every fact (recall 1.0); "insufficient" means an unanswerable question or recall 0. Verdicts
come from the identifier check and the LLM, on the same base top-5.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.evaluation.chunking_benchmark import DATASET
from omnirag.evaluation.dataset import EvalDataset
from omnirag.evaluation.embedding_benchmark import evaluate
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.retrieval import build_retrieval
from omnirag.workflows.retrieval_graph import Strategy, build_workflow

CONFIGS: dict[str, dict] = {
    "base": {"transform": "none", "grade": False, "max_retries": 0},
    "rewrite": {"transform": "rewrite", "grade": False, "max_retries": 0},
    "multi_query": {"transform": "multi_query", "grade": False, "max_retries": 0},
    "hyde": {"transform": "hyde", "grade": False, "max_retries": 0},
    "multi_query+hyde": {"transform": "multi_query+hyde", "grade": False, "max_retries": 0},
    "corrective": {"transform": "none", "grade": True, "max_retries": 1},
    "multi_query+corrective": {"transform": "multi_query", "grade": True, "max_retries": 1},
}


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.workflow_benchmark")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    config = settings.retrieval_config_id
    chunks = [c for r in store.list_records() if r.status is IngestionStatus.COMPLETED
              for c in store.read_chunks(r.document_id, config) if c.retrievable]  # fmt: skip
    index_of = {c.chunk_id: i for i, c in enumerate(chunks)}
    dataset = EvalDataset.load(args.dataset)
    answerable = [q for q in dataset.questions if q.answerable]
    unanswerable = [q for q in dataset.questions if not q.answerable]
    service = build_retrieval(settings)
    workflow = build_workflow(settings, service)
    llm = workflow.tx.llm
    llm.chat("Reply with OK.", "warm up", num_predict=2)
    service.retrievers["dense"].retrieve("warm up", 1)
    print(f"{len(chunks)} chunks · {len(answerable)} answerable + {len(unanswerable)} "
          f"unanswerable questions · K={args.k} · {settings.generation_model}\n")  # fmt: skip

    results: dict[str, dict] = {}
    for name in [c for c in args.configs.split(",") if c]:
        strategy = Strategy(**{**workflow.default.model_dump(), **CONFIGS[name]})
        calls_before = llm.stats["calls"]
        runs = {q.id: workflow.retrieve(q.question, args.k, strategy=strategy)
                for q in [*answerable, *unanswerable]}  # fmt: skip
        ranked = [[index_of[h.chunk_id] for h in runs[q.id].hits if h.chunk_id in index_of]
                  for q in answerable]  # fmt: skip
        r = evaluate(chunks, answerable, lambda qi, ranked=ranked: ranked[qi], args.k)
        latencies = sorted(run.latency_ms for run in runs.values())
        r.update({
            "latency_ms": {"p50": round(statistics.median(latencies), 1),
                           "p95": round(latencies[int(0.95 * len(latencies)) - 1], 1)},
            "llm_calls_per_question": round((llm.stats["calls"] - calls_before) / len(runs), 2),
            "attempts": dict(Counter(run.details["attempts"] for run in runs.values())),
            "verdicts": dict(Counter(run.details["verdict"] for run in runs.values())),
            "verdict_by_question": {qid: run.details["verdict"] for qid, run in runs.items()},
            "queries_by_question": {qid: run.details["queries"] for qid, run in runs.items()},
        })  # fmt: skip
        results[name] = r
        print(f"  {name:<24} R@5={r['recall']:.3f} MRR={r['mrr']:.3f} nDCG={r['ndcg']:.3f} "
              f"p50={r['latency_ms']['p50']}ms llm/q={r['llm_calls_per_question']} "
              f"verdicts={r['verdicts']}", flush=True)  # fmt: skip

    base = results.get("base")
    if base:
        for name, r in results.items():
            a, b = r["per_question_recall"], base["per_question_recall"]
            r["vs_base"] = {"wins": sorted(q for q in a if a[q] > b[q]),
                            "losses": sorted(q for q in a if a[q] < b[q])}  # fmt: skip
            if name != "base":
                print(f"  {name:<24} vs base: +{r['vs_base']['wins']} -{r['vs_base']['losses']}")

    # Grader accuracy on the base top-5 (first check of the corrective run).
    if base and "corrective" in results:
        truth = {q.id: ("sufficient" if base["per_question_recall"][q.id] == 1.0 else
                        "insufficient" if base["per_question_recall"][q.id] == 0 else "partial")
                 for q in answerable}  # fmt: skip
        truth |= {q.id: "insufficient" for q in unanswerable}
        verdict = {}
        for q in [*answerable, *unanswerable]:
            run = workflow.retrieve(
                q.question,
                args.k,
                strategy=Strategy(
                    **{**workflow.default.model_dump(), **CONFIGS["corrective"], "max_retries": 0}
                ),
            )
            verdict[q.id] = run.details["checks"][0]["verdict"]
        pairs = Counter((truth[q], verdict[q]) for q in truth)
        binary = sum((truth[q] == "sufficient") == (verdict[q] == "sufficient") for q in truth)
        results["grader"] = {
            "confusion": {f"{t}->{v}": n for (t, v), n in sorted(pairs.items())},
            "binary_accuracy": round(binary / len(truth), 3),
            "unanswerable": {q.id: verdict[q.id] for q in unanswerable},
            "disagreements": sorted(q for q in truth
                                    if (truth[q] == "sufficient") != (verdict[q] == "sufficient")),
        }  # fmt: skip
        print(
            f"\n  grader (sufficient vs not) accuracy {results['grader']['binary_accuracy']} "
            f"confusion {results['grader']['confusion']}"
        )
        print(f"  disagreements: {results['grader']['disagreements']}")
    workflow.close()
    service.close()
    if args.out:
        args.out.write_text(json.dumps({
            "created": datetime.now(UTC).isoformat(), "dataset": args.dataset.name,
            "k": args.k, "model": settings.generation_model, "results": results,
        }, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
