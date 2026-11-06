"""Upload an evaluation report to LangSmith (Phase 12, optional).

LangSmith is an external hosted service, so this is opt-in (``rag_eval --langsmith``) and
redacting by default:

* **dataset** (``omnirag-eval-v<version>``): question, reference answer, ``must_include``,
  expected source file names, type. No document text, no chunk content.
* **experiments** (one per E1-E7): the metrics computed locally (correctness, faithfulness,
  citation accuracy, hallucination, recall, context recall, judge verdict) as evaluator scores.
  Generated answers are replaced by ``"[redacted]"`` unless ``include_answers=True``: answers
  quote the documents.

The scores are the ones in the local report: LangSmith displays and compares them, it does not
recompute them (the judge and the checks run locally, on the local model).
Needs ``LANGSMITH_API_KEY`` in the environment (read by the ``langsmith`` SDK itself).
Each question result of each experiment is one LangSmith trace (550 for E1-E5 over 110
questions): a free plan's monthly trace limit is reached quickly (measured 2026-11-06).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omnirag.evaluation.dataset import EvalDataset

REDACTED = "[redacted]"
METRICS = ("correct", "unanswerable_refused", "faithfulness", "citation_accuracy",
           "hallucination", "recall", "context_recall", "judge_correct")  # fmt: skip


def dataset_examples(dataset: EvalDataset) -> list[dict[str, Any]]:
    """Examples without any document text (only question-side information)."""
    return [{"inputs": {"question": q.question, "qid": q.id},
             "outputs": {"reference_answer": q.answer, "must_include": q.must_include,
                         "expected_sources": q.expected_sources, "answerable": q.answerable},
             "metadata": {"type": q.type}} for q in dataset.questions]  # fmt: skip


def run_outputs(record: dict[str, Any], include_answers: bool = False) -> dict[str, Any]:
    """What one experiment run reports for one question: scores, never context text."""
    metrics = record["answer"]["metrics"]
    answerable = record["answerable"]
    return {
        "answer": record["answer"]["text"] if include_answers else REDACTED,
        "scores": {
            # same definitions as the report: correctness on answerable questions only,
            # refusals of unanswerable ones scored separately
            "correct": float(bool(record["correct"])) if answerable else None,
            "unanswerable_refused": None if answerable else float(bool(record["refused"])),
            "faithfulness": metrics.get("faithfulness"),
            "citation_accuracy": metrics.get("citation_accuracy"),
            "hallucination": float(bool(record["hallucination"])),
            "recall": (record.get("retrieval") or {}).get("recall"),
            "context_recall": (record.get("retrieval") or {}).get("context_recall"),
            "judge_correct": (None if not record.get("judge") or not record["judge"].get(
                "correctness") else float(record["judge"]["correctness"] == "correct")),
        },
        "failures": record["failures"],
    }  # fmt: skip


def _evaluator(key: str):
    def evaluate_metric(outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": key, "score": (outputs.get("scores") or {}).get(key)}

    evaluate_metric.__name__ = key
    return evaluate_metric


def upload_report(report: dict[str, Any], dataset: EvalDataset, partial: Path | None, *,
                  include_answers: bool = False, client: Any = None,
                  suffix: str = "") -> str:  # fmt: skip
    """Create/refresh the dataset and one experiment per E-run. Returns a status line."""
    if client is None:
        if not os.environ.get("LANGSMITH_API_KEY"):
            return "LangSmith upload skipped: LANGSMITH_API_KEY is not set"
        from langsmith import Client

        client = Client()
    if partial is None or not partial.exists():
        return "LangSmith upload skipped: no per-question records (run with --out)"
    name = f"omnirag-eval-v{dataset.version}"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description=dataset.description[:500])
        client.create_examples(dataset_id=ds.id, examples=dataset_examples(dataset))
    from omnirag.evaluation.rag_eval import rescore

    by_id = {q.id: q for q in dataset.questions}
    records: dict[str, dict[str, dict]] = {}
    for line in partial.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)  # re-scored exactly like the report (errata, refusal rule)
        records.setdefault(rec["experiment"], {})[rec["qid"]] = rescore(rec, by_id[rec["qid"]])
    from langsmith import evaluate

    uploaded = []
    for exp, by_qid in records.items():
        if exp not in report["experiments"]:
            continue

        def target(inputs: dict[str, Any], by_qid: dict = by_qid) -> dict[str, Any]:
            rec = by_qid.get(inputs["qid"])
            return run_outputs(rec, include_answers) if rec else {"answer": None, "scores": {}}

        evaluate(target, data=name, evaluators=[_evaluator(m) for m in METRICS],
                 experiment_prefix=f"{exp}-{report['experiments'][exp]['label']}{suffix}",
                 metadata={"prompt_version": report["prompt_version"],
                           "generation_model": report["generation_model"],
                           "chunk_config": report["chunk_config"]},
                 client=client, max_concurrency=1)  # fmt: skip
        # The SDK sends runs and feedback from a background thread: flush before the next
        # experiment and before the process exits (measured: without it, a short-lived
        # process lost the runs of 4 of 5 experiments).
        if hasattr(client, "flush"):
            client.flush()
        uploaded.append(exp)
    # The SDK reports ingestion errors (e.g. 429: monthly trace limit) only as log warnings
    # from its background thread: "sent" is not "stored". Check the experiments in the UI.
    return (
        f"LangSmith: dataset {name}, experiments sent {uploaded} "
        f"({len(dataset.questions)} traces each); check for 429 warnings above"
    )
