"""End-to-end RAG evaluation (Phase 12): retrieval, context, generation, judge, failures.

    uv run python -m omnirag.evaluation.rag_eval [--experiments E1,...,E7] [--limit N]
        [--no-judge] [--indexing] [--out docs/benchmarks/rag-eval-<date>.json] [--langsmith]

Experiments (the brief's E1-E6, plus the pipeline ``/chat`` actually runs):

====  ==========================================  ===========================================
E1    Dense only                                  ``dense``
E2    BM25 only                                   ``bm25``
E3    Dense + BM25 (RRF)                          ``dense+bm25``
E4    Dense + BM25 + Graph                        ``dense+bm25+graph``
E5    Hybrid + reranking                          ``dense+bm25+rerank``
E6    Hybrid + reranking + query rewriting        workflow: rewrite + rerank, no router/agent
E7    Full pipeline (default ``/chat``)           workflow defaults: router, agent, check
====  ==========================================  ===========================================

Every experiment builds the same context (1,024-token budget from 20 ranked candidates,
Phase 9c) and asks the same generator (Phase 10), so differences come from retrieval.
E1-E5 have no context check: the model alone decides to refuse.

Metrics per question (``record``):
* retrieval at K=5: Recall, Precision, MRR, nDCG, Hit, token-weighted context precision;
  **context recall** = facts whose evidence is in the assembled context;
* generation, deterministic: answer correctness (all ``must_include`` terms), completeness
  (share of terms), faithfulness and citation accuracy (quote verification, Phase 11),
  hallucination (an unsupported claim, or any answer to an unanswerable question),
  refusals (correct on unanswerable, false on answerable);
* generation, LLM judge (``judge.py``): correctness, relevance, supported;
* performance: end-to-end, retrieval, embedding, reranking, generation latency; throughput;
  memory; indexing time (``--indexing``).
* failures (``failures.py``): the root-cause stage for every failed question.

Results stream to ``<out>.partial.jsonl`` and are resumed from it, so a long run can be
interrupted. Never edit a report by hand: re-run.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.evaluation.dataset import EvalDataset, EvalQuestion, covered_facts
from omnirag.evaluation.failures import FailureClassifier, answer_includes, summarize
from omnirag.evaluation.judge import JUDGE_VERSION, judge
from omnirag.evaluation.metrics import Unit, ranking_metrics
from omnirag.generation.prompts import PROMPT_VERSION
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.retrieval.context import ContextOptions

DATASET = Path(__file__).parents[3] / "data" / "eval" / "omnirag_eval_v1.json"
K = 5
CANDIDATES = 20

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E1": {"label": "Dense only", "mode": "dense"},
    "E2": {"label": "BM25 only", "mode": "bm25"},
    "E3": {"label": "Dense + BM25", "mode": "dense+bm25"},
    "E4": {"label": "Dense + BM25 + Graph", "mode": "dense+bm25+graph"},
    "E5": {"label": "Hybrid + reranking", "mode": "dense+bm25+rerank"},
    "E6": {"label": "Hybrid + reranking + query rewriting",
           "workflow": {"transform": "rewrite", "rerank": True, "route": "off", "agent": "off",
                        "grade": False, "max_retries": 0, "self_query": "off"}},
    "E7": {"label": "Full pipeline (default /chat)", "workflow": {}},
}  # fmt: skip


def _find_ms(details: Any, key: str) -> float:
    """Sum every ``key`` in a nested details dict (fused results nest their members)."""
    if isinstance(details, dict):
        return sum((v if k == key and isinstance(v, int | float) else _find_ms(v, key))
                   for k, v in details.items())  # fmt: skip
    if isinstance(details, list):
        return sum(_find_ms(v, key) for v in details)
    return 0.0


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, round(p * len(v)) - 1))], 1)


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


class Evaluator:
    def __init__(self, settings: Any, dataset: EvalDataset, *, use_judge: bool = True) -> None:
        from omnirag.generation.chat import build_chat
        from omnirag.knowledge_base.graph_store import GraphUnavailableError
        from omnirag.retrieval import build_retrieval
        from omnirag.workflows.retrieval_graph import Strategy, build_workflow

        self.settings, self.dataset, self.use_judge = settings, dataset, use_judge
        self.Strategy = Strategy
        store = DocumentStore(settings.data_dir)
        records = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED]
        self.records = records
        self.chunks = [c for r in records for c in store.read_chunks(r.document_id,
                       settings.retrieval_config_id) if c.retrievable]  # fmt: skip
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        elements = {r.source_file: store.read_elements(r.document_id) for r in records}
        self.service = build_retrieval(settings)
        self.workflow = build_workflow(settings, self.service)
        self.generator = build_chat(settings, self.workflow).generator
        self.llm = self.generator.llm
        try:
            rows = self.service.store.query(
                "MATCH (c:Chunk {config_id: $cfg}) RETURN c.chunk_id AS id, "
                "c.embedding IS NOT NULL AS vec",
                cfg=settings.retrieval_config_id,
            )
            graph = {r["id"] for r in rows}
            embedded = {r["id"] for r in rows if r["vec"]}
        except GraphUnavailableError:
            graph = embedded = None
        self.classifier = FailureClassifier(elements, self.chunks, graph, embedded)

    def close(self) -> None:
        self.workflow.close()
        self.service.close()

    # ---------------------------------------------------------------- one question
    def _retrieve(self, exp: dict[str, Any], q: EvalQuestion) -> dict[str, Any]:
        started = time.perf_counter()
        if "mode" in exp:
            result = self.service.search(q.question, exp["mode"], CANDIDATES)
            retrieval_ms = result.latency_ms
            ctx = self.workflow.assembler.assemble(
                q.question,
                result.hits,
                ContextOptions(
                    budget_tokens=self.settings.context_budget_tokens, max_candidates=CANDIDATES
                ),
            )
            context = {**ctx.model_dump(mode="json"), "rendered": ctx.render()}
            return {"top": result.hits[:K], "candidates": result.hits, "context": context,
                    "verdict": None, "checks": None, "respect_check": False,
                    "retrieval_ms": retrieval_ms, "context_ms": ctx.ms,
                    "embed_ms": _find_ms(result.details, "embed_ms"),
                    "rerank_ms": _find_ms(result.details, "rerank_ms"),
                    "total_ms": (time.perf_counter() - started) * 1000}  # fmt: skip
        strategy = self.Strategy(**{**self.workflow.default.model_dump(), **exp["workflow"]})
        result = self.workflow.retrieve(q.question, K, strategy=strategy)
        d = result.details
        context = d.get("context") or {"blocks": [], "rendered": "", "skipped": []}
        ids = [cid for b in context["blocks"] for cid in b["hit_chunk_ids"]] + context["skipped"]
        candidates = [
            SimpleNamespace(chunk_id=cid, content=self.chunk_by_id[cid].content, sources={})
            for cid in ids
            if cid in self.chunk_by_id
        ]
        trace = Counter()
        for step in d["trace"]:
            trace[step["node"]] += step["ms"]
        return {"top": result.hits, "candidates": candidates, "context": context,
                "verdict": d["verdict"], "checks": d["checks"], "respect_check": None,
                "retrieval_ms": result.latency_ms - trace["assemble"],
                "context_ms": trace["assemble"],
                "embed_ms": sum(v.get("embed_ms") or 0 for v in d["lists"].values()),
                "rerank_ms": trace["rerank"], "llm_calls_retrieval": None,
                "total_ms": (time.perf_counter() - started) * 1000}  # fmt: skip

    def record(self, name: str, q: EvalQuestion) -> dict[str, Any]:
        exp = EXPERIMENTS[name]
        r = self._retrieve(exp, q)
        answer = self.generator.generate(
            q.question, r["context"], r["verdict"], r["checks"], respect_check=r["respect_check"]
        ).model_dump(mode="json")
        context_text = r["context"]["rendered"]
        rec: dict[str, Any] = {"experiment": name, "qid": q.id, "type": q.type,
                               "answerable": q.answerable, "verdict": r["verdict"]}  # fmt: skip
        if q.answerable:
            units = [Unit(h.chunk_id, self.chunk_by_id[h.chunk_id].token_count
                          if h.chunk_id in self.chunk_by_id else 1,
                          frozenset(covered_facts(h.content, q))) for h in r["top"]]  # fmt: skip
            n_rel = sum(bool(covered_facts(c.content, q)) for c in self.chunks)
            rec["retrieval"] = {
                k: round(v, 4) for k, v in ranking_metrics(units, len(q.facts), n_rel, K).items()
            }
            rec["retrieval"]["context_recall"] = round(
                len(covered_facts(context_text, q)) / len(q.facts), 4
            )
            ok, share = answer_includes(answer["text"], q.must_include)
            rec["correct"], rec["completeness"] = (
                ok and answer["answerable"],
                (share if answer["answerable"] else 0.0),
            )
        else:
            rec["correct"] = not answer["answerable"]
        rec["refused"] = not answer["answerable"]
        rec["hallucination"] = (answer["answerable"] if not q.answerable
                                else answer["metrics"]["unsupported"] > 0)  # fmt: skip
        rec["failures"] = self.classifier.classify(
            q, candidates=r["candidates"], context_text=context_text, answer=answer,
            dense="dense" in exp.get("mode", "dense"), reranked="rerank" in exp.get("mode", "")
            or exp.get("workflow", {}).get("rerank", False), k=K)  # fmt: skip
        rec["judge"] = (judge(self.llm, q.question, q.answer, answer["text"], answer["missing"],
                              context_text) if self.use_judge else None)  # fmt: skip
        rec["ranked"] = [h.chunk_id for h in r["top"]]
        rec["context"] = {
            "tokens": r["context"].get("tokens", 0),
            "chunks": [cid for b in r["context"]["blocks"] for cid in b["chunk_ids"]],
        }
        rec["answer"] = {
            "text": answer["text"], "answerable": answer["answerable"],
            "refused_by": answer["refused_by"], "missing": answer["missing"],
            "grounded": answer["grounded"], "metrics": answer["metrics"],
            "claims": [{k: c[k] for k in ("text", "status", "support", "sources",
                                          "model_sources", "issues")} for c in answer["claims"]],
            "citations": [{"number": c["number"], "source_file": c["source_file"],
                           "page_numbers": c["page_numbers"]} for c in answer["citations"]],
            "prompt_tokens": answer["prompt_tokens"], "eval_tokens": answer["eval_tokens"],
            "error": answer["error"]}  # fmt: skip
        rec["timings"] = {"retrieval_ms": round(r["retrieval_ms"], 1),
                          "context_ms": round(r["context_ms"], 1),
                          "embed_ms": round(r["embed_ms"], 1),
                          "rerank_ms": round(r["rerank_ms"], 1),
                          "generation_ms": answer["latency_ms"],
                          "end_to_end_ms": round(r["total_ms"] + answer["latency_ms"], 1),
                          "judge_ms": (rec["judge"] or {}).get("ms")}  # fmt: skip
        return rec


# ---------------------------------------------------------------- re-scoring
def rescore(rec: dict[str, Any], q: EvalQuestion) -> dict[str, Any]:
    """Recompute the answer-term verdict from the stored answer with the dataset's *current*
    ``must_include`` (errata are re-applied to every experiment alike, without regenerating),
    and the judge's effective verdict.

    Refusals are not left to the judge: measured on E1-E4, the 4B judge called most refusals
    of answerable questions "correct". A refusal is "refused" on an answerable question and
    "correct" on an unanswerable one; the raw judge verdict is kept as ``judge.raw``.
    """
    rec = {**rec}
    a = rec["answer"]
    if q.answerable:
        ok, share = answer_includes(a["text"], q.must_include)
        correct = ok and a["answerable"]
        failures = [f for f in rec["failures"] if f != "generation_failure"]
        upstream = any(f not in ("hallucination", "citation_failure") for f in failures)
        if not correct and not upstream:
            failures.insert(0, "generation_failure")
        rec.update(correct=correct, completeness=share if a["answerable"] else 0.0,
                   failures=failures)  # fmt: skip
    j = rec.get("judge")
    if j and j.get("correctness"):
        raw = j.get("raw", j["correctness"])
        effective = raw
        if not a["answerable"]:
            effective = "refused" if q.answerable else "correct"
        rec["judge"] = {**j, "raw": raw, "correctness": effective}
    return rec


def judge_refusal_errors(records: list[dict[str, Any]]) -> dict[str, int]:
    """How often the raw judge verdict was wrong on refusals (answerable questions)."""
    refused = [r for r in records if r["answerable"] and r["refused"] and r.get("judge")]
    return {
        "refusals_judged": len(refused),
        "raw_said_correct": sum(1 for r in refused if r["judge"].get("raw") == "correct"),
    }


# ---------------------------------------------------------------- aggregation
def _by_type(rs: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rs if r["answerable"]]
    return {
        "n": len(rs),
        "correct": _mean([float(bool(r["correct"])) for r in rs]),
        "recall": _mean([r["retrieval"]["recall"] for r in ans]),
        "context_recall": _mean([r["retrieval"]["context_recall"] for r in ans]),
    }


def aggregate(records: list[dict[str, Any]], wall_s: float | None = None) -> dict[str, Any]:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    retrieval = {m: _mean([r["retrieval"][m] for r in ans]) for m in
                 ("recall", "precision", "mrr", "ndcg", "hit", "context_precision",
                  "context_recall")}  # fmt: skip
    judged = [r["judge"] for r in records if r.get("judge") and r["judge"].get("correctness")]
    jc = Counter(j["correctness"] for j in judged)
    agree = [(r["judge"]["correctness"] == "correct") == bool(r["correct"]) for r in records
             if r.get("judge") and r["judge"].get("correctness")]  # fmt: skip
    t = {k: [r["timings"][k] for r in records if r["timings"].get(k) is not None]
         for k in ("end_to_end_ms", "retrieval_ms", "embed_ms", "rerank_ms", "generation_ms",
                   "context_ms", "judge_ms")}  # fmt: skip
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    failures = summarize({r["qid"]: r["failures"] for r in records})
    return {
        "questions": len(records), "answerable": len(ans), "unanswerable": len(una),
        "retrieval": retrieval,
        "generation": {
            "answer_correctness": _mean([float(bool(r["correct"])) for r in ans]),
            "completeness": _mean([r["completeness"] for r in ans]),
            "faithfulness": _mean([r["answer"]["metrics"]["faithfulness"] for r in answered]),
            "citation_accuracy": _mean([r["answer"]["metrics"]["citation_accuracy"]
                                        for r in answered]),
            "hallucination_rate": _mean([float(r["hallucination"]) for r in records]),
            "unanswerable_refused": sum(r["refused"] for r in una),
            "false_refusals": sum(r["refused"] for r in ans),
            "refused_by": dict(Counter(r["answer"]["refused_by"] for r in records
                                       if r["answer"]["refused_by"])),
            "claims": sum(r["answer"]["metrics"]["claims"] for r in records),
            "claim_status": dict(sum((Counter({s: r["answer"]["metrics"][s] for s in
                                               ("verified", "relocated", "partial", "unsupported")})
                                      for r in records), Counter())),
        },
        "judge": {
            "judged": len(judged), "correctness": dict(jc),
            "correct_rate": round(jc["correct"] / len(judged), 3) if judged else None,
            "relevance": _mean([j["relevance"] for j in judged]),
            "supported_rate": _mean([float(j["supported"]) for j in judged
                                     if j["supported"] is not None]),
            "agreement_with_answer_check": _mean([float(a) for a in agree]),
            "errors": sum(1 for r in records if r.get("judge") and r["judge"].get("error")),
            "refusal_overrides": judge_refusal_errors(records),
        },
        "performance": {
            **{f"{k}_p50": _pct(v, 0.5) for k, v in t.items()},
            **{f"{k}_p95": _pct(v, 0.95) for k, v in t.items()},
            # sequential: one question at a time, from each question's measured end-to-end
            # latency (judge excluded); comparable across experiments and resumed runs
            "throughput_q_per_min": (round(60_000 / statistics.mean(t["end_to_end_ms"]), 2)
                                     if t["end_to_end_ms"] else None),
            "wall_s": round(wall_s, 1) if wall_s else None,
            "prompt_tokens_p50": _pct([r["answer"]["prompt_tokens"] for r in records
                                       if r["answer"]["prompt_tokens"]], 0.5),
            "eval_tokens_p50": _pct([r["answer"]["eval_tokens"] for r in records
                                     if r["answer"]["eval_tokens"]], 0.5),
        },
        "by_type": {typ: _by_type(rs) for typ, rs in sorted(by_type.items())},
        "failures": failures,
        "per_question": {r["qid"]: {"correct": r["correct"], "refused": r["refused"],
                                    "hallucination": r["hallucination"],
                                    "recall": r.get("retrieval", {}).get("recall"),
                                    "context_recall": r.get("retrieval", {}).get("context_recall"),
                                    "judge": (r.get("judge") or {}).get("correctness"),
                                    "failures": r["failures"]} for r in records},
    }  # fmt: skip


def gpu_snapshot() -> dict[str, Any] | None:
    """GPU state via nvidia-smi (list args, no shell). A hot laptop GPU throttles its clock,
    so latency numbers are reported with it. None when nvidia-smi is not available."""
    import shutil
    import subprocess

    # In WSL the Linux nvidia-smi segfaults when started from Python (measured: exit -11);
    # the Windows binary works through interop, so it is tried first.
    windows = Path("/mnt/c/Windows/System32/nvidia-smi.exe")
    exe = str(windows) if windows.exists() else shutil.which("nvidia-smi")
    if not exe:
        return None
    fields = "temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu,memory.used,power.draw"
    try:
        out = subprocess.run(  # noqa: S603 - fixed binary from PATH, constant arguments
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    keys = ("temp_c", "sm_mhz", "sm_max_mhz", "util_pct", "mem_used_mb", "power_w")
    return dict(zip(keys, values, strict=False))


def memory_snapshot(settings: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "eval_process_max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        )
    }
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        total = int(mem["MemTotal"].split()[0]) / 1024
        avail = int(mem["MemAvailable"].split()[0]) / 1024
        info |= {"vm_total_mb": round(total), "vm_used_mb": round(total - avail)}
    except (OSError, KeyError, ValueError):
        pass
    try:
        ps = httpx.get(f"{settings.ollama_base_url}/api/ps", timeout=5).json()
        info["ollama_vram_mb"] = {m["name"]: round(m.get("size_vram", 0) / 2**20)
                                  for m in ps.get("models", [])}  # fmt: skip
    except (httpx.HTTPError, ValueError):
        pass
    return info


def indexing_times(ev: Evaluator) -> dict[str, Any]:
    """Ingestion time per document (from its record) and a fresh graph load of every document
    (idempotent, ADR-0011), timed."""
    from omnirag.ingestion import build_graph_indexer
    from omnirag.knowledge_base.__main__ import _load_one

    store = DocumentStore(ev.settings.data_dir)
    indexer = build_graph_indexer(ev.settings)
    graph_ms = {}
    try:
        for record in ev.records:
            stats = _load_one(store, indexer, record)
            if stats:
                graph_ms[record.source_file] = stats["ms"]
    finally:
        indexer.close()
    ingest = {r.source_file: r.duration_ms for r in ev.records if r.duration_ms}
    return {
        "documents": len(ev.records),
        "chunks": len(ev.chunks),
        "ingestion_ms_total": round(sum(ingest.values()), 1),
        "ingestion_ms": ingest,
        "graph_load_ms_total": round(sum(graph_ms.values()), 1),
        "graph_load_ms": graph_ms,
    }


def run(settings: Any, dataset: EvalDataset, experiments: list[str], *, use_judge: bool = True,
        limit: int | None = None, partial: Path | None = None, indexing: bool = False,
        progress: Any = print) -> dict[str, Any]:  # fmt: skip
    questions = dataset.questions[:limit] if limit else dataset.questions
    done: dict[tuple[str, str], dict] = {}
    if partial and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[(rec["experiment"], rec["qid"])] = rec
    ev = Evaluator(settings, dataset, use_judge=use_judge)
    ev.llm.chat("Reply with OK.", "warm up", num_predict=2)
    ev.service.retrievers["dense"].retrieve("warm up", 1)
    started_all = time.perf_counter()
    results: dict[str, Any] = {}
    try:
        for name in experiments:
            records, started = [], time.perf_counter()
            fresh, gpu_start = 0, gpu_snapshot()
            for i, q in enumerate(questions, 1):
                rec = done.get((name, q.id))
                if rec is None:
                    rec = ev.record(name, q)
                    fresh += 1
                    if partial:
                        with partial.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
                if i % 10 == 0 or i == len(questions):
                    progress(f"  {name} {i}/{len(questions)}")
            wall = time.perf_counter() - started
            by_id = {q.id: q for q in questions}
            records = [rescore(r, by_id[r["qid"]]) for r in records]
            summary = aggregate(records, wall if fresh == len(questions) else None)
            results[name] = {
                "label": EXPERIMENTS[name]["label"],
                **summary,
                "gpu": {"start": gpu_start, "end": gpu_snapshot()},
            }
            g, rt = summary["generation"], summary["retrieval"]
            progress(f"{name} {EXPERIMENTS[name]['label']:<40} R@5={rt['recall']} "
                     f"ctxR={rt['context_recall']} correct={g['answer_correctness']} "
                     f"faith={g['faithfulness']} halluc={g['hallucination_rate']} "
                     f"refused(unans)={g['unanswerable_refused']}/{summary['unanswerable']} "
                     f"e2e p50={summary['performance']['end_to_end_ms_p50']} ms")  # fmt: skip
        report = {
            "created": datetime.now(UTC).isoformat(), "dataset": dataset.name,
            "dataset_version": dataset.version, "questions": len(questions), "k": K,
            "candidates": CANDIDATES, "context_budget": settings.context_budget_tokens,
            "generation_model": settings.generation_model,
            "judge_model": settings.generation_model if use_judge else None,
            "prompt_version": PROMPT_VERSION, "judge_version": JUDGE_VERSION,
            "chunk_config": settings.retrieval_config_id,
            "wall_s": round(time.perf_counter() - started_all, 1),
            "experiments": results,
            "ground_truth": {q.id: {"type": q.type, "expected_sources": q.expected_sources,
                                    "relevant_chunk_ids": sorted(c.chunk_id for c in ev.chunks
                                                                 if covered_facts(c.content, q))}
                             for q in questions},
            "memory": memory_snapshot(settings),
        }  # fmt: skip
        if indexing:
            report["indexing"] = indexing_times(ev)
        return report
    finally:
        ev.close()


def rebuild(report: dict[str, Any], dataset: EvalDataset, partial: Path) -> dict[str, Any]:
    """Re-aggregate a report from its per-question records with the current dataset (errata,
    effective judge verdicts). Run metadata (memory, GPU, indexing, timings) is kept."""
    by_id = {q.id: q for q in dataset.questions}
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for line in partial.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        by_exp[rec["experiment"]].append(rescore(rec, by_id[rec["qid"]]))
    out = {
        **report,
        "rescored": datetime.now(UTC).isoformat(),
        "dataset_errata": dataset.errata,
        "experiments": {},
    }
    names = [n for n in EXPERIMENTS if n in by_exp]  # report order = E1..E7
    out["not_measured"] = [n for n in EXPERIMENTS if n not in by_exp]
    for name in names:
        old = report.get("experiments", {}).get(
            name, {"label": EXPERIMENTS[name]["label"], "performance": {}}
        )
        out["experiments"][name] = {
            "label": old["label"],
            **aggregate(by_exp[name]),
            "gpu": old.get("gpu"),
        }
        out["experiments"][name]["performance"]["wall_s"] = old["performance"].get("wall_s")
        out["experiments"][name]["complete"] = len(by_exp[name]) == len(dataset.questions)
    return out


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.rag_eval")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--indexing", action="store_true", help="also time a full graph reload")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--from-partial",
        action="store_true",
        help="rebuild --out from its .partial.jsonl (re-score, no model calls)",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="upload dataset + experiments (needs LANGSMITH_API_KEY)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    dataset = EvalDataset.load(args.dataset)
    names = [n for n in args.experiments.split(",") if n]
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments: {sorted(unknown)}")
    partial = args.out.with_suffix(".partial.jsonl") if args.out else None
    if args.from_partial:
        if not partial or not partial.exists():
            parser.error("--from-partial needs --out and its .partial.jsonl")
        if args.out.exists():
            base = json.loads(args.out.read_text(encoding="utf-8"))
        else:  # the run stopped before writing its report: metadata from the settings
            base = {
                "created": datetime.now(UTC).isoformat(),
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "questions": len(dataset.questions),
                "k": K,
                "candidates": CANDIDATES,
                "context_budget": settings.context_budget_tokens,
                "generation_model": settings.generation_model,
                "judge_model": settings.generation_model,
                "prompt_version": PROMPT_VERSION,
                "judge_version": JUDGE_VERSION,
                "chunk_config": settings.retrieval_config_id,
                "wall_s": None,
                "stopped_early": True,
                "memory": memory_snapshot(settings),
            }
        report = rebuild(base, dataset, partial)
        args.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n")
                             .encode("utf-8"))  # fmt: skip
        print(f"rebuilt {args.out} from {partial}")
        return 0
    report = run(
        settings,
        dataset,
        names,
        use_judge=not args.no_judge,
        limit=args.limit,
        partial=partial,
        indexing=args.indexing,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n")
                             .encode("utf-8"))  # fmt: skip
        print(f"wrote {args.out}")
    if args.langsmith:
        from omnirag.evaluation.langsmith_sync import upload_report

        print(upload_report(report, dataset, partial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
