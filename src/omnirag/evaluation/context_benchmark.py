"""Benchmark context assembly (Phase 9c): how much evidence survives per token.

    uv run python -m omnirag.evaluation.context_benchmark [--dataset ...] [--out x.json]

Retrieval runs once per question (``dense+bm25``, 20 candidates); only the assembly varies, so
differences come from assembly alone. For every budget (bge-m3 tokens, including each block's
source line) and every expansion × compression:

* **context recall**: share of a question's facts whose evidence quote is contained in the
  assembled text (the same containment rule as every retrieval benchmark, ADR-0009);
* tokens used, assembly latency, LLM calls.

Reference rows: ``top5`` (the five best chunks, whatever their size: what Phases 6-9 returned)
and ``all`` (every candidate, no budget). Per-question wins/losses are against
``none/none`` at the same budget.

``--calibrate`` also measures generator tokens (qwen3, Ollama ``prompt_eval_count`` of raw
prompts) against bge-m3 tokens on every chunk, because budgets are counted with bge-m3.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from omnirag.chunking import build_embedder, load_tokenizer
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.evaluation.chunking_benchmark import DATASET
from omnirag.evaluation.dataset import EvalDataset, covered_facts
from omnirag.llm.chat import OllamaChat
from omnirag.retrieval import build_retrieval
from omnirag.retrieval.compression import Compressor
from omnirag.retrieval.context import AssembledContext, ContextAssembler, ContextOptions

EXPANDS = ("none", "neighbors", "section")
COMPRESSES = ("none", "lexical", "embedding", "llm")


def _recall(text: str, q) -> float:
    return len(covered_facts(text, q)) / len(q.facts)


def calibrate(settings, count) -> dict[str, Any]:
    """qwen3 tokens / bge-m3 tokens over every chunk text (raw prompts, no template)."""
    from omnirag.ingestion.models import IngestionStatus
    from omnirag.ingestion.store import DocumentStore

    store = DocumentStore(settings.data_dir)
    texts = [c.content for r in store.list_records() if r.status is IngestionStatus.COMPLETED
             for c in store.read_chunks(r.document_id, settings.retrieval_config_id)
             if c.retrievable]  # fmt: skip
    pairs = []
    with httpx.Client(base_url=settings.ollama_base_url, timeout=60) as client:
        for t in texts:
            body = client.post("/api/generate", json={
                "model": settings.generation_model, "prompt": t, "raw": True, "stream": False,
                "options": {"num_predict": 1, "temperature": 0}}).json()  # fmt: skip
            pairs.append((count(t), body.get("prompt_eval_count", 0)))
    bge, qwen = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    ratios = sorted(q / b for b, q in pairs if b)
    return {"chunks": len(pairs), "bge_m3_tokens": bge, "qwen_tokens": qwen,
            "ratio_total": round(qwen / bge, 3), "ratio_p50": round(statistics.median(ratios), 3),
            "ratio_min": round(ratios[0], 3), "ratio_max": round(ratios[-1], 3)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.context_benchmark")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--budgets", default="256,512,1024")
    parser.add_argument("--expand", default=",".join(EXPANDS))
    parser.add_argument("--compress", default=",".join(COMPRESSES))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    tokenizer = load_tokenizer(settings)
    service = build_retrieval(settings)
    llm = OllamaChat(settings.ollama_base_url, settings.generation_model,
                     timeout_s=settings.workflow_llm_timeout_s)  # fmt: skip
    compressor = Compressor(embedder=build_embedder(settings), llm=llm)
    assembler = ContextAssembler(tokenizer.count, service.store, compressor)
    dataset = EvalDataset.load(args.dataset)
    answerable = [q for q in dataset.questions if q.answerable]
    llm.chat("Reply with OK.", "warm up", num_predict=2)

    hits = {q.id: service.search(q.question, "dense+bm25", 20).hits for q in answerable}
    results: dict[str, Any] = {}

    def record(name: str, runs: dict[str, AssembledContext | str], calls: int = 0,
               ms: list[float] | None = None) -> None:  # fmt: skip
        per_q, toks = {}, {}
        for q in answerable:
            r = runs[q.id]
            text = r if isinstance(r, str) else r.render()
            per_q[q.id] = round(_recall(text, q), 3)
            toks[q.id] = tokenizer.count(text)
        results[name] = {
            "context_recall": round(statistics.mean(per_q.values()), 3),
            "tokens_p50": statistics.median(toks.values()), "tokens_max": max(toks.values()),
            "per_question_recall": per_q, "tokens_by_question": toks,
            "llm_calls_per_question": round(calls / len(answerable), 2),
            "assembly_ms_p50": round(statistics.median(ms), 1) if ms else 0.0,
        }  # fmt: skip
        r = results[name]
        print(f"  {name:<34} recall={r['context_recall']:.3f} tokens p50={r['tokens_p50']:>6} "
              f"max={r['tokens_max']:>5} ms p50={r['assembly_ms_p50']:>7} "
              f"llm/q={r['llm_calls_per_question']}", flush=True)  # fmt: skip

    print(f"{len(answerable)} answerable questions · tokenizer {tokenizer.name}\n")
    top5 = {q.id: "\n\n".join(h.content for h in hits[q.id][:5]) for q in answerable}
    record("top5 (Phases 6-9)", top5)
    every = {q.id: "\n\n".join(h.content for h in hits[q.id]) for q in answerable}
    record("all 20 candidates", every)

    budgets = [int(b) for b in args.budgets.split(",")]
    for expand, compress in itertools.product(args.expand.split(","), args.compress.split(",")):
        for budget in budgets:
            calls_before = llm.stats["calls"]
            runs, ms = {}, []
            for q in answerable:
                started = time.perf_counter()
                runs[q.id] = assembler.assemble(q.question, hits[q.id], ContextOptions(
                    budget_tokens=budget, expand=expand, compress=compress))  # fmt: skip
                ms.append((time.perf_counter() - started) * 1000)
            record(f"{expand}/{compress}@{budget}", runs, llm.stats["calls"] - calls_before, ms)

    for name, r in results.items():
        if "@" not in name:
            continue
        base = results.get(f"none/none@{name.split('@')[1]}")
        if base and name != f"none/none@{name.split('@')[1]}":
            a, b = r["per_question_recall"], base["per_question_recall"]
            r["vs_none_none"] = {"wins": sorted(q for q in a if a[q] > b[q]),
                                 "losses": sorted(q for q in a if a[q] < b[q])}  # fmt: skip
    print("\nper-question vs none/none at the same budget:")
    for name, r in results.items():
        if r.get("vs_none_none", {}).get("wins") or r.get("vs_none_none", {}).get("losses"):
            print(f"  {name:<34} +{r['vs_none_none']['wins']} -{r['vs_none_none']['losses']}")

    out: dict[str, Any] = {"created": datetime.now(UTC).isoformat(), "dataset": args.dataset.name,
                           "model": settings.generation_model, "tokenizer": tokenizer.name,
                           "candidates": 20, "results": results}  # fmt: skip
    if args.calibrate:
        out["calibration"] = calibrate(settings, tokenizer.count)
        print(f"\ncalibration (qwen3 / bge-m3 tokens): {out['calibration']}")
    service.close()
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
