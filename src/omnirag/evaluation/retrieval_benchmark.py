"""Benchmark the Neo4j retrievers on the seed set against the exact in-memory baselines.

    uv run python -m omnirag.evaluation.retrieval_benchmark [--k 5] [--out x.json]
    # analyzer sweep (drops/recreates the full-text index: test server only, port 7688)
    export OMNIRAG_NEO4J_URI=bolt://localhost:7688
    uv run python -m omnirag.evaluation.retrieval_benchmark \\
        --analyzers standard-no-stop-words,standard,english,classic

Methods, all on the same chunk set and questions as Phases 3–4:
* ``memory:bm25`` / ``memory:dense``: the exact in-memory baselines (Phase 4 numbers);
* ``neo4j:dense``: query embedding + vector index SEARCH (HNSW, quantization NONE);
* ``neo4j:bm25:<analyzer>:<mode>``: full-text index with identifier mode terms / boost /
  require (see ``retrieval/sparse.py``);
* ``neo4j:graph:hops<n>``: entity linking + traversal (Phase 7);
* ``neo4j:dense+bm25[:+graph]:k<k>``: Reciprocal Rank Fusion with RRF constant k (and
  optional per-retriever weights).

Besides the ranking metrics:
* **identifier precision**: for the questions that name a technical identifier (AUR-C103,
  E-7, M3, ...), the share of top-k chunks that contain one of them verbatim;
* **unanswerable**: how many chunks each method returns for the 3 questions without an
  answer (a retriever always returns k chunks unless something like ``require`` stops it);
* **latency** per query (warm), dense split into embedding and search.

Reports per-question recall wins/losses against the in-memory baseline of the same family.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from omnirag.chunking import build_embedder
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.evaluation.chunking_benchmark import DATASET
from omnirag.evaluation.dataset import EvalDataset
from omnirag.evaluation.embedding_benchmark import evaluate
from omnirag.evaluation.retrieval import BM25Index
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore
from omnirag.knowledge_base.schema import ensure_schema
from omnirag.retrieval import build_retrieval
from omnirag.retrieval.fusion import HybridRetriever
from omnirag.retrieval.graph_search import GraphRetriever
from omnirag.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from omnirag.retrieval.sparse import identifiers

MODES = ("terms", "boost", "require")


def _contains_id(content: str, ids: list[str]) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", content, re.I) for i in ids)


def _latency(ms: list[float]) -> dict[str, float]:
    ordered = sorted(ms)
    return {"mean": round(statistics.mean(ms), 1), "p50": round(statistics.median(ms), 1),
            "p95": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 1)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.retrieval_benchmark")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--analyzers", default="", help="comma list; test server only")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-fusion", action="store_true", help="skip graph + RRF (Phase 7)")
    parser.add_argument(
        "--rerank-models",
        default="",
        help="comma list, e.g. bge-reranker-base,bge-reranker-v2-m3 (slow on CPU)",
    )
    parser.add_argument(
        "--rerank-bases",
        default="dense+bm25",
        help="first-stage modes to rerank, e.g. dense+bm25,dense+bm25+graph",
    )
    parser.add_argument("--rerank-depths", default="20", help="comma list of candidate depths")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="eval set (default: chunking_seed; relational_seed for Phase 7)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    config = settings.retrieval_config_id
    analyzers = [a for a in args.analyzers.split(",") if a]
    if analyzers and urlparse(settings.neo4j_uri).port != 7688:
        parser.error("--analyzers recreates the full-text index: only against port 7688")

    store = DocumentStore(settings.data_dir)
    records = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED]
    chunks = [c for r in records for c in store.read_chunks(r.document_id, config)
              if c.retrievable]  # fmt: skip
    index_of = {c.chunk_id: i for i, c in enumerate(chunks)}
    dataset = EvalDataset.load(args.dataset)
    questions = [q for q in dataset.questions if q.answerable]
    unanswerable = [q for q in dataset.questions if not q.answerable]
    with_ids = [(i, identifiers(q.question)) for i, q in enumerate(questions)]
    with_ids = [(i, ids) for i, ids in with_ids if ids]
    print(f"{len(chunks)} chunks of {config} · {len(questions)} questions "
          f"({len(with_ids)} name an identifier) · K={args.k}"
          f" · {settings.neo4j_uri}\n")  # fmt: skip

    results: dict[str, dict[str, Any]] = {}

    def record(name: str, ranked: list[list[int]], ms: list[float], extra=None) -> None:
        r = evaluate(chunks, questions, lambda qi: ranked[qi], args.k)
        top = {i: [chunks[j] for j in ranked[i][: args.k]] for i, _ in with_ids}
        shares = [sum(_contains_id(c.content, ids) for c in top[i]) / max(1, len(top[i]))
                  for i, ids in with_ids]  # fmt: skip
        r["identifier_precision"] = round(statistics.mean(shares), 3) if shares else None
        r["latency_ms"] = _latency(ms)
        r.update(extra or {})
        results[name] = r
        print(f"  {name:<44} R@{args.k}={r['recall']:.3f} MRR={r['mrr']:.3f} "
              f"nDCG={r['ndcg']:.3f} idP={r['identifier_precision']:.3f} "
              f"p50={r['latency_ms']['p50']}ms", flush=True)  # fmt: skip

    # ---- in-memory baselines (exact)
    texts = [c.embedding_text() for c in chunks]
    bm25 = BM25Index(texts)
    ms, ranked = [], []
    for q in questions:
        t = time.perf_counter()
        ranked.append(bm25.search(q.question, args.k))
        ms.append((time.perf_counter() - t) * 1000)
    record("memory:bm25", ranked, ms)

    embedder = build_embedder(settings)
    embedder.embed_query("warm up")  # model load is not query latency
    doc = embedder.embed_documents(texts)  # served from the embedding cache
    ms, ranked = [], []
    for q in questions:
        t = time.perf_counter()
        scores = doc @ embedder.embed_query(q.question)
        ranked.append([int(i) for i in np.argsort(-scores, kind="stable")[: args.k]])
        ms.append((time.perf_counter() - t) * 1000)
    record("memory:dense", ranked, ms)

    # ---- Neo4j retrievers
    service = build_retrieval(settings)
    graph = service.store

    def run(retriever, **kw) -> tuple[list[list[int]], list[float], dict]:
        keys = ("embed_ms", "search_ms", "rerank_ms", "top_score")
        ranked, ms, extra = [], [], {key: [] for key in keys}
        for q in questions:
            res = retriever.retrieve(q.question, args.k, **kw)
            ranked.append([index_of[h.chunk_id] for h in res.hits if h.chunk_id in index_of])
            ms.append(res.latency_ms)
            for key in keys:
                if res.details.get(key) is not None:
                    extra[key].append(res.details[key])
        unans = [retriever.retrieve(q.question, args.k, **kw) for q in unanswerable]
        out = {"unanswerable_hits": {q.id: len(r.hits) for q, r in zip(unanswerable, unans,
                                                                         strict=True)}}  # fmt: skip
        for key, values in extra.items():
            if values and key != "top_score":
                out[f"{key}_p50"] = round(statistics.median(values), 1)
        if extra["top_score"]:  # reranker: does its top score separate answerable questions?
            out["top_score_answerable"] = [round(v, 4) for v in extra["top_score"]]
            out["top_score_unanswerable"] = {q.id: r.details.get("top_score") for q, r in
                                             zip(unanswerable, unans, strict=True)}  # fmt: skip
        return ranked, ms, out

    dense = service.retrievers["dense"]
    dense.retrieve("warm up", 1)
    record("neo4j:dense", *run(dense))

    sweep = analyzers or [settings.graph_fulltext_analyzer]
    for analyzer in sweep:
        if analyzers:
            ensure_schema(graph, vector_dim=settings.embedding_dim, analyzer=analyzer,
                          recreate_fulltext_index=True)  # fmt: skip
        lexical = service.retrievers["bm25"]
        lexical.retrieve("warm up", 1)
        for mode in MODES:
            record(f"neo4j:bm25:{analyzer}:{mode}", *run(lexical, identifier_mode=mode))
    if analyzers:  # leave the configured analyzer in place
        default = settings.graph_fulltext_analyzer
        ensure_schema(graph, vector_dim=settings.embedding_dim, analyzer=default,
                      recreate_fulltext_index=True)  # fmt: skip

    # ---- Phase 7: graph retrieval and RRF fusion (the configured analyzer / identifier mode)
    if not args.no_fusion:
        members = service.retrievers
        depth = settings.retrieval_fusion_depth

        def fused(name: str, parts: dict, rrf_k: int, weights=None) -> None:
            mode = "dense+bm25+graph" if "graph" in parts else "dense+bm25"
            hybrid = HybridRetriever(parts, mode=mode, rrf_k=rrf_k, weights=weights,
                                     depth=depth)  # fmt: skip
            record(name, *run(hybrid))
            hybrid.close()

        for rrf_k in (60, 10):
            fused(f"neo4j:dense+bm25:k{rrf_k}", {n: members[n] for n in ("dense", "bm25")}, rrf_k)
        # Graph retrieval at 0-2 hops, alone and fused with dense + BM25 (k 60 / 10, and a
        # half-weight graph). Earlier variants: docs/benchmarks/graph-variants-*.json.
        for hops in (0, 1, 2):
            g = GraphRetriever(graph, config_id=config, max_hops=hops)
            g.retrieve("warm up", 1)
            ranked, ms, extra = run(g)
            extra["questions_with_graph_hits"] = sum(bool(r) for r in ranked)
            record(f"neo4j:graph:hops{hops}", ranked, ms, extra)
            parts = {"dense": members["dense"], "bm25": members["bm25"], "graph": g}
            for rrf_k in (60, 10):
                fused(f"neo4j:dense+bm25+graph:k{rrf_k}:hops{hops}", parts, rrf_k)
            fused(f"neo4j:dense+bm25+graph:k60:hops{hops}:wgraph=0.5", parts, 60, {"graph": 0.5})

    # ---- Phase 8: cross-encoder reranking of the fused candidates (CPU: slow, opt-in sweep)
    for model in [m for m in args.rerank_models.split(",") if m]:
        reranker = CrossEncoderReranker(
            settings.data_dir / "models", model, max_length=settings.rerank_max_length,
            batch_size=settings.rerank_batch_size, threads=settings.rerank_threads,
            device=settings.rerank_device,
        )  # fmt: skip
        reranker.score("warm up", ["load the weights first"])
        for base_name in [b for b in args.rerank_bases.split(",") if b]:
            base = service.retrievers[base_name]
            for depth in [int(d) for d in args.rerank_depths.split(",") if d]:
                rr = RerankingRetriever(base, reranker, mode=f"{base_name}+rerank", depth=depth)
                ranked, ms, extra = run(rr)
                extra.update(device=reranker.device, load_s=reranker.stats["load_s"])
                record(f"neo4j:{base_name}+rerank:{model}:d{depth}", ranked, ms, extra)
        del reranker
    service.close()

    # ---- per-question wins / losses against the in-memory baseline of the same family
    def diff(name: str, base: str) -> dict[str, list[str]]:
        a, b = results[name]["per_question_recall"], results[base]["per_question_recall"]
        return {"wins": sorted(q for q in a if a[q] > b[q]),
                "losses": sorted(q for q in a if a[q] < b[q])}  # fmt: skip

    best_bm25 = (
        f"neo4j:bm25:{settings.graph_fulltext_analyzer}:{settings.retrieval_identifier_mode}"
    )
    for name in results:
        if not name.startswith("neo4j:"):
            continue
        if name == "neo4j:dense":
            bases = ["memory:dense"]
        elif ":bm25:" in name:
            bases = ["memory:bm25"]
        else:  # graph and fused modes: against both single retrievers
            bases = [b for b in ("neo4j:dense", best_bm25) if b in results]
            if "+rerank:" in name:  # and against its own first stage
                first = name.split("+rerank:")[0]
                first = f"{first}:k60" if first.endswith("bm25") else f"{first}:k60:hops2"
                bases += [first] if first in results else []
        for base in bases:
            results[name]["vs_" + base] = d = diff(name, base)
            print(f"  {name:<44} vs {base}: +{d['wins']} -{d['losses']}")

    if args.out:
        payload = {
            "created": datetime.now(UTC).isoformat(),
            "setup": {"config_id": config, "k": args.k, "chunks": len(chunks),
                      "questions": len(questions), "identifier_questions": len(with_ids),
                      "neo4j_uri": settings.neo4j_uri,
                      "embedding_space": embedder.space.space_id,
                      "analyzers": sweep, "dataset": args.dataset.name},
            "results": results,
        }  # fmt: skip
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
