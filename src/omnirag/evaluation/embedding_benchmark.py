"""Compare embedding representations on the seed set (same chunks, questions and metrics).

    uv run python -m omnirag.evaluation.embedding_benchmark [--config multimodal-256-ctx]

Methods (all on the same stored chunk set):
* ``bm25``: lexical baseline (no model)
* ``dense:ollama:bge-m3``: production path (Ollama, GGUF F16)
* ``dense:hf:bge-m3``: the original weights on CPU (fp32): a reference for the Ollama path
* ``sparse:hf:bge-m3``: BGE-M3 learned lexical weights
* ``multivector:hf:bge-m3``: BGE-M3 ColBERT late interaction
* ``dense:ollama:nomic-embed-text``: a different model (768-d, needs query/document
  prefixes), plugged in through the same ``DenseEmbedder`` interface

Timings are measured without the embedding cache, so they're the real encoding cost.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from omnirag.chunking import load_tokenizer
from omnirag.chunking.models import Chunk
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.embeddings.base import sparse_dot
from omnirag.embeddings.bge_m3_local import BGEM3Local, colbert_score
from omnirag.embeddings.ollama import OllamaDenseEmbedder
from omnirag.evaluation.chunking_benchmark import DATASET, METRICS
from omnirag.evaluation.dataset import EvalDataset, EvalQuestion, covered_facts
from omnirag.evaluation.metrics import Unit, ranking_metrics
from omnirag.evaluation.retrieval import BM25Index
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore

Scorer = Callable[[int], list[int]]  # question index -> ranked chunk indexes


def evaluate(chunks: list[Chunk], questions: list[EvalQuestion], rank: Scorer, k: int) -> dict:
    per_q, by_type = [], defaultdict(list)
    for qi, q in enumerate(questions):
        top = [chunks[i] for i in rank(qi)[:k]]
        ranked = [Unit(c.chunk_id, c.token_count, frozenset(covered_facts(c.content, q)))
                  for c in top]  # fmt: skip
        n_relevant = sum(bool(covered_facts(c.content, q)) for c in chunks)
        m = ranking_metrics(ranked, len(q.facts), n_relevant, k)
        per_q.append(m)
        by_type[q.type].append(m["recall"])
    return {
        **{m: round(statistics.mean(r[m] for r in per_q), 3) for m in METRICS},
        "recall_by_type": {t: round(statistics.mean(v), 3) for t, v in sorted(by_type.items())},
        "per_question_recall": {q.id: per_q[i]["recall"] for i, q in enumerate(questions)},
    }


def _dense_rank(doc: np.ndarray, queries: np.ndarray) -> Scorer:
    scores = queries @ doc.T
    return lambda qi: [int(i) for i in np.argsort(-scores[qi], kind="stable")]


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.embedding_benchmark")
    parser.add_argument("--config", default="multimodal-256-ctx")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    counter = load_tokenizer(settings)
    dataset = EvalDataset.load(DATASET)
    questions = [q for q in dataset.questions if q.answerable]
    records = [r for r in store.list_records() if r.status is IngestionStatus.COMPLETED]
    chunks = [c for r in records for c in store.read_chunks(r.document_id, args.config)
              if c.retrievable]  # fmt: skip
    if not chunks:
        print(f"No chunks for {args.config}; run `python -m omnirag.chunking rechunk --all`")
        return 1
    texts = [c.embedding_text() for c in chunks]
    tokens = [counter.count(t) for t in texts]
    print(f"{len(chunks)} chunks of {args.config} ({sum(tokens)} tokens) · {len(questions)} "
          f"questions · K={args.k}\n")  # fmt: skip

    results: dict[str, dict[str, Any]] = {}

    def record(name: str, rank: Scorer, encode_s: float, query_ms: float, bytes_per_chunk: float,
               dim: int | str) -> None:  # fmt: skip
        results[name] = {
            **evaluate(chunks, questions, rank, args.k),
            "encode_s": round(encode_s, 2),
            "chunks_per_s": round(len(chunks) / encode_s, 1) if encode_s else None,
            "query_ms": round(query_ms, 1),
            "bytes_per_chunk": round(bytes_per_chunk),
            "dim": dim,
        }
        r = results[name]
        print(f"  {name:<32} R@{args.k}={r['recall']:.3f} MRR={r['mrr']:.3f} "
              f"enc={r['encode_s']}s query={r['query_ms']}ms", flush=True)  # fmt: skip

    # BM25
    t = time.perf_counter()
    bm25 = BM25Index(texts)
    build = time.perf_counter() - t
    t = time.perf_counter()
    rankings = [bm25.search(q.question, len(chunks)) for q in questions]
    record("bm25", lambda qi: rankings[qi], build,
           (time.perf_counter() - t) * 1000 / len(questions), 0, "-")  # fmt: skip

    # Ollama dense models (no cache: real encoding cost)
    models = [
        ("bge-m3", ("", ""), 1024),
        ("nomic-embed-text:v1.5", ("search_query: ", "search_document: "), 768),
    ]
    for model, prefixes, dim in models:
        embedder = OllamaDenseEmbedder(
            settings.ollama_base_url,
            model,
            expected_dim=dim,
            query_prefix=prefixes[0],
            document_prefix=prefixes[1],
        )
        embedder.embed_query("warm up")  # model load is not encoding cost
        t = time.perf_counter()
        doc = embedder.embed_documents(texts)
        enc = time.perf_counter() - t
        t = time.perf_counter()
        qv = np.stack([embedder.embed_query(q.question) for q in questions])
        qms = (time.perf_counter() - t) * 1000 / len(questions)
        record(f"dense:ollama:{model.split(':')[0]}", _dense_rank(doc, qv), enc, qms, dim * 4, dim)
        if model == "bge-m3":
            ollama_dense = doc

    # BGE-M3 local: one forward pass -> dense, sparse, multivector
    m3 = BGEM3Local(settings.data_dir / "models")
    m3.encode(["warm up"])
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.perf_counter()
    docs = m3.encode(texts)
    enc = time.perf_counter() - t
    t = time.perf_counter()
    qs = m3.encode([q.question for q in questions])
    qms = (time.perf_counter() - t) * 1000 / len(questions)
    dense = np.stack([o.dense for o in docs])
    record("dense:hf:bge-m3", _dense_rank(dense, np.stack([o.dense for o in qs])), enc, qms,
           4096, 1024)  # fmt: skip
    sparse_scores = np.array([[sparse_dot(q.sparse, d.sparse) for d in docs] for q in qs])
    record("sparse:hf:bge-m3", lambda qi: [int(i) for i in np.argsort(-sparse_scores[qi],
           kind="stable")], enc, qms,
           statistics.mean(len(d.sparse) for d in docs) * 8, "250002 (sparse)")  # fmt: skip
    colbert_scores = np.array([[colbert_score(q.colbert, d.colbert) for d in docs] for q in qs])
    record("multivector:hf:bge-m3", lambda qi: [int(i) for i in np.argsort(-colbert_scores[qi],
           kind="stable")], enc, qms,
           statistics.mean(len(d.colbert) for d in docs) * 1024 * 4, "1024 × tokens")  # fmt: skip
    agreement = np.sum(ollama_dense * dense, axis=1)
    meta = {
        "date": datetime.now(UTC).isoformat(),
        "config": args.config,
        "chunks": len(chunks),
        "questions": len(questions),
        "k": args.k,
        "ollama_vs_hf_dense_cosine": {
            "min": round(float(agreement.min()), 5),
            "mean": round(float(agreement.mean()), 5),
        },  # fmt: skip
        "hf_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "hf_rss_before_corpus_mb": round(rss_before / 1024),
        "cpu_threads": m3.threads,
    }

    print(f"\n| Method | Dim | Recall@{args.k} | MRR | nDCG@{args.k} | Hit@{args.k} | "
          f"Encode {len(chunks)} chunks | Chunks/s | Query ms | Bytes/chunk |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['dim']} | {r['recall']:.3f} | {r['mrr']:.3f} | {r['ndcg']:.3f} | "
              f"{r['hit']:.3f} | {r['encode_s']} s | {r['chunks_per_s']} | {r['query_ms']} | "
              f"{r['bytes_per_chunk']} |")  # fmt: skip
    types = sorted({t for r in results.values() for t in r["recall_by_type"]})
    print(f"\n| Method | {' | '.join(types)} |\n|---|{'---|' * len(types)}")
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(f"{r['recall_by_type'][t]:.2f}" for t in types) + " |")
    print(f"\nOllama vs HF dense cosine: {meta['ollama_vs_hf_dense_cosine']} · HF model RSS "
          f"{meta['hf_rss_mb']} MB")  # fmt: skip
    out = args.out or settings.data_dir / "benchmarks" / (
        f"embeddings-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
