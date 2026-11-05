"""Chunking benchmark: every configuration, same documents, same questions, same retrievers.

    uv run python -m omnirag.evaluation.chunking_benchmark                 # default grid
    uv run python -m omnirag.evaluation.chunking_benchmark --llm-context   # + LLM context run

For each configuration: chunk all completed documents, embed the retrievable chunks (bge-m3,
cached), search with dense (exact cosine) and BM25, and score the top K **returned units**.
For parent-child the returned unit is the child's parent, because that's what the LLM would
read. Unanswerable questions are skipped here (nothing to retrieve); generation-side checks
for them come in Phase 12.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from omnirag.chunking import build_embedder, load_tokenizer
from omnirag.chunking.enrichment import LLMContextualizer
from omnirag.chunking.models import Chunk, ChunkingConfig, ChunkingStrategy
from omnirag.chunking.strategies import chunk_document
from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.evaluation.dataset import EvalDataset, EvalQuestion, covered_facts, resolve
from omnirag.evaluation.metrics import Unit, ranking_metrics
from omnirag.evaluation.retrieval import BM25Index, DenseIndex
from omnirag.ingestion.models import IngestionStatus
from omnirag.ingestion.store import DocumentStore

S = ChunkingStrategy
DATASET = Path(__file__).parents[3] / "data" / "eval" / "chunking_seed.json"
METRICS = ("hit", "recall", "precision", "mrr", "ndcg", "context_precision")


def default_grid() -> list[ChunkingConfig]:
    grid = []
    for size in (128, 256, 512):
        for s in (S.FIXED, S.RECURSIVE, S.STRUCTURE, S.TABLE, S.MULTIMODAL, S.SEMANTIC):
            grid.append(ChunkingConfig(strategy=s, chunk_size=size, chunk_overlap=size // 8))
    grid += [
        ChunkingConfig(strategy=S.HIERARCHICAL, parent_size=1024, chunk_size=256, child_size=64),
        ChunkingConfig(strategy=S.HIERARCHICAL, parent_size=2048, chunk_size=512, child_size=128),
        ChunkingConfig(strategy=S.PARENT_CHILD, chunk_size=256, child_size=64, chunk_overlap=16),
        ChunkingConfig(strategy=S.PARENT_CHILD, chunk_size=512, child_size=128, chunk_overlap=32),
        ChunkingConfig(strategy=S.MULTIMODAL, chunk_size=256, contextual_headers=True),
    ]
    return grid


def run_config(
    config: ChunkingConfig,
    docs: list[tuple[Any, list]],
    questions: list[EvalQuestion],
    q_vectors: np.ndarray,
    counter: Any,
    embedder: Any,
    contextualizer: LLMContextualizer | None,
    k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    chunks: list[Chunk] = []
    for record, elements in docs:
        doc_chunks = chunk_document(record, elements, config, counter, embedder)
        if config.contextual_llm and contextualizer is not None:
            contextualizer.enrich(doc_chunks, elements)
        chunks += doc_chunks
    chunk_ms = (time.perf_counter() - started) * 1000

    by_id = {c.chunk_id: c for c in chunks}
    search = [c for c in chunks if c.retrievable]
    returned = [by_id[c.parent_chunk_id] if config.strategy is S.PARENT_CHILD and c.parent_chunk_id
                else c for c in search]  # fmt: skip
    texts = [c.embedding_text() for c in search]
    embed_before = embedder.stats["embedded"]
    vectors = embedder.embed(texts)
    indexes = {"dense": DenseIndex(vectors), "bm25": BM25Index(texts)}

    results: dict[str, Any] = {}
    for name, index in indexes.items():
        per_q, latencies, by_type = [], [], defaultdict(list)
        for qi, q in enumerate(questions):
            t = time.perf_counter()
            hits = index.search(q_vectors[qi] if name == "dense" else q.question, k * 6)
            latencies.append((time.perf_counter() - t) * 1000)
            seen, ranked = set(), []
            for i in hits:  # dedupe parents; keep rank order
                unit = returned[i]
                if unit.chunk_id not in seen:
                    seen.add(unit.chunk_id)
                    ranked.append(Unit(unit.chunk_id, unit.token_count,
                                       frozenset(covered_facts(unit.content, q))))  # fmt: skip
            candidates = {u.chunk_id: u for u in returned}.values()
            n_relevant = sum(bool(covered_facts(u.content, q)) for u in candidates)
            m = ranking_metrics(ranked, len(q.facts), n_relevant, k)
            per_q.append(m)
            by_type[q.type].append(m["recall"])
        results[name] = {
            **{m: round(statistics.mean(r[m] for r in per_q), 3) for m in METRICS},
            "search_ms": round(statistics.mean(latencies), 3),
            "recall_by_type": {t: round(statistics.mean(v), 3) for t, v in sorted(by_type.items())},
        }
    vector_bytes = vectors.nbytes
    text_bytes = sum(len(c.content.encode()) + len((c.context or "").encode()) for c in chunks)
    return {
        "config_id": config.config_id,
        "strategy": config.strategy.value,
        "chunks": len(chunks),
        "indexed": len(search),
        "median_tokens": statistics.median([c.token_count for c in search]) if search else 0,
        "index_kb": round((vector_bytes + text_bytes) / 1024, 1),
        "chunk_ms": round(chunk_ms, 1),
        "new_embeddings": embedder.stats["embedded"] - embed_before,
        **results,
    }


def print_table(rows: list[dict], retriever: str, k: int) -> None:
    print(f"\n### {retriever} retrieval, top {k}\n")
    print(f"| Config | Indexed | Med tok | Index KB | Hit@{k} | Recall@{k} | P@{k} | MRR | "
          f"nDCG@{k} | Ctx prec | Search ms |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r[retriever]
        cells = [r["config_id"], r["indexed"], f"{r['median_tokens']:.0f}", r["index_kb"],
                 *(f"{m[key]:.3f}" for key in METRICS), f"{m['search_ms']:.2f}"]  # fmt: skip
        print("| " + " | ".join(str(c) for c in cells) + " |")


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.chunking_benchmark")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--llm-context", action="store_true",
                        help="also run multimodal-256 with LLM-written context (slow)")  # fmt: skip
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    store = DocumentStore(settings.data_dir)
    counter = load_tokenizer(settings)
    embedder = build_embedder(settings)
    dataset = EvalDataset.load(DATASET)
    docs = [(r, store.read_elements(r.document_id)) for r in store.list_records()
            if r.status is IngestionStatus.COMPLETED]  # fmt: skip
    by_file = {r.source_file: elements for r, elements in docs}
    for q in dataset.questions:
        resolve(q, by_file)  # fails loudly if any evidence quote is missing from the corpus
    questions = [q for q in dataset.questions if q.answerable]
    t = time.perf_counter()
    q_vectors = embedder.embed([q.question for q in questions])
    q_embed_ms = (time.perf_counter() - t) * 1000 / len(questions)

    grid = default_grid()
    contextualizer = None
    if args.llm_context:
        grid.append(ChunkingConfig(strategy=S.MULTIMODAL, chunk_size=256, contextual_llm=True))
        contextualizer = LLMContextualizer(settings.ollama_base_url, settings.generation_model,
                                           counter)  # fmt: skip
    skipped = len(dataset.questions) - len(questions)
    print(
        f"{len(docs)} documents · {len(questions)} answerable questions ({skipped} unanswerable "
        f"skipped) · {len(grid)} configs · query embedding {q_embed_ms:.1f} ms/question"
    )
    rows = []
    for config in grid:
        row = run_config(config, docs, questions, q_vectors, counter, embedder, contextualizer,
                         args.k)  # fmt: skip
        rows.append(row)
        print(f"  {row['config_id']:<32} indexed={row['indexed']:<4} "
              f"chunk={row['chunk_ms']:>8.1f} ms new_emb={row['new_embeddings']:<4} "
              f"dense R@{args.k}={row['dense']['recall']:.3f} "
              f"bm25 R@{args.k}={row['bm25']['recall']:.3f}", flush=True)  # fmt: skip
    for retriever in ("dense", "bm25"):
        print_table(rows, retriever, args.k)
    print(f"\n### Recall@{args.k} by question type (dense)\n")
    types = sorted({t for r in rows for t in r["dense"]["recall_by_type"]})
    print("| Config | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for r in rows:
        rt = r["dense"]["recall_by_type"]
        print(f"| {r['config_id']} | " + " | ".join(f"{rt.get(t, 0):.2f}" for t in types) + " |")

    out = args.out or settings.data_dir / "benchmarks" / (
        f"chunking-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": datetime.now(UTC).isoformat(),
        "dataset": f"{dataset.name} v{dataset.version}",
        "documents": len(docs),
        "questions": len(questions),
        "k": args.k,
        "embedding_model": settings.embedding_model,
        "tokenizer": counter.name,
        "llm_context_model": settings.generation_model if args.llm_context else None,
        "llm_context_stats": contextualizer.stats if contextualizer else None,
    }
    out.write_text(json.dumps({"meta": meta, "results": rows}, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
