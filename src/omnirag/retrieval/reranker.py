"""Cross-encoder reranking with BGE rerankers (Phase 8).

A bi-encoder (bge-m3 dense retrieval) embeds the question and each chunk *separately* and
compares vectors. A **cross-encoder** reads question and chunk *together*, with attention
across both, and outputs one relevance logit. That is much more accurate, but it needs one
forward pass per (question, chunk) pair, so it only re-scores the candidates a cheap
retriever already found (``RerankingRetriever``: fetch ``depth``, re-score, keep ``k``).

* **Models** (pinned revision + SHA-256 of every LFS file, safetensors weights only):
  ``bge-reranker-v2-m3`` (568M, XLM-R large, built on bge-m3, Apache-2.0; the default) and
  ``bge-reranker-base`` (278M, XLM-R base, MIT; smaller and faster).
* **Runtime.** transformers + torch. The project's torch is the CPU build (ADR-0005: the 6
  GB GPU belongs to Ollama, which has no rerank endpoint: measured 2026-11-05, Ollama 0.40.14
  returns 404 for /api/rerank). ``device="auto"`` uses CUDA when a CUDA torch is installed
  and falls back to the CPU otherwise. ``"cuda"`` without CUDA also falls back, with a
  warning, instead of failing a request.
* **Input** is the pair ``(question, chunk context + content)``, tokenized by the model's own
  ``tokenizer.json`` as ``<s> q </s></s> passage </s>``. Only the passage is truncated, to
  ``max_length`` tokens.
* **Scores**: the raw logit and ``sigmoid(logit)`` in [0, 1], the model card's "normalize".
  The sigmoid is a relevance estimate for *this* model, not a calibrated probability.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import structlog

from omnirag.chunking.tokenizer import ensure_hf_file
from omnirag.retrieval.models import (
    RetrievalFilters,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
)

log = structlog.get_logger(module=__name__)

Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class RerankerSpec:
    repo: str
    revision: str
    files: dict[str, str]  # file -> SHA-256 (LFS oid; config.json hashed on first download)
    max_positions: int  # model limit, incl. special tokens
    license: str


MODELS: dict[str, RerankerSpec] = {
    "bge-reranker-v2-m3": RerankerSpec(
        repo="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        files={
            "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
            "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
            "config.json": "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
        },
        max_positions=8192,
        license="apache-2.0",
    ),
    "bge-reranker-base": RerankerSpec(
        repo="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        files={
            "model.safetensors": "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd",
            "tokenizer.json": "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e",
            "config.json": "289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708",
        },
        max_positions=512,
        license="mit",
    ),
}  # fmt: skip
_PAD_ID = 1  # XLM-RoBERTa


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def passage_text(hit: RetrievedChunk) -> str:
    return f"{hit.context}\n\n{hit.content}" if hit.context else hit.content


class CrossEncoderReranker:
    def __init__(
        self,
        models_dir: Path,
        model: str = "bge-reranker-v2-m3",
        *,
        max_length: int = 512,
        batch_size: int = 4,
        threads: int = 8,
        device: Device = "auto",
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"Unknown reranker {model!r}; choose one of {sorted(MODELS)}")
        self.spec = MODELS[model]
        self.name = model
        self.models_dir = models_dir
        self.max_length = min(max_length, self.spec.max_positions)
        self.batch_size = batch_size
        self.threads = threads
        self.requested_device = device
        self._lock = threading.Lock()  # one forward pass at a time on one model
        self.stats = {"pairs": 0, "seconds": 0.0, "load_s": 0.0}

    @cached_property
    def _model(self) -> dict[str, Any]:
        import torch
        from tokenizers import Tokenizer
        from transformers import XLMRobertaForSequenceClassification
        from transformers.utils import logging as hf_logging

        started = time.perf_counter()
        hf_logging.disable_progress_bar()
        paths = {f: ensure_hf_file(self.models_dir, self.spec.repo, self.spec.revision, f, sha)
                 for f, sha in self.spec.files.items()}  # fmt: skip
        device = self.requested_device
        if device in ("auto", "cuda") and not torch.cuda.is_available():
            if device == "cuda":
                log.warning("reranker_cuda_unavailable", fallback="cpu", model=self.name)
            device = "cpu"
        elif device == "auto":
            device = "cuda"
        torch.set_num_threads(self.threads)
        model = XLMRobertaForSequenceClassification.from_pretrained(
            paths["config.json"].parent, torch_dtype=torch.float32
        ).eval().to(device)  # fmt: skip
        tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        tokenizer.no_padding()
        tokenizer.enable_truncation(self.max_length, strategy="only_second")
        self.stats["load_s"] = round(time.perf_counter() - started, 2)
        log.info("reranker_loaded", model=self.name, device=device, s=self.stats["load_s"])
        return {"torch": torch, "model": model, "tokenizer": tokenizer, "device": device}

    @property
    def device(self) -> str:
        return self._model["device"]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw relevance logits, one per passage, in input order."""
        if not passages:
            return []
        m = self._model
        torch = m["torch"]
        encodings = m["tokenizer"].encode_batch([(query, p) for p in passages])
        order = sorted(range(len(passages)), key=lambda i: len(encodings[i].ids))
        logits = [0.0] * len(passages)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                width = max(len(encodings[i].ids) for i in idx)
                ids = torch.full((len(idx), width), _PAD_ID, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    seq = encodings[i].ids
                    ids[row, : len(seq)] = torch.tensor(seq)
                    mask[row, : len(seq)] = 1
                out = m["model"](input_ids=ids.to(m["device"]),
                                 attention_mask=mask.to(m["device"])).logits  # fmt: skip
                for row, i in enumerate(idx):
                    logits[i] = float(out[row, 0])
        self.stats["pairs"] += len(passages)
        self.stats["seconds"] += time.perf_counter() - started
        return logits


class RerankingRetriever:
    """Retrieve ``depth`` candidates with ``base``, re-score them with the cross-encoder."""

    def __init__(self, base: Any, reranker: CrossEncoderReranker, *, mode: RetrievalMode,
                 depth: int = 20) -> None:  # fmt: skip
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.base = base
        self.reranker = reranker
        self.mode = mode
        self.depth = depth

    def retrieve(
        self, query: str, k: int = 5, filters: RetrievalFilters | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        depth = max(self.depth, k)
        first = self.base.retrieve(query, depth, filters)
        rerank_started = time.perf_counter()
        logits = self.reranker.score(query, [passage_text(h) for h in first.hits])
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        order = sorted(range(len(first.hits)), key=lambda i: (-logits[i], first.hits[i].rank))
        hits = []
        for new_rank, i in enumerate(order[:k], start=1):
            hit = first.hits[i]
            sources = dict(hit.sources) or {first.retriever: {"rank": hit.rank,
                                                               "score": hit.score}}  # fmt: skip
            sources["rerank"] = {"rank": new_rank, "score": round(logits[i], 4),
                                 "first_stage_rank": hit.rank}  # fmt: skip
            hits.append(hit.model_copy(update={"rank": new_rank, "retriever": self.mode,
                                               "score": round(sigmoid(logits[i]), 6),
                                               "sources": sources}))  # fmt: skip
        return RetrievalResult(
            query=query, retriever=self.mode, k=k, hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            details={"reranker": self.reranker.name, "device": self.reranker.device,
                     "depth": depth, "candidates": len(first.hits),
                     "rerank_ms": round(rerank_ms, 2),
                     "first_stage_ms": first.latency_ms,
                     "top_score": hits[0].score if hits else None,
                     "first_stage": first.details},
        )  # fmt: skip
