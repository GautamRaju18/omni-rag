"""BGE-M3 run locally (CPU) for its **sparse** and **multi-vector** outputs.

Ollama's ``/api/embed`` returns only BGE-M3's dense vector (verified 2026-11-05: the response
has a single ``embeddings`` key). The other two outputs need the original model's extra heads
from the Hugging Face release (``sparse_linear.pt``, ``colbert_linear.pt``). One forward pass
of the XLM-RoBERTa encoder gives all three:

* **dense**: the CLS token's hidden state, L2-normalised (what Ollama returns too);
* **sparse** (learned lexical weights): ``relu(sparse_linear(hidden))`` per token, keeping
  the maximum weight per token ID and dropping special tokens. Score = Σ w_q·w_d over shared
  tokens. It's like BM25, but the model learned which tokens matter;
* **multi-vector** (ColBERT): ``colbert_linear(hidden)`` for every token, L2-normalised.
  Score = mean over query tokens of the best-matching document token (late interaction).

This is implemented here (≈60 lines) instead of adding the FlagEmbedding package, and is
verified against the lexical weights and score published on the model card.
Optional: needs the ``layout`` dependency group (torch + transformers, CPU).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from omnirag.chunking.tokenizer import ensure_hf_file
from omnirag.embeddings.base import EmbeddingKind, EmbeddingSpace, SparseVector, validate_dense

REPO = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FILES = {  # SHA-256 from the Hugging Face LFS pointers (config.json: hashed on download)
    "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    "sparse_linear.pt": "45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9",
    "colbert_linear.pt": "19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb",
    "config.json": "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
}
_SPECIAL_IDS = {0, 1, 2, 3}  # <s>, <pad>, </s>, <unk> (XLM-RoBERTa)
_PAD_ID = 1


@dataclass(frozen=True)
class M3Output:
    dense: np.ndarray  # (1024,)
    sparse: SparseVector  # token id -> weight
    colbert: np.ndarray  # (n_tokens, 1024), L2-normalised rows


def colbert_score(query: np.ndarray, doc: np.ndarray) -> float:
    return float((query @ doc.T).max(axis=1).mean()) if len(query) and len(doc) else 0.0


class BGEM3Local:
    def __init__(
        self, models_dir: Path, *, max_length: int = 8192, batch_size: int = 8, threads: int = 8
    ) -> None:
        self.models_dir = models_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.threads = threads
        self._lock = threading.Lock()  # torch modules: one forward pass at a time
        self.stats = {"texts": 0, "seconds": 0.0}

    # --------------------------------------------------------------- spaces
    def _space(self, kind: EmbeddingKind, dim: int) -> EmbeddingSpace:
        return EmbeddingSpace(provider="hf", model="bge-m3", revision=REVISION, kind=kind,
                              dim=dim, max_tokens=self.max_length)  # fmt: skip

    @property
    def space(self) -> EmbeddingSpace:  # DenseEmbedder protocol
        return self._space(EmbeddingKind.DENSE, 1024)

    @property
    def sparse_space(self) -> EmbeddingSpace:
        return self._space(EmbeddingKind.SPARSE, 250002)

    @property
    def multivector_space(self) -> EmbeddingSpace:
        return self._space(EmbeddingKind.MULTIVECTOR, 1024)

    # ----------------------------------------------------------------- model
    @cached_property
    def _model(self) -> dict[str, Any]:
        import torch
        from tokenizers import Tokenizer
        from transformers import XLMRobertaModel
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # "Loading weights" bars would corrupt CLI output
        paths = {f: ensure_hf_file(self.models_dir, REPO, REVISION, f, sha)
                 for f, sha in FILES.items()}  # fmt: skip
        torch.set_num_threads(self.threads)
        encoder = XLMRobertaModel.from_pretrained(paths["config.json"].parent).eval()
        sparse = torch.nn.Linear(1024, 1)
        sparse.load_state_dict(torch.load(paths["sparse_linear.pt"], map_location="cpu",
                                          weights_only=True))  # fmt: skip
        colbert = torch.nn.Linear(1024, 1024)
        colbert.load_state_dict(torch.load(paths["colbert_linear.pt"], map_location="cpu",
                                           weights_only=True))  # fmt: skip
        tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        tokenizer.enable_truncation(self.max_length)
        return {"torch": torch, "encoder": encoder, "sparse": sparse, "colbert": colbert,
                "tokenizer": tokenizer}  # fmt: skip

    def encode(self, texts: Sequence[str]) -> list[M3Output]:
        m = self._model
        torch = m["torch"]
        out: list[M3Output | None] = [None] * len(texts)
        encodings = m["tokenizer"].encode_batch(list(texts))
        order = sorted(range(len(texts)), key=lambda i: len(encodings[i].ids))  # less padding
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
                hidden = m["encoder"](input_ids=ids, attention_mask=mask).last_hidden_state
                dense = hidden[:, 0].numpy()
                weights = torch.relu(m["sparse"](hidden)).squeeze(-1).numpy()
                colbert = m["colbert"](hidden[:, 1:]) * mask[:, 1:, None]
                colbert = torch.nn.functional.normalize(colbert, dim=-1).numpy()
                for row, i in enumerate(idx):
                    n = int(mask[row].sum())
                    sparse: SparseVector = {}
                    for token, w in zip(ids[row, :n].tolist(), weights[row, :n].tolist(),
                                        strict=True):  # fmt: skip
                        if token not in _SPECIAL_IDS and w > 0 and w > sparse.get(token, 0.0):
                            sparse[token] = w
                    out[i] = M3Output(
                        dense=validate_dense(
                            dense[row : row + 1], count=1, dim=1024, source="bge-m3 (hf)"
                        )[0],  # fmt: skip
                        sparse=sparse,
                        colbert=colbert[row, : n - 1].copy(),  # drop the CLS position
                    )
        self.stats["texts"] += len(texts)
        self.stats["seconds"] += time.perf_counter() - started
        return [o for o in out if o is not None]

    # --------------------------------------------------- embedder protocols
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([o.dense for o in self.encode(texts)])

    def embed_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0].dense

    def embed_documents_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        return [o.sparse for o in self.encode(texts)]

    def embed_query_sparse(self, text: str) -> SparseVector:
        return self.encode([text])[0].sparse

    def token_strings(self, sparse: SparseVector) -> dict[str, float]:
        tokenizer = self._model["tokenizer"]
        return {tokenizer.id_to_token(t).lstrip("▁"): round(w, 5) for t, w in sparse.items()}
